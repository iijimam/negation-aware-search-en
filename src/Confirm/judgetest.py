#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple LLM-as-a-Judge runner (JSONL in -> JSONL out)

Input JSONL (1 line = 1 query):
{
  "query_text": "...",
  "query_flags": { "HasOxygenTherapy": {"value":0,...}, ... },
  "ranked_top3": [
     {"DocId": 203, "SectionText": "...", "FlagsJson": [{"FlagName":"HasOxygenTherapy","Value":0}, ...]},
     {"DocId": 20,  ...},
     {"DocId": 12,  ...}
  ]
}

Output JSONL:
{
  "query_text": "...",
  "ranked_top3_docids": [203,20,12],
  "judge_result": {...},
  "meta": {...}
}

Requires:
  pip install openai
Env:
  export OPENAI_API_KEY="..."
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用
MODEL = "gpt-4.1-mini"  #すでに決まった Top3 を、ルールに沿って機械的に判定するため、4.1-miniではないモデルを利用

SYSTEM_PROMPT = """
You are a Judge that evaluates the validity of ranking candidates (top3).

You MUST evaluate candidates using ONLY:
- query_flags.<FlagName>.value
- ranked_top3[i].FlagsJson.<FlagName>.value

You MUST NOT read, interpret, infer, or use:
- query_text
- SectionText
- evidence
- any clinical or medical knowledge

This is a STRICT, MECHANICAL rule-execution task.

==================================================
VALUE DEFINITIONS
==================================================

Each flag value is exactly one of:
- 1    : explicitly present
- 0    : explicitly absent
- null : unknown / not stated

CRITICAL:
- null is NOT a negation
- null is NEVER a contradiction
- You MUST NEVER treat null as 0

==================================================
FLAG CATEGORIES
==================================================

STRONG FLAGS (the ONLY flags that can affect verdict/decision):
- HasICUCare
- HasNPPV
- HasMechanicalVentilation
- HasIntubation
- HasDialysis
- HasVasopressor

WEAK FLAGS (EXPLANATION ONLY; MUST NOT affect verdict/decision):
- HasOxygenTherapy
- HasAntibioticsIV
- HasAntibioticsPO
- HasSteroidSystemic
- HasHFNC
- HasSepsis
- HasShock
- HasAKI
- HasDiabetes
- HasInsulinUse

==================================================
STATUS DETERMINATION (MECHANICAL)
==================================================

For EACH flag, determine status using ONLY the following rules:

1) If query value is null -> status = neutral
2) Else if document value is null -> status = neutral
3) Else if query value == document value -> status = match
4) Else if flag is STRONG AND query=1 AND document=0 -> status = contradict
5) Else -> status = neutral

RULE:
- WEAK flags MUST NEVER output "contradict" (even if values differ)

==================================================
MISMATCH — THE ONLY ALLOWED CONDITION
==================================================

A candidate is "mismatch" IF AND ONLY IF:
- At least ONE STRONG flag has status = contradict
(i.e., STRONG AND query=1 AND doc=0)

IMPORTANT:
- doc=null is NEVER mismatch
- No other condition may cause mismatch

==================================================
VERDICT DETERMINATION (PER CANDIDATE)
==================================================

Use ONLY STRONG flags for verdict.

Let STRONG_EXPLICIT = the set of STRONG flags where query value is 0 or 1.

- mismatch:
    If at least one STRONG flag has status = contradict

- match:
    If NO STRONG contradict exists
    AND STRONG_EXPLICIT is NOT empty
    AND at least ONE flag in STRONG_EXPLICIT has status = match

- partial:
    Otherwise (includes cases where STRONG_EXPLICIT is empty)

NOTE:
- WEAK flags MUST NOT move a candidate from partial->match
- WEAK flags MUST NOT move a candidate to mismatch

==================================================
DECISION (TOP-1 ONLY, NO EXCEPTIONS)
==================================================

- decision.top_doc_id MUST equal ranking[0].doc_id
- decision.is_similar_enough is determined ONLY by ranking[0].verdict:
    - match    -> true
    - mismatch -> false
    - partial  -> true or false allowed

- ranking[1] and ranking[2] MUST NOT affect decision

==================================================
SUMMARY CONSTRAINT
==================================================

decision.summary MUST be EXACTLY one of:
- "match: no STRONG-flag contradiction exists"
- "partial: no STRONG-flag contradiction exists"
- "mismatch: STRONG-flag contradiction detected"

NO additional text.

==================================================
OUTPUT FORMAT
==================================================

Output MUST be valid JSON ONLY.

ranking MUST:
- contain exactly 3 items
- be ordered by rank = 1, 2, 3
- have strictly decreasing relevance scores

reasons:
- up to 5 short strings
- MUST be derived ONLY from (flag name, query value, doc value, status)
- Do NOT mention any text fields.

If ANY rule is violated, you MUST correct the output BEFORE responding.
"""

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "ranking"],
    "properties": {
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": ["top_doc_id","is_similar_enough", "confidence", "summary", "missing_info"],
            "properties": {
                "top_doc_id": {"type": "number"},
                "is_similar_enough": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "summary": {"type": "string"},
                "missing_info": {"type": "array", "items": {"type": "string"}},
            },
        },
        "ranking": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["doc_id", "rank", "relevance", "verdict", "reasons"],
                "properties": {
                    "doc_id": {"type": "number"},
                    "rank": {"type": "number"},
                    "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "verdict": {"type": "string", "enum": ["match", "partial", "mismatch"]},
                    "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                },
            },
        },
    },
}

def build_user_prompt(item: dict) -> str:
    return f"""Judge the following input JSON.

Input:
<<<
{json.dumps(item, ensure_ascii=False)}
>>>

FINAL CHECK (MANDATORY):
- Use ONLY query_flags.<flag>.value and ranked_top3[i].FlagsJson.<flag>.value
- doc=null is NEVER mismatch
- WEAK flags NEVER output "contradict" and NEVER affect verdict/decision
- mismatch requires STRONG AND query=1 AND doc=0
- decision depends ONLY on rank1
- summary MUST be exactly one of the 3 allowed strings
- Output JSON ONLY
"""


def call_judge(
    client: OpenAI,
    model: str,
    item: dict,
    temperature: float = 0.0,
    max_output_tokens: int = 900,
    retries: int = 4,
) -> dict:
    last_err = None
    user_prompt = build_user_prompt(item)

    for attempt in range(1, retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                #  JSON Schemaで出力を制約
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            )

            # 返ってきたテキスト(JSON)をパース
            return json.loads(resp.output_text)

        except Exception as e:
            last_err = e
            time.sleep(min(6.0, 0.6 * (2 ** (attempt - 1))))

    raise RuntimeError(f"Judge failed: {last_err}") from last_err

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {i}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_flagsjson_to_dict(flagsjson: Any) -> Dict[str, Any]:
    """
    入力 flagsjson が
      - flags.v2 形式の dict
      - 旧形式 list[{"FlagName":..., "Value":...}]
      - JSON文字列（dict or list）
    のどれでも受けて、最終的に {"HasX": {"value": ...}, ...} を返す
    """
    if flagsjson is None:
        return {}

    # 1) JSON文字列ならパースを試す
    if isinstance(flagsjson, str):
        s = flagsjson.strip()
        if not s:
            return {}
        try:
            flagsjson = json.loads(s)
        except json.JSONDecodeError:
            # JSON文字列でないなら諦めて空
            return {}

    # 2) すでに dict の場合
    if isinstance(flagsjson, dict):
        # flags.v2 の入れ子（{"flags": {...}}）なら中身を返す
        if "flags" in flagsjson and isinstance(flagsjson["flags"], dict):
            return flagsjson["flags"]
        # すでに {"HasX": {"value": ...}} 形式ならそのまま
        return flagsjson

    # 3) list の場合（旧形式想定）
    if isinstance(flagsjson, list):
        d: Dict[str, Any] = {}
        for it in flagsjson:
            if not isinstance(it, dict):
                continue
            k = it.get("FlagName")
            if not k:
                continue
            d[k] = {"value": it.get("Value")}
        return d

    # 4) その他は空
    return {}

def main() -> int:
    input_file="/opt/src/flag_querytestRerank_results.jsonl"
    output_file="/opt/src/flag_judge_results.jsonl"

    items = read_jsonl(input_file)
    out_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        top3_raw = item.get("ranked_top3")
        if not isinstance(top3_raw, list) or len(top3_raw) != 3:
            raise ValueError(f"Input row {idx}: ranked_top3 must have exactly 3 items.")

        # ===== ここで judge 用に FlagsJson を dict 化する =====
        top3_for_judge = []
        for c in top3_raw:
            c2 = dict(c)
            if isinstance(c.get("FlagsJson"), str):
                print(f"[row {idx}] FlagsJson is str (DocId={c.get('DocId')}), head={c.get('FlagsJson')[:80]}")
            c2["FlagsJson"] = normalize_flagsjson_to_dict(c.get("FlagsJson"))
            top3_for_judge.append(c2)

        #item_for_judge = dict(item)
        #item_for_judge["ranked_top3"] = top3_for_judge
        item_for_judge = {
            "query_flags": item["query_flags"],
            "ranked_top3": [
                {
                    "DocId": c["DocId"],
                    "FlagsJson": c["FlagsJson"],
                }
                for c in top3_for_judge
            ]
        }
        # =====================================================

        temperature = 0.0  # 固定
        judge_result = call_judge(
            client=client,
            model=MODEL,
            item=item_for_judge,
            temperature=temperature,
            max_output_tokens=900,
        )

        out_rows.append({
            "query_text": item.get("query_text",""),
            "ranked_top3_docids": [c.get("DocId") for c in top3_raw],
            "stage1": item.get("stage1", {}),
            "ranked_top3_meta": [
                {
                    "DocId": c.get("DocId"),
                    "score_text": c.get("score_text"),
                    "score_text_norm": c.get("score_text_norm"),
                    "final_score": c.get("final_score"),
                } for c in top3_raw
            ],
            "judge_result": judge_result,
            "meta": {"model": MODEL, "temperature": temperature},
        })

        dec = judge_result.get("decision", {}).get("is_similar_enough")
        print(f"[{idx}/{len(items)}] decision={dec} query={item.get('query_text','')[:30]}...")

    write_jsonl(output_file, out_rows)
    print(f"Saved: {output_file} (n={len(out_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
