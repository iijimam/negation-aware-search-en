# /usr/irissys/bin/irispython /home/irisowner/.local/bin/streamlit run /src/UI_embedded_python/app.py --server.port 8080 --logger.level=debug
#
import streamlit as st
from openai import OpenAI
from typing import Any, Dict, List, Tuple,Optional
import json
import time,datetime
import sys
sys.path+=["/src/UI_embedded_python"]
import search

client = OpenAI()
MODEL = "gpt-4.1-mini"

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
                # Constraining Output with JSON Schema
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            )

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
    Input flagsjson accepts any of the following formats:
      - A dict in flags.v2 format
      - An older format list[{“FlagName”:..., “Value”:...}]
      - A JSON string (dict or list)
    and ultimately returns {“HasX”: {“value”: ...}, ...}
    """
    if flagsjson is None:
        return {}

    # 1) If it's a JSON string, attempt parsing.
    if isinstance(flagsjson, str):
        s = flagsjson.strip()
        if not s:
            return {}
        try:
            flagsjson = json.loads(s)
        except json.JSONDecodeError:
            # If it's not a JSON string, give up and return empty.
            return {}

    # 2) In the case of dict
    if isinstance(flagsjson, dict):
        #  If it's a nested `flags.v2` (`{“flags”: {...}}`), return its contents.
        if "flags" in flagsjson and isinstance(flagsjson["flags"], dict):
            return flagsjson["flags"]
        # If it's already in the format {“HasX”: {“value”: ...}}
        return flagsjson

    # 3) list
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

    # 4) others
    return {}

def excerpt(s: str, max_chars: int = 700) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…(truncated)"

from datetime import date

def fmt_date(d):
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()  # 'YYYY-MM-DD'
    return str(d)

st.set_page_config(page_title="Discharge Summary - Similar Search", layout="wide")
st.title("Similarity Search + Re-ranking + LLM as Judge for Discharge Summaries")


# input
if query := st.chat_input("Please enter a question for similar searches.>>"):
    st.markdown(f"### Input question:\n{query}")
    with st.spinner("Flag extraction + similarity search via LLM, re-ranking in progress... Please wait a moment."):
        ranking_result,search_top3 = search.get_simirality_ranking(query)
        with st.expander("🔍 Vector Search Results (After Relisting): Debug", expanded=False):
            search_map = {
                r["DocId"]: r
                for r in search_top3
            }
            ranked_top3_with_text = []
            for r in ranking_result["ranked_top3"]:
                docid = r["DocId"]
                src = search_map.get(docid, {})

                ranked_top3_with_text.append({
                    "PatientId": src.get("PatientId"),
                    "DischargeDate": fmt_date(src.get("DischargeDate")),
                    "SectionText": src.get("SectionText"),
                    **r,  # judge_top3
                })
            ranking_out={
                "query_text": ranking_result["query_text"],
                "query_flags": ranking_result["query_flags"],
                "stage1": ranking_result["stage1"],
                "ranking":ranked_top3_with_text
            }
            st.write(ranking_out)

    # LLM as a Judge
    with st.spinner("Reviewing... Please wait a moment."):
        #top3
        top3_raw = ranking_result.get("ranked_top3")
        if not isinstance(top3_raw, list) or len(top3_raw) != 3:
            raise ValueError(f"ranked_top3 must have exactly 3 items.")
        out_rows: List[Dict[str, Any]] = []
        #
        top3_for_judge = []
        for c in top3_raw:
            c2 = dict(c)
            if isinstance(c.get("FlagsJson"), str):
                print(f"FlagsJson is str (DocId={c.get('DocId')}), head={c.get('FlagsJson')[:80]}")
            c2["FlagsJson"] = normalize_flagsjson_to_dict(c.get("FlagsJson"))
            top3_for_judge.append(c2)

        item_for_judge = {
            "query_flags": ranking_result["query_flags"],
            "ranked_top3": [
                {
                    "DocId": c["DocId"],
                    "FlagsJson": c["FlagsJson"],
                }
                for c in top3_for_judge
            ]
        }
        # =====================================================

        temperature = 0.0  # fixed
        judge_result = call_judge(
            client=client,
            model=MODEL,
            item=item_for_judge,
            temperature=temperature,
            max_output_tokens=900,
        )

        out_rows.append({
            "query_text": ranking_result.get("query_text",""),
            "ranked_top3_docids": [c.get("DocId") for c in top3_raw],
            "stage1": ranking_result.get("stage1", {}),
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

        # the result of judge
        st.markdown("### 🏆 Review Results (Verification of Ranking Accuracy by LLM)")
        # Displayed Content: Key information extracted from the top DocId
        ranking = judge_result.get("ranking", [])
        if len(ranking) == 0:
            st.write("No review results are available.")

        else:
            is_similar = judge_result.get("decision", {}).get("is_similar_enough")
            if is_similar is True:
                st.success("✅ The ranking is correct.")
            elif is_similar is False:
                st.error("❌ The ranking is incorrect.")
            else:
                st.info("ℹ️ Unable to determine")
            
            st.markdown(f"**{judge_result.get('decision').get('summary')}**")
            top_rank = ranking[0]
            top_docid = top_rank.get("doc_id")
            top_candidate = next((c for c in search_top3 if c["DocId"] == top_docid), None)
            if top_candidate:
                st.markdown(f"#### ❓Question：{query}")
                st.markdown(f"**☆Most Similar Candidate☆ DocId: {top_docid}／{excerpt(top_candidate.get('SectionText'))}**")
                ranktbl = []
                ranktbl.append("Ranking|DocId | Patient ID | Discharge Date | Section Content")
                ranktbl.append("--| -- | -- | -- | --")
                num=0
                for reco in search_top3:
                    num+=1
                    ranktbl.append(
                        f"{num}|{reco.get('DocId')}|{reco.get('PatientId')}|{reco.get('DischargeDate')}|{reco.get('SectionText')}"                        
                    )
                st.markdown("\n".join(ranktbl))
            else:
                st.write("No details were found for the most similar candidate.")

        with st.expander("🔍 Ranking Review Results: Debugging", expanded=False):
            st.write(judge_result)
