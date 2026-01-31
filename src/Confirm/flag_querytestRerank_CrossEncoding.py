# 修正版（ランキング改善版）
# 変更点（重要）：
# - 「矛盾」をランキング段階で強く落とす（queryで明示されたフラグのみ評価）
# - Oxygen は “強フラグ” から外す（Judge方針と整合）
# - doc側が null の場合は「矛盾ではなく中立」（= missing扱いだがペナルティは弱く/0）
# - query側が null は完全に評価対象外（中立）
# - 既存の compare_detail_weighted をシンプルな一致/矛盾スコアに置換（安定・調整容易）
# - build_optional_filters は “queryで=1明示” の救済のみ（既存維持）
#

from sqlalchemy import create_engine, text
import json
from typing import Any, Dict, List, Tuple,Optional
from openai import OpenAI
from sentence_transformers import CrossEncoder
import re

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

client = OpenAI()
MODEL = "gpt-4.1-mini"

TOPK = 100


# ===== LLM抽出プロンプト =====
SYSTEM_PROMPT = """You are an information extraction system that extracts structured clinical flags
ONLY based on what is explicitly stated in the input medical text.

Your task is to extract predefined flags from hospital course notes or query text.
Output MUST be a single JSON object and nothing else.
If something is not explicitly stated, set its value to null.

==================================================
CORE PRINCIPLES (NO INFERENCE — MOST IMPORTANT)
==================================================

- ZERO inference, guessing, or completion based on clinical knowledge.
- If something is NOT explicitly written in the text, value MUST be null.
- Do NOT infer from other flags.
  - Example: "No ICU admission" does NOT imply "no intubation".
  - Example: "Oxygen not required" does NOT imply "no HFNC / no NPPV / no intubation / no mechanical ventilation".
- Negation NEVER propagates across flags.
- Absence of mention ≠ negation.
  - "not mentioned", "no evidence", "unclear" → value = null (NOT 0).

==================================================
RULES FOR value = 1 / 0 / null
==================================================

- value = 1 (affirmed)
  - Only when the intervention/condition is explicitly stated as present.
- value = 0 (negated)
  - ONLY when the SAME flag is explicitly negated in the text.
  - Implicit negation or indirect reasoning is forbidden.
- value = null (unknown)
  - Use when the flag is not explicitly affirmed or explicitly negated.

==================================================
EVIDENCE RULES (STRICT)
==================================================

- When value is 1 or 0, you MUST include evidence.
- Evidence MUST be a direct quote (1–2 continuous sentences max) from the input text.
- Do NOT paraphrase, summarize, merge sentences, or use "...".
- If value = null: evidence MUST be null (never include supporting quotes for suspected/possible mentions).
  - polarity = "unknown"
  - evidence = null
  - confidence = 0.0 (default)
- Evidence MUST be a single minimal phrase whenever possible.
- Do NOT include multiple sentences unless absolutely unavoidable.
- Prefer the shortest exact span that directly supports the value.

==================================================
CONFIDENCE
==================================================

- confidence range: 0.0 – 1.0
- 1.0: direct, explicit statement
- <1.0: wording is indirect but still explicit
- Do NOT justify confidence using inference.

==================================================
NOTE FIELD
==================================================

- Use note ONLY when strictly necessary to explain ambiguity.
- Do NOT write reasoning, assumptions, or negative explanations like:
  - "not mentioned"
  - "probably"
  - "therefore"
- Keep note very short.

==================================================
SCOPE (MANDATORY)
==================================================

You MUST output scope for every flag:

- inpatient: occurred during hospitalization
- history: pre-existing or prior to admission
- discharge_plan: explicitly planned after discharge
- unknown: cannot be determined

Rules:
- Default scope is inpatient unless future wording is explicit.
- Use discharge_plan ONLY when future intent is clearly written
  (e.g., "will continue after discharge", "prescribed at discharge").

If value = null, scope MUST still follow the default scope rule:
set scope = "inpatient" unless the text explicitly indicates "history" or "discharge_plan";
do NOT set scope to "unknown" just because value is null.
For HasDiabetes: set scope="history" ONLY when diabetes is explicitly stated; if value=null and diabetes is not explicitly mentioned, set scope="unknown" (do NOT default to history).

==================================================
CRITICAL NEGATION SAFEGUARDS
==================================================

# Flag-specific negation only
- value=0 requires explicit negation of THAT EXACT FLAG.
- Negation does NOT propagate to other flags.

Examples:
- "oxygen therapy was not required"
  - HasOxygenTherapy = 0
  - HFNC / NPPV / Intubation / MV = null
- "ICU admission was not required"
  - HasICUCare = 0
  - Intubation / Vasopressor = null

==================================================
HANDLING "ROOM AIR"
==================================================

- "on room air" / "maintained on room air" MAY be used ONLY for:
  - HasOxygenTherapy = 0
    when it clearly describes the clinical course
    (e.g., "remained stable on room air throughout hospitalization")
- "room air" MUST NOT be used to negate:
  - HasHFNC
  - HasNPPV
  - HasIntubation
  - HasMechanicalVentilation
  - HasICUCare
  - HasVasopressor
- If those are not explicitly mentioned → value = null

==================================================
SYSTEMIC STEROIDS (VERY IMPORTANT)
==================================================

- HasSteroidSystemic refers ONLY to systemic steroids
  (oral or IV):
  - prednisolone / PSL
  - methylprednisolone
  - hydrocortisone
  - dexamethasone
- The following are NOT systemic steroids:
  - inhalation therapy
  - nebulizer therapy
  - inhaled bronchodilators
  - inhaled corticosteroids (ICS)

Rules:
- If ONLY inhalation therapy is mentioned:
  - HasSteroidSystemic = null
- Set HasSteroidSystemic = 0 ONLY with explicit statements like:
  - "no systemic steroids"
  - "systemic steroids were not used"
  - "steroids were not administered" (clearly systemic)

==================================================
SHOCK FLAG
==================================================

- HasShock = 1 ONLY if explicitly stated:
  - "shock"
  - "circulatory shock"
  - "shock state"
- Hypotension or elevated lactate ALONE is NOT sufficient.
- Absence of the word "shock" is NOT a reason for 0.
- Set HasShock = 0 ONLY with explicit denial:
  - "no shock"
  - "not in shock"
  - "shock was ruled out"

==================================================
AKI FLAG
==================================================

- HasAKI = 1 ONLY if explicitly stated:
  - "AKI"
  - "acute kidney injury"
- Renal dysfunction or nephrology consult alone → null
- Explicit denial required for value=0.

==================================================
SEPSIS FLAG (CLARIFIED)
==================================================

- HasSepsis = 1 ONLY when sepsis is explicitly stated as a diagnosis.
- If sepsis is described as "suspected", "suggested", "possible", or similar:
  - HasSepsis = null (NOT 1)
  - You may add a short note such as "suspected only".
- HasSepsis = 0 is allowed ONLY when the text explicitly states:
  "no sepsis", "sepsis was ruled out", "sepsis was excluded", or equivalent direct denial.
- Statements about pneumonia not being the primary cause, alternative diagnoses,
  or non-infectious etiologies MUST NOT be interpreted as sepsis negation.
  In such cases, set HasSepsis = null.


==================================================
DICTIONARY (EXPLICIT TERMS ONLY — NO INFERENCE)
==================================================

HasOxygenTherapy:
- 1: oxygen therapy, nasal cannula, mask, reservoir, O2 X L/min
- 0: oxygen not required, no oxygen therapy, managed on room air (course-level)

HasHFNC:
- 1: HFNC, high-flow nasal cannula, high-flow oxygen
- 0: no HFNC, HFNC not initiated

HasNPPV:
- 1: NPPV, CPAP, BiPAP, non-invasive ventilation
- 0: no NPPV, NPPV not initiated

HasIntubation:
- 1: intubation, endotracheal intubation
- 0: no intubation

HasMechanicalVentilation:
- 1: mechanical ventilation, ventilator management
- 0: no mechanical ventilation

HasTracheostomy:
- 1: tracheostomy
- 0: no tracheostomy

HasICUCare:
- 1: ICU admission, ICU care, ICU-level management
- 0: ICU admission not required, no ICU admission

HasSepsis:
- 1: sepsis
- 0: no sepsis (explicit only)

HasShock:
- see Shock rules above

HasVasopressor:
- 1: vasopressors, norepinephrine, dopamine, vasopressin
- 0: vasopressors not used

HasAKI:
- see AKI rules above

HasDialysis:
- 1: dialysis, CRRT, CHDF, HD, HDF
- 0: no dialysis

HasDiabetes:
- 1: diabetes mellitus, DM
- 0: no diabetes

HasInsulinUse:
- 1: insulin started, insulin therapy initiated
- 0: insulin not used, insulin discontinued

HasAntibioticsIV:
- value = 1 ONLY when intravenous administration is explicitly stated
  (e.g., "intravenous antibiotics", "IV antibiotics").
- Generic phrases such as "antibiotic treatment" or "improved with antibiotics"
  WITHOUT IV specification → value = null.

HasAntibioticsPO:
- value = 0 ONLY with explicit denial of oral antibiotics
  (e.g., "no oral antibiotics", "oral antibiotics were not used").
- Statements about treatment priority or primary management
  MUST NOT be interpreted as oral antibiotic negation.

==================================================
ANTIBIOTICS PO SCOPE RULE (CLARIFIED)
==================================================

- The phrase "switched to oral therapy" ALONE implies inpatient treatment.
- Set scope = "discharge_plan" ONLY when continuation after discharge is explicitly stated
  (e.g., "discharged on oral antibiotics", "continue oral antibiotics after discharge").

==================================================
OUTPUT CONSTRAINTS
==================================================

- Output ONLY a single JSON object.
- Do NOT output explanations or text outside JSON.
- All specified flags MUST be included.
- JSON must start with { and end with }.
"""

# 返すdictの形式（このキー名のまま）
OUTPUT_FORMAT1=f"""
{{
  "schema_version": "flags.v2",
  "doc_type": "hospital_course",
  "flags": {{
    "<FlagName>": {{
      "value": 1 or 0 or null,
      "polarity": "affirmed" or "negated" or "unknown",
      "evidence": "<Excerpt from basis. If value is None, null is acceptable>",
      "confidence": 0.0-1.0,
      "scope": "inpatient" or "history" or "discharge_plan" or "unknown",
      "note": "<If you're unsure, keep it short. If not, leave it blank.>"
    }}
  }}
}}
"""

OUTPUT_FORMAT2=f"""
{{
  "schema_version": "flags.v2",
  "doc_type": "query",
  "flags": {{
    "<FlagName>": {{
      "value": 1 or 0 or null,
      "polarity": "affirmed" or "negated" or "unknown",
      "evidence": "<Excerpt from basis. If value is None, null is acceptable>",
      "confidence": 0.0-1.0,
      "scope": "inpatient" or "history" or "discharge_plan" or "unknown",
      "note": "<If you're unsure, keep it short. If not, leave it blank.>"
    }}
  }}
}}
"""

FLAGNAME="""
# Extraction Target Flag (Output everything with this key name intact)
- HasOxygenTherapy
- HasHFNC
- HasNPPV
- HasIntubation
- HasMechanicalVentilation
- HasTracheostomy
- HasICUCare
- HasSepsis
- HasShock
- HasVasopressor
- HasAKI
- HasDialysis
- HasDiabetes
- HasInsulinUse
- HasAntibioticsIV
- HasAntibioticsPO
- HasSteroidSystemic
"""

#-----------------------------------
# LLMを利用して重要用語とその状態を抽出
#-----------------------------------
def extract_flags(input_text: str, which: str) -> Dict[str, Any]:
    if which == "query":
        system_prompt = f"{SYSTEM_PROMPT}\n{OUTPUT_FORMAT2}\n{FLAGNAME}\n"
    else:
        raise ValueError("extract_flags(which) は 'query' のみ想定（ランキング用）")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        max_tokens=1500,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content.strip())

#-----------------------------------
# DB FlagsJson が
#  - {"flags": {...}} wrapper
#  - {...} 直下
#  - JSON文字列
# のどれでも flags辞書（HasXXX -> {value...}）を返す
#-----------------------------------
def normalize_flags_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}
    if "flags" in obj and isinstance(obj["flags"], dict):
        return obj["flags"]
    if any(k.startswith("Has") for k in obj.keys()):
        return obj
    return {}

#-----------------------------------
# LLMで抽出したフラグの情報全体（Dict）を返す
#-----------------------------------
def _get_flag_obj(flags: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = flags.get(key, {})
    return v if isinstance(v, dict) else {}

#-----------------------------------
# LLMで抽出したフラグのvalue（1/0/null）を返す
#-----------------------------------
def _get_value(flags: Dict[str, Any], key: str):
    return _get_flag_obj(flags, key).get("value", None)

#-----------------------------------
# LLMで抽出したフラグのconfidenceの値を返す
#-----------------------------------
def _get_conf(flags: Dict[str, Any], key: str) -> float:
    c = _get_flag_obj(flags, key).get("confidence", 0.0)
    try:
        return float(c)
    except Exception:
        return 0.0

#-----------------------------------
# LLMで抽出したフラグのevidenceを返す（なければ空文字）
#-----------------------------------
def _get_evidence(flags: Dict[str, Any], key: str) -> str:
    ev = _get_flag_obj(flags, key).get("evidence", "")
    if ev is None:
        return ""
    return str(ev)

#-----------------------------------
# value を null に落とす（Queryガード用）
#-----------------------------------
def _set_null(flags: Dict[str, Any], key: str, note: str = "") -> None:
    obj = _get_flag_obj(flags, key)
    obj["value"] = None
    obj["polarity"] = "unknown"
    obj["evidence"] = None
    obj["confidence"] = 0.0
    # scope は残す（抽出結果の情報としては有用）
    if note:
        obj["note"] = note
    flags[key] = obj


# "Strong negation" — allowed to adopt value=0 ONLY when:
#   (a) evidence has a strong negation expression AND
#   (b) the target concept is explicitly mentioned (see mention dictionary)
STRONG_NEG_PAT = re.compile(
    r"(?i)\b(?:"
    # explicit denial / absence
    r"no|not|without|denies?|declines?|refused?|"
    r"absent|negative for|ruled out|excluded|"
    # not required / not needed
    r"not required|unnecessary|not indicated|"
    r"did not require|no need for|"
    # not performed / not initiated
    r"not performed|wasn't performed|were not performed|"
    r"not done|wasn't done|were not done|"
    r"not initiated|wasn't initiated|were not initiated|"
    r"not started|wasn't started|were not started|"
    # not used / not administered / not given / not provided
    r"not used|wasn't used|were not used|"
    r"not administered|wasn't administered|were not administered|"
    r"not given|wasn't given|were not given|"
    r"not provided|wasn't provided|were not provided|"
    # stopped / discontinued / withheld
    r"stopped|discontinued|held|withheld|deferred|"
    # room air course-level negation (oxygen only; do NOT propagate)
    r"on room air|remained on room air|maintained on room air"
    r")\b"
)

# "Not the main cause / not primary" — NOT a negation; should force NULL (do not filter to 0)
RELATIVE_NOT_MAIN_PAT = re.compile(
    r"(?i)\b(?:"
    r"not (?:the )?main|not (?:the )?primary|not the principal|"
    r"not the predominant|not the leading|"
    r"secondary|adjunctive|supportive|"
    r"less likely|unlikely to be the primary|"
    r"was not considered the primary cause|"
    r"not considered primary"
    r")\b"
)

# ============================================================
# Mention dictionary (English version)
#   - Keep this as the only domain-specific swap point.
#   - Use simple substring checks (short texts).
# ============================================================

FLAG_MENTION_KEYWORDS: Dict[str, List[str]] = {
    "HasOxygenTherapy": [
        "oxygen", "o2", "nasal cannula", "nc", "face mask", "mask", "non-rebreather",
        "room air"
    ],
    "HasHFNC": ["hfnc", "high-flow", "high flow", "high-flow nasal cannula"],
    "HasNPPV": ["nppv", "cpap", "bipap", "non-invasive ventilation", "noninvasive ventilation"],
    "HasIntubation": ["intubation", "endotracheal", "et tube", "ett"],
    "HasMechanicalVentilation": ["mechanical ventilation", "ventilator", "ventilation", "mv"],
    "HasTracheostomy": ["tracheostomy", "trach"],
    "HasICUCare": ["icu", "intensive care"],
    "HasSepsis": ["sepsis", "septic"],
    "HasShock": ["shock", "circulatory shock", "shock state"],
    "HasVasopressor": ["vasopressor", "norepinephrine", "noradrenaline", "dopamine", "vasopressin", "epinephrine"],
    "HasAKI": ["aki", "acute kidney injury"],
    "HasDialysis": ["dialysis", "crrt", "chdf", "hd", "hdf", "hemodialysis", "haemodialysis"],
    "HasDiabetes": ["diabetes", "dm", "diabetes mellitus", "type 1", "type 2"],
    "HasInsulinUse": ["insulin"],
    "HasAntibioticsIV": ["iv antibiotics", "intravenous antibiotics", "intravenous", "iv"],
    "HasAntibioticsPO": ["oral antibiotics", "po antibiotics", "oral", "po", "by mouth"],
    "HasSteroidSystemic": ["prednisolone", "psl", "methylprednisolone", "hydrocortisone", "dexamethasone", "systemic steroid", "systemic steroids"],
}

# ============================================================
# Decision function: adopt (force) 0 for filtering (WHERE)
# ============================================================

STRONG_FLAGS = [
    "HasICUCare",
    "HasNPPV",
    "HasMechanicalVentilation",
    "HasIntubation",
    "HasDialysis",
    "HasVasopressor",
]
MID_FLAGS = ["HasHFNC"]
WEAK_FLAGS = [
    "HasOxygenTherapy",
    "HasSepsis",
    "HasShock",
    "HasAKI",
    "HasDiabetes",
    "HasInsulinUse",
    "HasAntibioticsIV",
    "HasAntibioticsPO",
    "HasSteroidSystemic",
]

#-----------------------------------
# 第1引数の文字列にFLAG_MENTION_KEYWORDSの指定フラグの値が含まれているかどうか
#-----------------------------------
def _contains_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for kw in keywords:
        if kw and kw.lower() in t:
            return True
    return False

#-----------------------------------
# value=0 を採用してよいか（一般ルール）
# 強い否定であるかどうかLLMチェック後のconfidence値でチェック
#-----------------------------------
def _should_accept_negation(flag_name: str, flag_obj: Dict[str, Any], full_text: str) -> bool:
    if not isinstance(flag_obj, dict):
        return False

    v = flag_obj.get("value", None)
    if v != 0:
        return False

    ev = str(flag_obj.get("evidence") or "")

    # 相対評価/主ではないは「否定」ではないので落とす
    if ev and RELATIVE_NOT_MAIN_PAT.search(ev):
        return False

    # evidence に強い否定がない 0 は採用しない（保守的）
    if ev and not STRONG_NEG_PAT.search(ev):
        return False

    # mention がなければ 0 を採用しない（0の誤爆を防ぐ）
    kws = FLAG_MENTION_KEYWORDS.get(flag_name, [])
    if kws and not (_contains_any(ev, kws) or _contains_any(full_text, kws)):
        return False

    return True

#-----------------------------------
# 質問文（Query）のフラグからWHERE作成時の情報を落としすぎないようにするための安全策
# - value=1 は基本そのまま採用（=肯定は誤爆してもその他処理で救えることが多い）
# - value=0 は誤爆が致命傷になりやすいので、一般ルールで厳格にガードしてダメなら null へ
#-----------------------------------
def postprocess_query_flags(query_text: str, flags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flags, dict):
        return {}

    for k, obj in list(flags.items()):
        if not (isinstance(k, str) and isinstance(obj, dict)):
            continue
        if obj.get("value", None) == 0:
            if not _should_accept_negation(k, obj, query_text):
                _set_null(flags, k, note="Queryガード: 0(否定)の根拠が弱い/mention不足/相対表現のためnullへ")

    return flags

#-----------------------------------
# 質問文（Query）のフラグからWHERE作成
# フラグは OR で条件指定する
#-----------------------------------
def build_optional_filters(query_text: str, query_flags: Dict[str, Any]) -> str:
    q = normalize_flags_dict(query_flags)
    conds = []

    # ---- Strong flags: 1 は必須条件として採用 ----
    for k in STRONG_FLAGS:
        if _get_value(q, k) == 1:
            conds.append(f"c.{k} = 1")

    # ---- Strong flags: 0 も（厳格ガード通過時のみ）採用 ----
    for k in STRONG_FLAGS:
        obj = q.get(k, {})
        if isinstance(obj, dict) and obj.get("value") == 0:
            conf = float(obj.get("confidence", 0.0))
            # “0の誤爆は致命傷”方針なら、oxygenと同程度に厳しく
            if conf >= 0.9 and _should_accept_negation(k, obj, query_text):
                conds.append(f"c.{k} IS NULL OR c.{k} = 0")

    # ---- Oxygen: 0 は今の特例を維持 ----
    oxy = q.get("HasOxygenTherapy", {})
    if isinstance(oxy, dict) and oxy.get("value") == 0:
        conf = float(oxy.get("confidence", 0.0))
        if conf >= 0.9 and _should_accept_negation("HasOxygenTherapy", oxy, query_text):
            conds.append("c.HasOxygenTherapy IS NULL OR c.HasOxygenTherapy = 0")

    if not conds:
        return ""

    # 重要：AND
    return "(" + " AND ".join(conds) + ")"

#-----------------------------------
# 2回の類似検索結果と質問文を利用して
# 特定用語に対する否定（0）が質問文に食い違いがないかのチェック
# (絶対に入れたくない矛盾を除外する：WHEREで取り除けないもの、マージ後の混入を除外)
# 質問文が強く否定(0)なのにDB側で肯定(1)している場合、結果から取り除く
# （LLMが回答したconfidenceの値を利用：パラメータでどの値以上なら取り除くを指定可）
# DB側が value==1 でも confidence が低い場合は誤爆の可能性があるため除外しない
#-----------------------------------
def hard_exclude_contradictions(
    results,
    query_text,
    query_flags,
    hard_flags=None,
    conf_th=0.9,         # query側
    doc_conf_th=0.8,     # doc側
    bypass_accept_if_query_conf_ge=0.99,  # ★追加：queryが超高信頼なら_acceptチェックをバイパス
):
    q = normalize_flags_dict(query_flags)
    hard_flags = hard_flags or ["HasOxygenTherapy"]

    out = results
    for flag_name in hard_flags:
        fq = q.get(flag_name, {})
        if not (isinstance(fq, dict) and fq.get("value", None) == 0):
            continue

        q_conf = float(fq.get("confidence", 0.0) or 0.0)
        if q_conf < conf_th:
            continue

        # 超高信頼なら _should_accept_negation を必須にしない
        if q_conf < bypass_accept_if_query_conf_ge:
            if not _should_accept_negation(flag_name, fq, query_text):
                continue

        kept = []
        for r in out:
            d = normalize_flags_dict(r.get("FlagsJson"))
            dv = _get_value(d, flag_name)

            d_obj = d.get(flag_name, {})
            d_conf = 0.0
            if isinstance(d_obj, dict):
                try:
                    d_conf = float(d_obj.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    d_conf = 0.0

            # 確からしい「1」だけ落とす
            if dv == 1 and d_conf >= doc_conf_th:
                continue

            kept.append(r)

        out = kept

    return out



#-----------------------------------
# Embedding
#-----------------------------------
def text_embedding(text_: str):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text_)
    return resp.data[0].embedding

#-----------------------------------
# 類似検索結果の出力作成（FlagJsonの中身をDictにする）
#-----------------------------------
def make_output_topk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        # rerank側で統一的に参照できるよう、ベクトル類似度を vec_score にも入れる
        if "vec_score" not in rr and "score_text" in rr:
            rr["vec_score"] = rr.get("score_text")
        if isinstance(rr.get("FlagsJson"), str):
            try:
                rr["FlagsJson"] = json.loads(rr["FlagsJson"])
            except Exception:
                rr["FlagsJson"] = None
        out.append(rr)
    return out


#-----------------------------------
# IRISの類似検索
#-----------------------------------
def search_topk(query_vec: str, topn: int, where_extra: str = "") -> List[Dict[str, Any]]:
    sql = f"""
SELECT TOP :topN
  c.DocId, c.SectionText, c.FlagsJson,
  VECTOR_COSINE(c.Embedding, TO_VECTOR(:query_vec, FLOAT, 1536)) AS score_text,
  d.PatientId,d.DischargeDate
FROM Demo.DischargeSummaryChunk c, Demo.DischargeSummaryDoc d
WHERE c.DocId=d.DocId AND c.SectionType = 'hospital_course'
  {("AND " + where_extra) if where_extra else ""}
ORDER BY score_text DESC
"""
    print(sql)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"topN": topn, "query_vec": query_vec}).mappings().all()
    return make_output_topk([dict(r) for r in rows])

#-----------------------------------
# 2つの類似検索結果のマージ（DocIdをキーにチェック）
#-----------------------------------
def merge_unique_by_docid(*lists_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for lst in lists_:
        for r in lst:
            docid = r.get("DocId")
            if docid is None:
                continue
            docid = str(docid)   # 型を正規化
            if docid in seen:
                continue
            seen.add(docid)
            out.append(r)
    return out


#-----------------------------------
# リランキング
#-----------------------------------
Candidate = Dict[str, Any]
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        model_name 例:
          - "BAAI/bge-reranker-v2-m3"
          - "jinaai/jina-reranker-v2-base-multilingual"
          - "Qwen/Qwen3-Reranker-0.6B" など（名称は環境に合わせて）
        """
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        text_key: str = "SectionText",
        batch_size: int = 16,
        top_n: int = 80,
    ) -> List[Candidate]:
        # 1) rerank対象を絞る（候補が多いと遅くなるので）
        cands = sorted(candidates, key=lambda x: float(x.get("vec_score", 0.0)), reverse=True)
        pool = cands[: min(top_n, len(cands))]

        # 2) (query, doc) のペアを作る
        pairs: List[Tuple[str, str]] = [(query, str(c.get(text_key, ""))) for c in pool]

        # 3) Cross-Encoderでスコア推論
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # 4) スコアを付与して並べ替え
        for c, s in zip(pool, scores):
            c["ce_score"] = float(s)

        pool.sort(key=lambda x: float(x["ce_score"]), reverse=True)
        return pool

# ===== 28クエリ出力（judge入力ファイルを作る） =====
def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

#-----------------------------------
# フラグに対する値の数を取得
#-----------------------------------
def count_doc_flag_value(results, flag_name, value):
    n = 0
    for r in results:
        d = normalize_flags_dict(r.get("FlagsJson"))
        if _get_value(d, flag_name) == value:
            n += 1
    return n

def test28query():
    query1=[
         "The patient had fever and cough but did not require supplemental oxygen during the hospital course. The pneumonia was relatively mild.",
    ]
    query28 = [
 "The patient had fever and cough but did not require supplemental oxygen during the hospital course. The pneumonia was relatively mild.",
 "The patient was admitted for community-acquired pneumonia but remained on room air throughout hospitalization; supplemental oxygen was not required.",
 "The patient was admitted for pneumonia; respiratory status was stable and no oxygen therapy was administered.",
 "The patient was admitted with dyspnea and received supplemental oxygen, but ICU-level care was not necessary.",
 "Hypoxemia was noted on admission and was managed with nasal cannula oxygen; the patient was managed on the general ward.",
 "Supplemental oxygen was provided, but the patient did not deteriorate and did not require ICU admission.",
 "The patient was admitted with severe pneumonia and required ICU-level care due to worsening respiratory status.",
 "The patient developed respiratory failure after admission and required management in the intensive care unit.",
 "Community-acquired pneumonia progressed to severe disease, requiring comprehensive ICU management.",
 "During treatment for pneumonia, glycemic control for diabetes mellitus became necessary and insulin therapy was administered.",
 "With a history of diabetes mellitus, glycemic management was implemented in parallel with treatment of the infection.",
 "In this patient with diabetes mellitus, inpatient management included initiation of insulin therapy.",
 "Antibiotics were started intravenously and, after clinical improvement, transitioned to oral therapy.",
 "The patient was treated with intravenous antibiotics during hospitalization and was subsequently switched to oral agents.",
 "Intravenous antibiotics were initiated, and the patient was transitioned to oral antibiotics before discharge.",
 "Symptoms improved with treatment, and the patient was discharged home in stable condition.",
 "Inflammatory markers improved, and the patient was discharged home without issues.",
 "After confirming clinical improvement, the patient was discharged home.",
 "The patient was admitted with dyspnea and received supplemental oxygen for desaturation. Chest X-ray showed congestion, and the condition improved with diuretics. Antibiotic therapy for pneumonia was not the main focus.",
 "The patient was brought in by ambulance with orthopnea; hypoxemia persisted despite oxygen therapy, so NPPV was initiated. Respiratory status improved with intravenous diuretics, and acute heart failure exacerbation—not infection—was the primary cause.",
 "The patient was admitted with cough and dyspnea and received supplemental oxygen. The condition improved with bronchodilators and steroids; imaging showed no clear pneumonia, and COPD exacerbation was the predominant issue.",
 "The patient was admitted with worsening respiratory status and started on supplemental oxygen for desaturation. NPPV was initiated for hypercapnia, and treatment targeted COPD exacerbation more than antibiotic therapy.",
 "The patient was admitted with high fever and cough and received short-term oxygen for desaturation. A rapid test was positive for influenza, and the patient improved primarily with supportive care rather than antibiotics.",
 "The patient was admitted with cough, sputum, and fever, but chest imaging did not clearly show pneumonia. The patient improved with symptomatic treatment; supplemental oxygen and ICU care were not required.",
 "The patient was admitted with worsening exertional dyspnea and received supplemental oxygen for desaturation. There were few signs of infection; treatment was centered on steroids (antibiotics were adjunctive).",
 "After choking, the patient developed fever and cough and was admitted. Intravenous antibiotics were started for aspiration pneumonia, and swallowing evaluation with diet modification was performed. Antibiotics were later transitioned to oral therapy after improvement.",
 "The patient was admitted with high fever and elevated inflammatory markers and started on intravenous antibiotics. Respiratory symptoms were mild with little evidence of pneumonia; pyelonephritis was the primary cause. After improvement, antibiotics were switched to oral therapy and the patient was discharged.",
 "The patient was admitted with high fever and altered mental status, and intravenous antibiotics and fluids were initiated. Supplemental oxygen was given for desaturation, but the suspected source was outside the lungs; the patient was treated for sepsis.",
    ]

    result28 = []
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", device="cpu")
    for query_text in query28:
        # 1) ベクトル検索 TOP100
        query_emb = text_embedding(query_text)
        query_vec = ",".join(map(str, query_emb))
        results100 = search_topk(query_vec, 100)

        # 2) LLMで query_flags 抽出
        query_flags = extract_flags(query_text, "query")["flags"]
        query_flags = postprocess_query_flags(query_text,query_flags) #抽出後の再調整

        # 3) 条件文ができるか確認（あればTOP50追加）
        conds = build_optional_filters(query_text,query_flags)

        if conds:
            results50 = search_topk(query_vec, 50, conds)
            merged = merge_unique_by_docid(results100, results50)
            filtered_n = len(results50)
            where_sql = conds
        else:
            merged = results100
            filtered_n = 0
            where_sql = ""

        #マージ直後の数
        merged_raw_n = len(merged)
        # 特定用語のvalue=1の数
        o2_pos_before = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
        #（必要に応じて）特定のフラグに対して質問文とDB登録のフラグ値が強く食い違うものを取り除く
        merged = hard_exclude_contradictions(
            merged, query_text, query_flags,
            hard_flags=["HasOxygenTherapy"],
            conf_th=0.9,
            doc_conf_th=0.8,   # 例：0.7にするともっと落ちる／0.9にすると落ちにくい
            )
        merged_final_n = len(merged)
        dropped_n=merged_raw_n-merged_final_n
        # hard_exclude_contradictions()実行後の特定用語のvalue=1の数
        o2_pos_after  = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
        # 4) リランク
        reranked = reranker.rerank(query_text, merged, top_n=50)
        top3 = reranked[:3]

        # 5) judge 用 top3 生成（FlagsJson は dictのまま）
        judge_top3 = []
        for r in top3:
            doc_flags = r.get("FlagsJson")
            if isinstance(doc_flags, dict) and "flags" in doc_flags and isinstance(doc_flags["flags"], dict):
                doc_flags = doc_flags["flags"]

            judge_top3.append(
                {
                    "DocId": r.get("DocId"),
                    "score_text": r.get("score_text"),
                    "score_rerank": r.get("ce_score"),
                    #"score_text_norm": r.get("score_text_norm"),
                    #"final_score": r.get("final_score"),
                    # SectionText は使わない方針なら保存しない（必要ならコメント解除）
                    #"SectionText": r.get("SectionText"),
                    "FlagsJson": doc_flags,
                }
            )

        result28.append(
            {
                "query_text": query_text,
                "query_flags": query_flags,
                "stage1": {
                    "base_k": 100,
                    "base_n": len(results100),
                    "filtered_k": 50,
                    "filtered_n": filtered_n,
                    "where_sql": where_sql,
                    "merged_raw_n": merged_raw_n,
                    "excluded_n":dropped_n,
                    "merged_final_n":merged_final_n,
                    "o2_pos_before": o2_pos_before,
                    "o2_pos_after": o2_pos_after,
                },
                "ranked_top3": judge_top3,
            }
        )

    dump_jsonl("/opt/src/flag_querytestRerank_results.jsonl", result28)

if __name__ == "__main__":
    test28query()
