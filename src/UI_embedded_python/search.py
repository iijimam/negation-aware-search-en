from openai import OpenAI
import json
import re
from typing import Any, Dict, List, Tuple,Optional
from string import Template
from sentence_transformers import CrossEncoder

import iris

client = OpenAI()  # Use the environment variable OPENAI_API_KEY
MODEL = "gpt-4.1-mini"  # Prioritize stability for the judge. Upgrade to GPT-4.1 if necessary.
TOPK = 100

#-----------------------------------
# Rerank
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
        # 1) Narrow down the reranking targets (as too many candidates slow things down)
        cands = sorted(candidates, key=lambda x: float(x.get("vec_score", 0.0)), reverse=True)
        pool = cands[: min(top_n, len(cands))]

        # 2) pare of (query, doc)
        pairs: List[Tuple[str, str]] = [(query, str(c.get(text_key, ""))) for c in pool]

        # 3) Score Inference with Cross-Encoder
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # 4) Assign scores and sort
        for c, s in zip(pool, scores):
            c["ce_score"] = float(s)

        pool.sort(key=lambda x: float(x["ce_score"]), reverse=True)
        return pool

def initial():
    global reranker
    #Create a Relank instance
    #BAAI/bge-reranker-v2-m3 is loaded into /opt/src/models/bge-reranker within the Dockerfile.
    reranker = CrossEncoderReranker(model_name="/opt/src/models/bge-reranker", device="cpu")       

initial()

# ===== Prompt for extracting key terms from an LLM =====
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

#Format of the returned dictionary (with these key names unchanged)
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
# Using LLM to extract key terms and their states
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
# DB FlagsJson returns a flags dictionary (HasXXX -> {value...}) regardless of whether it is:
#  - Wrapped in {“flags”: {...}}
#  - Directly placed in {...}
#  - A JSON string
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
# Returns the entire set of flag information extracted by the LLM (Dict)
#-----------------------------------
def _get_flag_obj(flags: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = flags.get(key, {})
    return v if isinstance(v, dict) else {}

#-----------------------------------
# Return the value (1/0/null) of the flag extracted by the LLM.
#-----------------------------------
def _get_value(flags: Dict[str, Any], key: str):
    return _get_flag_obj(flags, key).get("value", None)

#-----------------------------------
# Returns the confidence value of the flag extracted by the LLM
#-----------------------------------
def _get_conf(flags: Dict[str, Any], key: str) -> float:
    c = _get_flag_obj(flags, key).get("confidence", 0.0)
    try:
        return float(c)
    except Exception:
        return 0.0

#-----------------------------------
# Return evidence for flags extracted by LLM (empty string if none found)
#-----------------------------------
def _get_evidence(flags: Dict[str, Any], key: str) -> str:
    ev = _get_flag_obj(flags, key).get("evidence", "")
    if ev is None:
        return ""
    return str(ev)

#-----------------------------------
# Set value to null (for Query guards)
#-----------------------------------
def _set_null(flags: Dict[str, Any], key: str, note: str = "") -> None:
    obj = _get_flag_obj(flags, key)
    obj["value"] = None
    obj["polarity"] = "unknown"
    obj["evidence"] = None
    obj["confidence"] = 0.0
    # Scope is retained (useful as information for extraction results)
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
# Whether the string in the first argument contains the value of the specified flag FLAG_MENTION_KEYWORDS
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
# Should we adopt value=0? (General rule)
# Whether it is a strong negation is checked using the confidence value after LLM verification
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
# Safeguard to prevent excessive loss of information when creating WHERE clauses from query flags
# - value=1 is generally adopted as-is (=affirmative results can often be recovered by other processing even if mistakenly included)
# - value=0 is prone to causing fatal errors if mistakenly included, so strictly guard using general rules; if unsuccessful, set to null
#-----------------------------------
def postprocess_query_flags(query_text: str, flags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flags, dict):
        return {}

    for k, obj in list(flags.items()):
        if not (isinstance(k, str) and isinstance(obj, dict)):
            continue
        if obj.get("value", None) == 0:
            if not _should_accept_negation(k, obj, query_text):
                _set_null(flags, k, note="Query Guard: Weak basis for 0 (negation)/insufficient mentions/null due to relative expression")

    return flags

#-----------------------------------
# Create WHERE clause from query flags
# Flags specify conditions using OR
#-----------------------------------
def build_optional_filters(query_text: str, query_flags: Dict[str, Any]) -> str:
    q = normalize_flags_dict(query_flags)
    conds = []

    # ---- Strong flags: 1 is a mandatory requirement for hiring ----
    for k in STRONG_FLAGS:
        if _get_value(q, k) == 1:
            conds.append(f"c.{k} = 1")

    # ---- Strong flags: 0 Also adopted (only when passing strict guard) ----
    for k in STRONG_FLAGS:
        obj = q.get(k, {})
        if isinstance(obj, dict) and obj.get("value") == 0:
            conf = float(obj.get("confidence", 0.0))
            # If the policy is that “zero misfires are fatal,” then it should be enforced as strictly as oxygen.
            if conf >= 0.9 and _should_accept_negation(k, obj, query_text):
                conds.append(f"c.{k} IS NULL OR c.{k} = 0")

    # ---- Oxygen:  0 Maintain the current special provisions  ----
    oxy = q.get("HasOxygenTherapy", {})
    if isinstance(oxy, dict) and oxy.get("value") == 0:
        conf = float(oxy.get("confidence", 0.0))
        if conf >= 0.9 and _should_accept_negation("HasOxygenTherapy", oxy, query_text):
            conds.append("c.HasOxygenTherapy IS NULL OR c.HasOxygenTherapy = 0")

    if not conds:
        return ""

    # importtant：AND
    return "(" + " AND ".join(conds) + ")"


#-----------------------------------
# Merge two similar search results (check using DocId as key)
#-----------------------------------
def merge_unique_by_docid(*lists_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for lst in lists_:
        for r in lst:
            docid = r.get("DocId")
            if docid is None:
                continue
            docid = str(docid)   # Normalize the schema
            if docid in seen:
                continue
            seen.add(docid)
            out.append(r)
    return out

#-----------------------------------
# Using two sets of similar search results and the query text
# Check for discrepancies in negation (0) for specific terms within the query text
# (Exclude absolute contradictions: those not removable via WHERE clauses, and post-merge inclusions)
# If the query strongly negates (0) but the DB affirms (1), remove from results
# (Utilizes the confidence value from the LLM's response: Can specify via parameter which threshold triggers removal)
# If the DB shows value==1 but low confidence, it may be a false positive and is not excluded
#-----------------------------------
def hard_exclude_contradictions(
    results,
    query_text,
    query_flags,
    hard_flags=None,
    conf_th=0.9,         # query
    doc_conf_th=0.8,     # doc
    bypass_accept_if_query_conf_ge=0.99,
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

        #  For ultra-high reliability, _should_accept_negation_ is not required.
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

            # Drop only the likely “1”
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
# Generating Similar Search Results Output (Converting FlagJson Contents to Dict)
#-----------------------------------
def make_output_topk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        # To enable unified referencing on the rerank side, include the vector similarity in vec_score as well.
        if "vec_score" not in rr and "score_text" in rr:
            rr["vec_score"] = rr.get("score_text")
        if isinstance(rr.get("FlagsJson"), str):
            try:
                rr["FlagsJson"] = json.loads(rr["FlagsJson"])
            except Exception:
                rr["FlagsJson"] = None
        out.append(rr)
    return out


def search_topk(query_vec: str, topn: int, where_extra: str = "") -> List[Dict[str, Any]]:
    sql = f"""
SELECT TOP ?
  c.DocId, c.SectionText, c.FlagsJson,
  VECTOR_COSINE(c.Embedding, TO_VECTOR(?, FLOAT, 1536)) AS score_text,
  d.PatientId,d.DischargeDate
FROM Demo.DischargeSummaryChunk c, Demo.DischargeSummaryDoc d
WHERE d.DocId=c.DocId AND (c.SectionType = 'hospital_course')
  {("AND " + where_extra) if where_extra else ""}
ORDER BY score_text DESC
"""
    #with engine.connect() as conn:
    #    rows = conn.execute(text(sql), {"topN": topn, "query_vec": query_vec}).mappings().all()
    #return make_output_topk([dict(r) for r in rows])
    print(sql)

    stmt=iris.sql.prepare(sql)
    rset=stmt.execute(topn,query_vec)

    cols=["DocId","SectionText","FlagsJson","score_text","PatientId","DischargeDate"]
    return make_output_topk([dict(zip(cols,r)) for r in rset])

#-----------------------------------
# Get the number of values for a flag
#-----------------------------------
def count_doc_flag_value(results, flag_name, value):
    n = 0
    for r in results:
        d = normalize_flags_dict(r.get("FlagsJson"))
        if _get_value(d, flag_name) == value:
            n += 1
    return n

#-----------------------------------
#Similarity Search Functions
# 1) Extract key terms (flags) and their states from the query using LLM
# 2) Verify if WHERE conditions can be specified from step 2
#    If available from step 3, execute TOP50 similarity search
#    After TOP50 execution, merge results with TOP100
# 3) TOP100
# 4) Execute re-ranking (CrossEncoding)
#    Use the TOP3 after re-ranking to create JSON for LLM as a Judge
#-----------------------------------
def get_simirality_ranking(query_text):
    result = {}

    # 1) extract query_flags with LLM
    query_flags = extract_flags(query_text, "query")["flags"]
    query_flags = postprocess_query_flags(query_text,query_flags) #抽出後の再調整

    # 2) Verify if conditional statements are possible (if so, add to TOP50)
    conds = build_optional_filters(query_text,query_flags)

    print(f"***{query_flags}\n***{conds}***")

    # 3) vector search TOP100
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    results100 = search_topk(query_vec, 100)

    if conds:
        results50 = search_topk(query_vec, 50, conds)
        merged = merge_unique_by_docid(results100, results50)
        filtered_n = len(results50)
        where_sql = conds
    else:
        merged = results100
        filtered_n = 0
        where_sql = ""

    #Immediately after merging
    merged_raw_n = len(merged)
    #Number of instances where value=1 for a specific term
    o2_pos_before = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
    #(If necessary) Remove entries where the query text and the flag value registered in the database strongly conflict for specific flags.
    merged = hard_exclude_contradictions(
        merged, query_text, query_flags,
        hard_flags=["HasOxygenTherapy"],
        conf_th=0.9,
        doc_conf_th=0.8,
        )

    merged_final_n = len(merged)
    dropped_n=merged_raw_n-merged_final_n
    # Number of instances where value=1 for specific terms after executing hard_exclude_contradictions()
    o2_pos_after  = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
    # 5) rerank
    reranked = reranker.rerank(query_text, merged, top_n=50)
    top3 = reranked[:3]

    # 5) top3
    judge_top3 = []
    search_top3 = []
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
                "SectionText": r.get("SectionText"),
                "FlagsJson": doc_flags,
            }
        )
        #Information for display on screen
        search_top3.append(
            {
                "DocId": r.get("DocId"),
                "score_text": r.get("score_text"),
                "score_text_norm": r.get("score_text_norm"),
                "final_score": r.get("final_score"),
                "PatientId": r.get("PatientId"),
                "DischargeDate": r.get("DischargeDate"),
                "SectionText": r.get("SectionText"),
                "FlagsJson": doc_flags,
            }
        )

    result.update(
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
    print(result)
    return result,search_top3

if __name__ == "__main__":
    initial()