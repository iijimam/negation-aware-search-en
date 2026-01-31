from sqlalchemy import create_engine,text
from openai import OpenAI
import json
from typing import Dict, Any,List
# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "FTEST"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

# DB接続
engine = create_engine(DATABASE_URL,echo=False)

# OpenAI
#API_KEY = ""
#client = OpenAI(api_key=API_KEY)
client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用
MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT="""You are an information extraction system that extracts structured clinical flags
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

def text_embedding(text :str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding

def extract_flags(input_text: str,which) -> Dict[str, Any]:
    if which=="hospital_course":
        system_pompt=f"""
    {SYSTEM_PROMPT}
    {OUTPUT_FORMAT1}
    {FLAGNAME}
    """
    else:
        system_pompt=f"""
    {SYSTEM_PROMPT}
    {OUTPUT_FORMAT2}
    {FLAGNAME}
    """
    response = client.chat.completions.create(
        model= MODEL,
        messages=[
            {"role": "system", "content": system_pompt},
            {"role": "user", "content": input_text}
        ],
        max_tokens=1500,
        temperature=0.0,
    )
    output_text = response.choices[0].message.content.strip()
    output_text = output_text.strip()
    if not output_text:
      raise ValueError("LLM returned empty output")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as e:
        print("---- JSON PARSE ERROR ----")
        print("OUTPUT_TEXT:", repr(output_text[:500]))
        print("FULL_LEN:", len(output_text))
        raise

# Return the hospital course as a list
def get_hospital_course():
    sql = "select DocId,SectionText from Demo.DischargeSummaryChunk WHERE SectionType='hospital_course'"
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]

# Reading the JSONL of including the hospital course
def load_file(path: str):
    result=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            result.append(obj)
    return result

def output_flag(input : List[Any], out_jsonl:str,which="hospital_course"):
    n=0
    # write to jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for text in input:
            if which == "hospital_course":
                flags=extract_flags(text.get("SectionText"),which)
                row = {
                    "DocId": (text or {}).get("DocId"),
                    "Hospital_Course": text.get("SectionText"),
                    "schema_version": flags.get("schema_version", "flags.v2"),
                    "doc_type": flags.get("doc_type", "hospital_course"),
                    "flags": flags.get("flags", {}),
                }
            else:
                flags=extract_flags(text,which)
                row = {
                    "query": text,
                    "schema_version": flags.get("schema_version", "flags.v2"),
                    "doc_type": flags.get("doc_type", "query"),
                    "flags": flags.get("flags", {}),
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            print(f"Processed: {n}")

    print(f"Saved: {out_jsonl} (n={n})")
    return


if __name__ == "__main__":
    # DBから入院経過を取る場合
    #input_texts = get_hospital_course()
    input_texts = load_file("/src/sectiontext-en.jsonl")
    output_flag(input_texts, "flag_hospital_courseall_flg-en.jsonl","hospital_course")

