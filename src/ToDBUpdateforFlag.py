# 入院経過の文字列をLLMに渡してフラグ抽出した結果をDBに登録する
# このファイルでは、事前抽出した情報を利用する
# ファイル名：flag_hospital_courseall3.jsonl

from sqlalchemy import create_engine,text
import json
from typing import Dict, Any, Literal,Optional,Tuple,List

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

# DB接続
engine = create_engine(DATABASE_URL,echo=False)

FILTERABLE_FLAGS = ["HasICUCare","HasOxygenTherapy","HasHFNC","HasNPPV","HasIntubation","HasMechanicalVentilation","HasDialysis","HasVasopressor"]

# フィルター対象フラグがJSONに含まれているかチェック。ある場合のみ1を返す
def build_optional_filters(query_flags: Dict[str, Any]) -> List:
    conds = []
    for k in FILTERABLE_FLAGS:
        if query_flags.get(k).get("value") == 1:
            conds.append((k, 1))
    return conds

def update():
    basesql="""
    UPDATE Demo.DischargeSummaryChunk
    SET FlagsJson = :flagsjson
    """
    basewhere="WHERE DocId = :docid"

    with open("flag_hospital_courseall_flg-en.jsonl", "r", encoding="utf-8") as f:
        with engine.connect() as conn:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                flagjson=rec.get("flags", {})

                #フィルター用フラグチェック
                conds=build_optional_filters(flagjson)
                extra_sql=""
                if conds:
                    if ("HasICUCare", 1) in conds:
                        extra_sql+=",HasICUCare = 1"
                    if ("HasOxygenTherapy", 1) in conds:
                        extra_sql+=",HasOxygenTherapy = 1"
                    if ("HasHFNC", 1) in conds:
                        extra_sql+=",HasHFNC = 1"
                    if ("HasNPPV", 1) in conds:
                        extra_sql+=",HasNPPV = 1"
                    if ("HasIntubation", 1) in conds:
                        extra_sql+=",HasIntubation = 1"
                    if ("HasMechanicalVentilation", 1) in conds:
                        extra_sql+=",HasMechanicalVentilation = 1"
                    if ("HasDialysis", 1) in conds:
                        extra_sql+=",HasDialysis = 1"
                    if ("HasVasopressor", 1) in conds:
                        extra_sql+=",HasVasopressor = 1"
                
                if extra_sql=="":
                    sql=basesql + basewhere
                else :
                    sql=basesql + extra_sql + basewhere
                # DB 更新
                para={
                    "flagsjson":json.dumps(flagjson,ensure_ascii=False),
                    "docid":rec.get("DocId")
                }
                conn.execute(text(sql),para)
                conn.commit()

if __name__ == "__main__":
    update()