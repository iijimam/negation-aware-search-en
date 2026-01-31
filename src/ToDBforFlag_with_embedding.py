# =======================================
# 入院経過のEmbedding済のファイルをロード
# =======================================
# 読み込むJSONLの中のProcedureCodeの意味
# 1. IV-ANTIBIOTIC
#    意味:点滴による抗菌薬治療を実施
#    臨床的ニュアンス:入院治療が必要な感染症
#    軽症外来ではない可能性が高い
#   PoCでの役割：重症度・入院経過の「厚み」を出す要素
#   検索除外条件には使わない
#
# 2. INHALATION
#   意味:吸入療法（気管支拡張薬など）
#   臨床的ニュアンス:咳嗽・喘鳴・気道症状あり
#    COPD / 喘息合併の可能性
#
# 3. OXYGEN
#   意味:酸素投与を実施（経路不問）
#   含まれる想定:鼻カニュラ,マスク,HFNC（高流量）
#
# 4. INSULIN
#   意味:インスリン治療を実施
#   臨床的ニュアンス:糖尿病の既往 or ステロイド等による高血糖、感染時の血糖管理強化
#
# 5. ICU
#   意味:ICU（集中治療室）での管理
#   臨床的ニュアンス:重症〜最重症/人工呼吸・昇圧剤・厳密管理の可能性
# 6. NPPV
#   意味:非侵襲的陽圧換気（BiPAP / CPAP）
#   臨床的ニュアンス:酸素だけでは不十分/ICU相当の呼吸管理レベル
#
# 7. INTUBATION
#   意味:気管挿管を実施
#   臨床的ニュアンス:自発呼吸が困難/人工呼吸管理前提
#
# 8. VENTILATION
#   意味:人工呼吸器管理
#   臨床的ニュアンス:長期管理・鎮静・全身管理
#
# 9. VPRESSOR
#   意味:昇圧剤（ノルアドレナリン等）使用
#   臨床的ニュアンス:敗血症性ショックなど循環不全


from sqlalchemy import create_engine,text
import json
from typing import Dict, Any, Literal,Optional,Tuple
import re
from dataclasses import dataclass

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

# DB接続
engine = create_engine(DATABASE_URL,echo=False)


def load_file(path: str,type:str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if (type=="section"):
                section_insert(engine,obj)
            else:
                chunk_insert(engine,obj)
    return


def chunk_insert(engine, input: Dict[str, Any]):
    #この時点ではフラグとなるHasXXXは未処理（LLMから抽出して更新する予定）
    sql = """
    INSERT INTO Demo.DischargeSummaryChunk (
        DocKey, DocId, SectionType, SectionTitle, SectionText,
        StartOffset, EndOffset, Embedding
    )
    VALUES (
        :DocKey, :DocId, :SectionType, :SectionTitle, :SectionText,
        :StartOffset, :EndOffset, TO_VECTOR(:Embedding, FLOAT, 1536)
    )
    """.strip()

    # デフォルトはNULL（= None）を入れる
    # Embeddingはlist
    embeddingstr=",".join(map(str, input.get("Embedding")))
    para = {
        "DocKey": input.get("DocKey"),
        "DocId": input.get("DocId"),
        "SectionType": input.get("SectionType"),
        "SectionTitle": input.get("SectionTitle"),
        "SectionText": input.get("SectionText"),
        "StartOffset": input.get("StartOffset"),
        "EndOffset": input.get("EndOffset"),
        "Embedding": embeddingstr,
    }

    with engine.connect() as conn:
        conn.execute(text(sql), para)
        conn.commit()


def section_insert(engine,input :Dict[str,Any]):
    sql="""
    INSERT INTO Demo.DischargeSummaryDoc (
        DocId,PatientId, AdmitDate, DischargeDate,AgeBand,Sex,Department,DischargeDisposition,
        DiagnosisCodes,ProcedureCodes,SourceFilename,SourceUri,SourceText,Lang)
        VALUES (
        :DocId,:PatientId, :AdmitDate, :DischargeDate,:AgeBand,:Sex,:Department,:DischargeDisposition,
        :DiagnosisCodes,:ProcedureCodes,:SourceFilename,:SourceUri,:SourceText,:Lang)
    """.strip()
    
    para={
        "DocId":input.get("DocId"),
        "PatientId":input.get("PatientId"),
        "AdmitDate":input.get("AdmitDate"),
        "DischargeDate":input.get("DischargeDate"),
        "AgeBand":input.get("AgeBand"),
        "Sex":input.get("Sex"),
        "Department":input.get("Department"),
        "DischargeDisposition":input.get("DischargeDisposition"),
        "DiagnosisCodes":input.get("DiagnosisCodes"),
        "ProcedureCodes":input.get("ProcedureCodes"),
        "SourceFilename":input.get("SourceFilename"),
        "SourceUri":input.get("SourceUri"),
        "SourceText":input.get("SourceText"),
        "Lang":input.get("Lang")
    }
    with engine.connect() as conn:
        conn.execute(text(sql),para)
        conn.commit()

load_file("sectionsample-en.jsonl","section")
load_file("sectiontext_with_embedding-en.jsonl","chunk")
