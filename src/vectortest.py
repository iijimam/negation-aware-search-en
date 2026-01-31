from sqlalchemy import create_engine,text
from typing import Any, Dict, List, Tuple,Optional
from sqlalchemy import create_engine,text
from openai import OpenAI
client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用

engine = None
conn = None

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

def initial():
    global engine, conn
    # DB接続
    engine = create_engine(DATABASE_URL,echo=False)
    if engine is None:
        engine =create_engine(DATABASE_URL,echo=True, future=True)
    if conn is None:
        conn = engine.connect()
        


def text_embedding(text :str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding


def search_topk(query_vec: str, topn: int, where_extra: str = "") -> List[Dict[str, Any]]:
    sql = f"""
SELECT TOP :topN
  c.DocId, c.SectionText,
  VECTOR_COSINE(c.Embedding, TO_VECTOR(:query_vec, FLOAT, 1536)) AS score_text,
  d.PatientId,d.DischargeDate
FROM Demo.DischargeSummaryChunk c, Demo.DischargeSummaryDoc d
WHERE d.DocId=c.DocId AND (c.SectionType = 'hospital_course')
  {("AND " + where_extra) if where_extra else ""}
ORDER BY score_text DESC
"""

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"topN": topn, "query_vec": query_vec}).mappings().all()
    return rows


if __name__ == "__main__":
    initial()
    #query_text="咳嗽と痰、発熱で入院しましたが、胸部画像で肺炎所見は明確ではありませんでした。対症療法で改善し、酸素投与やICU管理は不要でした"
    query_text="The patient had fever and cough but did not require supplemental oxygen during the hospital course. The pneumonia was relatively mild."
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    rows=search_topk(query_vec, 10)
    print(f"***Patter1: this query said 'did not require supplemental oxygen' / the resuls includes ' Supplemental oxygen was provided'. query >>>  {query_text} \n")
    for reco in rows:
        print(f"{str(reco["DocId"])} - {reco["SectionText"]}")
    
    #query_text="酸素投与は行いましたが、重症化せずICUには入室していません。"
    query_text="Supplemental oxygen was provided, but the patient did not deteriorate and did not require ICU admission."
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    rows=search_topk(query_vec, 10)
    print(f"----\n\n***Patter2: this query has 'oxygen was provided' / the resuls includes 'oxygen therapy was not required'. query >>> {query_text} \n")
    for reco in rows:
        print(f"{str(reco["DocId"])} - {reco["SectionText"]}")