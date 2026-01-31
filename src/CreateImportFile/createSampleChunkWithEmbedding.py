import json
from typing import Optional, Any, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"


def get_engine() -> Engine:
    engine = create_engine(DATABASE_URL,echo=False)
    return engine


def fetch_embedding_by_docid(conn, doc_id: int) -> Optional[Any]:
    """
    doc_idでEmbeddingを1件取得して返す。
    返り値はDBの型次第:
      - JSON/JSONBなら list が返ることが多い
      - TEXTなら str (JSON文字列) のことが多い
      - バイナリ/独自型なら bytes/driver依存
    """
    sql = text(f"""
        SELECT Embedding AS embedding
        FROM Demo.DischargeSummaryChunk
        WHERE DocId = :doc_id AND SectionType='hospital_course'
    """)
    row = conn.execute(sql, {"doc_id": doc_id}).mappings().first()
    if not row:
        return None
    return row["embedding"]


def normalize_embedding(emb: Any) -> Optional[list]:
    """
    出力JSONLに入れられるように、embeddingを list[float] に寄せる。
    DBがJSON文字列で持っているケースにも対応。
    """
    if emb is None:
        return None
    if isinstance(emb, list):
        return emb
    if isinstance(emb, str):
        # JSON文字列として入っている場合
        try:
            v = json.loads(emb)
            return v if isinstance(v, list) else None
        except json.JSONDecodeError:
            return None
    # それ以外（bytesなど）はDB側の実体に合わせて処理を追加
    return None


def enrich_jsonl_with_embedding(
    in_path: str,
    out_path: str,
    add_field: str = "Embedding",
    missing_policy: str = "keep_null",  # "skip" | "keep_null" | "error"
) -> None:
    """
    JSONLを読み、DocIdでEmbeddingを引いて行に追加し、別JSONLへ出力。
    """
    engine = get_engine()

    n_in = 0
    n_out = 0
    n_missing = 0

    with engine.connect() as conn, \
         open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            obj: Dict[str, Any] = json.loads(line)

            doc_id = obj.get("DocId")
            type=obj.get("SectionType")
            if doc_id is None:
                raise ValueError(f"DocIdがありません (line={n_in})")

            if type!='hospital_course':
                continue
            
            emb_raw = fetch_embedding_by_docid(conn, int(doc_id))
            emb=[float(x) for x in emb_raw.split(",") if x]
            if emb is None:
                n_missing += 1
                if missing_policy == "skip":
                    continue
                if missing_policy == "error":
                    raise ValueError(f"Embeddingが見つからない/変換不可: DocId={doc_id} (line={n_in})")
                # keep_null
                obj[add_field] = None
            else:
                obj[add_field] = emb

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"done: in={n_in}, out={n_out}, missing={n_missing}, out_path={out_path}")


if __name__ == "__main__":
    enrich_jsonl_with_embedding(
        in_path="sectiontext-en.jsonl",
        out_path="sectiontext_with_embedding-en.jsonl",
        add_field="Embedding",
        missing_policy="keep_null",
    )
