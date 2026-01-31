from sqlalchemy import create_engine,text
import json
from typing import Dict, Any, Literal,Optional,Tuple,List
import iris
import json

def get_sectiontext(outfile:str):
    sql="""
SELECT DocKey, DocId,SectionType, SectionTitle,SectionText
FROM Demo.DischargeSummaryChunk
    """
    rset=iris.sql.exec(sql)

    with open(outfile, "w", encoding="utf-8") as f:
        for row in rset:
            # Row → dict
            record = {
                "DocKey":row[0],
                "DocId":row[1],
                "SectionType":row[2],
                "SectionTitle":row[3],
                "SectionText":row[4]
            }

            # JSONLとして1行ずつ書き出し
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return

get_sectiontext("/opt/src/sectiontext.jsonl")