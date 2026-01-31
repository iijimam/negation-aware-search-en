ARG IMAGE=containers.intersystems.com/intersystems/irishealth-community:2025.3
FROM $IMAGE

USER root
WORKDIR /opt/src
RUN chown ${ISC_PACKAGE_MGRUSER}:${ISC_PACKAGE_IRISGROUP} /opt/src
USER ${ISC_PACKAGE_MGRUSER}

# ビルド中に実行したいスクリプトがあるファイルをコンテナにコピーしています
COPY iris.script .
COPY src .

# IRISを開始し、IRISにログインし、iris.scriptに記載のコマンドを実行しています
RUN iris start IRIS \
    && pip install -r requirements.txt --break-system-packages \
    && iris session IRIS < iris.script \
    && python3 ToDBforFlag_with_embedding.py \
    && python3 ToDBUpdateforFlag.py \
    && iris stop IRIS quietly 

# リランクモデルのロード
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="/opt/src/models/bge-reranker",
    local_dir_use_symlinks=False,
)
PY

