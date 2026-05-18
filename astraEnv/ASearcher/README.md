# ASearcher

This directory contains the local ASearcher environment used by AstraFlow to
train retrieval-augmented search agents.

The environment code here is referred from the original ASearcher project and
adapted for AstraFlow training workflows in this repo.

Original project:
- https://github.com/inclusionAI/ASearcher

## Env Setup

### Online Search Mode

No more dependencies needed for online search, except for these two APIs:

```bash
export SERPER_API_KEY=YOUR_SERPER_API_KEY
export JINA_API_KEY=YOUR_JINA_API_KEY
```

`JINA_API_KEY` is optional. Without it, online search still works, but page
reading falls back to non-Jina behavior.

### Local Retrieval Mode

#### Install RAG server dependencies
```bash
conda create -n rag-retriever python=3.10 -y
conda activate rag-retriever
pip install -r astraEnv/ASearcher/requirements-rag-server.txt
```

#### Download local knowledge and build the index

```bash
cd astraEnv/ASearcher
conda activate rag-retriever

export WIKI2018_WORK_DIR=data/wiki2018
mkdir -p "$WIKI2018_WORK_DIR"

huggingface-cli download inclusionAI/ASearcher-Local-Knowledge \
  --repo-type dataset \
  --local-dir "$WIKI2018_WORK_DIR" \
  --local-dir-use-symlinks False

bash scripts/build_index.sh
```

#### Launch Local retrieval server

```bash
cd astraEnv/ASearcher
conda activate rag-retriever

export RAG_SERVER_ADDR_DIR=./tmp-log/rag_server_addrs
export PORT=7000
export USE_FAISS_GPU=1 # set 0 to disable using GPUs

bash scripts/launch_rag_server.sh 6,7
# bash scripts/launch_rag_server.sh <gpu_ids>
```

#### Cleanup

If you launched the retrieval server with `launch_rag_server.sh`, stop it with
`Ctrl+C` in that terminal.

If you launched background retrieval servers by port range, use:

```bash
bash astraEnv/ASearcher/scripts/kill_multi_local_server.sh <start_port> <num_servers>
```

## Training
Please refer to `docs/en/recipes/search.md` for more details.