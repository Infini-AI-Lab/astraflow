# Search Recipes

Search recipes train models for search-augmented workflows.

## Environment Server

Install the local retrieval server dependencies:

```bash
conda create -n rag-retriever python=3.10 -y
conda activate rag-retriever
pip install -r astraEnv/ASearcher/requirements-rag-server.txt
```

Download local knowledge and build the index:

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

Start the retrieval server before training:

```bash
cd astraEnv/ASearcher
conda activate rag-retriever

export RAG_SERVER_ADDR_DIR=./tmp-log/rag_server_addrs
export PORT=7000
export USE_FAISS_GPU=1 # set 0 to disable GPU FAISS

bash scripts/launch_rag_server.sh 6,7
```

Run one example from the repo root:

```bash
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh
```

Complete guidance: [`docs/en/recipes/search.md`](../../docs/en/recipes/search.md).

---
**GPU Resources**

These recipes default to an 8xH100 node, with the launcher using 6 GPUs for
training and inference and leaving 2 GPUs available for the retrieval server.
