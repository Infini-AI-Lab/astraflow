# Search

Reinforcement learning for search-augmented agents ([ASearcher](https://github.com/inclusionAI/ASearcher)) that interleave reasoning with local retrieval against a Wikipedia knowledge base.

**Search recipes**: [`examples/search/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/search)

Each recipe ships an all-in-one launch script under `scripts/` and its config under `yaml/`.

## Environment setup

The search recipes query a local FAISS retrieval server over the Wikipedia 2018 corpus. Set it up once before training.

Install the retrieval server dependencies:

```bash
conda create -n rag-retriever python=3.10 -y
conda activate rag-retriever
pip install -r astraEnv/ASearcher/requirements-rag-server.txt
```

Download the knowledge corpus and build the index (this can take hours to build the index):

```bash
cd astraEnv/ASearcher
conda activate rag-retriever
export WIKI2018_WORK_DIR=data/wiki2018
mkdir -p "$WIKI2018_WORK_DIR"
huggingface-cli download inclusionAI/ASearcher-Local-Knowledge \
  --repo-type dataset --local-dir "$WIKI2018_WORK_DIR" --local-dir-use-symlinks False
bash scripts/build_index.sh
```

Start the retrieval server before training — it uses the 2 GPUs the launcher leaves free:

```bash
cd astraEnv/ASearcher
conda activate rag-retriever
export RAG_SERVER_ADDR_DIR=./tmp-log/rag_server_addrs
export PORT=7000
export USE_FAISS_GPU=1   # set 0 to disable GPU FAISS
bash scripts/launch_rag_server.sh 6,7
```

## Qwen2.5-7B-Instruct — 8 GPUs

The search recipe trains an ASearcher agent on an 8-GPU node — 4 GPUs for inference, 2 for training, with 2 GPUs left for a local retrieval (RAG) server. It comes in two variants that differ **only** in weight transfer mode:

- [`qwen2.5-7b-instruct-m2po-full/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/search/qwen2.5-7b-instruct-m2po-full) — full weight transfer
- [`qwen2.5-7b-instruct-m2po-delta/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/search/qwen2.5-7b-instruct-m2po-delta) — delta weight transfer (only changed weights are sent)

The agent uses the `async-search-access` search client, which queries the local FAISS retrieval server from [Environment setup](#environment-setup) above. The launch script reads the server addresses from `astraEnv/ASearcher/tmp-log/rag_server_addrs` and aborts if the server is not running.

### Run

With the retrieval server already running, one script launches all three processes — the AstraFlow service, the RaaS inference server, and the trainer:

```bash
# delta weight transfer
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh

# full weight transfer
bash examples/search/qwen2.5-7b-instruct-m2po-full/scripts/run_qwen2.5-7b-instruct-m2po-full.sh
```

### Settings

| Setting | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| GPUs | 6 of 8 — RaaS ×4 (SGLang, DP=4), Trainer ×2 (FSDP, DP=2) |
| Algorithm | M2PO (`m2_threshold` 0.004) |
| Weight transfer | TCP — full, or delta (`delta_full_sync_interval` 10) |
| Context length | 16384 |
| Max new tokens | 1024 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Train batch size | 256 |
| Learning rate | 5e-6 (Adam, constant schedule) |
| Train steps | 1000 |
| Workflow / reward | `asearcher` (`max_turns` 32) / F1 |
| Retrieval | Local FAISS RAG server over Wikipedia 2018, `async-search-access` client (`topk` 5) |
| Train dataset | ASearcher-Base-35k |
| Eval datasets | TriviaQA, PopQA, HotpotQA, Bamboogle |
