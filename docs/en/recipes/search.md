# Search (ASearcher)

Train retrieval-augmented search agents with AstraFlow using local retrieval.

## Overview

- **Model**: Qwen2.5-7B
- **Algorithm**: M2PO (m2_threshold=0.004), LR: 5e-6
- **Workflow**: `asearcher`, max_turns: 32, reward_type: F1
- **Eval datasets**: TriviaQA, PopQA, HotpotQA, Bamboogle
- **Eval frequency**: every 10 steps

Recipe:
- **Local retrieval**: `examples/search/qwen2.5-7b-instruct-m2po-delta/`

Common settings:
- Buffer: 32768, replay_ratio: 0, max_staleness: 8
- Batch size: 256, max_new_tokens: 1024, max_length: 4096
- RaaS: SGLang TP=1 on 4 GPUs
- Trainer: FSDP on 2 GPUs
- n_trajs: 8, topk: 5

## Common Data Setup

Download the training dataset from Hugging Face from the project root into
`astraEnv/ASearcher`:

```bash
huggingface-cli download inclusionAI/ASearcher-train-data \
  --repo-type dataset \
  --local-dir astraEnv/ASearcher/ASearcher-train-data \
  --local-dir-use-symlinks False
```

The training dataset path wired in this recipe is:

```bash
astraEnv/ASearcher/ASearcher-train-data/ASearcher-Base-35k.jsonl
```

Download the eval datasets from the repo root:

```bash
huggingface-cli download inclusionAI/ASearcher-test-data \
  --repo-type dataset \
  --local-dir astraEnv/ASearcher/data \
  --local-dir-use-symlinks False \
  --include "TriviaQA_rand1000/*"

huggingface-cli download inclusionAI/ASearcher-test-data \
  --repo-type dataset \
  --local-dir astraEnv/ASearcher/data \
  --local-dir-use-symlinks False \
  --include "PopQA_rand1000/*"

huggingface-cli download inclusionAI/ASearcher-test-data \
  --repo-type dataset \
  --local-dir astraEnv/ASearcher/data \
  --local-dir-use-symlinks False \
  --include "HotpotQA_rand1000/*"

huggingface-cli download inclusionAI/ASearcher-test-data \
  --repo-type dataset \
  --local-dir astraEnv/ASearcher/data \
  --local-dir-use-symlinks False \
  --include "Bamboogle/*"
```

## Local Retrieval Mode

**Config**: `examples/search/qwen2.5-7b-instruct-m2po-delta/`

Key settings:
- search_client_type: `async-search-access`
- Local retrieval server required
- Reads server addresses from `astraEnv/ASearcher/tmp-log/rag_server_addrs`

### Install RAG server dependencies

```bash
conda create -n rag-retriever python=3.10 -y
conda activate rag-retriever
pip install -r astraEnv/ASearcher/requirements-rag-server.txt
```

### Download local knowledge and build the index

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

### Split launch

Launch 4 processes in separate terminals: 1 local retrieval server, then RaaS,
AstraFlow, and Trainer.

#### 0. Local retrieval server

Run this from `astraEnv/ASearcher/`:

```bash
cd astraEnv/ASearcher
conda activate rag-retriever

export RAG_SERVER_ADDR_DIR=./tmp-log/rag_server_addrs
export PORT=7000
export USE_FAISS_GPU=1 # set 0 to disable using GPUs

bash scripts/launch_rag_server.sh 6,7
# bash scripts/launch_rag_server.sh <gpu_ids>
```

Optional smoke test after the server starts:

```bash
cd astraEnv/ASearcher
bash scripts/test_rag_server.sh
```

#### 1. RaaS

Run this from the repo root:

```bash
conda activate astraflow
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/2_raas.sh
```

#### 2. AstraFlow

Run this from the repo root:

```bash
conda activate astraflow
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/1_astraflow.sh
```

#### 3. Trainer

Run this from the repo root:

```bash
conda activate astraflow
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/3_trainer_model0.sh
```

### One-shot launch

The one-shot script starts RaaS, AstraFlow, and Trainer, but it still expects
the local retrieval server to already be running.

#### Terminal 0: start the local retrieval server first

```bash
cd astraEnv/ASearcher
conda activate rag-retriever

export RAG_SERVER_ADDR_DIR=./tmp-log/rag_server_addrs
export PORT=7000
export USE_FAISS_GPU=1 # set 0 to disable using GPUs

bash scripts/launch_rag_server.sh 6,7
# bash scripts/launch_rag_server.sh <gpu_ids>
```

#### Terminal 1: start training from the repo root

```bash
conda activate astraflow
bash examples/search/qwen2.5-7b-instruct-m2po-delta/scripts/run_qwen2.5-7b-instruct-m2po-delta.sh
```

## Cleanup

If you launched the retrieval server with `launch_rag_server.sh`, stop it with
`Ctrl+C` in that terminal.

If you launched background retrieval servers by port range, use:

```bash
bash astraEnv/ASearcher/scripts/kill_multi_local_server.sh <start_port> <num_servers>
```
