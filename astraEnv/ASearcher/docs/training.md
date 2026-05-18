# ASearcher - Fully Asynchronous Agentic RL Training
A fully asynchronous agentic RL training framework for search agent.	

## 🎯 Key Features
+ **Fully Asynchronous RL Training**: trajectory generation and model update are fully decoupled, speeding up training & reducing training cost
+ **Diverse Choices of Search Tools**: Search agent training can use either local knowledge base, web search APIs, or MCP clients.
+ **Async RL Training is especially suitable for cases** where:
    - Excution time of a trajectory is very long.
    - Trajectories can not be stopped, e.g. the server state is hard to save and load
+ **User-friendly Development**: users can implement their own agent without touching any system-level codes

# Preparation

**Step 1:** Prepare the runtime environment and install AReaL.

Please refer to the installation guide in the project documentation.


**Step 2:** download training data from [ASearcher-train-data](https://huggingface.co/datasets/inclusionAI/ASearcher-train-data).

# Train a Search Agent


## A. Train a Search Agent with Web Search
**Step 1.** Setup Environment Variable

```shell
export SERPER_API_KEY=YOUR_SERPER_API_KEY
export JINA_API_KEY=YOUR_JINA_API_KEY
```

Here `SERPER_API_KEY` is for the [serper](https://serper.dev/api-keys) API used for Web search. The underlying search engine is Google search, `JINA_API_KEY` is for the [Jina](https://jina.ai/api-dashboard/reader) API used for read the content from thr URLs.

**Step 2**. Launch Training


Run the following command to launch training on a single node:

```shell
cd AReaL

python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B \
    train_dataset.path=/path/to/training_data.jsonl \
    trial_name=<your trial name>
```

You can run distributed experiments with Ray or Slurm

```shell
cd AReaL

python3 -m areal.launcher.ray ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web_16nodes.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B \
    train_dataset.path=/path/to/training_data.jsonl \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```



## B. Training a Search Agent with Local Knowledge Base
**Step 1.** Setup Environment Variable

```shell
export RAG_SERVER_ADDR_DIR=PATH_TO_DUMP_LOCAL_SERVER_ADDRESS
```

Here `RAG_SERVER_ADDR_DIR` is the directory to dump the address of the launched local RAG server, which will be loaded during training.

**Step 2**. Set up and launch the local RAG server

+ Step 2.1. Download the [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) model, [corpus file and webpage file](https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge)

+ Step 2.2 Build the index (need e5-base-v2 model and wiki corpus):

```shell
bash scripts/build_index.sh
```

+ Step 2.3. Launch the local RAG server

```shell
bash scripts/launch_local_server.sh $PORT $RAG_SERVER_ADDR_DIR
```

**Step 3**. Launch Training

Run the following command to launch training on a single node:

```shell
cd AReaL
python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_local.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
```

You can run distributed experiments with Ray or Slurm

```shell
cd AReaL
python3 -m areal.launcher.slurm ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_local.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    actor.path=Qwen/Qwen2.5-7B/ \
    train_dataset.path=/path/to/training_data.jsonl \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```


## C. Fine-tuning a LRM Agent

**Step 1.** Launch Qwen2.5-72B-Instruct for LLM-as-Judge:

```shell
python3 -m areal.launcher.ray ASearcher/train/asearcher_reasoning.py \
    --config ASearcher/configs/asearcher_web_qwq.yaml \
    experiment_name=asearcher-qwen72b-inst-server-only \
    trial_name=run1 \
    cluster.n_nodes=1 allocation_mode=sglang.d2t4p1 \
    actor.path=Qwen/Qwen2.5-72B-Instruct 
```

**Step 2.** Launch QwQ-32B agent training:

```shell
python3 -m areal.launcher.ray \
    ASearcher/train/asearcher_reasoning.py \
    --config ASearcher/configs/asearcher_web_qwq.yaml \
    experiment_name=asearcher-qwq-train \
    trial_name=run1 cluster.n_nodes=6 allocation_mode=sglang.d2t8+d4t8 \
    actor.path=Qwen/QwQ-32B \
    train_dataset.path=path_to_ASearcher-LRM-35k.jsonl \
    judge_engine.experiment_name=asearcher-qwen72b-inst-server-only \
    judge_engine.trial_name=run1
```

P.S. You could also try using smaller models, e.g. <=8B, to train a search agent with limited compute.

P.S. Users can run RL training with user-defined agent workflow with only minimal modifications by replacing `OpenAIClient` with `AReaLOpenAIClient`. See [ASearcher/train/reasoning_agent.py](ASearcher/train/reasoning_agent.py) for a concret example.

# Customization

Please refer to our [guideline](../docs/guideline.md) for more information about building a custom agent.

