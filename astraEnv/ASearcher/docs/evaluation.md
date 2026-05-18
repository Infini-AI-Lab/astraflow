# ASearcher Evaluation

A comprehensive framework for testing and evaluating LLM Agent performance. Supports multiple agent architectures, search engine integrations, and evaluation methods.


# Preparation

**Step 1:** Prepare the runtime environment.

Please refer to the installation guide in the project documentation for `Runtime Environment`.


**Step 2:** download test data from [ASearcher-test-data](https://huggingface.co/datasets/inclusionAI/ASearcher-test-data).


# Evaluate a Search Agent
We can evaluate different agent workflows by specifying the agent-type and search-client-type.

```bash
python3 -m evaluation.search_eval_async \
    ...
    --prompt_type ${PROMPT_TYPE} \
    --agent-type ${AGENT_TYPE} \
    --search_client_type ${SEARCH_CLIENT_TYPE} \
    ...

```
We list several examples as follows:

### A. Evaluate an Reasoning Model with Web Search
```bash
cd evaluation/

MODEL_PATH=/path/to/models 
DATA_DIR=/path/to/test_set # Could be downloaded from [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]

DATA_NAMES=GAIA,xbench-deepsearch,Frames
AGENT_TYPE=asearcher-reasoning
PROMPT_TYPE=asearcher-reasoning
SEARCH_CLIENT_TYPE=async-web-search-access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
SERPER_API_KEY=${your_serper_api_key} \
JINA_API_KEY=${your_jina_api_key} \
TOKENIZERS_PARALLELISM=false \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --search_client_type ${SEARCH_CLIENT_TYPE} \
    --tensor_parallel_size 4 \
    --temperature 0.6 \
    --parallel-mode seed \
    --seed 1 \
    --use-jina \
    --llm_as_judge \
    --pass-at-k 1 \ # if you want get more stable result, please increase it
```

### B. Evaluate a Non-reasoning Search Agent with Web Search

```bash
cd evaluation/

MODEL_PATH=/path/to/models 
DATA_DIR=/path/to/test_set # Could be downloaded from [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]

DATA_NAMES=GAIA,xbench-deepsearch,Frames
AGENT_TYPE=asearcher
PROMPT_TYPE=asearcher
SEARCH_CLIENT_TYPE=async-web-search-access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
SERPER_API_KEY=${your_serper_api_key} \
JINA_API_KEY=${your_jina_api_key} \
TOKENIZERS_PARALLELISM=false \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --search_client_type ${SEARCH_CLIENT_TYPE} \
    --tensor_parallel_size 4 \
    --temperature 0.6 \
    --parallel-mode seed \
    --seed 1 \
    --use-jina \
    --llm_as_judge \
    --pass-at-k 1 \ 
```

### C. Evaluate Search-R1 with Local Knowledge Base
**Step 0.** Build the image and start the container following  `evaluation/Dockerfile` 

**Step 1.** Setup Environment Variable

```shell
export RAG_SERVER_ADDR_DIR=PATH_TO_DUMP_LOCAL_SERVER_ADDRESS
export PORT=8000
```

Here `RAG_SERVER_ADDR_DIR` is the directory to dump the address of the launched local RAG server, which will be loaded during training.

**Step 2**. Set up and launch the local RAG server

+ Step 2.1. Download the [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) model, [e5 retriver index file, corpus file, and webpage file](https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge):


+ Step 2.2. Launch the local RAG server

```shell
bash scripts/launch_local_server.sh $PORT $RAG_SERVER_ADDR_DIR
```

**Step 3**: 
```bash
cd evaluation/

MODEL_PATH=/path/to/models 
DATA_DIR=/path/to/test_set # Could be downloaded from [https://huggingface.co/datasets/inclusionAI/ASearcher-test-data]

DATA_NAMES=Bamboogle,NQ
AGENT_TYPE=search-r1
PROMPT_TYPE=search-r1
SEARCH_CLIENT_TYPE=async-search-access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
SERPER_API_KEY=${your_serper_api_key} \
JINA_API_KEY=${your_jina_api_key} \
TOKENIZERS_PARALLELISM=false \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --search_client_type ${SEARCH_CLIENT_TYPE} \
    --tensor_parallel_size 4 \
    --temperature 0.6 \
    --parallel-mode seed \
    --seed 1 \
    --use-jina \
    --llm_as_judge \
    --pass-at-k 1 \ 
```


## ⚙️ Configuration Parameters

### Core Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--data_names` | Test dataset names | `GAIA,HotpotQA,NQ,TriviaQA`, etc. |
| `--agent-type` | Agent type | `search-r1`, `asearcher-reasoning`, `asearcher` |
| `--search_client_type` | Search client type | `async-search-access`, `async-web-search-access` |
| `--model_name_or_path` | LLM model path | Local model path or HuggingFace model name |
| `--pass-at-k` | Count of evaluation | For multiple tests (serial) |

### Model Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--temperature` | Generation temperature | `0` |
| `--top_p` | Top-p sampling | `1` |
| `--top_k` | Top-k sampling | `-1` |
| `--max-tokens-per-call` | Maximum tokens to generate | `4096` |

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_test_sample` | Number of test samples | `-1` (all) |
| `--concurrent` | Number of concurrent requests | `128` |
| `--llm_as_judge` | Enable LLM-as-Judge evaluation | `False` |
| `--judge-prompt` | LLM Judge prompt type | default |

## 🔧 Evaluation Metrics

The framework supports the following evaluation metrics:

- **EM (Exact Match)**: Exact matching
- **F1 Score**: F1 score
- **CEM (Cover Exact Match)**: Cover exact matching
- **LLM-as-Judge**: Using LLM as evaluator

## 🌐 Search Integration

- **[Serper API](https://serper.dev/)**: For web search
- **[Jina API](https://jina.ai/)**: For web content extraction and processing
- **Custom Search Clients**: Support for extending other search engines
