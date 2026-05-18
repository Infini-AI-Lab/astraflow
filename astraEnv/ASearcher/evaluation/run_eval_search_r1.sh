set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


MODEL_PATH=/path/to/model

DATA_DIR=/path/to/data

SPLIT=1
MAX_GEN_TOKENS=4097
DATA_NAMES=Bamboogle,NQ
AGENT_TYPE=search-r1
PROMPT_TYPE=local-rag
SEARCH_CLIENT_TYPE=async-search-access
temperature=0.6
top_p=0.95
top_k=-1

echo "MODEL PATH: $MODEL_PATH"
echo "Temperature: ${temperature}"
echo "top_p: ${top_p}"
echo "top_k: ${top_k}"
echo "split: ${SPLIT}/"

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --data_dir ${DATA_DIR} \
    --split test \
    --search-client-type ${SEARCH_CLIENT_TYPE} \
    --max-tokens-per-call ${MAX_GEN_TOKENS} \
    --tensor_parallel_size 4 \
    --num_test_sample 10 \
    --n_sampling 1 \
    --temperature ${temperature} \
    --top_p $top_p \
    --top_k $top_k \
    --start 0 \
    --end -1 \
    --seed 1 \
    --parallel-mode seed \
    --use-jina \
    --llm_as_judge \
    --pass-at-k 2 \
