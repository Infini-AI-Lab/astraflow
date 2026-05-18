import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print()
                raise RuntimeError("Error in loading: {}".format(line))

PROMPT_TYPES = {
    "asearcher-reasoning": "{question}",
    "asearcher": """A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls a search engine to find necessary information, accesses web pages with certain urls, and provides the user with the answer. The Assistant conducts search by <search> query </search>, access cerain url by <access> url </access>, and the top search results and url page will be returned between <information> and </information>.  The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>, i.e. <answer> answer here </answer>. If there are multiple queries, ensure all answers are enclosed within <answer> </answer>, seperated with comma. \n\nNote: the question is a valid question and you should try to find a correct answer. \n\nUser: {question}\n\nAssistant: \n<think>""",
    "local-rag": """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
<|im_end|>
<|im_start|>assistant""",
}

def make_prompt(question: str, prompt_type):
    prompt = PROMPT_TYPES[prompt_type].format(question=question)
    return prompt

def prepare_data(data_name, args, save_async=False):
    output_dir = args.output_dir
    
    if os.path.exists(f"{args.data_dir}/{data_name}/{args.split}.jsonl"):
        print(f"Loading data from {args.data_dir}/{data_name}/{args.split}.jsonl")
        processes = [json.loads(ff) for ff in open(f"{args.data_dir}/{data_name}/{args.split}.jsonl", "r")]
    else:
        print(f"Loading data from {args.data_dir}/{data_name}/{args.split}.json")
        data_file = f"{args.data_dir}/{data_name}/{args.split}.json"
        processes = json.load(open(data_file, "r"))
    
    for idx, process in enumerate(processes):
        process["id"] = str(idx)

    if args.shuffle:
        np.random.shuffle(processes)
    
    if args.num_test_sample != -1 and args.num_test_sample < len(processes):
        processes = processes[:args.num_test_sample]

    if args.parallel_mode == "split":
        split_size = len(processes) // args.n_splits
        if args.split_id == args.n_splits:
            processes = processes[split_size * (args.n_splits - 1):]
        else:
            processes = processes[split_size * (args.split_id - 1): split_size * args.split_id]

    for p in processes:
        if "answer" not in p and "gt" in p:
            p["answer"] = p["gt"]
        if isinstance(p["answer"], list) and len(p["answer"]) == 1:
            p["answer"] = p["answer"][0]
        p["gt"] = p["answer"]
        p["prompt"] = make_prompt(p["question"], args.prompt_type)

    eval_dir = f"agent_eval_{args.max_tokens_per_call}"

    if args.parallel_mode == "seed":
        out_file_prefix = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
    else:
        out_file_prefix = f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}_{args.num_test_sample}_split{args.split_id}_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
    out_file = f"{output_dir}/{eval_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_n{args.n_sampling}.jsonl"

    os.makedirs(f"{output_dir}/{eval_dir}/{data_name}", exist_ok=True)

    if not save_async:
        process_id2data = {p['id']: p for p in processes}
        if not args.overwrite and os.path.exists(out_file):
            for ff in open(out_file, "r"):
                d = json.loads(ff)
                process_id2data[d["id"]] = d
        processes = list(process_id2data.values())
        return processes, out_file

    else:
        out_dir = out_file.rstrip(".jsonl")
        if args.overwrite or not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            return processes, out_dir
        else:
            process_id2data = {p['id']: p for p in processes}
            import glob
            for fname in glob.glob(f"{out_dir}/*.json"):
                try:
                    cur_id = os.path.basename(fname).rstrip(".json")
                    
                    with open(fname) as f:
                        cur_process = json.load(f)
                    
                    process_id2data[cur_id] = cur_process
                except:
                    print(f"remove {fname}")
                    os.system(f"rm -f {fname}")
            
            processes = list(process_id2data.values())
            return processes, out_dir

    