#!/usr/bin/env python3
"""Convert LiveCodeBench code_generation_lite JSONL to AstraFlow JSONL."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any


INTRO = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

STDIN_FORMAT = """Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```"""

STARTER_FORMAT = """You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```"""


def _load_json_maybe_compressed(value: str) -> Any:
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        payload = pickle.loads(zlib.decompress(base64.b64decode(value)))
        if isinstance(payload, str):
            return json.loads(payload)
        return payload


def _ensure_trailing_newline(value: Any) -> str:
    text = str(value)
    return text if text.endswith("\n") else text + "\n"


def _build_question(record: dict[str, Any]) -> str:
    starter_code = (record.get("starter_code") or "").rstrip()
    if starter_code:
        format_block = STARTER_FORMAT.format(starter_code=starter_code)
    else:
        format_block = STDIN_FORMAT

    question_content = (record.get("question_content") or "").strip()
    return (
        f"{INTRO}\n\n"
        f"### Question:\n{question_content}\n\n"
        f"### Format: {format_block}\n\n"
        "### Answer: (use the provided format with backticks)\n\n\n"
    )


def _build_input_output(
    cases: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    fn_name = metadata.get("func_name")
    if fn_name:
        return {
            "inputs": [str(case["input"]) for case in cases],
            "outputs": [str(case["output"]) for case in cases],
            "fn_name": fn_name,
        }

    return {
        "inputs": [_ensure_trailing_newline(case["input"]) for case in cases],
        "outputs": [_ensure_trailing_newline(case["output"]) for case in cases],
        "remote": False,
    }


def convert(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            record = json.loads(line)
            public_cases = json.loads(record.get("public_test_cases") or "[]")
            private_cases = _load_json_maybe_compressed(
                record.get("private_test_cases") or ""
            )
            metadata = json.loads(record.get("metadata") or "{}")
            cases = public_cases + private_cases

            query_id = record.get("question_id") or str(idx)
            output_record = {
                "question": _build_question(record),
                "input_output": json.dumps(
                    _build_input_output(cases, metadata),
                    ensure_ascii=False,
                ),
                "query_id": query_id,
                "idx": idx,
                "question_id": record.get("question_id"),
                "question_title": record.get("question_title"),
                "contest_id": record.get("contest_id"),
                "contest_date": record.get("contest_date"),
                "platform": record.get("platform"),
                "difficulty": record.get("difficulty"),
            }
            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    count = convert(args.src, args.dst)
    print(f"wrote {count} examples to {args.dst}")


if __name__ == "__main__":
    main()
