"""Runnable example for astraEnv.judge.

Run with:
    # source your Fireworks key first
    set -a && source ~/.fireworks_key && set +a

    python astraEnv/judge_example.py

To try your own cases, edit the CASES list below or pass arbitrary
(system, user) text via --system / --user flags.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from astraEnv.judge import extract_json, judge

# ----------------------------------------------------------------------
# Default rubric — a generic "did the output satisfy the goal" grader.
# Replace with whatever fits your task.
# ----------------------------------------------------------------------

DEFAULT_SYSTEM = """You grade a sub-agent's output against its delegated goal.
Return ONLY JSON in this exact format:
{"score": float in [0, 1], "reason": "<one short sentence>"}

Scoring guide:
  1.0 = output fully and correctly satisfies the goal
  0.5 = output partially correct or partially complete
  0.0 = output is wrong, empty, or a refusal

Do not include any other text — JSON only."""


# ----------------------------------------------------------------------
# Example cases — (goal, output, what you expect)
# Add your own here.
# ----------------------------------------------------------------------

CASES: list[tuple[str, str, str]] = [
    # math
    ("Compute the sum of [3, 7, 12].", "22", "high"),
    ("Compute the sum of [3, 7, 12].", "21", "low (off by one)"),
    ("Compute the sum of [3, 7, 12].", "I am not sure.", "low (refusal)"),
    # listing
    ("List the first 3 prime numbers.", "[2, 3, 5]", "high"),
    ("List the first 3 prime numbers.", "[1, 2, 3]", "low (1 is not prime)"),
    # translation
    ('Translate "hello" to French.', "bonjour", "high"),
    ('Translate "hello" to French.', "hola", "low (Spanish, not French)"),
    # extraction
    (
        "Extract all dates from: 'meeting on 2024-03-15, follow-up 2024-04-02'.",
        '["2024-03-15", "2024-04-02"]',
        "high",
    ),
    (
        "Extract all dates from: 'meeting on 2024-03-15, follow-up 2024-04-02'.",
        "There are no dates.",
        "low",
    ),
]


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


async def grade_one(
    system: str, goal: str, output: str
) -> tuple[str, str, float, str]:
    """Return (user_message, raw_response, score, reason)."""
    user = f"GOAL: {goal}\n\nOUTPUT: {output}"
    response = await judge(system=system, user=user)
    parsed = extract_json(response)
    return user, response, float(parsed["score"]), str(parsed.get("reason", "")).strip()


def _hr(char: str = "─", n: int = 80) -> str:
    return char * n


async def run_batch(system: str, cases: list[tuple[str, str, str]]) -> None:
    # Print the system prompt once at the top — it's the same for every case.
    print(_hr("═"))
    print("SYSTEM PROMPT (sent with every case)")
    print(_hr("═"))
    print(system)
    print()

    # Fire all grades in parallel.
    tasks = [grade_one(system, goal, output) for goal, output, _ in cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Per-case detail: user message + raw response + parsed.
    for i, ((goal, output, expected), result) in enumerate(zip(cases, results), 1):
        print(_hr("═"))
        print(f"CASE {i}/{len(cases)}  (expected: {expected})")
        print(_hr("═"))
        print("[USER MESSAGE]")
        print(f"GOAL: {goal}")
        print()
        print(f"OUTPUT: {output}")
        print()

        if isinstance(result, Exception):
            print(f"[FAIL] {type(result).__name__}: {result}")
            print()
            continue

        user_msg, raw, score, reason = result
        print("[RAW MODEL RESPONSE]")
        print(raw)
        print()
        print("[PARSED]")
        print(f"  score:  {score:.2f}")
        print(f"  reason: {reason}")
        print()

    # Compact summary table at the end.
    print(_hr("═"))
    print("SUMMARY")
    print(_hr("═"))
    print(f'{"score":>5}  {"output":40}  {"expected":35}')
    print(_hr("-"))
    for (goal, output, expected), result in zip(cases, results):
        if isinstance(result, Exception):
            print(f" FAIL  {output[:40]:40}  {expected[:35]:35}")
            continue
        _, _, score, _ = result
        print(f"{score:5.2f}  {output[:40]:40}  {expected[:35]:35}")


async def run_single(system: str, user: str) -> None:
    print(_hr("═"))
    print("[SYSTEM MESSAGE]")
    print(_hr("═"))
    print(system)
    print()
    print(_hr("═"))
    print("[USER MESSAGE]")
    print(_hr("═"))
    print(user)
    print()
    response = await judge(system=system, user=user)
    print(_hr("═"))
    print("[RAW MODEL RESPONSE]")
    print(_hr("═"))
    print(response)
    print()
    try:
        parsed = extract_json(response)
        print(_hr("═"))
        print("[PARSED JSON]")
        print(_hr("═"))
        for k, v in parsed.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"=== Could not parse JSON: {e} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        help="Custom system prompt. If omitted, uses the built-in default rubric.",
    )
    parser.add_argument(
        "--user",
        help="Custom user prompt. If given, runs a single grade instead of the batch.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (e.g. accounts/fireworks/models/deepseek-v4-pro).",
    )
    args = parser.parse_args()

    system = args.system or DEFAULT_SYSTEM

    # Single-call mode if --user is provided
    if args.user:
        if args.model:
            # judge() doesn't take model via this helper — simplest: monkey-patch
            from astraEnv import judge as judge_module
            judge_module._DEFAULT_MODEL = args.model
        asyncio.run(run_single(system, args.user))
        return 0

    # Batch mode
    if args.model:
        from astraEnv import judge as judge_module
        judge_module._DEFAULT_MODEL = args.model
    asyncio.run(run_batch(system, CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
