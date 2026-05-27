"""Rule-based graders for oolong-synth (ported from platoon).

Source:
  https://github.com/abertsch72/oolong/blob/main/src/eval/eval_helpers.py
  platoon/plugins/oolong/platoon/oolong/eval_helpers.py

We only port the synth scorer for now; oolong-real requires an LLM judge
which we defer to a follow-up (see Appendix A.7 of arxiv 2605.06639).
"""

from __future__ import annotations

import ast
import re
from datetime import datetime
from typing import Any

try:
    import dateutil.parser  # type: ignore[import-untyped]
    _HAS_DATEUTIL = True
except ImportError:
    _HAS_DATEUTIL = False


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Extract the candidate answer string from the agent's finish() output.

    Returns (candidate, parse_confidence in {"low","med","high","vhigh"}).
    """
    parse_confidence = "low"
    if ":" not in answer:  # bad start
        if len(answer) < 20:  # short, return the whole thing
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence

    candidate = answer.split(":")[-1].strip()
    candidate = candidate.replace("*", "")  # strip markdown bold
    candidate = candidate.replace("[", "").replace("]", "")  # strip brackets

    parse_confidence = "med"
    if any(tok in answer for tok in ("User:", "Answer:", "Date:", "Label")):
        parse_confidence = "high"

    if len(candidate) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"

    return candidate, parse_confidence


def synth_process_response(datapoint: dict[str, Any], output: str) -> dict[str, Any]:
    """Compute (score, parse_confidence) for an oolong-synth task.

    Matches platoon's `synth_process_response` semantics:
      - exact-string match -> 1.0
      - NUMERIC: 0.75 ** |gold - pred| partial credit
      - DATE: dateutil-parsed equality
      - COMPARISON wording: substring match
      - else: 0.0
    """
    answer_str = datapoint["answer"]
    answer_type = datapoint.get("answer_type", "")

    # Parse gold: stored as a Python list literal like "[47]" or "['spam']",
    # or a datetime stamp for date answer types.
    if "datetime" not in answer_str:
        try:
            gold = ast.literal_eval(answer_str)[0]
        except (ValueError, SyntaxError, IndexError, TypeError):
            return {"score": 0.0, "parse_confidence": "low",
                    "reason": f"Could not parse gold answer: {answer_str!r}"}
    else:
        try:
            gold = datetime.strptime(answer_str, "[datetime.date(%Y, %m, %d)]")
        except (ValueError, TypeError):
            return {"score": 0.0, "parse_confidence": "low",
                    "reason": f"Could not parse gold date: {answer_str!r}"}

    trimmed, parse_confidence = synth_attempt_answer_parse(output)

    score = 0.0
    if str(trimmed) == str(gold):
        score = 1.0
    elif str(trimmed) in ("more common", "less common", "same frequency"):
        if str(trimmed) in str(gold):
            score = 1.0
    elif answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            t = int(trimmed)
            g = int(gold)
            score = 0.75 ** abs(g - t)
        except (ValueError, TypeError):
            parse_confidence = "low"  # didn't parse as a number — bad sign
    elif answer_type == "ANSWER_TYPE.DATE":
        if _HAS_DATEUTIL:
            try:
                t = dateutil.parser.parse(str(trimmed))
                score = 1.0 if t == gold else 0.0
            except (ValueError, TypeError):
                parse_confidence = "low"
        else:
            parse_confidence = "low"

    return {
        "score": float(score),
        "parse_confidence": parse_confidence,
        "candidate": trimmed,
        "gold": str(gold),
    }


# Placeholder for oolong-real (D&D) — would need an LLM judge.
# We define the function so the workflow can dispatch on dataset name,
# but it always returns score=0 for now. Wire up GPT-5-mini-style judge
# in a follow-up when we add `oolong-real` support.
def dnd_process_response(datapoint: dict[str, Any], output: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "parse_confidence": "n/a",
        "reason": "oolong-real LLM judge not yet implemented; score=0 placeholder.",
    }
