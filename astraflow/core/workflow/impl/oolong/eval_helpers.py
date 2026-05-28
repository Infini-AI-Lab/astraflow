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


# --------------------------------------------------------------------------
# oolong-real (D&D) grader — rule-based, ported faithfully from platoon's
# `dnd_process_response` (plugins/oolong/platoon/oolong/eval_helpers.py).
#
# Contrary to what one might expect, oolong-real does NOT use an LLM judge:
# the dataset's gold answers are typed (int / str / list[str]) and admit
# rule-based scoring with partial credit. The model is expected to wrap its
# final answer in \boxed{...} (math-style); if missing we fall back to the
# raw output with lower parse_confidence.
# --------------------------------------------------------------------------


def dnd_parse_answer(answer: str) -> int | str | list[str]:
    """Coerce a string into int / str / list[str] based on content shape."""
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[int | str | list[str], str]:
    """Extract the candidate answer + parse_confidence label.

    Order of preference:
      1. ``\\boxed{\\text{X}}``  -> high confidence
      2. ``\\boxed{X}``          -> high confidence
      3. raw stripped output     -> med confidence
      4. empty                   -> low confidence
    """
    answer = answer.strip()
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer)
    if not match:
        match = re.search(r"\\boxed\{([^}]*)\}", answer)
    if match:
        return dnd_parse_answer(match.group(1)), "high"
    if not answer:
        return answer, "low"
    return dnd_parse_answer(answer), "med"


def dnd_process_response(datapoint: dict[str, Any], output: str) -> dict[str, Any]:
    """Score a model answer against a D&D gold answer.

    Type-aware scoring:
      - int  vs int          : ``0.75 ** |gap|`` (partial credit, decay)
      - str  vs str          : exact match after strip().lower()  -> 0 or 1
      - list vs list         : Jaccard overlap |gold & pred| / |gold|
      - type mismatch        : 0.0
    """
    gold = dnd_parse_answer(datapoint["answer"])
    trimmed_output, parse_confidence = dnd_parse_response(output)

    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = 0.75 ** abs(gold - trimmed_output)
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = len(overlap) / len(gold) if gold else 0.0

    return {
        "score": float(score),
        "parse_confidence": parse_confidence,
        "attempted_parse": trimmed_output,
        "gold": gold,
        "full_answer": output,
    }
