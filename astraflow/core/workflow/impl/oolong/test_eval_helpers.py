"""Unit tests for the rule-based oolong-real (D&D) grader.

Ported faithfully from platoon's `dnd_process_response` and verified
against the documented scoring rules.

Run:
    pytest astraflow/core/workflow/impl/oolong/test_eval_helpers.py -v
"""

from __future__ import annotations

import pytest

from astraflow.core.workflow.impl.oolong.eval_helpers import (
    dnd_parse_answer,
    dnd_parse_response,
    dnd_process_response,
)


# --------------------------------------------------------------------------
# dnd_parse_answer — type dispatch
# --------------------------------------------------------------------------


def test_parse_answer_int():
    assert dnd_parse_answer("42") == 42


def test_parse_answer_negative_int():
    assert dnd_parse_answer("-7") == -7


def test_parse_answer_list_via_comma():
    assert dnd_parse_answer("alice, bob, carol") == ["alice", "bob", "carol"]


def test_parse_answer_list_drops_empty_after_split():
    assert dnd_parse_answer("a,,b,") == ["a", "b"]


def test_parse_answer_falls_back_to_string():
    assert dnd_parse_answer("the killer") == "the killer"


# --------------------------------------------------------------------------
# dnd_parse_response — \boxed extraction + fallback
# --------------------------------------------------------------------------


def test_parse_response_boxed_text_wrapped():
    """\\boxed{\\text{X}} should be matched FIRST and stripped to X."""
    val, conf = dnd_parse_response(r"my final answer: \boxed{\text{alice}}")
    assert val == "alice"
    assert conf == "high"


def test_parse_response_boxed_plain():
    val, conf = dnd_parse_response(r"the count is \boxed{17}")
    assert val == 17
    assert conf == "high"


def test_parse_response_boxed_list():
    val, conf = dnd_parse_response(r"\boxed{alice, bob}")
    assert val == ["alice", "bob"]
    assert conf == "high"


def test_parse_response_no_box_uses_raw_with_med_confidence():
    val, conf = dnd_parse_response("the answer is 5")
    # No box, no comma, not parseable as int → string fallback
    assert val == "the answer is 5"
    assert conf == "med"


def test_parse_response_empty_returns_low_confidence():
    val, conf = dnd_parse_response("   ")
    assert val == ""
    assert conf == "low"


# --------------------------------------------------------------------------
# dnd_process_response — type-aware scoring
# --------------------------------------------------------------------------


def test_score_int_exact_match():
    r = dnd_process_response({"answer": "10"}, r"\boxed{10}")
    assert r["score"] == 1.0
    assert r["parse_confidence"] == "high"


def test_score_int_off_by_one_partial_credit():
    r = dnd_process_response({"answer": "10"}, r"\boxed{11}")
    assert r["score"] == pytest.approx(0.75)


def test_score_int_off_by_three():
    r = dnd_process_response({"answer": "10"}, r"\boxed{13}")
    # 0.75 ** 3 = 0.421875
    assert r["score"] == pytest.approx(0.75**3)


def test_score_str_exact_match_case_insensitive():
    r = dnd_process_response(
        {"answer": "Alice"}, r"my guess is \boxed{\text{alice}}"
    )
    assert r["score"] == 1.0


def test_score_str_mismatch():
    r = dnd_process_response({"answer": "alice"}, r"\boxed{\text{bob}}")
    assert r["score"] == 0.0


def test_score_list_full_overlap():
    r = dnd_process_response(
        {"answer": "alice, bob"}, r"\boxed{alice, bob}"
    )
    assert r["score"] == 1.0


def test_score_list_partial_overlap():
    """Jaccard-like: |gold ∩ pred| / |gold|. Half of gold is recovered → 0.5."""
    r = dnd_process_response(
        {"answer": "alice, bob"}, r"\boxed{alice, carol}"
    )
    assert r["score"] == pytest.approx(0.5)


def test_score_list_no_overlap():
    r = dnd_process_response(
        {"answer": "alice, bob"}, r"\boxed{x, y}"
    )
    assert r["score"] == 0.0


def test_score_type_mismatch_returns_zero():
    """Gold is int, model returns a string → score 0 (no partial credit)."""
    r = dnd_process_response({"answer": "10"}, r"\boxed{\text{ten}}")
    assert r["score"] == 0.0


def test_score_unboxed_int_still_works():
    """Without \\boxed{}, raw output is parsed with med confidence; still scores."""
    r = dnd_process_response({"answer": "10"}, "10")
    assert r["score"] == 1.0
    assert r["parse_confidence"] == "med"


def test_score_empty_output_is_zero():
    r = dnd_process_response({"answer": "10"}, "")
    assert r["score"] == 0.0
    assert r["parse_confidence"] == "low"


def test_result_includes_metadata():
    r = dnd_process_response({"answer": "10"}, r"\boxed{11}")
    assert "attempted_parse" in r
    assert "gold" in r
    assert "full_answer" in r
    assert r["gold"] == 10
    assert r["attempted_parse"] == 11
