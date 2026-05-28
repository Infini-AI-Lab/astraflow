"""Single-call auto-checklist grader — local replacement for
``ai-rubric``'s ``rubric.core.checklist.RubricChecklistFast``.

Matches the upstream package's behavior:

  - **One LLM call** (not two) — the model generates the checklist and
    scores every item in a single response.
  - **Continuous per-item scores** (0-1, not binary pass/fail) so the LLM
    can reflect partial satisfaction.
  - **Holistic ``overall_score``** chosen by the LLM, not mechanical
    ``passed / total`` (lets critical items dominate non-critical ones).
  - **No caching** — fresh checklist every call (matches upstream).

System prompt ported verbatim from
``ai_rubric-0.2.4/rubric/prompts/generate-rubric-checklist-fast-system.jinja``.

Usage::

    from astraEnv.checklist import ChecklistGrader

    grader = ChecklistGrader(goal="find the actor's birth year")
    score, reason = await grader.aevaluate(context=trajectory_text)

Uses ``astraEnv.judge.judge`` for the LLM call so we keep our retry / key
handling. Temperature defaults to 1.0 for parity with the upstream package.
"""

from __future__ import annotations

from typing import Any

from astraEnv.judge import extract_json, judge


# Verbatim from ai-rubric 0.2.4:
# rubric/prompts/generate-rubric-checklist-fast-system.jinja
_SYSTEM_PROMPT = (
    "We are building a rubric to evaluate a task. We will do this by "
    "decomposing success criteria for the task into a checklist\n"
    "and reasoning about the task success using this checklist. The "
    "checklist should comprehensively test that the task is successfully "
    "completed.\n\n"
    "The rubric checklist should be as comprehensive as possible, and "
    "should be able to evaluate the task in a way that is fair and accurate.\n\n"
    "The rubric checklist should be as concise as possible, and should be "
    "able to be easily understood by a human.\n\n"
    "The rubric checklist should be as easy to evaluate as possible.\n\n"
    "To evaluate a task on a checklist, you may consider the following "
    "procedure:\n"
    "1. For each criterion, reason whether it is critical or non-critical.\n"
    "2. For each criterion, provide a score between 0 and 1 for how well "
    "the task satisfies the criterion.\n"
    "3. Consider the overall progress towards task completion and allow "
    "for partial credit when generating the overall score.\n\n"
    "# Output Format\n"
    "```json\n"
    "{\n"
    '    "checklist": [\n'
    '        "...", // a list of strings\n'
    "    ],\n"
    '    "checklist_scores": [\n'
    "        0.0, // between 0 and 1\n"
    "    ],\n"
    '    "reasoning": "...",\n'
    '    "overall_score": 0.0 // between 0 and 1\n'
    "}\n"
    "```"
)


def _build_user_prompt(task: str, context: str) -> str:
    """Mirrors generate-rubric-checklist-fast-user.jinja."""
    return f"# Task\n{task}\n\n{context}\n\n# Your Evaluation Output"


class ChecklistGrader:
    """Single-call checklist grader matching ai-rubric's RubricChecklistFast.

    Parameters
    ----------
    goal : str
        The task goal the agent was given.
    judge_model : str | None
        Optional override for the judge model. None = astraEnv.judge default.
    temperature : float
        Sampling temperature. 1.0 matches the upstream package's default.
    """

    def __init__(
        self,
        goal: str,
        *,
        judge_model: str | None = None,
        temperature: float = 1.0,
    ):
        self.goal = goal
        self.judge_model = judge_model
        self.temperature = temperature
        # Most-recent parsed response — exposed for inspection / debugging.
        self.last_checklist: list[str] = []
        self.last_checklist_scores: list[float] = []
        self.last_reasoning: str = ""
        self.last_overall_score: float | None = None

    def _judge_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"temperature": self.temperature}
        if self.judge_model:
            kw["model"] = self.judge_model
        return kw

    async def aevaluate(self, *, context: str) -> tuple[float, str]:
        """Run one LLM call that generates+scores the checklist.

        Returns
        -------
        score : float in [0, 1]
            The LLM's holistic ``overall_score``.
        reason : str
            The LLM's reasoning. Empty string on failure.

        On any failure (network, parse, out-of-range score) returns
        ``(0.0, error_message)`` — never raises.
        """
        user = _build_user_prompt(self.goal, context)
        try:
            raw = await judge(
                system=_SYSTEM_PROMPT, user=user, **self._judge_kwargs()
            )
        except Exception as e:
            return 0.0, f"checklist call failed: {e}"

        try:
            parsed = extract_json(raw)
        except Exception as e:
            return 0.0, f"checklist response unparseable: {e}"

        try:
            overall = float(parsed.get("overall_score", 0.0))
        except (TypeError, ValueError) as e:
            return 0.0, f"overall_score not a number: {e}"

        # Clamp defensively; the upstream package raises if out of [0,1],
        # but we prefer to log and continue so a flaky judge response
        # never crashes the rollout.
        overall = max(0.0, min(1.0, overall))

        # Stash for inspection.
        checklist = parsed.get("checklist") or []
        scores = parsed.get("checklist_scores") or []
        self.last_checklist = [str(x) for x in checklist if isinstance(x, (str, int, float))]
        self.last_checklist_scores = []
        for s in scores:
            try:
                self.last_checklist_scores.append(float(s))
            except (TypeError, ValueError):
                continue
        self.last_reasoning = str(parsed.get("reasoning", ""))
        self.last_overall_score = overall

        return overall, self.last_reasoning


async def grade_with_checklist(
    goal: str,
    context: str,
    *,
    judge_model: str | None = None,
    temperature: float = 1.0,
) -> tuple[float, str]:
    """Convenience wrapper: build a grader and evaluate in one call."""
    grader = ChecklistGrader(goal, judge_model=judge_model, temperature=temperature)
    return await grader.aevaluate(context=context)
