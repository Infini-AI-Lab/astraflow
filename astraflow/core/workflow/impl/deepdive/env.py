"""Per-agent action-based env for DeepDive web research.

Action format (TextCraft-style): each turn the agent emits exactly one
``<action type="NAME">{JSON args}</action>`` block. The workflow parses
the block and calls the matching method on this env, which returns a
text observation that becomes the next user message.

Actions (the only valid types):
  - ``search``  args: {"query": str, "n_docs": int = 5}
                Calls astraEnv.search; observation is the formatted list of passages.
  - ``spawn``   args: {"goal": str}
                Spawns one sub-agent (recursive researcher); observation is
                the sub's finish_message string.
  - ``finish``  args: {"answer": str}
                Submits the final answer and ends the episode.

Reward at finish() time:
  - Root tasks (id starts with "deepdive." and no "/sub_" marker) →
    binary-success LLM judge compares the agent's answer to the ground
    truth (platoon's verbatim rubric).
  - Sub-agent tasks (id contains "/sub_") → ChecklistGrader: a single
    LLM call that generates a per-goal checklist and scores it (ai-rubric
    RubricChecklistFast semantics).

The judge is always required (no rule-based grader exists for DeepDive).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from astraEnv.checklist import ChecklistGrader
from astraEnv.judge import extract_json, judge
from astraEnv.search import search as cmu_search
from astraflow.core.workflow.impl.deepdive.tasks import Task


# Rubric for grading the ROOT agent's answer against the ground truth.
# Ported verbatim from platoon's deepdive judge prompt (plugins/deepdive/
# platoon/deepdive/env.py:323-334). Binary success — matches platoon's
# Stage 1 calibration for parity.
_ROOT_RUBRIC_SYSTEM = (
    "We need to judge the performance of an deepresearch agent on a task. "
    "The task requires searching the web for information across various sources "
    "and synthesizing information together to answer a question.\n"
    "The agent may use subagents to solve parts of the task. Do not penalize the "
    "model for relying on subagents, unless the subtasks delegated to the subagents "
    "are not meaningful or useful for the task.\n"
    "You will be given the ground truth answer to the task and the agent's answer "
    "to the task.\n"
    "When comparing the agent's answer to the ground truth answer, it is acceptable "
    "to have minor formatting differences as long as the core information is equivalent.\n"
    "Please provide a reason and success flag (boolean value) in the following format:\n"
    "```json\n"
    "{\n"
    '    "reason": "Brief reasoning for success flag here.",\n'
    '    "success": <true|false>\n'
    "}\n"
    "```"
)

# Sub-agent grading uses ChecklistGrader (astraEnv.checklist) — an auto-
# generated per-goal checklist via two LLM calls (generate + evaluate).
# This matches platoon's RubricChecklistFast pattern. No fixed rubric
# constant needed here.


# Cap how many search results' text we surface in one observation. The
# CMU server returns ~1-2KB passages; with n_docs=5 that's ~10KB per
# search. We truncate each passage to keep contexts bounded.
DEFAULT_PASSAGE_TRUNCATE = 600

# Hard caps for action args.
MAX_SEARCH_N_DOCS = 20
DEFAULT_SEARCH_N_DOCS = 5


# Signature: launch_subagent(goal) -> finish_message string.
SpawnCallback = Callable[[str], Awaitable[str]]


class DeepDiveEnv:
    """Action-based env for one DeepDive research agent.

    No Python sandbox — the workflow calls action methods directly based
    on the parsed action block. Each method returns a text observation.
    """

    def __init__(
        self,
        task: Task,
        spawn_callback: SpawnCallback | None = None,
        passage_truncate: int = DEFAULT_PASSAGE_TRUNCATE,
        judge_model: str | None = None,
    ):
        self.task = task
        self.passage_truncate = passage_truncate
        self._spawn_cb = spawn_callback
        self.judge_model = judge_model

        self.finished: bool = False
        self.finish_payload: Any | None = None

        # Per-agent telemetry.
        self.subagent_launched: int = 0
        self.subagent_succeeded: float = 0.0
        self.search_calls: int = 0

    # ----------------------------------------------------------------- actions

    async def search(self, query: str, n_docs: int = DEFAULT_SEARCH_N_DOCS) -> str:
        """Run a CMU RAG search; return formatted passages as one text block.

        Errors are returned as text observations (NOT raised), so a flaky
        search call never crashes the rollout.
        """
        n = max(1, min(MAX_SEARCH_N_DOCS, int(n_docs)))
        q = str(query).strip()
        if not q:
            return "ERROR: search query is empty"
        self.search_calls += 1
        try:
            passages = await cmu_search(q, n_docs=n)
        except Exception as e:
            return f"ERROR: search failed: {type(e).__name__}: {e}"
        if not passages:
            return f"(no results for query: {q!r})"
        return self._format_passages(q, passages)

    async def spawn(self, goal: str) -> str:
        """Spawn one sub-agent; return its finish_message string."""
        if self._spawn_cb is None:
            return "ERROR: spawn unavailable (no spawn_callback)"
        g = str(goal).strip()
        if not g:
            return "ERROR: spawn goal is empty"
        self.subagent_launched += 1
        try:
            return await self._spawn_cb(g)
        except Exception as e:
            return f"ERROR: sub-agent crashed: {type(e).__name__}: {e}"

    def finish(self, answer: Any) -> None:
        """Mark this agent finished with `answer` as the payload."""
        self.finished = True
        self.finish_payload = answer

    # ----------------------------------------------------------------- format

    def _format_passages(self, query: str, passages: list[dict]) -> str:
        """Render a list of passages into one text observation."""
        lines = [f"Search results for: {query!r}"]
        for i, p in enumerate(passages, 1):
            src = p.get("source", "?")
            score = p.get("score")  # None when missing — distinguishable from a real NaN
            text = str(p.get("text", "")).strip()
            if len(text) > self.passage_truncate:
                text = text[: self.passage_truncate] + f"... [truncated, total {len(p.get('text', ''))} chars]"
            if score is None:
                score_str = "?"
            else:
                try:
                    f_score = float(score)
                    score_str = "?" if f_score != f_score else f"{f_score:.2f}"  # NaN check
                except (TypeError, ValueError):
                    score_str = "?"
            lines.append(f"[{i}] (source={src} score={score_str}) {text}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- evaluation

    async def evaluate(self) -> tuple[float, dict[str, Any]]:
        """Reward for THIS agent. Always uses an LLM judge.

        Routes:
          - root tasks: id starts with "deepdive." and has NO "/sub_" marker
                        → root rubric vs ground_truth
          - sub-agent tasks: id contains "/sub_"
                        → sub-agent rubric vs goal
        """
        if not self.finished:
            return 0.0, {"reason": "agent never called finish()"}

        task_id = self.task.id or ""
        output = "" if self.finish_payload is None else str(self.finish_payload)

        is_subagent = "/sub_" in task_id
        if not is_subagent and task_id.startswith("deepdive."):
            r = await self._grade_root_with_llm(output)
        else:
            r = await self._grade_subagent_with_llm(output)

        return float(r.get("score", 0.0)), r

    async def _grade_root_with_llm(self, output: str) -> dict[str, Any]:
        ground_truth = str(self.task.misc.get("ground_truth", "")).strip()
        user = (
            f"QUESTION: {self.task.goal}\n\n"
            f"GROUND TRUTH ANSWER: {ground_truth}\n\n"
            f"AGENT'S ANSWER: {output}"
        )
        return await self._call_judge(_ROOT_RUBRIC_SYSTEM, user)

    async def _grade_subagent_with_llm(self, output: str) -> dict[str, Any]:
        """Grade a sub-agent via the ai-rubric-style auto-generated checklist.

        Single LLM call (per `ai_rubric.core.checklist.RubricChecklistFast`):
        the model generates 3-5 atomic criteria from the goal AND scores
        them in one response. Returns its holistic `overall_score` in [0,1].

        Falls back gracefully on any failure (returns {"score": 0.0, ...}).
        """
        grader = ChecklistGrader(
            goal=self.task.goal,
            judge_model=self.judge_model,
            temperature=1.0,  # platoon parity
        )
        # Context: just the finish_message for now. Adding action history
        # (turns the agent took) would match platoon more closely; deferred.
        context = f"Final Message:\n{output}"
        score, reason = await grader.aevaluate(context=context)
        return {
            "score": max(0.0, min(1.0, float(score))),
            "reason": reason,
        }

    async def _call_judge(self, system: str, user: str) -> dict[str, Any]:
        """Common judge invocation. Parses platoon-style binary success.

        Expected response JSON: {"reason": str, "success": bool}.
        Score = 1.0 if success is True, else 0.0.

        Falls back to legacy continuous "score" field if "success" missing,
        so test fixtures with continuous scores still parse — but the
        production rubrics now ask for binary success.
        """
        judge_kwargs: dict[str, Any] = {"temperature": 1.0}  # platoon parity
        if self.judge_model:
            judge_kwargs["model"] = self.judge_model
        try:
            raw = await judge(system=system, user=user, **judge_kwargs)
        except Exception as e:
            return {"score": 0.0, "reason": f"judge call failed: {e}"}
        try:
            parsed = extract_json(raw)
            if "success" in parsed:
                score = 1.0 if bool(parsed["success"]) else 0.0
            elif "score" in parsed:
                # Legacy continuous score path — kept for test fixtures and
                # for any future rubric that returns continuous scores.
                score = float(parsed["score"])
                score = max(0.0, min(1.0, score))
            else:
                raise KeyError("response missing both 'success' and 'score'")
        except Exception as e:
            return {
                "score": 0.0,
                "reason": f"judge response unparseable: {e}",
                "judge_raw": raw,
            }
        return {
            "score": score,
            "reason": str(parsed.get("reason", "")),
            "judge_raw": raw,
        }
