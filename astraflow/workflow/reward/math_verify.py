from astraflow.workflow.registry import register_reward
from astraflow.workflow.reward import get_math_verify_worker
from astraflow.workflow.utils import logging

logger = logging.getLogger(__name__)


# Sympy-based math equivalence check, applicable to any boxed-answer
# math dataset (DeepScaleR, MATH-500, AIME, AMC, Minerva, OlympiadBench,
# GSM8K).
@register_reward("math_verify")
def math_verify_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in math_verify_reward_fn", exc_info=True)
        return 0.0
