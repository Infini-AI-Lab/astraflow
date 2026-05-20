from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from astraflow.core.workflow.utils import logging

logger = logging.getLogger(__name__)

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn

        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn

        return geometry3k_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )


class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision."""

    def __init__(self, try_extract_without_anchor=True, precision: int = 6):
        import logging as _logging
        _logging.getLogger("math_verify").setLevel(_logging.CRITICAL)

        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            precision=precision,
        )

    def verify(self, response: str, ground_truth: str) -> float:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                ret_score, _ = self.verify_func([ground_truth], [response])
                return float(ret_score)
            except TimeoutException:
                logger.debug(
                    f"Math-Verify timeout for ground_truth={ground_truth[:50]}...; treating as 0 reward"
                )
                return 0.0
            except Exception:
                logger.debug(
                    f"Math-Verify extraction failed for ground_truth={ground_truth[:50]}..."
                )
                return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER
