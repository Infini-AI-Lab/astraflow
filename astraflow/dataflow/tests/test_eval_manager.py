import pytest
import torch

from astraflow.dataflow.eval_manager import EvalManager


def _result(task_id: int, reward: float) -> dict:
    return {
        "task_id": task_id,
        "ok": True,
        "result": {"rewards": torch.tensor([reward], dtype=torch.float32)},
    }


def test_eval_manager_keeps_mean_success_rate_and_adds_pass_at_k():
    manager = EvalManager()
    manager.configure_agent(
        agent_name="agent0",
        eval_datasets={"aime24": (object(), 4, {"workflow_cls": "rlvr"})},
    )

    task_id_to_run = {
        1: ("aime24", 0, 0),
        2: ("aime24", 0, 1),
        3: ("aime24", 1, 0),
        4: ("aime24", 1, 1),
        5: ("aime24", 2, 0),
        6: ("aime24", 2, 1),
        7: ("aime24", 3, 0),
        8: ("aime24", 3, 1),
    }
    results = [
        _result(1, 1.0),
        _result(2, 0.0),
        _result(3, 0.0),
        _result(4, 0.0),
        _result(5, 0.0),
        _result(6, 0.0),
        _result(7, 0.0),
        _result(8, 0.0),
    ]

    eval_results = manager._aggregate_results(
        agent_name="agent0",
        results=results,
        task_id_to_run=task_id_to_run,
    )

    ds = eval_results["datasets"]["aime24"]
    assert ds["avg@k"] == pytest.approx(12.5)
    assert ds["pass_k"] == 4
    assert ds["pass@k"] == pytest.approx(50.0)
    assert ds["sample_count"] == 2
    assert eval_results["overall_avg@k"] == pytest.approx(12.5)
    assert eval_results["overall_pass@k"] == pytest.approx(50.0)
