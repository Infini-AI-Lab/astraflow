from __future__ import annotations

import json

import pytest

from astraflow.dataflow.dataset.terminal_bench import get_harbor_task_path_dataset
from astraflow.core.workflow.impl.terminal_bench_harbor import (
    TerminalBenchHarborRLWorkflow,
    TerminalBenchHarborWorkflow,
    _collect_harbor_rewards,
    _extract_reward_from_result,
    _harbor_result_to_training_sequence,
    _load_harbor_trial_result,
)


def test_extract_reward_from_harbor_trial_result():
    result = {
        "verifier_result": {
            "rewards": {
                "reward": 1.0,
            }
        }
    }

    assert _extract_reward_from_result(result) == pytest.approx(1.0)


def test_collect_harbor_rewards_reads_trial_result_json(tmp_path):
    run_dir = tmp_path / "job" / "2026-05-15__17-17-54"
    result_file = run_dir / "trial-a" / "result.json"
    result_file.parent.mkdir(parents=True)
    result_file.write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}})
    )

    with pytest.raises(RuntimeError, match="missing agent_result/verifier_result"):
        _collect_harbor_rewards(tmp_path / "job")

    result_file.write_text(
        json.dumps(
            {
                "agent_result": {},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )

    assert _collect_harbor_rewards(tmp_path / "job") == [pytest.approx(1.0)]


def test_load_harbor_trial_result_uses_single_trial_result(tmp_path):
    aggregate = tmp_path / "job" / "2026-05-15__17-17-54" / "result.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(json.dumps({"n_total_trials": 1, "stats": {}}))

    trial = aggregate.parent / "trial-a" / "result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "agent_result": {"rollout_details": []},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )

    assert _load_harbor_trial_result(tmp_path / "job") == (
        trial,
        json.loads(trial.read_text()),
    )


def test_harbor_task_path_dataset_loads_skyrl_layout(tmp_path, monkeypatch):
    root = tmp_path / "CodeContests"
    task = root / "task-a"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("do task\n")
    monkeypatch.setenv("HARBOR_DATA", str(root))

    dataset = get_harbor_task_path_dataset(
        path="$HARBOR_DATA",
        dataset_name="test_harbor_tasks",
    )

    assert len(dataset) == 1
    assert dataset[0]["task_path"] == str(task)
    assert dataset[0]["prompt"] == str(task)
    assert dataset[0]["task_name"] == "task-a"


def test_harbor_result_to_training_sequence_uses_rollout_details():
    result = {
        "agent_result": {
            "rollout_details": [
                {
                    "prompt_token_ids": [[10, 11], [10, 11, 12, 20, 30]],
                    "completion_token_ids": [[12, 20], [31]],
                    "logprobs": [[-0.1, -0.2], [-0.3]],
                }
            ]
        },
        "verifier_result": {"rewards": {"reward": 0.5}},
    }

    seq = _harbor_result_to_training_sequence(result, reward=0.5, version=7)

    assert seq["input_ids"].tolist() == [[10, 11, 12, 20, 30, 31]]
    assert seq["loss_mask"].tolist() == [[0, 0, 1, 1, 0, 1]]
    assert seq["logprobs"].tolist()[0] == pytest.approx(
        [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]
    )
    assert seq["versions"].tolist() == [[-1, -1, 7, 7, -1, 7]]
    assert seq["attention_mask"].tolist() == [[True, True, True, True, True, True]]
    assert seq["rewards"].tolist() == [pytest.approx(0.5)]


def test_harbor_result_to_training_sequence_requires_rollout_details():
    with pytest.raises(ValueError, match="collect_rollout_details=true"):
        _harbor_result_to_training_sequence(
            {"agent_result": {"rollout_details": []}},
            reward=0.0,
            version=1,
        )


def test_build_command_supports_conda_wrapped_harbor(tmp_path):
    class DummyEngine:
        addresses = ["127.0.0.1:12345"]

    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
        harbor_command=[
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "harbor-tb2",
            "harbor",
        ],
    )

    cmd = workflow._build_command(DummyEngine(), "build-pmars", tmp_path)

    assert cmd[:7] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "harbor-tb2",
        "harbor",
        "run",
    ]
    assert "--include-task-name" in cmd
    assert "--yes" in cmd
    assert "build-pmars" in cmd
    assert "api_base=http://127.0.0.1:12345/v1" in cmd


def test_build_command_supports_harbor_task_path(tmp_path):
    class DummyEngine:
        addresses = ["127.0.0.1:12345"]

    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
    )

    cmd = workflow._build_command(
        DummyEngine(),
        "task-a",
        tmp_path / "job",
        task_path=task_dir,
    )

    assert "--path" in cmd
    assert str(task_dir) in cmd
    assert "--include-task-name" not in cmd
    assert "task-a" not in cmd


def test_build_command_round_robins_inferred_api_bases(tmp_path):
    class DummyEngine:
        addresses = ["127.0.0.1:12345", "127.0.0.1:12346"]

    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
    )

    cmd0 = workflow._build_command(DummyEngine(), "task-a", tmp_path / "a")
    cmd1 = workflow._build_command(DummyEngine(), "task-b", tmp_path / "b")
    cmd2 = workflow._build_command(DummyEngine(), "task-c", tmp_path / "c")

    assert "api_base=http://127.0.0.1:12345/v1" in cmd0
    assert "api_base=http://127.0.0.1:12346/v1" in cmd1
    assert "api_base=http://127.0.0.1:12345/v1" in cmd2


def test_build_command_round_robins_configured_api_bases(tmp_path):
    class DummyEngine:
        addresses = ["127.0.0.1:12345"]

    workflow = TerminalBenchHarborWorkflow(
        gconfig=object(),
        tokenizer=None,
        api_base=[
            "http://127.0.0.1:20001/v1",
            "http://127.0.0.1:20002/v1",
        ],
    )

    cmd0 = workflow._build_command(DummyEngine(), "task-a", tmp_path / "a")
    cmd1 = workflow._build_command(DummyEngine(), "task-b", tmp_path / "b")

    assert "api_base=http://127.0.0.1:20001/v1" in cmd0
    assert "api_base=http://127.0.0.1:20002/v1" in cmd1


def test_rl_workflow_enables_rollout_details_and_disables_summarize(tmp_path):
    class DummyEngine:
        addresses = ["127.0.0.1:12345"]

    workflow = TerminalBenchHarborRLWorkflow(
        gconfig=object(),
        tokenizer=None,
    )

    cmd = workflow._build_command(DummyEngine(), "task-a", tmp_path)

    assert "collect_rollout_details=true" in cmd
    assert "enable_summarize=false" in cmd
