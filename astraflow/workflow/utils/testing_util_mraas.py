"""MRAAS-specific wrapper around the single-turn code verifier.

Two modes:

* Legacy (no --case_index): runs ALL test cases in a single ``run_test`` call
  and writes per-case feedback for the 2-model retry prompt. Kept for callers
  that haven't migrated.
* Per-case (--case_index N): slices the input to a single case, runs it, and
  writes ``{tmp_id}-{N}-output.json``. The orchestrator
  (``code_execution_mraas.call_verify_collect_all``) uses this mode with a
  hard wall-clock deadline + SIGKILL escalation per case.

To avoid the ~14 s ``astraflow/__init__.py`` import cost on every per-case
subprocess spawn, ``run_test`` is loaded lazily by file path via
``importlib.util`` (no package init triggered). Combined with ``python -P``
so the script's directory isn't injected into ``sys.path``, the standalone
startup is ~0.25 s.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback


def _load_run_test():
    """Load ``run_test`` from sibling testing_util.py without triggering
    astraflow.__init__ (which transitively pulls in torch/sglang and costs
    ~14 s)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_testing_util_standalone",
        os.path.join(here, "testing_util.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to locate testing_util.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_test


def run_test_collect_all(sample, test=None, debug=False, timeout=6):
    io_spec = json.loads(sample["input_output"])
    total_cases = len(io_spec["inputs"])

    run_test = _load_run_test()

    # Run ALL cases in one call (reliability_guard only fires once).
    try:
        results, info = run_test(
            sample=sample,
            test=test,
            debug=debug,
            timeout=timeout,
        )
    except Exception as exc:
        results = [False] * max(1, total_cases)
        info = {
            "error": repr(exc),
            "error_code": -4,
            "error_message": f"Verifier wrapper exception: {exc}",
        }

    if not isinstance(results, list):
        results = [results]
    results = list(results[:total_cases]) + [False] * max(0, total_cases - len(results))

    details = []
    for case_index in range(total_cases):
        passed = bool(case_index < len(results) and results[case_index] is True)
        detail = {
            "passed": passed,
            "inputs": io_spec["inputs"][case_index],
            "expected": io_spec["outputs"][case_index],
            "output": io_spec["outputs"][case_index] if passed else None,
            "error": None if passed else info.get("error"),
            "error_code": 0 if passed else info.get("error_code", -2),
            "error_message": "Passed" if passed else info.get("error_message", "Wrong Answer"),
        }
        details.append(detail)

    return results, {"details": details}


def run_one_case(sample, test, case_index, debug=False, timeout=6):
    """Run a single test case and return (passed, detail_dict)."""
    io_spec = json.loads(sample["input_output"])
    if not (0 <= case_index < len(io_spec["inputs"])):
        return False, {
            "passed": False,
            "inputs": None,
            "expected": None,
            "output": None,
            "error": f"case_index out of range: {case_index}",
            "error_code": -4,
            "error_message": "Invalid case_index",
        }

    sliced_io = dict(io_spec)
    sliced_io["inputs"] = [io_spec["inputs"][case_index]]
    sliced_io["outputs"] = [io_spec["outputs"][case_index]]
    sliced_sample = dict(sample)
    sliced_sample["input_output"] = json.dumps(sliced_io)

    run_test = _load_run_test()
    try:
        results, info = run_test(
            sample=sliced_sample,
            test=test,
            debug=debug,
            timeout=timeout,
        )
    except Exception as exc:
        results = [False]
        info = {
            "error": repr(exc),
            "error_code": -4,
            "error_message": f"Verifier wrapper exception: {exc}",
        }

    if not isinstance(results, list) or not results:
        results = [False]
    passed = results[0] is True

    detail = {
        "passed": passed,
        "inputs": io_spec["inputs"][case_index],
        "expected": io_spec["outputs"][case_index],
        "output": io_spec["outputs"][case_index] if passed else None,
        "error": None if passed else info.get("error"),
        "error_code": 0 if passed else info.get("error_code", -2),
        "error_message": "Passed" if passed else info.get("error_message", "Wrong Answer"),
    }
    return passed, detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_id", type=str, required=True)
    parser.add_argument(
        "--case_index",
        type=int,
        default=None,
        help="If set, run only this single test case index.",
    )
    args = parser.parse_args()

    with open(f"/tmp/{args.tmp_id}-input.json", encoding="utf-8") as temp_file:
        input_data = json.load(temp_file)

    if args.case_index is not None:
        # Per-case mode: write {tmp_id}-{N}-output.json.
        try:
            passed, detail = run_one_case(
                sample=input_data["sample"],
                test=input_data["test"],
                case_index=args.case_index,
                debug=input_data.get("debug", False),
                timeout=input_data.get("timeout", 6),
            )
            saved_result = {"result": [passed], "info": {"detail": detail}}
        except Exception as exc:
            saved_result = {
                "result": [False],
                "info": {
                    "detail": {
                        "passed": False,
                        "inputs": None,
                        "expected": None,
                        "output": None,
                        "error": repr(exc),
                        "error_code": -4,
                        "error_message": f"Verifier wrapper crashed: {exc}",
                    },
                    "traceback": traceback.format_exc(),
                },
            }
        with open(
            f"/tmp/{args.tmp_id}-{args.case_index}-output.json",
            "w",
            encoding="utf-8",
        ) as temp_file:
            json.dump(saved_result, temp_file)
    else:
        # Legacy whole-program mode.
        try:
            result, info = run_test_collect_all(**input_data)
            saved_result = {"result": result, "info": info}
        except Exception as exc:
            try:
                io_spec = json.loads(input_data["sample"]["input_output"])
                n_cases = len(io_spec.get("inputs", []))
            except Exception:
                n_cases = 1
            saved_result = {
                "result": [False] * max(1, n_cases),
                "info": {
                    "details": [],
                    "error": repr(exc),
                    "error_code": -4,
                    "error_message": f"Verifier wrapper crashed: {exc}",
                    "traceback": traceback.format_exc(),
                },
            }
        with open(
            f"/tmp/{args.tmp_id}-output.json", "w", encoding="utf-8"
        ) as temp_file:
            json.dump(saved_result, temp_file)
