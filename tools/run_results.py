"""Shared checkpoint selection and evaluation history for local reproduction runs."""

import csv
import json
import math
from pathlib import Path


EVAL_FIELDS = ["iteration", "train_wall_h", "saved_at", "eval_at", "norm_return",
               "norm_return_framew", "n_failed", "failed"]


def meets_target(result):
    return result["norm_return"] >= 0.82 and result["n_failed"] == 0


def validate_result(result):
    assert result["n_clips"] == len(result["per_clip"]) == 40, "incomplete evaluation"
    assert math.isfinite(result["norm_return"]), "nonfinite aggregate return"
    assert math.isfinite(result["norm_return_framew"]), "nonfinite frame-weighted return"
    assert all(math.isfinite(v) for v in result["per_clip"].values()), "nonfinite per-clip return"
    assert math.isclose(result["norm_return"], sum(result["per_clip"].values()) / 40, abs_tol=1e-4), "aggregate return mismatch"
    assert len(set(result["failed"])) == result["n_failed"], "failure count mismatch"
    assert set(result["failed"]) <= result["per_clip"].keys(), "unknown failed clip"
    details = result.get("per_clip_details")
    if details is not None:
        assert details.keys() == result["per_clip"].keys(), "incomplete diagnostics"
        for name, detail in details.items():
            assert 0 < detail["steps"] <= detail["total_steps"], "invalid rollout length"
            assert math.isclose(detail["completion_fraction"], detail["steps"] / detail["total_steps"], abs_tol=1e-6)
            for key in ("height_error_m", "tilt_error", "max_height_error_m", "max_tilt_error"):
                assert math.isfinite(detail[key]) and detail[key] >= 0, "invalid tracking error"
            if name in result["failed"]:
                assert detail["termination_reasons"], "missing failure cause"
                assert math.isclose(detail["first_failure_time_s"], detail["steps"] * 0.02, abs_tol=1e-6)
            else:
                assert detail["first_failure_time_s"] is None and not detail["termination_reasons"]
                assert detail["steps"] == detail["total_steps"], "unfinished successful clip"


def read_evaluations(run_dir):
    results = []
    for path in sorted(Path(run_dir).glob("ckpt_*_eval.json")):
        result = json.loads(path.read_text())
        result.setdefault("iteration", int(path.name.split("_")[1]))
        validate_result(result)
        results.append((path.with_name(path.name.replace("_eval.json", ".pt")), result))
    return results


def select_checkpoints(results):
    available = [(p, r) for p, r in results if p.is_file()]
    if not available:
        raise ValueError("No evaluated checkpoint weights available")
    best_return = max(available, key=lambda item: (item[1]["norm_return"], -item[1]["n_failed"], item[1]["iteration"]))
    fewest_failures = max(available, key=lambda item: (-item[1]["n_failed"], item[1]["norm_return"], item[1]["iteration"]))
    best = max(available, key=lambda item: (meets_target(item[1]), item[1]["norm_return"],
                                           -item[1]["n_failed"], item[1]["iteration"]))
    return best, best_return, fewest_failures


def write_eval_csv(run_dir, results):
    path = Path(run_dir) / "eval.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for _, result in sorted(results, key=lambda item: item[1]["iteration"]):
            writer.writerow({k: "|".join(result[k]) if k == "failed" else result.get(k, "") for k in EVAL_FIELDS})
    tmp.replace(path)


def prune_checkpoints(run_dir, results, eval_interval):
    """Only called by the controller for its newly created run; keep JSON history."""
    paths = sorted(Path(run_dir).glob("ckpt_*.pt"))
    protected = set(paths[-2:])
    protected.update(item[0] for item in select_checkpoints(results))
    scored = {p for p, _ in results}
    for path in paths:
        iteration = int(path.stem.split("_")[1])
        if path in protected or (iteration % eval_interval == 0 and path not in scored):
            continue
        path.unlink()
