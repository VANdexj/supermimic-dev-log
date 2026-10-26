"""Regression tests for joint-target selection, history and new-run retention."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finish_run
from run_results import (meets_target, prune_checkpoints, read_evaluations,
                         select_checkpoints, validate_result, write_eval_csv)


def result(iteration, reward, failures):
    names = [f"clip_{i}" for i in range(40)]
    return {"iteration": iteration, "n_clips": 40, "norm_return": reward,
            "norm_return_framew": reward, "n_failed": failures, "failed": names[:failures],
            "per_clip": dict.fromkeys(names, reward), "train_wall_h": iteration / 1000,
            "saved_at": "saved", "eval_at": "evaluated", "config": {"num_envs": iteration}}


class RunResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def checkpoint(self, iteration, reward, failures):
        path = self.folder / f"ckpt_{iteration:06d}.pt"
        path.touch()
        data = result(iteration, reward, failures)
        path.with_name(path.stem + "_eval.json").write_text(json.dumps(data))
        return path, data

    def test_joint_target_beats_higher_return_with_failure(self):
        high = self.checkpoint(2000, 0.90, 1)
        success = self.checkpoint(4000, 0.83, 0)
        low = self.checkpoint(6000, 0.80, 0)
        best, best_return, fewest = select_checkpoints([high, success, low])
        self.assertEqual(best, success)
        self.assertEqual(best_return, high)
        self.assertEqual(fewest, success)
        self.assertFalse(meets_target(high[1]))
        self.assertFalse(meets_target(low[1]))

    def test_history_survives_weight_pruning(self):
        old = self.checkpoint(2000, 0.5, 20)
        new = self.checkpoint(8000, 0.6, 10)
        old[0].unlink()
        history = read_evaluations(self.folder)
        write_eval_csv(self.folder, history)
        write_eval_csv(self.folder, read_evaluations(self.folder))
        with (self.folder / "eval.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r["iteration"]) for r in rows], [2000, 8000])
        self.assertEqual(select_checkpoints(history)[0], new)

    def test_retention_keeps_best_latest_and_pending_candidates(self):
        best = self.checkpoint(2000, 0.8, 1)
        failures = self.checkpoint(4000, 0.7, 0)
        discarded = self.checkpoint(6000, 0.4, 20)
        pending = self.folder / "ckpt_008000.pt"
        pending.touch()
        for it in (10000, 12000, 14000):
            (self.folder / f"ckpt_{it:06d}.pt").touch()
        prune_checkpoints(self.folder, read_evaluations(self.folder), 8000)
        self.assertTrue(best[0].exists())
        self.assertTrue(failures[0].exists())
        self.assertTrue(pending.exists())
        self.assertFalse(discarded[0].exists())
        self.assertTrue(discarded[0].with_name(discarded[0].stem + "_eval.json").exists())
        self.assertFalse((self.folder / "ckpt_010000.pt").exists())
        self.assertTrue((self.folder / "ckpt_012000.pt").exists())
        self.assertTrue((self.folder / "ckpt_014000.pt").exists())

    def test_invalid_metrics_are_rejected(self):
        for key, value in (("n_failed", 1), ("norm_return", float("nan")), ("n_clips", 39), ("norm_return", 0.8)):
            with self.subTest(key=key, value=value):
                data = result(2000, 0.5, 0)
                data[key] = value
                with self.assertRaises(AssertionError):
                    validate_result(data)

    def test_failure_diagnostics_must_agree_with_lengths_and_causes(self):
        data = result(2000, 0.5, 1)
        data["per_clip_details"] = {
            name: {"steps": 50 if name == "clip_0" else 100, "total_steps": 100,
                   "completion_fraction": 0.5 if name == "clip_0" else 1.0,
                   "first_failure_time_s": 1.0 if name == "clip_0" else None,
                   "termination_reasons": ["height"] if name == "clip_0" else [],
                   "height_error_m": 0.26, "tilt_error": 0.1,
                   "max_height_error_m": 0.26, "max_tilt_error": 0.2}
            for name in data["per_clip"]}
        validate_result(data)
        data["per_clip_details"]["clip_0"]["first_failure_time_s"] = 2.0
        with self.assertRaises(AssertionError):
            validate_result(data)

    def test_stage_summary_uses_checkpoint_config_and_requires_confirmations(self):
        high = self.checkpoint(2000, 0.9, 1)
        success = self.checkpoint(8000, 0.83, 0)
        (self.folder / "args.json").write_text('{"num_envs": 999}')
        out = self.folder / "output"
        argv = ["finish_run.py", str(self.folder), "--out", str(out), "--metrics_only"]
        with patch.object(sys, "argv", argv), patch.object(finish_run, "run"):
            finish_run.main()
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["best_checkpoint"], str(success[0]))
        self.assertEqual(summary["highest_return_checkpoint"], str(high[0]))
        self.assertEqual(summary["config"], success[1]["config"])
        self.assertTrue(summary["candidate_target_reached"])
        self.assertFalse(summary["target_reached"])
        self.assertEqual(len(summary["evaluation_history"]), 2)

    def test_three_fresh_evaluations_must_all_pass(self):
        self.checkpoint(8000, 0.83, 0)
        for failures in (0, 1):
            with self.subTest(validation_failures=failures):
                out = self.folder / f"output_{failures}"
                evaluations = []

                def fake_run(*command):
                    if "supermimic/evaluate.py" in command:
                        output = Path(command[command.index("--output") + 1])
                        data = result(8000, 0.83, failures if len(evaluations) == 2 else 0)
                        output.write_text(json.dumps(data))
                        evaluations.append(output)

                argv = ["finish_run.py", str(self.folder), "--out", str(out)]
                with patch.object(sys, "argv", argv), patch.object(finish_run, "run", side_effect=fake_run), \
                     patch.object(finish_run.subprocess, "check_output", return_value='{"format":{"duration":"40"}}'):
                    finish_run.main()
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(len(set(evaluations)), 3)
                self.assertEqual(len(summary["validation_evaluations"]), 3)
                self.assertEqual(summary["target_reached"], failures == 0)


if __name__ == "__main__":
    unittest.main()
