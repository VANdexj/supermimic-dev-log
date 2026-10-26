"""Controller budget and terminal-state checks without starting GPU work."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import retrain_local
from test_run_results import result


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.previous_cwd)
        self.run_dir = self.repo / "logs" / "test_run"

    def resume_state(self, deadline):
        self.run_dir.mkdir(parents=True)
        state = {"status": "training", "deadline_epoch": deadline, "stage_hours": 8,
                 "preflight_passed": True, "num_envs": 256, "restore_checks_passed": True,
                 "export_check_passed": True, "stages": [], "target_reached": False}
        (self.run_dir / "controller_state.json").write_text(json.dumps(state))

    def test_resume_does_not_reset_expired_deadline(self):
        deadline = time.time() - 10
        self.resume_state(deadline)
        with patch.object(retrain_local, "REPO", self.repo), \
             patch.object(sys, "argv", ["retrain_local.py", "test_run", "--resume"]), \
             patch.object(retrain_local, "run") as runner:
            retrain_local.main()
        state = json.loads((self.run_dir / "controller_state.json").read_text())
        self.assertEqual(state["deadline_epoch"], deadline)
        self.assertEqual(state["status"], "budget_exhausted")
        runner.assert_not_called()

    def test_deadline_terminates_child_process(self):
        with self.assertRaises(TimeoutError):
            retrain_local.run([sys.executable, "-c", "import time; time.sleep(60)"],
                              self.repo / "child.log", time.time() - 1)

    def test_failed_capacity_probes_prevent_formal_training(self):
        calls = []

        def fake_run(command, *args, **kwargs):
            calls.append(command)
            return {"returncode": -9, "interrupted": None, "capacity_failure": True}

        with patch.object(retrain_local, "REPO", self.repo), \
             patch.object(sys, "argv", ["retrain_local.py", "test_run", "--hours", "2"]), \
             patch.object(retrain_local, "capture_environment"), \
             patch.object(retrain_local, "run", side_effect=fake_run):
            retrain_local.main()
        state = json.loads((self.run_dir / "controller_state.json").read_text())
        self.assertEqual(state["status"], "resource_limit")
        self.assertEqual(len(state["calibrations"]), 2)
        train_calls = [c for c in calls if "supermimic/train.py" in c]
        self.assertEqual(len(train_calls), 2)
        self.assertTrue(all("calibration_" in c[c.index("--log_dir") + 1] for c in train_calls))
        self.assertFalse((self.run_dir / "last.pt").exists())

    def test_confirmed_target_stops_after_one_stage_with_scaled_schedule(self):
        self.resume_state(time.time() + 7200)
        checkpoint = self.run_dir / "ckpt_008000.pt"
        checkpoint.touch()
        calls = []

        def fake_run(command, *args, **kwargs):
            calls.append(command)
            if "supermimic/train.py" in command:
                (self.run_dir / "last.pt").touch()
            if "tools/finish_run.py" in command:
                out = self.repo / "results" / "test_run"
                out.mkdir(parents=True, exist_ok=True)
                (out / "summary.json").write_text('{"target_reached": true}')
            return {"returncode": 0, "interrupted": None}

        with patch.object(retrain_local, "REPO", self.repo), \
             patch.object(sys, "argv", ["retrain_local.py", "test_run", "--resume"]), \
             patch.object(retrain_local, "run", side_effect=fake_run), \
             patch.object(retrain_local, "read_evaluations", return_value=[(checkpoint, result(8000, 0.83, 0))]):
            retrain_local.main()
        state = json.loads((self.run_dir / "controller_state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertTrue(state["target_reached"])
        self.assertEqual(len(state["stages"]), 1)
        train = next(c for c in calls if "supermimic/train.py" in c)
        self.assertEqual(train[train.index("--wm_only_iters") + 1], "8000")
        self.assertEqual(train[train.index("--lr_decay_iters") + 1], "4000000")
        self.assertEqual(train[train.index("--warmup_steps") + 1], "1000")


if __name__ == "__main__":
    unittest.main()
