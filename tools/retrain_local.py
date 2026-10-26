"""Bounded, serial calibration -> fresh training -> full-clip evaluation on the local GPU."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from run_results import meets_target, prune_checkpoints, read_evaluations, select_checkpoints


REPO = Path(__file__).resolve().parents[1]
COMMON = ["--buffer_steps", "1000", "--wm_batches", "10", "--horizon", "32", "--wm_window", "8",
          "--goal_frames", "10", "--hist", "10", "--amp", "bf16", "--no_compile", "--warmup_steps", "1000"]


def profile_args(num_envs):
    return ["--num_envs", str(num_envs), "--wm_batch", str(num_envs // 2), "--pi_batch", str(num_envs)]


def run(command, log, deadline, monitor=False, graceful=False):
    print("+", " ".join(map(str, command)), flush=True)
    peak_gpu_mib, peak_rss, interrupted = 0, 0, None
    with Path(log).open("a") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if monitor:
                    used = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                    peak_gpu_mib = max(peak_gpu_mib, int(used.splitlines()[0]))
                    memory_stat = Path("/sys/fs/cgroup/memory.stat")
                    if memory_stat.exists():
                        stats = dict(line.split() for line in memory_stat.read_text().splitlines())
                        peak_rss = max(peak_rss, int(stats.get("anon", 0)) + int(stats.get("shmem", 0)))
                if time.time() >= deadline:
                    interrupted = "deadline"
                    break
                if graceful and shutil.disk_usage(REPO).free < 2 * 1024**3:
                    interrupted = "disk_reserve"
                    break
                time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=90 if graceful else 15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        code = process.wait()
    result = {"returncode": code, "peak_gpu_mib": peak_gpu_mib, "peak_container_rss_bytes": peak_rss,
              "interrupted": interrupted}
    if interrupted == "deadline":
        raise TimeoutError(f"Budget exhausted running {command[1]}; see {log}")
    if code != 0:
        tail = Path(log).read_text(errors="replace")[-6000:]
        if monitor and (code == -9 or "out of memory" in tail.lower()):
            result["capacity_failure"] = True
        else:
            raise RuntimeError(f"Command exited {code}; see {log}\n{tail}")
    return result


def save_state(path, state):
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def capture_environment(run_dir):
    commands = {"gpu.txt": ["nvidia-smi"], "packages.txt": [sys.executable, "-m", "pip", "freeze"],
                "isaaclab_commit.txt": ["git", "-C", "IsaacLab", "rev-parse", "HEAD"],
                "git_commit.txt": ["git", "rev-parse", "HEAD"], "source.diff": ["git", "diff", "--binary"]}
    for name, command in commands.items():
        (run_dir / name).write_bytes(subprocess.check_output(command))
    (run_dir / "python_version.txt").write_text(sys.version + "\n")
    shutil.copyfile("data/source.json", run_dir / "data_source.json")
    # Include untracked implementation files as well as the working-tree diff.
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("supermimic", "tools", "tests", "docker"):
            for path in Path(folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path)
        for path in [*Path(".").glob("run*.sh"), Path("README.md"), Path("AGENTS.md")]:
            if path.is_file():
                archive.write(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", nargs="?", default="local_3060_v2")
    parser.add_argument("--hours", type=float, default=48)
    parser.add_argument("--stage_hours", type=float, default=8)
    parser.add_argument("--calibration_iterations", type=int, default=200)
    parser.add_argument("--calibration_seconds", type=float, default=900)
    parser.add_argument("--resume", action="store_true", help="resume this controller with its original deadline")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        parser.error("invalid run name")
    if not 1 < args.hours <= 48 or args.stage_hours <= 0 or args.calibration_iterations < 2 or args.calibration_seconds <= 0:
        parser.error("require 1 < hours <= 48, positive stage/calibration time, and at least two calibration iterations")
    os.chdir(REPO)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    run_dir = Path("logs") / args.run_name
    out = Path("results") / args.run_name
    if not args.resume:
        if out.exists():
            parser.error(f"Results already exist: {out}")
        run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "controller_state.json"
    with (run_dir / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            state = json.loads(state_path.read_text())
            if state["status"] in ("complete", "budget_exhausted", "resource_limit"):
                parser.error(f"Run has ended: {state['status']}")
        else:
            started = time.time()
            state = {"status": "preflight", "started_at_epoch": started,
                     "deadline_epoch": started + args.hours * 3600, "stage_hours": args.stage_hours,
                     "calibration_iterations": args.calibration_iterations,
                     "calibration_seconds": args.calibration_seconds, "calibrations": [], "stages": [],
                     "target_reached": False, "baseline": {"norm_return": 0.5375851988792419, "n_failed": 20}}
        save_state(state_path, state)
        deadline = state["deadline_epoch"]
        try:
            if not state.get("preflight_passed"):
                capture_environment(run_dir)
                for script in (["-m", "unittest", "discover", "-s", "tests", "-v"],
                               ["tools/test_env.py", "--num_envs", "16"], ["tools/test_reset.py"]):
                    run([sys.executable, *script], run_dir / "preflight.log", deadline)
                state["preflight_passed"] = True
                save_state(state_path, state)
            if "num_envs" not in state:
                state["status"] = "calibrating"
                save_state(state_path, state)
                # Successful completed probes can be reused after a controller restart.
                done = {item["num_envs"] for item in state["calibrations"]}
                for num_envs in (256, 128):
                    if num_envs in done:
                        continue
                    probe = run_dir / f"calibration_{num_envs}"
                    if probe.exists():
                        # Never overwrite an interrupted probe on restart.
                        probe = run_dir / f"calibration_{num_envs}_{int(time.time())}"
                    probe.mkdir()
                    report_path = probe / "resources.json"
                    command = [sys.executable, "supermimic/train.py", *COMMON, *profile_args(num_envs),
                               "--wm_only_iters", "1", "--iterations", str(state["calibration_iterations"]),
                               "--max_train_seconds", str(state["calibration_seconds"]), "--log_dir", str(probe),
                               "--ckpt_every", "1000000", "--log_every", "50", "--resource_report", str(report_path)]
                    measured = run(command, probe / "stdout.log", deadline, monitor=True, graceful=True)
                    measured.update(num_envs=num_envs, probe_dir=str(probe))
                    passed = measured["returncode"] == 0 and measured["interrupted"] is None
                    if passed:
                        run([sys.executable, "tools/verify_run.py", probe], probe / "verify.log", deadline)
                        measured.update(json.loads(report_path.read_text()))
                        rows = list(csv.DictReader((probe / "train_log.csv").open()))
                        active = [r for r in rows if int(r["iteration"]) > 1]
                        measured["transitions_per_second"] = num_envs * 2.5 * float(active[-1]["it_per_s"])
                        passed = (measured["peak_gpu_mib"] <= 5120
                                  and measured["peak_cuda_reserved_bytes"] <= 5 * 1024**3
                                  and measured["peak_container_rss_bytes"] <= 7 * 1024**3
                                  and measured["iterations"] >= 2)
                    measured["passed"] = passed
                    state["calibrations"].append(measured)
                    save_state(state_path, state)
                valid = [item for item in state["calibrations"] if item["passed"]]
                if not valid:
                    state["status"] = "resource_limit"
                    save_state(state_path, state)
                    return
                selected = max(valid, key=lambda item: item["transitions_per_second"])
                state["num_envs"] = selected["num_envs"]
                # Verify restore and off-interval time-limit saving on disposable probe weights.
                probe = Path(selected["probe_dir"])
                rows = list(csv.DictReader((probe / "train_log.csv").open()))
                iteration = int(rows[-1]["iteration"])
                for limit, target in ((0, iteration + 2), (0.001, iteration + 3)):
                    run([sys.executable, "supermimic/train.py", *COMMON, *profile_args(state["num_envs"]),
                         "--wm_only_iters", "1", "--iterations", str(target), "--log_dir", probe,
                         "--resume", probe / "last.pt", "--max_train_seconds", str(limit),
                         "--ckpt_every", "1000000", "--log_every", "50"], probe / "resume.log", deadline)
                    run([sys.executable, "tools/verify_run.py", probe, "--expected_iteration", str(target)],
                        probe / "verify.log", deadline)
                state["restore_checks_passed"] = True
                save_state(state_path, state)
            if not state.get("restore_checks_passed"):
                raise RuntimeError("Restore checks incomplete; inspect calibration logs before resuming")
            if not state.get("export_check_passed"):
                selected = next(item for item in state["calibrations"] if item["num_envs"] == state["num_envs"] and item["passed"])
                run([sys.executable, "tools/finish_run.py", selected["probe_dir"], "--out", run_dir / "smoke_results", "--smoke"],
                    run_dir / "smoke_export.log", deadline)
                state["export_check_passed"] = True
                save_state(state_path, state)
            scale = 4096 // state["num_envs"]
            base = [sys.executable, "supermimic/train.py", *COMMON, *profile_args(state["num_envs"]),
                    "--wm_only_iters", str(500 * scale), "--lr_decay_iters", str(250000 * scale),
                    "--iterations", "1000000", "--ckpt_every", "2000", "--log_every", "50", "--log_dir", str(run_dir)]
            exported = False
            while deadline - time.time() > 3600:
                stage = len(state["stages"]) + 1
                seconds = min(state["stage_hours"] * 3600, deadline - time.time() - 3600)
                state["status"] = "training"
                state["active_stage"] = stage
                save_state(state_path, state)
                command = [*base, "--max_train_seconds", str(seconds),
                           "--resource_report", str(run_dir / f"stage_{stage}_resources.json")]
                if (run_dir / "last.pt").exists():
                    command.extend(["--resume", str(run_dir / "last.pt")])
                measured = run(command, run_dir / "stdout.log", deadline - 3300, monitor=True, graceful=True)
                if measured["returncode"] != 0:
                    raise RuntimeError(f"Training failed: {measured}; see {run_dir / 'stdout.log'}")
                state["status"] = "evaluating"
                save_state(state_path, state)
                run([sys.executable, "tools/finish_run.py", run_dir, "--out", out, "--metrics_only"],
                    run_dir / "evaluation.log", deadline)
                results = read_evaluations(run_dir)
                (best_path, best), _, _ = select_checkpoints(results)
                state["stages"].append({"stage": stage, "finished_at_epoch": time.time(),
                                        "best_checkpoint": str(best_path), "norm_return": best["norm_return"],
                                        "n_failed": best["n_failed"], "resources": measured})
                if meets_target(best):
                    run([sys.executable, "tools/finish_run.py", run_dir, "--out", out],
                        run_dir / "export.log", deadline)
                    exported = True
                    if json.loads((out / "summary.json").read_text())["target_reached"]:
                        state["target_reached"] = True
                        break
                prune_checkpoints(run_dir, results, 8000)
                save_state(state_path, state)
                if measured["interrupted"] == "disk_reserve":
                    if shutil.disk_usage(REPO).free < 3 * 1024**3:
                        raise RuntimeError("Insufficient disk after checkpoint retention")
                exported = False
            if not exported and (run_dir / "last.pt").exists():
                state["status"] = "exporting"
                save_state(state_path, state)
                run([sys.executable, "tools/finish_run.py", run_dir, "--out", out], run_dir / "export.log", deadline)
                state["target_reached"] = json.loads((out / "summary.json").read_text())["target_reached"]
            state["status"] = "complete" if state["target_reached"] else "budget_exhausted"
        except (Exception, KeyboardInterrupt) as exc:
            state["status"] = ("budget_exhausted" if isinstance(exc, TimeoutError) else
                               "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            state["error"] = str(exc)
            save_state(state_path, state)
            raise
        save_state(state_path, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
