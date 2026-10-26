"""Evaluate stage candidates and export the selected policy with complete evaluation history."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_results import (meets_target, read_evaluations, select_checkpoints,
                         validate_result, write_eval_csv)


def run(*command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="evaluate only the final checkpoint and render 2s")
    parser.add_argument("--eval_interval", type=int, default=8000)
    parser.add_argument("--metrics_only", action="store_true", help="score stage candidates without rendering")
    args = parser.parse_args()
    if args.eval_interval < 1:
        parser.error("--eval_interval must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "tools/verify_run.py", args.run_dir)
    paths = sorted(args.run_dir.glob("ckpt_*.pt"))
    if not paths:
        raise ValueError("No checkpoints found")
    candidates = [p for p in paths if not args.smoke and int(p.stem.split("_")[1]) % args.eval_interval == 0]
    if paths[-1] not in candidates:
        candidates.append(paths[-1])
    for path in candidates:
        result_path = path.with_name(path.stem + "_eval.json")
        if not result_path.exists():
            run(sys.executable, "supermimic/evaluate.py", "--run_dir", args.run_dir, "--ckpt", path)
    results = read_evaluations(args.run_dir)
    write_eval_csv(args.run_dir, results)
    (best_path, best), best_return, fewest_failures = select_checkpoints(results)
    config = best.get("config")
    if config is None:
        import torch
        config = torch.load(best_path, map_location="cpu")["args"]
    confirmations = []
    trajectory = args.out / "trajectories.npz"
    if not args.metrics_only:
        # Record in a fresh process, preserving the original scored JSON/history.
        recorded_path = args.out / "recorded_eval.json"
        run(sys.executable, "supermimic/evaluate.py", "--run_dir", args.run_dir, "--ckpt", best_path,
            "--record", trajectory, "--output", recorded_path)
        recorded = json.loads(recorded_path.read_text())
        validate_result(recorded)
        candidate_target = meets_target(best)
        best = recorded
        config = recorded["config"]
        if candidate_target:
            confirmations.append(recorded)
            for i in (2, 3):
                validation_path = args.out / f"validation_{best_path.stem}_{i}.json"
                run(sys.executable, "supermimic/evaluate.py", "--run_dir", args.run_dir, "--ckpt", best_path,
                    "--output", validation_path)
                result = json.loads(validation_path.read_text())
                validate_result(result)
                confirmations.append(result)
        run(sys.executable, "tools/plot.py", args.run_dir, "--out", args.out / "success_curve.png")
        video = args.out / "highlight.mp4"
        clips = ["dance2_subject1"] if args.smoke else ["dance2_subject1", "jumps1_subject1", "fallAndGetUp1_subject1", "sprint1_subject2"]
        run(sys.executable, "tools/render.py", trajectory, video, "--clips", *clips,
            "--seconds", 2 if args.smoke else 10, "--title", f"Local policy it {best['iteration']}")
        run("ffmpeg", "-y", "-loglevel", "error", "-i", video, "-filter_complex",
            "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            args.out / "highlight.gif")
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "json", str(video)], text=True)
        assert float(json.loads(probe)["format"]["duration"]) > 0, "empty video"
    target_reached = len(confirmations) == 3 and all(meets_target(r) for r in confirmations)
    summary = {
        "best_checkpoint": str(best_path), "evaluation": best, "config": config,
        "highest_return_checkpoint": str(best_return[0]),
        "fewest_failures_checkpoint": str(fewest_failures[0]),
        "readme_target": {"norm_return": 0.82, "n_failed": 0},
        "candidate_target_reached": meets_target(best), "target_reached": target_reached,
        "validation_evaluations": confirmations,
        "evaluation_history": [{"checkpoint": str(p), "iteration": r["iteration"],
                                "norm_return": r["norm_return"], "n_failed": r["n_failed"]} for p, r in results],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "REPORT.md").write_text(
        "# Local reproduction results\n\n"
        f"Selected checkpoint: `{best_path}`; resume training with `{args.run_dir / 'last.pt'}`.\n\n"
        f"Evaluation return: **{best['norm_return']:.4f}**; failed clips: **{best['n_failed']}/40**.\n\n"
        f"Target (return >= 0.82, 0/40 failures, three fresh evaluations) reached: **{target_reached}**.\n\n"
        f"Highest return: `{best_return[0]}`; fewest failures: `{fewest_failures[0]}`.\n\n"
        "[summary.json](summary.json) contains the selected checkpoint's configuration, per-clip "
        "diagnostics, confirmation evaluations and complete stage evaluation history.\n\n"
        + ("Stage metrics only; video export is pending.\n" if args.metrics_only else
           "![Curve](success_curve.png)\n\n![Tracking](highlight.gif)\n"))
    print(f"Artifacts: {args.out}; return {best['norm_return']:.4f}, failed {best['n_failed']}/40; confirmed={target_reached}", flush=True)


if __name__ == "__main__":
    main()
