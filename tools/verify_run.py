"""Check actual training logs and a resumable checkpoint (run inside Docker)."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected_iteration", type=int)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "args.json").read_text())
    with (args.run_dir / "train_log.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "empty training log"
    for row in rows:
        assert all(math.isfinite(float(v)) for v in row.values()), f"nonfinite training metric: {row}"
    assert any(int(r["iteration"]) > config["wm_only_iters"] for r in rows), "no policy/critic update logged"
    ck = torch.load(args.run_dir / "last.pt", map_location="cpu")
    assert ck["iteration"] == int(rows[-1]["iteration"]), "final checkpoint/log iteration mismatch"
    if args.expected_iteration is not None:
        assert ck["iteration"] == args.expected_iteration, "resume or budget stop failed"
    for model in ("wm", "pi", "critic"):
        assert all(torch.isfinite(t).all() for t in ck[model].values()), f"nonfinite {model} weights"
    assert all(key in ck for key in ("opt_wm", "opt_pi", "opt_v")), "optimizer states missing"
    assert (args.run_dir / f"ckpt_{ck['iteration']:06d}.pt").is_file(), "final evaluation checkpoint missing"
    print(f"Verified training log, model weights and resume state at iteration {ck['iteration']}")


if __name__ == "__main__":
    main()
