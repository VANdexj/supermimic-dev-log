"""Fetch the LAFAN1 G1 data and validate it before converting the robot (container only)."""

import json
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data"
BASE = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"


def fetch(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def download(path, dest, revision, size=None):
    if dest.exists() and dest.stat().st_size and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    content = fetch(f"{BASE}/resolve/{revision}/{path}")
    if size is not None and len(content) != size:
        raise ValueError(f"Download size mismatch: {path}")
    temp.write_bytes(content)
    os.replace(temp, dest)
    print(f"downloaded {path}", flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = ROOT / "source.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
    else:
        source = {"dataset": BASE, "revision": json.loads(fetch(API))["sha"]}
        manifest.write_text(json.dumps(source, indent=2) + "\n")
    revision = source["revision"]
    for folder, dest, suffix, count in (
        ("g1", ROOT / "lafan1_g1", ".csv", 40),
        ("robot_description/g1/meshes", ROOT / "g1_description/meshes", ".stl", 38),
    ):
        entries = json.loads(fetch(f"{API}/tree/{revision}/{folder}?limit=1000"))
        files = [x for x in entries if x["type"] == "file" and x["path"].lower().endswith(suffix)]
        if len(files) != count:
            raise ValueError(f"Expected {count} files in {folder}, got {len(files)}")
        for entry in files:
            path = entry["path"]
            download(path, dest / Path(path).name, revision, entry["size"])
        if {p.name for p in dest.iterdir() if p.suffix.lower() == suffix} != {Path(x["path"]).name for x in files}:
            raise ValueError(f"Unexpected extra files in {dest}")
    download("robot_description/g1/g1_29dof_rev_1_0.urdf",
             ROOT / "g1_description/g1_29dof_rev_1_0.urdf", revision)
    frames = 0
    for path in sorted((ROOT / "lafan1_g1").glob("*.csv")):
        data = np.loadtxt(path, delimiter=",")
        if data.ndim != 2 or data.shape[1] != 36 or not np.isfinite(data).all():
            raise ValueError(f"Invalid motion data: {path}")
        frames += len(data)
    if frames != 264705:
        raise ValueError(f"Expected 264705 source frames, got {frames}")
    print(f"Validated 40 clips, {frames} frames, 38 meshes; dataset revision {revision}")


if __name__ == "__main__":
    main()
