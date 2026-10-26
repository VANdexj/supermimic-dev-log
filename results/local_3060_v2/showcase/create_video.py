"""Render selected real policy rollouts using the project's MuJoCo renderer.

Run from the repository root inside the supermimic container:
    python results/local_3060_v2/showcase/create_video.py --preview
    python results/local_3060_v2/showcase/create_video.py
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import render
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FPS, WIDTH, HEIGHT, SECONDS = 25, 1280, 720, 10
SCENES = [
    ("DANCE / FULL BODY CONTROL", "dance2_subject3", 128),
    ("FIGHT / STRIKES & FOOTWORK", "fight1_subject5", 147),
    ("JUMPS / TAKEOFF & LANDING", "jumps1_subject1", 50),
    ("SPRINT / DYNAMIC LOCOMOTION", "sprint1_subject2", 193),
    ("RECOVERY / FALL & GET UP", "fallAndGetUp3_subject1", 64),
    ("DANCE / FINISH IN MOTION", "dance1_subject3", 105),
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    evaluation = json.loads((OUT / "recorded_eval.json").read_text())
    with np.load(OUT / "trajectories.npz") as source:
        d = {k: source[k] for k in source.files}
    names = list(d["names"])
    model = mujoco.MjModel.from_xml_string(render.two_robot_xml())
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.3, 135, -12
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 25)
    small_font = ImageFont.truetype(font_path, 17)
    video = OUT / "showcase.mp4"
    encoder = None
    if not args.preview:
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-metadata", "title=SuperMimic G1 Motion Showcase", str(video),
        ], stdin=subprocess.PIPE)
    previews = []
    scenes = []
    for chapter, (label, clip, start) in enumerate(SCENES):
        c = names.index(clip)
        t0, t1 = start * 50, (start + SECONDS) * 50
        assert t1 < int(d["clip_len"][c])
        assert d["alive"][t0:t1, c].all(), clip
        assert evaluation["per_clip_details"][clip]["completion_fraction"] == 1, clip
        steps = [0, 50, 100, 150, 200] if args.preview else range(FPS * SECONDS)
        for frame in steps:
            t = t0 + frame * 2
            f = int(d["f"][t, c])
            data.qpos[:3] = d["root_p"][t, c]
            data.qpos[3:7] = d["root_quat"][t, c]
            data.qpos[7:36] = d["q"][t, c]
            assert np.isfinite(data.qpos[:36]).all()
            data.qpos[36:39] = d["ref_root_p"][f]
            data.qpos[36:38] = data.qpos[:2]
            data.qpos[39:43] = render._mat_to_quat(d["ref_root_R"][f])
            data.qpos[43:72] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.65]
            renderer.update_scene(data, camera=camera)
            img = Image.fromarray(renderer.render()).convert("RGBA")
            overlay = Image.new("RGBA", img.size)
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((0, 0, WIDTH, 96), fill=(16, 24, 37, 235))
            draw.rectangle((0, HEIGHT - 60, WIDTH, HEIGHT), fill=(16, 24, 37, 235))
            draw.text((26, 12), "SUPERMIMIC  /  G1 MOTION SHOWCASE", font=small_font, fill=(133, 192, 255))
            draw.text((26, 40), label, font=title_font, fill="white")
            draw.text((955, 17), "V2  /  POLICY 048000", font=small_font, fill="white")
            draw.text((955, 48), f"SEQUENCE {chapter + 1:02d} / {len(SCENES):02d}    |    1x SPEED", font=small_font, fill=(195, 207, 223))
            draw.text((26, HEIGHT - 47), "SOLID: simulated G1 policy    /    BLUE: reference motion", font=small_font, fill="white")
            draw.text((26, HEIGHT - 25), clip, font=small_font, fill=(195, 207, 223))
            draw.text((940, HEIGHT - 38), f"SOURCE  {start + frame / FPS:06.2f}s", font=small_font, fill=(195, 207, 223))
            progress = (chapter * FPS * SECONDS + frame + 1) / (len(SCENES) * FPS * SECONDS)
            draw.rectangle((0, HEIGHT - 4, int(WIDTH * progress), HEIGHT), fill=(86, 172, 255))
            img = Image.alpha_composite(img, overlay).convert("RGB")
            if args.preview:
                img.thumbnail((384, 216))
                previews.append(img)
            else:
                encoder.stdin.write(np.asarray(img).tobytes())
        scenes.append({"label": label, "clip": clip, "source_start_s": start,
                       "source_end_s": start + SECONDS, "video_start_s": chapter * SECONDS,
                       "full_clip_return": evaluation["per_clip"][clip],
                       "full_clip_completed": True})
        print(f"Rendered {chapter + 1}/{len(SCENES)}: {clip}", flush=True)
    renderer.close()
    if args.preview:
        sheet = Image.new("RGB", (384 * 5, 216 * len(SCENES)))
        for i, img in enumerate(previews):
            sheet.paste(img, ((i % 5) * 384, (i // 5) * 216))
        sheet.save(OUT / "preview.jpg", quality=92)
    else:
        encoder.stdin.close()
        assert encoder.wait() == 0, "Video encoding failed"
        manifest = {"checkpoint": "logs/local_3060_v2/ckpt_048000.pt",
                    "evaluation": "recorded_eval.json", "trajectory": "trajectories.npz",
                    "duration_s": len(SCENES) * SECONDS, "fps": FPS,
                    "resolution": [WIDTH, HEIGHT], "playback_speed": 1,
                    "reference_xy_anchored_to_policy": True, "scenes": scenes,
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {video}", flush=True)


if __name__ == "__main__":
    main()
