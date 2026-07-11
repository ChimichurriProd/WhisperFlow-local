#!/usr/bin/env python3
"""Process the per-STT-model idle videos into app clips.

Input: four 960x960 MP4s (Kling 2.5, black background) — one per model mood
(sleepy base .. electric large). Output: assets/marvin/idle_<model>/ with
61 RGBA frames at 256x256, background keyed out, aura/sparks kept with
smooth luminance alpha (same look as the super_saiyan clip).

Keying per frame:
  1. flood-fill from the borders over near-black pixels -> alpha 0
  2. inside the head disc (r < 0.46) -> fully opaque
  3. outside the disc, not flooded -> alpha ramps with brightness, so the
     electric aura/sparks fade smoothly instead of ending in a hard edge

Usage: build_model_idles.py <videos_dir> [<out_assets_marvin_dir>]
Expects <videos_dir>/idle_{base,small,medium,large}.mp4.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation

N_OUT = 61          # frames kept per clip (every 2nd of ~121)
SIZE = 256
BG_V = 20           # border flood threshold (max-channel value)
GLOW_V = 70         # outside-head brightness that maps to full alpha
HEAD_R = 0.46       # head disc radius (fraction of frame width)

CLIPS = ("idle_base", "idle_small", "idle_medium", "idle_large")


def key_frame(img):
    a = np.asarray(img.convert("RGBA")).copy()
    v = a[..., :3].max(axis=2).astype(np.float32)
    h, w = v.shape

    # 1. background: near-black region connected to the borders
    dark = v < BG_V
    seed = np.zeros_like(dark)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    bg = binary_propagation(seed & dark, mask=dark)

    # 2./3. alpha: opaque head disc, luminance ramp for the glow outside it
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0) / w
    alpha = np.where(r < HEAD_R, 255.0,
                     np.clip(v / GLOW_V, 0.0, 1.0) * 255.0)
    alpha[bg] = 0.0
    a[..., 3] = alpha.astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def process(video, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video),
             "-vsync", "0", f"{td}/f_%04d.png"],
            check=True,
        )
        frames = sorted(Path(td).glob("f_*.png"))
        # every 2nd frame, capped at N_OUT
        picked = frames[::2][:N_OUT]
        for i, f in enumerate(picked):
            img = key_frame(Image.open(f))
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
            img.save(out_dir / f"frame_{i:03d}.png")
    return len(picked)


def main():
    vids = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "assets" / "marvin")
    for name in CLIPS:
        mp4 = vids / f"{name}.mp4"
        if not mp4.exists():
            print(f"skip {name} (no {mp4})")
            continue
        n = process(mp4, dest / name)
        print(f"{name}: {n} frames -> {dest / name}")


if __name__ == "__main__":
    main()
