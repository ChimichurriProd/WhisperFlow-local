#!/usr/bin/env python3
"""Process the per-STT-model idle videos into app clips.

Input: four 960x960 MP4s (Kling 2.5, black background) — one per model mood
(sleepy base .. electric large). Output: assets/marvin/idle_<model>/ with
61 RGBA frames at 256x256, background keyed out, aura/sparks kept with
smooth luminance alpha (same look as the super_saiyan clip).

Keying per frame (transparency-safe — no dark box/ring behind Marvin):
  1. Find the SOLID HEAD as the largest bright connected region (head + eyes),
     holes filled. This follows Marvin's real silhouette, so the black
     background just outside it is never forced opaque (the old fixed-disc
     approach left an opaque black ring between the head and the disc edge).
  2. Head -> fully opaque.
  3. Everything else -> alpha ramps with brightness, so the aura/sparks/glow
     fade smoothly to transparent and pure black goes to alpha 0. Dark pixels
     never get meaningful alpha, so nothing composites as a black tint.

Usage: build_model_idles.py <videos_dir> [<out_assets_marvin_dir>]
Processes every <videos_dir>/<name>.mp4 into assets/marvin/<name>/ — works
for the idle_* model loops and for one-shot gesture clips (angry, love,
stressed, glow, ...) alike.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label

N_OUT = 61          # frames kept per clip (every 2nd of ~121)
SIZE = 256
HEAD_V = 60         # brightness that counts as solid foreground (head/eyes)
RAMP_LO = 24        # below this brightness -> transparent
RAMP_HI = 92        # at/above this -> fully opaque glow


def key_frame(img):
    a = np.asarray(img.convert("RGBA")).copy()
    v = a[..., :3].max(axis=2).astype(np.float32)

    # 1. solid head = largest bright connected component, holes filled. This
    #    tracks the actual silhouette instead of a disc, so no black ring.
    bright = v > HEAD_V
    lbl, n = label(bright)
    head = np.zeros_like(bright)
    if n:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        head = binary_fill_holes(lbl == sizes.argmax())

    # 2./3. brightness ramp everywhere; head forced opaque. Dark bg -> ~0
    #       alpha, so it never shows as a black tint under the transparency.
    alpha = np.clip((v - RAMP_LO) / (RAMP_HI - RAMP_LO), 0.0, 1.0)
    alpha[head] = 1.0
    a[..., 3] = (alpha * 255).astype(np.uint8)
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
    mp4s = sorted(vids.glob("*.mp4"))
    if not mp4s:
        sys.exit(f"no .mp4 files in {vids}")
    for mp4 in mp4s:
        n = process(mp4, dest / mp4.stem)
        print(f"{mp4.stem}: {n} frames -> {dest / mp4.stem}")


if __name__ == "__main__":
    main()
