#!/usr/bin/env python3
"""Process the per-STT-model idle videos into app clips.

Input: four 960x960 MP4s (Kling 2.5, black background) — one per model mood
(sleepy base .. electric large). Output: assets/marvin/idle_<model>/ with
61 RGBA frames at 256x256, background keyed out, aura/sparks kept with
smooth luminance alpha (same look as the super_saiyan clip).

Keying per frame (transparency-safe — no dark box/ring behind Marvin, and no
see-through chin):
  1. BACKGROUND = the dark region connected to the frame border (flood from
     the edges over near-black pixels). Only that is background — a dark
     *shadow on the head* (the chin) is not border-connected darkness, so it
     stays foreground. (The old rule "head = pixels brighter than 60" let the
     chin shadow fall through to the brightness ramp -> semi-transparent jaw.)
  2. HEAD = largest non-background component, small bays closed + holes
     filled, then eroded a couple of px -> that interior is fully opaque. The
     outermost rim keeps the brightness-ramp alpha so the anti-aliased edge
     stays soft (dark edge blends never become an opaque dark outline).
  3. Everything else -> alpha ramps with brightness, so the aura/sparks/glow
     fade smoothly to transparent and pure black goes to alpha 0. Dark pixels
     never get meaningful alpha, so nothing composites as a black tint.

Usage:
  build_model_idles.py <videos_dir> [<out_assets_marvin_dir>]
      Process every <videos_dir>/<name>.mp4 into assets/marvin/<name>/ —
      works for the idle_* model loops and for one-shot gesture clips
      (angry, love, stressed, glow, ...) alike.
  build_model_idles.py --rekey <clip_dir> [<clip_dir> ...]
      Recompute the alpha of existing frame_*.png in place from their RGB
      (the RGB survives under a bad key, so no source video is needed).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes,
                           label)

N_OUT = 61          # frames kept per clip (every 2nd of ~121)
SIZE = 256
RAMP_LO = 24        # below this brightness -> transparent
RAMP_HI = 92        # at/above this -> fully opaque glow
CLOSE_ITERS = 3     # close bays this size in the head mask (chin pinch-offs)
EDGE_ERODE = 2      # px of head rim left on the soft brightness ramp


def key_frame(img):
    a = np.asarray(img.convert("RGBA")).copy()
    v = a[..., :3].max(axis=2).astype(np.float32)

    # 1. background = near-black pixels connected to the frame border.
    dark = v < RAMP_LO
    lbl, n = label(dark)
    bg = np.zeros_like(dark)
    if n:
        border = np.unique(np.concatenate([
            lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        bg = np.isin(lbl, border[border != 0])

    # 2. head = largest non-background component; close small dark bays
    #    (chin shadow pinch-offs), fill enclosed holes, erode for a soft rim.
    fg = ~bg
    lbl, n = label(fg)
    head = np.zeros_like(fg)
    if n:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        head = lbl == sizes.argmax()
        head = binary_fill_holes(binary_closing(head, iterations=CLOSE_ITERS))
    interior = binary_erosion(head, iterations=EDGE_ERODE)

    # 3. brightness ramp everywhere; head interior forced opaque. Dark bg ->
    #    ~0 alpha, so it never shows as a black tint under the transparency.
    alpha = np.clip((v - RAMP_LO) / (RAMP_HI - RAMP_LO), 0.0, 1.0)
    alpha[interior] = 1.0
    a[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def rekey(clip_dirs):
    for d in clip_dirs:
        d = Path(d)
        frames = sorted(d.glob("frame_*.png"))
        if not frames:
            print(f"{d.name}: no frames, skipped")
            continue
        for f in frames:
            key_frame(Image.open(f)).save(f)
        print(f"{d.name}: re-keyed {len(frames)} frames")


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
    if sys.argv[1] == "--rekey":
        rekey(sys.argv[2:])
        return
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
