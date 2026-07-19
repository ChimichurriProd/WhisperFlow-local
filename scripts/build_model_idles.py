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
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           gaussian_filter, label)

N_OUT = 61          # frames kept per clip (every 2nd of ~121)
SIZE = 256
BG_LO = 24          # near-black: background flood threshold
HEAD_V = 55         # head shell = brighter than this ...
HEAD_SAT = 0.35     #   ... and less saturated than this (grey, not eye/glow)
GLOW_SAT = 0.22     # a beyond-head pixel counts as glow only if this saturated
GLOW_GAMMA = 0.85   # glow alpha = (v/255)^gamma * gain: smooth fade, no cutoff
GLOW_GAIN = 1.15
CLOSE_ITERS = 3     # close bays this size in the head mask (chin pinch-offs)
HEAD_GROW = 0       # extra dilation of the solid mask (feather replaces it)
EDGE_SIGMA = 1.0    # gaussian feather of the mask -> anti-aliased, stable edge
SHADOW_FLOOR = 68   # lift the head's black point so the chin isn't a dark smudge


def key_frame(img):
    """Key a render (Marvin on black) to RGBA that composites cleanly on ANY
    background. Three layers:
      SOLID  — every pixel inside Marvin's outline (incl. the near-black chin):
               opaque, shadow-lifted so the dark underside reads grey not black.
      GLOW   — COLOURED (saturated) pixels beyond the outline (cyan halo, green
               bloom, gold aura): forced to their full-bright hue, alpha fading
               smoothly with luminance all the way to zero (no hard thresholds).
      Everything else — fully transparent. Neutral-dark pixels outside the
               outline are dropped entirely, so no dark ring ever fringes the
               glow, and the chin can't fall through (it's inside the outline).
    """
    a = np.asarray(img.convert("RGBA")).copy()
    rgb = a[..., :3].astype(np.float32)
    v = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(v > 1, (v - mn) / np.maximum(v, 1.0), 0.0)

    # 1. background = near-black pixels connected to the frame border. The chin
    #    is near-black too but NOT border-connected, so it survives this flood.
    dark = v < BG_LO
    lbl, n = label(dark)
    bg = np.zeros_like(dark)
    if n:
        border = np.unique(np.concatenate([
            lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        bg = np.isin(lbl, border[border != 0])
    sil = ~bg                       # everything inside the outline + the glow

    # 2. head core = the unmistakable grey shell (bright + low-saturation).
    grey = (v > HEAD_V) & (sat < HEAD_SAT) & sil
    grey = binary_closing(grey, iterations=CLOSE_ITERS)
    lbl, n = label(grey)
    core = np.zeros_like(grey)
    if n:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        core = lbl == sizes.argmax()

    # 3. SOLID = the neutral (non-glow) region connected to that core. This
    #    walks down into the near-black chin (neutral, touching the core) but
    #    NOT into the halo (saturated = a barrier) or the dark band beyond it
    #    (not connected to the core through neutral pixels). Eyes are saturated
    #    holes inside the outline -> filled back in.
    neutral = sil & (sat <= GLOW_SAT)
    lbl, n = label(neutral | core)
    solid = core
    if n:
        ids = np.unique(lbl[core])
        ids = ids[ids != 0]
        if ids.size:
            solid = np.isin(lbl, ids)
    solid = binary_fill_holes(binary_closing(solid, iterations=CLOSE_ITERS))
    if HEAD_GROW:
        solid = binary_dilation(solid, iterations=HEAD_GROW)

    # 4. FEATHER the mask into a real anti-aliased edge. A hard binary edge
    #    wobbles ±1px frame to frame ("marching ants"); a ~1px gaussian ramp is
    #    sub-pixel stable and matches the render's own anti-aliasing.
    alpha = gaussian_filter(solid.astype(np.float32), EDGE_SIGMA)
    alpha = np.clip((alpha - 0.15) / 0.7, 0.0, 1.0)   # keep the ramp tight

    # 5. Un-premultiply the edge band: an anti-aliased pixel on black stores
    #    head-colour * coverage. Dividing by our alpha recovers the true head
    #    colour, so the soft edge blends to ANY background with no dark fringe.
    edge = (alpha > 0.02) & (alpha < 0.98)
    unpre = np.clip(1.0 / np.maximum(alpha, 0.25), 1.0, 4.0)
    for c in range(3):
        rgb[..., c] = np.where(edge, np.clip(rgb[..., c] * unpre, 0, 255),
                               rgb[..., c])

    # 6. lift the head's deep shadows so the chin reads grey, not black.
    body = alpha > 0.02
    vv = rgb.max(axis=2)
    vt = SHADOW_FLOOR + vv * (255 - SHADOW_FLOOR) / 255.0
    lift = np.where(vv > 1, vt / np.maximum(vv, 1.0), 1.0)
    for c in range(3):
        rgb[..., c] = np.where(body, np.clip(rgb[..., c] * lift, 0, 255),
                               rgb[..., c])

    # 7. GLOW: coloured pixels beyond the solid. Full-bright hue, alpha from
    #    luminance with a smooth exponent curve that breathes out to zero —
    #    no cut-off edge, so the halo fades naturally on any background.
    # glow = anything BRIGHT beyond the head: saturated colour (cyan/green
    # halo) OR white-hot bloom (bright but unsaturated). Only DIM neutral
    # pixels are dropped — they are what caused the dark fringe.
    colored = (~solid) & (v > 8) & ((sat > GLOW_SAT) | (v > 120))
    scale = np.clip(np.where(v > 1, 255.0 / np.maximum(v, 1.0), 1.0), 1.0, 8.0)
    galpha = np.clip((v / 255.0) ** GLOW_GAMMA * GLOW_GAIN, 0.0, 1.0)
    for c in range(3):
        rgb[..., c] = np.where(colored, np.clip(rgb[..., c] * scale, 0, 255),
                               rgb[..., c])
    alpha = np.where(colored, np.maximum(alpha, galpha), alpha)

    a[..., :3] = rgb.astype(np.uint8)
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
