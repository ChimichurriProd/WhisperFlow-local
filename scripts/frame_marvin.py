#!/usr/bin/env python3
"""Normalize a Marvin render to canonical framing, and key it to RGBA.

The image model won't frame Marvin consistently (heads come out anywhere from
20% to 90% of the frame), so we normalize in post instead of fighting the
prompt: detect the GREY HEAD silhouette (bright, low-saturation — this ignores
the green/cyan eyes and any glow, which would otherwise inflate the box), then
scale + center so the head is a fixed fraction of the output at a fixed centre.
Both the green scribe face and the cyan oracle face normalize to the SAME head
size and position this way, so the spin between them never jumps.

TARGET_HEAD_W leaves a margin around the head so gesture motion (a lean, a
droop) and a contained glow/aura have room and never clip at the frame edge.

Usage:
  frame_marvin.py <in.png> <out.png> [--size N] [--head F] [--rgba]
  --rgba keys the black background to transparency (for app assets); without
  it the output stays on black (for Kling start/end keyframes).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           gaussian_filter, label)

TARGET_HEAD_W = 0.70     # head width as a fraction of the output frame
CENTER = (0.50, 0.52)    # head-bbox centre in the output (x, y); y biased down
BG_LO = 24               # bg flood threshold (see build_model_idles)
HEAD_V, HEAD_SAT = 55, 0.35     # head shell = bright + low-saturation (grey)
GLOW_SAT = 0.22          # beyond-head pixel is glow only if this saturated
GLOW_GAMMA, GLOW_GAIN = 0.85, 1.15   # smooth glow alpha curve (no cutoff)
CLOSE_ITERS, HEAD_GROW = 3, 0
EDGE_SIGMA = 1.0         # gaussian feather -> anti-aliased, temporally stable edge
SHADOW_FLOOR = 68        # lift the head's black point (chin smudge on light bg)


def grey_head_bbox(a):
    """Bounding box (x0,x1,y0,y1) of the grey head, ignoring coloured eyes/glow."""
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    v = np.maximum(np.maximum(R, G), B)
    sat = v - np.minimum(np.minimum(R, G), B)
    grey = (v > 55) & (sat < 45)          # bright and near-neutral => the shell
    grey = binary_fill_holes(binary_closing(grey, iterations=2))
    lbl, n = label(grey)
    if not n:
        ys, xs = np.where(v > 40)          # fallback: any non-black
    else:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        ys, xs = np.where(lbl == sizes.argmax())
    return xs.min(), xs.max(), ys.min(), ys.max()


def normalize(img, size, head_w=TARGET_HEAD_W):
    a = np.asarray(img.convert("RGB"))
    x0, x1, y0, y1 = grey_head_bbox(a)
    hw = x1 - x0
    scale = (head_w * size) / hw
    new = img.convert("RGB").resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS)
    # head-bbox centre in the scaled image, placed at CENTER of the output
    cx = (x0 + x1) / 2 * scale
    cy = (y0 + y1) / 2 * scale
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(new, (round(CENTER[0] * size - cx), round(CENTER[1] * size - cy)))
    return canvas


def key_rgba(img):
    """Key a render to RGBA that composites cleanly on ANY background.
    Same three-layer logic as scripts/build_model_idles.key_frame: SOLID =
    everything inside Marvin's outline (border-flood; includes the near-black
    chin), opaque + shadow-lifted; GLOW = coloured pixels beyond the outline at
    full-bright hue with a smooth luminance fade; the rest transparent."""
    a = np.asarray(img.convert("RGBA")).copy()
    rgb = a[..., :3].astype(np.float32)
    v = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(v > 1, (v - mn) / np.maximum(v, 1.0), 0.0)

    dark = v < BG_LO
    lbl, n = label(dark)
    bg = np.zeros_like(dark)
    if n:
        border = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        bg = np.isin(lbl, border[border != 0])
    sil = ~bg

    grey = (v > HEAD_V) & (sat < HEAD_SAT) & sil
    grey = binary_closing(grey, iterations=CLOSE_ITERS)
    lbl, n = label(grey)
    core = np.zeros_like(grey)
    if n:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        core = lbl == sizes.argmax()

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

    # feathered anti-aliased edge + un-premultiply of the edge band (see
    # build_model_idles.key_frame for the reasoning)
    alpha = gaussian_filter(solid.astype(np.float32), EDGE_SIGMA)
    alpha = np.clip((alpha - 0.15) / 0.7, 0.0, 1.0)
    edge = (alpha > 0.02) & (alpha < 0.98)
    unpre = np.clip(1.0 / np.maximum(alpha, 0.25), 1.0, 4.0)
    for c in range(3):
        rgb[..., c] = np.where(edge, np.clip(rgb[..., c] * unpre, 0, 255),
                               rgb[..., c])

    body = alpha > 0.02
    vv = rgb.max(axis=2)
    vt = SHADOW_FLOOR + vv * (255 - SHADOW_FLOOR) / 255.0
    lift = np.where(vv > 1, vt / np.maximum(vv, 1.0), 1.0)
    for c in range(3):
        rgb[..., c] = np.where(body, np.clip(rgb[..., c] * lift, 0, 255),
                               rgb[..., c])

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--head", type=float, default=TARGET_HEAD_W)
    ap.add_argument("--rgba", action="store_true")
    args = ap.parse_args()
    out = normalize(Image.open(args.src), args.size, args.head)
    if args.rgba:
        out = key_rgba(out)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.dst)
    print(f"{args.src} -> {args.dst} ({args.size}px, head~{args.head:.0%}, "
          f"{'rgba' if args.rgba else 'black'})")


if __name__ == "__main__":
    main()
