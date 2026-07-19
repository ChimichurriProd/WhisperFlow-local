#!/usr/bin/env python3
"""Erase hallucinated mouth features from keyed Marvin frames.

Kling sometimes stitches a small dark mouth onto Marvin's blank lower face
(the plush skin is especially prone). Post-fix deterministically: within the
lower-central face zone, EVERY locally-dark anomaly (anything sitting well
below a large-scale blur of its neighbourhood — i.e. features bigger than the
knit-stitch scale; a smooth shading gradient never triggers) is replaced with
smooth surrounding texture via normalised convolution over the clean pixels,
plus matched grain so the patch reads as wool/shell at pill size. No blob
heuristics — earlier size-filtered versions kept missing mouth fragments and
left flickering shards.

Usage: erase_mouth.py <clip_dir> [<clip_dir> ...]   (rewrites frames in place)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label

ROI_Y = (0.55, 0.96)   # mouth zone within the head bbox (fraction of height)
ROI_X = (0.15, 0.85)
BLUR = 8               # anomaly detection scale (> stitch texture)
DARK_DIFF = 10         # below local mean by this much = anomaly
FILL_SIGMA = 7.0       # normalised-convolution fill smoothness
GRAIN = 9.0            # matched-noise amplitude cap


def erase(path):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).copy()
    rgb = a[..., :3].astype(np.float32)
    alpha = a[..., 3]
    v = rgb.max(axis=2)

    solid = alpha >= 245
    ys, xs = np.where(solid)
    if not len(xs):
        return False
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = y1 - y0, x1 - x0
    roi = np.zeros_like(solid)
    roi[int(y0 + ROI_Y[0] * h):int(y0 + ROI_Y[1] * h),
        int(x0 + ROI_X[0] * w):int(x0 + ROI_X[1] * w)] = True
    roi &= solid

    # keep clear of the glowing eyes. HUE-aware: eyes are GREEN/CYAN — a red/
    # maroon mouth interior is saturated too and must NOT be protected (that
    # was how mouth shards kept surviving earlier passes).
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    greenish = (G > R * 1.05) | (B > R * 1.2)
    glowing = binary_dilation(greenish & (v > 95) & solid, iterations=4)
    roi &= ~glowing

    local = gaussian_filter(v, BLUR)
    mask = roi & ((local - v) > DARK_DIFF)
    if mask.sum() < 20:
        return False
    mask = binary_dilation(mask, iterations=3) & roi

    # normalised convolution over clean pixels -> smooth local texture colour
    clean = (solid & ~mask & ~glowing).astype(np.float32)
    filled = np.empty_like(rgb)
    denom = gaussian_filter(clean, FILL_SIGMA) + 1e-6
    for c in range(3):
        filled[..., c] = gaussian_filter(rgb[..., c] * clean, FILL_SIGMA) / denom

    # matched grain so the patch isn't a flat smudge on the knit texture
    ring = binary_dilation(mask, iterations=8) & ~mask & solid & ~glowing
    std = float(rgb[ring].std(axis=0).mean()) if ring.sum() > 30 else 6.0
    rng = np.random.default_rng(0)
    my, mx = np.where(mask)
    noise = rng.normal(0.0, min(std, GRAIN) * 0.6, size=(len(my), 1))
    rgb[my, mx] = np.clip(filled[my, mx] + noise, 0, 255)

    a[..., :3] = rgb.astype(np.uint8)
    Image.fromarray(a, "RGBA").save(path)
    return True


def main():
    for d in sys.argv[1:]:
        d = Path(d)
        frames = sorted(d.glob("frame_*.png"))
        fixed = sum(erase(f) for f in frames)
        print(f"{d.name}: cleaned {fixed}/{len(frames)} frames")


if __name__ == "__main__":
    main()
