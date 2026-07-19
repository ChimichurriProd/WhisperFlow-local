#!/usr/bin/env python3
"""Erase hallucinated mouth lines from keyed Marvin frames.

Kling sometimes stitches a small dark mouth onto Marvin's blank lower face
(the plush skin is especially prone). Rather than re-rolling generations,
remove it deterministically. The mouth is a COHERENT dark blob much wider
than the wool-stitch texture scale, in the lower-central face:

  1. dark = pixels well below a large-scale blur of their neighbourhood
     (the blur rides over the fine knit texture, so individual stitches
     don't trigger — only features bigger than the texture scale do)
  2. keep blobs that are mouth-shaped: wide (>= MIN_W px), not huge
  3. fill each blob by copying REAL texture from just above it (same wool /
     shell shading), so the patch keeps the material look

Usage: erase_mouth.py <clip_dir> [<clip_dir> ...]
Rewrites frame_*.png in place. No-op on clean frames.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, label

ROI_Y = (0.55, 0.95)   # mouth zone within the head bbox (fraction of height)
ROI_X = (0.18, 0.82)
BLUR = 8               # neighbourhood scale; > knit-stitch scale
DARK_DIFF = 13         # how far below the local mean counts as "a dark feature"
MIN_W = 12             # blob min width  -> ignores stitches/speckle
MAX_H = 38             # blob max height -> ignores big shadow regions
MIN_AREA = 30


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

    # Never touch the eyes: exclude anything at or near the GLOWING coloured
    # eye regions (bright + saturated) — their dark rims would otherwise read
    # as "wide dark blobs" and get patched over. Brightness matters: the
    # hallucinated mouth stitches are dark maroon (saturated but DIM), and
    # must stay targetable.
    rgbmax = rgb.max(axis=2)
    sat = np.where(rgbmax > 1,
                   (rgbmax - rgb.min(axis=2)) / np.maximum(rgbmax, 1.0), 0.0)
    eyes = binary_dilation((sat > 0.18) & (v > 95) & solid, iterations=7)
    roi &= ~eyes

    local = gaussian_filter(v, BLUR)
    dark = roi & ((local - v) > DARK_DIFF)
    lbl, n = label(dark)
    if not n:
        return False

    changed = False
    for i, sl in enumerate(find_objects(lbl), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        blob = lbl[sl] == i
        if bw < MIN_W or bh > MAX_H or blob.sum() < MIN_AREA:
            continue
        patch = binary_dilation(blob, iterations=2)
        dy = bh + 6                      # copy source: just above the blob
        src_top = sl[0].start - dy
        if src_top < y0:
            continue
        py, px = np.where(patch)
        ty = py + sl[0].start
        tx = px + sl[1].start
        sy = ty - dy
        ok = solid[sy, tx]               # only copy from real head pixels
        rgb[ty[ok], tx[ok]] = rgb[sy[ok], tx[ok]]
        changed = True

    if changed:
        a[..., :3] = rgb.astype(np.uint8)
        Image.fromarray(a, "RGBA").save(path)
    return changed


def main():
    for d in sys.argv[1:]:
        d = Path(d)
        frames = sorted(d.glob("frame_*.png"))
        fixed = sum(erase(f) for f in frames)
        print(f"{d.name}: cleaned {fixed}/{len(frames)} frames")


if __name__ == "__main__":
    main()
