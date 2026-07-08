#!/usr/bin/env python3
"""Produce the eyeless Marvin head + record eye geometry for the procedural eyes.

Input : a front/neutral render of Marvin on a solid black background
        (grey head, two solid glowing-green eyes).
Output: assets/marvin/center_eyeless.png  — the head cut out to transparent,
        resized to SIZE x SIZE, with the green eyes masked out and the grey
        shell inpainted behind them.
Also  : prints the base eye geometry (normalized left/right eye centre + size
        in the FINAL image's coordinates) — paste these into pill.py as the
        procedural eyes' rest positions.

`marvin_live` draws its own eyes on top of this head, so the important thing is
that the eye area reads as clean grey shell (for the blink/lid look), not that
the fill is photographic.

Pure PIL + numpy (no OpenCV needed). Run from the repo root:
    .venv/bin/python scripts/make_eyeless_head.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "marvin" / "_design" / "expressions" / "exp_1_neutral.png"
OUT = ROOT / "assets" / "marvin" / "center_eyeless.png"
GEOM = ROOT / "assets" / "marvin" / "center_eyeless.eyes.json"
SIZE = 256          # final head image size (matches the old center.png)
MARGIN = 0.03       # black margin kept around the head in the final image


def green_mask(rgb: np.ndarray) -> np.ndarray:
    """Green-dominant eye pixels. Loose enough to catch the pale glow (not just
    the saturated core), but g>r/g>b excludes the neutral grey shell."""
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (g > r + 12) & (g > b + 12) & (g > 85)


def dilate(mask: np.ndarray, it: int = 3) -> np.ndarray:
    m = mask.copy()
    for _ in range(it):
        m[1:, :] |= mask[:-1, :]
        m[:-1, :] |= mask[1:, :]
        m[:, 1:] |= mask[:, :-1]
        m[:, :-1] |= mask[:, 1:]
        mask = m.copy()
    return m


def diffusion_inpaint(rgb: np.ndarray, hole: np.ndarray, iters: int = 150) -> np.ndarray:
    """Fill the hole to match the shell. First a per-column vertical gradient
    (interpolate between the nearest non-hole pixel above and below each hole
    pixel — recovers the sphere's top-light/bottom-dark shading), then a light
    Laplace smoothing to remove column banding and blend the seam."""
    out = rgb.astype(np.float32).copy()
    H, W = hole.shape
    big = float(H * 4)

    # nearest non-hole value above each hole pixel (+ its distance)
    above = out.copy()
    d_above = np.where(hole, big, 0.0)
    for y in range(1, H):
        h = hole[y]
        above[y][h] = above[y - 1][h]
        d_above[y][h] = d_above[y - 1][h] + 1

    below = out.copy()
    d_below = np.where(hole, big, 0.0)
    for y in range(H - 2, -1, -1):
        h = hole[y]
        below[y][h] = below[y + 1][h]
        d_below[y][h] = d_below[y + 1][h] + 1

    w = (d_above / np.clip(d_above + d_below, 1e-6, None))[..., None]
    grad = above * (1.0 - w) + below * w
    out[hole] = grad[hole]

    for _ in range(iters):
        acc = np.zeros_like(out)
        acc[1:, :] += out[:-1, :]
        acc[:-1, :] += out[1:, :]
        acc[:, 1:] += out[:, :-1]
        acc[:, :-1] += out[:, 1:]
        acc /= 4.0
        out[hole] = acc[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def background_alpha(img: Image.Image) -> Image.Image:
    """Flood-fill the pure-black border to transparent; interior shading stays."""
    rgb = np.asarray(img.convert("RGB"))
    lum = rgb.max(axis=2)
    # sentinel flood from the corners over near-black pixels
    flood = Image.new("L", img.size, 0)
    fd = ImageDraw.Draw(flood)
    del fd
    mark = Image.fromarray((lum < 26).astype(np.uint8) * 255)  # candidate bg
    # keep only the border-connected black region
    ImageDraw.floodfill(mark, (0, 0), 128, thresh=10)
    ImageDraw.floodfill(mark, (img.size[0] - 1, 0), 128, thresh=10)
    ImageDraw.floodfill(mark, (0, img.size[1] - 1), 128, thresh=10)
    ImageDraw.floodfill(mark, (img.size[0] - 1, img.size[1] - 1), 128, thresh=10)
    bg = np.asarray(mark) == 128
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    return Image.fromarray(alpha, "L")


def main() -> None:
    img = Image.open(SRC).convert("RGB")
    rgb = np.asarray(img)
    H, W = rgb.shape[:2]

    # 1) eye geometry from the green pixels, split L/R by image centre
    gm = green_mask(rgb)
    ys, xs = np.nonzero(gm)
    if len(xs) == 0:
        raise SystemExit("No green eyes detected in the source render.")
    cx_img = W / 2.0
    eyes_px = {}
    for side, sel in (("left", xs < cx_img), ("right", xs >= cx_img)):
        exs, eys = xs[sel], ys[sel]
        eyes_px[side] = dict(
            cx=float(exs.mean()), cy=float(eys.mean()),
            w=float(exs.max() - exs.min()), h=float(eys.max() - eys.min()),
        )

    # 2) blank the whole eye region (expanded ellipse per eye covers the pale
    #    rim + the dark socket recess), then inpaint the grey shell behind it.
    hole_img = Image.new("L", (W, H), 0)
    hd = ImageDraw.Draw(hole_img)
    for e in eyes_px.values():
        ex = e["w"] * 0.75 + 6      # expand generously beyond the green bbox
        ey = e["h"] * 0.95 + 6
        hd.ellipse(
            (e["cx"] - e["w"] / 2 - ex, e["cy"] - e["h"] / 2 - ey,
             e["cx"] + e["w"] / 2 + ex, e["cy"] + e["h"] / 2 + ey),
            fill=255,
        )
    hole = (np.asarray(hole_img) > 0) | dilate(gm, 2)
    filled = diffusion_inpaint(rgb, hole, iters=300)
    head = Image.fromarray(filled, "RGB")

    # 3) cut the head out of black -> transparent
    alpha = background_alpha(img)
    head.putalpha(alpha)

    # 4) crop to the head bbox (+margin) and resize to SIZE
    a = np.asarray(alpha)
    ys2, xs2 = np.nonzero(a > 8)
    x0, x1, y0, y1 = xs2.min(), xs2.max(), ys2.min(), ys2.max()
    side = max(x1 - x0, y1 - y0)
    pad = int(side * MARGIN)
    ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side / 2.0 + pad
    box = (int(ccx - half), int(ccy - half), int(ccx + half), int(ccy + half))
    head = head.crop(box).resize((SIZE, SIZE), Image.LANCZOS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    head.save(OUT)

    # 5) eye geometry in the FINAL cropped/resized normalized frame (0..1)
    bx0, by0, bside = box[0], box[1], (box[2] - box[0])
    geom = {}
    for side_name, e in eyes_px.items():
        geom[side_name] = dict(
            cx=(e["cx"] - bx0) / bside,
            cy=(e["cy"] - by0) / bside,
            w=e["w"] / bside,
            h=e["h"] / bside,
        )
    GEOM.write_text(json.dumps(geom, indent=2))

    print(f"wrote {OUT.relative_to(ROOT)}  ({SIZE}x{SIZE}, RGBA)")
    print(f"wrote {GEOM.relative_to(ROOT)}")
    print("eye geometry (normalized, final frame):")
    print(json.dumps(geom, indent=2))


if __name__ == "__main__":
    main()
