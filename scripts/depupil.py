import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

def depupil(im):
    """Fill dark pupil-spots enclosed by the green eyes with local green."""
    a = np.asarray(im.convert("RGBA")).astype(np.uint8)
    rgb = a[:, :, :3].astype(int); al = a[:, :, 3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (g > r + 12) & (g > b + 3) & (g > 80) & (al > 50)
    if green.sum() < 30:
        return im
    # close small gaps so an edge-touching pupil counts as enclosed, then find holes
    closed = ndimage.binary_closing(green, iterations=3)
    filled = ndimage.binary_fill_holes(closed)
    lum = rgb.max(2)
    # pupil = inside the eye area, notably darker than the green core
    gcore = np.median(lum[green])
    pupil = filled & (lum < gcore - 30)
    pupil = ndimage.binary_dilation(pupil, iterations=1) & filled
    if pupil.sum() == 0:
        return im
    fill = np.array([int(np.median(g[green])), int(np.median(g[green])), int(np.median(g[green]))])
    # use median green RGB
    fill = np.array([int(np.median(rgb[...,c][green])) for c in range(3)])
    out = a.copy()
    for c in range(3):
        ch = out[..., c]; ch[pupil] = fill[c]
    out[..., 3][pupil] = 255
    res = Image.fromarray(out, "RGBA")
    # soft blur only over the patch to blend
    blurred = res.filter(ImageFilter.GaussianBlur(1.0))
    mask = Image.fromarray((ndimage.binary_dilation(pupil, iterations=2) * 255).astype("uint8"))
    res.paste(blurred, (0, 0), mask)
    return res

if __name__ == "__main__":
    import sys
    S = sys.argv[1]
    from pathlib import Path
    fs = sorted(Path(f"{S}/emote2_full").glob("f_*.png"))
    # clean-crop compare on frame 172 (before/after), eye region
    cell = 300
    sheet = Image.new("RGB", (cell*2, cell), (26,26,28))
    im = Image.open(fs[172]).convert("RGBA")
    w,h = im.size
    box = (int(w*0.2),int(h*0.28),int(w*0.8),int(h*0.62))
    before = im.crop(box).resize((cell,cell)).convert("RGB")
    after = depupil(im).crop(box).resize((cell,cell)).convert("RGB")
    sheet.paste(before,(0,0)); sheet.paste(after,(cell,0))
    sheet.save(f"{S}/depupil_test.png"); print("wrote depupil_test.png (left=before, right=after)")


# --- CLI: depupil a file or a directory of frame_*.png / f_*.png in place ---
def _run_cli():
    import sys
    from pathlib import Path
    args = sys.argv[1:]
    if not args:
        print("usage: depupil.py <image.png | dir> [out_dir]"); return
    src = Path(args[0]); out = Path(args[1]) if len(args) > 1 else None
    files = ([src] if src.is_file()
             else sorted(list(src.glob("frame_*.png")) + list(src.glob("f_*.png"))))
    if out: out.mkdir(parents=True, exist_ok=True)
    for p in files:
        fixed = depupil(Image.open(p))
        dst = (out / p.name) if out else p
        fixed.save(dst)
    print(f"depupiled {len(files)} image(s)"
          + (f" -> {out}" if out else " in place"))


if __name__ == "__main__":
    _run_cli()
