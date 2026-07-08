#!/usr/bin/env python3
"""Preview / tune the procedural marvin_live mascot.

Two modes:

  .venv/bin/python scripts/marvin_live_preview.py           # live window
      A floating Marvin drives itself and cycles idle -> recording ->
      transcribing, and plays all expressions, so you can watch him come alive
      and tune the timing constants in app/pill.py.

  .venv/bin/python scripts/marvin_live_preview.py --shots    # offscreen grid
      Renders every eye state to scratch PNGs + a montage (no window), for a
      quick visual check of the eye drawing.

Run from the repo root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AppKit import (  # noqa: E402
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
    NSMakeRect, NSTimer, NSWindow,
)
import objc  # noqa: E402

from app import pill  # noqa: E402

SIZE = 200


class _FakeC:
    """A minimal stand-in for the _Pill controller, for offscreen rendering."""
    style = "marvin_live"

    def __init__(self):
        self.eyeless = pill._load_marvin_eyeless()
        self.gaze = [0.0, 0.0]
        self.eye_l = dict(pill._EXPR["neutral"])
        self.eye_r = dict(pill._EXPR["neutral"])
        self.blink = 1.0
        self.live_glow = 0.0
        self.live_bob = 0.0
        self.live_tilt = 0.0


def _set_expr(c, name):
    if name in pill._EXPR_ASYM:
        l, r = pill._EXPR_ASYM[name]
    else:
        l = r = pill._EXPR.get(name, pill._EXPR["neutral"])
    c.eye_l = dict(l)
    c.eye_r = dict(r)


def _save_rep(rep, path):
    data = rep.representationUsingType_properties_(4, {})  # 4 = PNG
    data.writeToFile_atomically_(str(path), True)


def render_shots():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    # Render through a real _WaveView in an offscreen window (never ordered
    # front) so we exercise the exact drawRect_ path the app uses.
    c = _FakeC()
    view = pill._WaveView.alloc().initWithController_(c)
    view.setFrame_(NSMakeRect(0, 0, SIZE, SIZE))
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, SIZE, SIZE), 0, NSBackingStoreBuffered, False)
    win.setContentView_(view)

    states = [
        ("neutral", lambda c: _set_expr(c, "neutral")),
        ("alert", lambda c: _set_expr(c, "alert")),
        ("blink", lambda c: (_set_expr(c, "neutral"), setattr(c, "blink", 0.06))),
        ("skeptic", lambda c: _set_expr(c, "skeptic")),
        ("happy", lambda c: _set_expr(c, "happy")),
        ("sad", lambda c: _set_expr(c, "sad")),
        ("look-left", lambda c: (_set_expr(c, "neutral"), setattr(c, "gaze", [-0.08, 0.0]))),
        ("look-right", lambda c: (_set_expr(c, "neutral"), setattr(c, "gaze", [0.08, 0.0]))),
        ("recording-glow", lambda c: (_set_expr(c, "alert"), setattr(c, "live_glow", 0.85))),
    ]

    out = Path("/private/tmp/claude-501/-Users-skonheten-Documents-WISPER-FLOW/"
               "6a34bf1d-24b0-4350-8cc1-c8b339698377/scratchpad")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, setup in states:
        c = _FakeC()
        setup(c)
        view._c = c
        rep = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), rep)
        p = out / f"live_{name}.png"
        _save_rep(rep, p)
        paths.append((name, p))
        print("rendered", name)

    _montage(paths, out / "marvin_live_states.png")
    print("montage:", out / "marvin_live_states.png")


def _montage(paths, dest):
    from PIL import Image, ImageDraw, ImageFont
    cols, cell, lab = 3, SIZE, 26
    rows = (len(paths) + cols - 1) // cols
    W, H = cols * cell, rows * (cell + lab)
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        f = ImageFont.load_default()
    for i, (name, p) in enumerate(paths):
        r, cc = divmod(i, cols)
        x, y = cc * cell, r * (cell + lab)
        im = Image.open(p).convert("RGBA").resize((cell, cell), Image.LANCZOS)
        bg = Image.new("RGBA", (cell, cell), (16, 16, 18, 255))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (x, y))
        d.text((x + 8, y + cell + 4), name, font=f, fill=(220, 220, 225))
    sheet.save(dest)


class _Driver(objc.lookUpClass("NSObject")):
    """Drives a live Marvin window through a mode cycle for visual tuning."""
    def initWithPill_(self, p):
        self = objc.super(_Driver, self).init()
        self._p = p
        self._n = 0
        return self

    def tickMode_(self, _timer):
        # 12 s loop: idle 5s, recording 4s (fake mic level), transcribing 2s, blip
        self._n += 1
        phase = (self._n // 20) % 12
        import math
        if phase < 5:
            mode, level = "idle", 0.0
        elif phase < 9:
            mode = "recording"
            level = 0.35 + 0.3 * abs(math.sin(self._n * 0.5))
        else:
            mode, level = "transcribing", 0.0
        self._p.tick(mode, level)
        if self._n == 40:          # showcase all expressions once early on
            self._p.play_all()


def run_live():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    p = pill.create_pill(style="marvin_live")
    if p is None:
        print("create_pill failed (no UI)")
        return
    drv = _Driver.alloc().initWithPill_(p)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.05, drv, "tickMode:", None, True)
    print("Marvin live — Ctrl-C to quit.")
    app.run()


if __name__ == "__main__":
    if "--shots" in sys.argv:
        render_shots()
    else:
        run_live()
