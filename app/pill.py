"""Floating desktop 'pill': the primary WhisperFlow UI.

A small borderless always-on-top window with a modern frosted-glass look.
Draggable anywhere; remembers where you put it. Click (without dragging)
cycles the model. Hover reveals the current model.

States:
- idle:         a small glowing round dot (color+size reflect the model)
- idle+hover:   expands to show the model ("Model: small")
- recording:    expands, gradient waveform driven by mic level
- transcribing: a gentle animated shimmer ("thinking")

All AppKit objects are created/mutated on the main thread only. If PyObjC UI
isn't available, create_pill() returns None and the app still works headlessly.
"""

import math
import random

_HEIGHT = 32.0
_WIDTH_IDLE = 32.0  # equals height => a round dot when idle
_WIDTH_REC = 250.0  # hover width is measured from the label (see _target_width)
_BARS = 24
_MARVIN_SIZE = 280.0  # big window so the super_saiyan aura can flare FAR beyond the head box
# without clipping. The extra window is transparent/invisible — only Marvin + his aura show.
_MARVIN_INSET = 0.386  # normal head ~58px in the window
# super_saiyan: Marvin's head stays FIXED at the idle size; the golden aura flares far BEYOND the
# head box into the big window and never clips. A fixed (larger) inset keeps the head == idle size
# while the aura fills out to ~125px from centre — well inside the 280px window.
_SS_CLIP = "super_saiyan"
_SS_INSET = 0.114

# Clip names eligible as idle micro-gestures: while resting, Marvin randomly
# plays one of these every so often to read as alive. A name only joins the
# pool if a matching assets/marvin/<name>/ clip exists, so dropping in a new
# folder (e.g. "blink") auto-enrolls it. Excludes nod/shake/wake/spin, which
# have their own event triggers.
_IDLE_GESTURE_CLIPS = ("skeptic", "curious", "glance", "blink", "yawn", "emote")
# Per-clip one-shot playback speed (frames advanced per 20 Hz tick; default 1.0).
# spin is a 120-frame full 360°; 2.4/tick plays it in ~2.5 s as a quick flourish.
_ONESHOT_SPEED = {"spin": 2.4}

# ---------------------------------------------------------------------------
# marvin_live: real-time procedural eyes drawn over an eyeless grey head.
# The eyes are the code's, not baked frames — they blink, dart, follow the
# cursor and glow with your voice. See scripts/make_eyeless_head.py for the
# head asset + eye geometry these numbers come from.
# ---------------------------------------------------------------------------
# Base eye centres + size in the eyeless head's normalized frame (y from top).
_EYE_L = (0.342, 0.545)
_EYE_R = (0.658, 0.545)
_EYE_W = 0.205          # eye width  (fraction of head)
_EYE_H = 0.150          # eye full height (fraction of head)
_EYE_GREEN = (0.53, 1.0, 0.44)

# Per-eye expression targets. ap=aperture 0..1 (1 wide open, 0 shut),
# off=vertical centre shift (+down = sleepy), bow=edge curvature (+up happy /
# -down sad), outer=outer-corner drop (+down). Symmetric unless in _EXPR_ASYM.
_EXPR = {
    "neutral": {"ap": 0.44, "off": 0.14, "bow": 0.00, "outer": 0.00},
    "alert":   {"ap": 1.00, "off": 0.00, "bow": 0.00, "outer": 0.00},
    "happy":   {"ap": 0.34, "off": -0.10, "bow": 0.95, "outer": 0.00},
    "sad":     {"ap": 0.50, "off": 0.16, "bow": -0.35, "outer": 0.55},
    "curious": {"ap": 0.92, "off": -0.06, "bow": 0.12, "outer": -0.12},
}
# Explicit per-eye (left, right) for asymmetric expressions.
_EXPR_ASYM = {
    "skeptic": ({"ap": 0.16, "off": 0.22, "bow": 0.00, "outer": 0.10},
                {"ap": 0.95, "off": -0.04, "bow": 0.00, "outer": -0.18}),
}
# Blink aperture multipliers over successive ticks (20 Hz): close ~2, hold,
# open ~3 — a natural ~250 ms blink applied on top of any expression.
_BLINK_SEQ = (0.55, 0.12, 0.0, 0.0, 0.35, 0.7, 0.9, 1.0)


try:
    from AppKit import (
        NSAffineTransform,
        NSAnimationContext,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSGraphicsContext,
        NSEvent,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSGradient,
        NSImage,
        NSEventModifierFlagControl,
        NSMakePoint,
        NSMakeRect,
        NSScreen,
        NSTrackingActiveAlways,
        NSTrackingArea,
        NSTrackingInVisibleRect,
        NSTrackingMouseEnteredAndExited,
        NSView,
        NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectStateActive,
        NSVisualEffectView,
        NSWindow,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
    )
    from Foundation import NSMutableDictionary, NSObject
    import objc

    NSWindowStyleMaskBorderless = 0
    NSStatusWindowLevel = 25
    NSVisualEffectMaterialHUDWindow = 13
    _DRAG_THRESHOLD = 4.0

    def _rgb(r, g, b, a=1.0):
        return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)

    # Vibrant gradients for the waveform (top -> bottom).
    _GRAD = {
        "recording": ((0.99, 0.45, 0.66), (0.55, 0.36, 1.0)),   # pink -> violet
        "transcribing": ((0.40, 0.88, 0.99), (0.36, 0.52, 0.99)),  # cyan -> blue
    }
    # Idle dot reflects model size: blue+small (base) -> red+big (large).
    _MODEL_STYLE = {
        "base":           ((0.36, 0.62, 1.00), 3.3),   # blue, smallest
        "small":          ((0.28, 0.82, 0.80), 4.3),   # teal
        "medium":         ((0.99, 0.68, 0.24), 5.3),   # amber
        "large-v3-turbo": ((0.98, 0.33, 0.35), 6.4),   # red, biggest
    }

    def _model_style(model):
        return _MODEL_STYLE.get(model, ((0.60, 0.80, 1.0), 4.5))

    def _measure(text, size=11):
        """Rendered width of *text* in the pill's font (for dynamic sizing)."""
        attrs = NSMutableDictionary.dictionary()
        attrs[NSFontAttributeName] = NSFont.systemFontOfSize_(size)
        s = objc.lookUpClass("NSString").stringWithString_(text)
        return float(s.sizeWithAttributes_(attrs).width)

    def _hover_label(model):
        return f"Model: {model}"

    def _load_marvin_center():
        """Load the resting head image (center.png), shown when idle."""
        from pathlib import Path

        p = Path(__file__).resolve().parent.parent / "assets" / "marvin" / "center.png"
        return NSImage.alloc().initWithContentsOfFile_(str(p)) if p.exists() else None

    def _load_marvin_eyeless():
        """Load the eyeless grey head (center_eyeless.png) that marvin_live draws
        its procedural eyes over. Falls back to center.png if absent."""
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "assets" / "marvin"
        p = base / "center_eyeless.png"
        if not p.exists():
            p = base / "center.png"
        return NSImage.alloc().initWithContentsOfFile_(str(p)) if p.exists() else None

    def _load_marvin_clips():
        """Load frame-sequence clips from assets/marvin/<name>/frame_*.png plus
        per-frame eye positions from <name>/eyes.json.

        Returns (clips, eyes): clips={name:[NSImage,...]}, eyes={name:[[(lx,ly),
        (rx,ry)],...]} for the voice-reactive glow.
        """
        import json
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "assets" / "marvin"
        clips, eyes = {}, {}
        if base.is_dir():
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                imgs = []
                for f in sorted(sub.glob("frame_*.png")):
                    img = NSImage.alloc().initWithContentsOfFile_(str(f))
                    if img is not None:
                        imgs.append(img)
                if imgs:
                    clips[sub.name] = imgs
                    ej = sub / "eyes.json"
                    if ej.exists():
                        try:
                            eyes[sub.name] = json.loads(ej.read_text())
                        except Exception:
                            pass
        return clips, eyes

    def _screen_with_mouse():
        p = NSEvent.mouseLocation()
        for s in NSScreen.screens():
            f = s.frame()
            if (f.origin.x <= p.x <= f.origin.x + f.size.width
                    and f.origin.y <= p.y <= f.origin.y + f.size.height):
                return s
        return NSScreen.mainScreen()

    class _WaveView(NSView):
        def initWithController_(self, controller):
            self = objc.super(_WaveView, self).initWithFrame_(
                NSMakeRect(0, 0, _WIDTH_REC, _HEIGHT)
            )
            if self is None:
                return None
            self._c = controller
            self._down = None
            self._dragged = False
            self._tracking = None
            return self

        def isFlipped(self):
            return True

        # -------- hover tracking ------------------------------------------

        def updateTrackingAreas(self):
            objc.super(_WaveView, self).updateTrackingAreas()
            if self._tracking is not None:
                self.removeTrackingArea_(self._tracking)
            opts = (
                NSTrackingMouseEnteredAndExited
                | NSTrackingActiveAlways
                | NSTrackingInVisibleRect
            )
            self._tracking = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                self.bounds(), opts, self, None
            )
            self.addTrackingArea_(self._tracking)

        def mouseEntered_(self, event):
            self._c.set_hover(True)

        def mouseExited_(self, event):
            self._c.set_hover(False)

        # -------- drag to move / click to cycle ---------------------------

        def rightMouseDown_(self, event):
            if self._c.on_menu:
                self._c.on_menu(self, event)

        def mouseDown_(self, event):
            # Control-click = right-click: open the menu instead of dragging.
            if (event.modifierFlags() & NSEventModifierFlagControl) and self._c.on_menu:
                self._c.on_menu(self, event)
                self._down = None
                return
            self._down = NSEvent.mouseLocation()
            self._win0 = self.window().frame().origin
            self._dragged = False

        def mouseDragged_(self, event):
            if self._down is None:  # e.g. a control-click opened the menu
                return
            cur = NSEvent.mouseLocation()
            dx, dy = cur.x - self._down.x, cur.y - self._down.y
            if abs(dx) + abs(dy) > _DRAG_THRESHOLD:
                self._dragged = True
            self.window().setFrameOrigin_((self._win0.x + dx, self._win0.y + dy))

        def mouseUp_(self, event):
            if self._dragged:
                if self._c.on_move:
                    o = self.window().frame().origin
                    try:
                        self._c.on_move(float(o.x), float(o.y))
                    except Exception:
                        pass
                self._down = None
                return
            # A plain click: route single vs double. The single-click action is
            # deferred briefly and cancelled by a second click, so a double-click
            # (talk) doesn't also fire the single-click (cycle model).
            try:
                clicks = int(event.clickCount())
            except Exception:
                clicks = 1
            NSObject.cancelPreviousPerformRequestsWithTarget_(self)
            if clicks >= 2:
                if self._c.on_double_click:
                    try:
                        self._c.on_double_click()
                    except Exception:
                        pass
            elif self._c.on_click:
                self.performSelector_withObject_afterDelay_(
                    "fireSingleClick:", None, 0.28)
            self._down = None

        def fireSingleClick_(self, _arg):
            if self._c.on_click:
                try:
                    self._c.on_click()
                except Exception:
                    pass

        # -------- drawing --------------------------------------------------

        def drawRect_(self, rect):
            c = self._c
            w = self.frame().size.width
            h = self.frame().size.height

            if c.style == "marvin_live":
                self._draw_marvin_live(c, w, h)  # eyeless head + procedural eyes
                return

            if c.style == "marvin":
                if c.clips or c.center is not None:
                    self._draw_marvin_image(c, w, h)  # video flipbook / still
                    return
                # fallback (no assets): 2D fake tilt/nod on the vector face
                NSGraphicsContext.saveGraphicsState()
                t = NSAffineTransform.transform()
                t.translateXBy_yBy_(0.0, c.nod)
                t.translateXBy_yBy_(w / 2.0, h / 2.0)
                t.rotateByDegrees_(c.tilt)
                t.translateXBy_yBy_(-w / 2.0, -h / 2.0)
                t.concat()
                self._draw_marvin_vector(c, w, h)
                NSGraphicsContext.restoreGraphicsState()
                return

            # subtle dark tint over the frosted glass for contrast + a hairline
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(0.5, 0.5, w - 1, h - 1), (h - 1) / 2.0, (h - 1) / 2.0
            )
            _rgb(0.05, 0.05, 0.08, 0.30 if c.mode != "idle" else 0.18).setFill()
            path.fill()
            _rgb(1, 1, 1, 0.12).setStroke()
            path.setLineWidth_(1.0)
            path.stroke()

            if c.mode == "idle":
                color, radius = _model_style(c.model)
                if c.hover:
                    self._glow_dot(18, h / 2.0, radius, color)
                    self._text(_hover_label(c.model), 34, h, size=11)
                else:
                    self._glow_dot(w / 2.0, h / 2.0, radius, color)
                return

            # recording / transcribing: status dot + label + gradient bars
            top, bot = _GRAD.get(c.mode, _GRAD["recording"])
            self._glow_dot(17, h / 2.0, 4.5, top)
            label = "Lyssnar…" if c.mode == "recording" else "Skriver…"
            self._text(label, w - 66, h, size=10, dim=True)

            grad = NSGradient.alloc().initWithStartingColor_endingColor_(
                _rgb(*top), _rgb(*bot)
            )
            x0, x1 = 32.0, w - 74.0
            span = max(x1 - x0, 10.0)
            n = len(c.levels)
            bar_w = max(2.0, span / max(n, 1) * 0.5)
            for i, lvl in enumerate(c.levels):
                bh = 3.0 + max(0.0, min(1.0, lvl)) * (h - 12.0)
                x = x0 + span * (i / max(n - 1, 1))
                y = (h - bh) / 2.0
                bar = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, y, bar_w, bh), bar_w / 2.0, bar_w / 2.0
                )
                grad.drawInBezierPath_angle_(bar, 90.0)

        @objc.python_method
        def _draw_marvin_image(self, c, w, h):
            """Draw one Marvin frame (the current flipbook frame while a clip is
            active, else the still centre pose) and, while dictating, add a
            voice-reactive green bloom over the frame's eyes."""
            shown = (c._oneshot if (c._oneshot and c._oneshot in c.clips)
                     else c._active_clip_name(c.mode))
            # super_saiyan: draw at a fixed larger rect so the head == idle size while the aura
            # flares beyond the head box; every other clip uses the normal head-box inset.
            base_inset = _SS_INSET if shown == _SS_CLIP else _MARVIN_INSET
            m = w * base_inset
            rx, ry, rw, rh = m, m, w - 2 * m, h - 2 * m
            rect = NSMakeRect(rx, ry, rw, rh)

            # A one-shot gesture (wake/spin/react) overrides everything.
            if c._oneshot and c._oneshot in c.clips:
                frames = c.clips[c._oneshot]
                frames[min(len(frames) - 1, int(c._oneshot_f))].drawInRect_(rect)
                return

            name = c._active_clip_name(c.mode)
            clip = c.clips.get(name) if name else None
            fi = 0
            if clip:
                fi = max(0, min(len(clip) - 1, int(c._clip_f)))
                img = clip[fi]
            else:
                img = c.center
            if img is None:
                return
            img.drawInRect_(rect)  # simple draw is flip-safe and opaque

            level = c.levels[-1] if c.levels else 0.0
            if c.mode == "recording" and level > 0.05 and name:
                eyes_seq = c.clip_eyes.get(name)
                if eyes_seq and fi < len(eyes_seq):
                    NSGraphicsContext.saveGraphicsState()
                    NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(w * 0.02, h * 0.02, w * 0.96, h * 0.96)
                    ).addClip()
                    self._eye_glow(rx, ry, rw, rh, eyes_seq[fi], level)
                    NSGraphicsContext.restoreGraphicsState()

        @objc.python_method
        def _eye_glow(self, rx, ry, rw, rh, eyes, glow):
            """Soft feathered green bloom over each eye (stacked radial gradients
            so it fades gradually), scaled by loudness."""
            core = min(0.65, glow * 0.85)
            for nx, ny in eyes:
                gx, gy = rx + nx * rw, ry + ny * rh
                for scale, alpha in ((0.08, core), (0.15, core * 0.5),
                                     (0.24, core * 0.25)):
                    rad = rw * scale * (1.0 + glow * 0.4)
                    grad = NSGradient.alloc().initWithColors_([
                        _rgb(0.60, 1.0, 0.45, alpha),
                        _rgb(0.60, 1.0, 0.45, 0.0),
                    ])
                    path = NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(gx - rad, gy - rad, 2 * rad, 2 * rad)
                    )
                    grad.drawInBezierPath_relativeCenterPosition_(
                        path, NSMakePoint(0.0, 0.0)
                    )

        @objc.python_method
        def _draw_marvin_live(self, c, w, h):
            """Eyeless grey head under a gentle 2D head transform, then the two
            procedural green eyes (blink, gaze, expression, voice glow) on top."""
            NSGraphicsContext.saveGraphicsState()
            t = NSAffineTransform.transform()
            t.translateXBy_yBy_(w / 2.0, h / 2.0 + c.live_bob)
            t.rotateByDegrees_(c.live_tilt)
            t.translateXBy_yBy_(-w / 2.0, -h / 2.0)
            t.concat()

            m = w * _MARVIN_INSET
            rx, ry, rw, rh = m, m, w - 2 * m, h - 2 * m
            if c.eyeless is not None:
                c.eyeless.drawInRect_(NSMakeRect(rx, ry, rw, rh))

            gx, gy = c.gaze
            hw, hh = _EYE_W * rw / 2.0, _EYE_H * rh / 2.0
            for side, base, p in (("l", _EYE_L, c.eye_l), ("r", _EYE_R, c.eye_r)):
                ex = rx + (base[0] + gx) * rw
                ey = ry + (base[1] + gy) * rh + p["off"] * hh
                self._draw_live_eye(c, side, ex, ey, hw, hh, p)
            NSGraphicsContext.restoreGraphicsState()

        @objc.python_method
        def _draw_live_eye(self, c, side, ex, ey, hw, hh, p):
            """One eye as a green 'lens' between an upper and lower curved lid.
            aperture (with blink) sets openness; bow arcs it (happy/sad); outer
            drops the outer corner (skeptic/sad). Grey head shows where the lens
            doesn't cover, which reads as the eyelids."""
            glow = c.live_glow
            a = max(0.0, min(1.0, p["ap"] * c.blink)) * hh
            bow = p["bow"] * hh
            outer = p["outer"] * hh
            lc_y = ey + (outer if side == "l" else 0.0)
            rc_y = ey + (0.0 if side == "l" else outer)
            g = _EYE_GREEN
            bright = min(1.0, 0.88 + glow * 0.35)

            # voice bloom behind the eye
            if glow > 0.05:
                self._live_bloom(ex, ey, hw, glow)

            if a < 1.6:  # shut → a thin glowing lid seam
                seam = NSBezierPath.bezierPath()
                seam.moveToPoint_(NSMakePoint(ex - hw, lc_y))
                cM = NSMakePoint(ex, ey - bow)
                seam.curveToPoint_controlPoint1_controlPoint2_(
                    NSMakePoint(ex + hw, rc_y), cM, cM)
                seam.setLineWidth_(max(1.4, hh * 0.16))
                _rgb(g[0] * bright, g[1], g[2] * bright,
                     0.5 + glow * 0.4).setStroke()
                seam.setLineCapStyle_(1)  # round
                seam.stroke()
                return

            # flatter top (heavy upper lid), rounder bottom -> a weary eye
            path = NSBezierPath.bezierPath()
            path.moveToPoint_(NSMakePoint(ex - hw, lc_y))
            cU = NSMakePoint(ex, ey - a * 0.80 - bow)
            path.curveToPoint_controlPoint1_controlPoint2_(
                NSMakePoint(ex + hw, rc_y), cU, cU)
            cL = NSMakePoint(ex, ey + a * 1.10 - bow)
            path.curveToPoint_controlPoint1_controlPoint2_(
                NSMakePoint(ex - hw, lc_y), cL, cL)
            path.closePath()
            _rgb(g[0] * bright, g[1], g[2] * bright, 0.97).setFill()
            path.fill()

        @objc.python_method
        def _live_bloom(self, cx, cy, hw, glow):
            core = min(0.6, glow * 0.8)
            for scale, alpha in ((0.9, core), (1.6, core * 0.5), (2.6, core * 0.22)):
                rad = hw * scale
                grad = NSGradient.alloc().initWithColors_([
                    _rgb(0.6, 1.0, 0.5, alpha), _rgb(0.6, 1.0, 0.5, 0.0)])
                path = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - rad, cy - rad, 2 * rad, 2 * rad))
                grad.drawInBezierPath_relativeCenterPosition_(
                    path, NSMakePoint(0.0, 0.0))

        @objc.python_method
        def _draw_marvin_vector(self, c, w, h):
            cx, cy = w / 2.0, h / 2.0
            R = min(w, h) / 2.0 - 3.0
            dark = (0.13, 0.13, 0.16)

            # head
            face = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - R, cy - R, 2 * R, 2 * R)
            )
            _rgb(0.93, 0.92, 0.87, 1.0).setFill()
            face.fill()
            _rgb(*dark).setStroke()
            face.setLineWidth_(max(2.0, R * 0.06))
            face.stroke()

            # brow line (slightly above the equator, like the reference)
            brow_y = cy - R * 0.02
            half = R * 0.80
            brow = NSBezierPath.bezierPath()
            brow.moveToPoint_(NSMakePoint(cx - half, brow_y))
            brow.lineToPoint_(NSMakePoint(cx + half, brow_y))
            brow.setLineWidth_(max(2.0, R * 0.05))
            _rgb(*dark).setStroke()
            brow.stroke()

            # two big downward green triangle eyes that glow/flicker with voice
            level = c.levels[-1] if c.levels else 0.0
            glow = level if c.mode in ("recording", "transcribing") else 0.0
            ew, eh = R * 0.42, R * 0.60
            for ex in (cx - R * 0.36, cx + R * 0.36):
                # soft green bloom behind the eye, intensity = loudness
                if glow > 0.04:
                    s = 1.0 + glow * 0.8
                    bloom = NSBezierPath.bezierPath()
                    bloom.moveToPoint_(NSMakePoint(ex - ew * s / 2, brow_y + 1))
                    bloom.lineToPoint_(NSMakePoint(ex + ew * s / 2, brow_y + 1))
                    bloom.lineToPoint_(NSMakePoint(ex, brow_y + eh * s))
                    bloom.closePath()
                    _rgb(0.55, 1.0, 0.35, min(0.6, glow * 0.7)).setFill()
                    bloom.fill()
                tri = NSBezierPath.bezierPath()
                tri.moveToPoint_(NSMakePoint(ex - ew / 2, brow_y + 1))
                tri.lineToPoint_(NSMakePoint(ex + ew / 2, brow_y + 1))
                tri.lineToPoint_(NSMakePoint(ex, brow_y + eh))
                tri.closePath()
                # brighter green as it speaks
                g = (min(0.65, 0.49 + glow * 0.3), min(1.0, 0.83 + glow * 0.15),
                     0.13 + glow * 0.2)
                _rgb(*g).setFill()
                tri.fill()
                _rgb(*dark).setStroke()
                tri.setLineWidth_(1.5)
                tri.stroke()

            # no mouth: he "speaks" with his eyes (the glow above). Just a faint
            # glum resting curve for character.
            my = cy + R * 0.66
            mouth = NSBezierPath.bezierPath()
            mouth.moveToPoint_(NSMakePoint(cx - R * 0.30, my))
            mouth.curveToPoint_controlPoint1_controlPoint2_(
                NSMakePoint(cx + R * 0.30, my),
                NSMakePoint(cx - R * 0.10, my + R * 0.12),
                NSMakePoint(cx + R * 0.10, my + R * 0.12),
            )
            mouth.setLineWidth_(max(1.6, R * 0.04))
            _rgb(0.13, 0.13, 0.16, 0.5).setStroke()
            mouth.stroke()

        @objc.python_method
        def _glow_dot(self, cx, cy, radius, rgb):
            r, g, b = rgb
            # soft halo
            _rgb(r, g, b, 0.28).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - radius * 2.1, cy - radius * 2.1,
                           radius * 4.2, radius * 4.2)
            ).fill()
            # core
            _rgb(r, g, b, 1.0).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
            ).fill()

        @objc.python_method
        def _text(self, text, x, h, size=11, dim=False):
            attrs = NSMutableDictionary.dictionary()
            attrs[NSFontAttributeName] = NSFont.systemFontOfSize_(size)
            shade = 0.6 if dim else 0.95
            attrs[NSForegroundColorAttributeName] = (
                NSColor.colorWithCalibratedWhite_alpha_(shade, 1.0)
            )
            s = objc.lookUpClass("NSString").stringWithString_(text)
            tsize = s.sizeWithAttributes_(attrs)
            s.drawAtPoint_withAttributes_(
                NSMakePoint(x, (h - tsize.height) / 2.0), attrs
            )

    class _Pill:
        def __init__(self, on_click=None, on_move=None, on_menu=None, pos=None,
                     style="waveform", on_double_click=None):
            self.on_click = on_click
            self.on_double_click = on_double_click
            self.on_move = on_move
            self.on_menu = on_menu
            self.style = style
            self.mode = "idle"
            self.model = "small"
            self.hover = False
            self.center = _load_marvin_center() if style == "marvin" else None
            self.clips, self.clip_eyes = (
                _load_marvin_clips() if style == "marvin" else ({}, {})
            )
            # marvin_live: eyeless head + procedural eye state (see _tick_live)
            self.eyeless = _load_marvin_eyeless() if style == "marvin_live" else None
            self.gaze = [0.0, 0.0]         # current gaze offset (normalized)
            self._gaze_t = [0.0, 0.0]      # gaze target
            self.blink = 1.0               # 1 open, 0 shut (multiplies aperture)
            self._blink_seq = []
            self._blink_t = random.randint(40, 140)
            self._sacc_t = random.randint(30, 80)
            self.live_glow = 0.0           # eased voice level -> eye glow
            self.live_bob = 0.0            # head bob (px)
            self.live_tilt = 0.0           # head roll (deg)
            self.eye_l = dict(_EXPR["neutral"])   # interpolated per-eye params
            self.eye_r = dict(_EXPR["neutral"])
            self._expr = "neutral"
            self._expr_hold = 0            # ticks left on a one-shot expression
            self.levels = [0.0] * _BARS
            self._phase = 0.0
            self._anim = 0.0
            self.tilt = 0.0  # head-roll degrees (vector fallback only)
            self.nod = 0.0   # vertical nod offset px (vector fallback only)
            self._clip_f = 0.0   # looping-clip frame (float, ping-ponged)
            self._clip_dir = 1
            self._rec_clip = "shake"   # alternates shake/nod each dictation
            self._prev_mode = "idle"
            self._oneshot = None       # a gesture clip playing once (wake/spin/react)
            self._oneshot_f = 0.0
            # Idle micro-gestures: the subset of _IDLE_GESTURE_CLIPS actually
            # present, one played at random every ~15-35s (20 ticks/sec) so
            # Marvin reads as alive rather than frozen while resting.
            self._idle_gestures = [
                n for n in _IDLE_GESTURE_CLIPS if n in self.clips
            ]
            self._idle_gesture_t = random.randint(300, 700)
            self._demo_queue = []      # remaining clips in a "play all" showcase
            _marv = style in ("marvin", "marvin_live")
            self._h = _MARVIN_SIZE if _marv else _HEIGHT
            self._w = _MARVIN_SIZE if _marv else _WIDTH_IDLE

            if pos and len(pos) == 2 and pos[0] is not None:
                x, y = float(pos[0]), float(pos[1])
            else:
                screen = _screen_with_mouse().frame()
                x = screen.origin.x + (screen.size.width - self._w) / 2.0
                y = screen.origin.y + screen.size.height - self._h - 64.0

            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, self._w, self._h),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setLevel_(NSStatusWindowLevel)
            # No window shadow for Marvin — it would render as a square around
            # the circular head. The waveform pill keeps its shadow.
            self.window.setHasShadow_(not _marv)
            self.window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
            )

            # frosted-glass content, rounded (a pill for waveform, circle for
            # Marvin since width == height). Marvin draws an opaque face on top.
            fx = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, 0, self._w, self._h)
            )
            fx.setMaterial_(NSVisualEffectMaterialHUDWindow)
            fx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
            fx.setState_(NSVisualEffectStateActive)
            fx.setWantsLayer_(True)
            fx.layer().setCornerRadius_(self._h / 2.0)
            fx.layer().setMasksToBounds_(True)
            if _marv:
                fx.setHidden_(True)  # face is opaque; no glass needed behind it
            self.window.setContentView_(fx)

            self.view = _WaveView.alloc().initWithController_(self)
            self.view.setFrame_(NSMakeRect(0, 0, self._w, self._h))
            self.view.setAutoresizingMask_(1 << 1 | 1 << 4)  # width | height
            self.window.setContentView_(self.view) if _marv else fx.addSubview_(self.view)
            self.window.orderFrontRegardless()

        def _resize(self, width, animate=True):
            if abs(width - self._w) < 0.5:
                return
            self._w = width
            f = self.window.frame()
            new = NSMakeRect(f.origin.x, f.origin.y, width, self._h)  # left-anchored
            if animate:
                NSAnimationContext.beginGrouping()
                NSAnimationContext.currentContext().setDuration_(0.16)
                self.window.animator().setFrame_display_(new, True)
                NSAnimationContext.endGrouping()
            else:
                self.window.setFrame_display_(new, True)

        # -------- main-thread updates -------------------------------------

        def set_model(self, model):
            self.model = model
            self._render()

        def set_hover(self, value):
            self.hover = bool(value)
            self._render()

        def _target_width(self):
            if self.style in ("marvin", "marvin_live"):
                return _MARVIN_SIZE  # face stays a fixed circle
            if self.mode in ("recording", "transcribing"):
                return _WIDTH_REC
            if self.hover:
                # dot area + measured text + right padding, so the bubble
                # hugs the label ("Model: small" vs "Model: large-v3-turbo").
                return 34.0 + _measure(_hover_label(self.model)) + 16.0
            return _WIDTH_IDLE

        def _render(self):
            self._resize(self._target_width())
            self.view.setNeedsDisplay_(True)

        def tick(self, mode, level):
            self.mode = mode
            if self.style == "marvin_live":
                self._tick_live(mode, level)
                self._render()
                return
            if mode == "recording":
                self.levels = self.levels[1:] + [max(0.03, level)]
            elif mode == "transcribing":
                self._phase += 0.5
                self.levels = [
                    0.25 + 0.22 * math.sin(self._phase + i * 0.45)
                    for i in range(_BARS)
                ]
            else:
                if any(v > 0.001 for v in self.levels):
                    self.levels = [v * 0.6 for v in self.levels]

            # head-tilt animation (marvin): gentle "listening" sway while
            # dictating, a barely-there bob when idle. Eased for smoothness.
            self._anim += 0.05
            if mode in ("recording", "transcribing"):
                target = 9.0 * math.sin(self._anim * 2.2)          # attentive roll
                nod_target = 2.6 * math.sin(self._anim * 3.1 + 1)  # gentle nod
            else:
                target = 2.0 * math.sin(self._anim * 0.9)          # subtle idle
                nod_target = 1.0 * math.sin(self._anim * 1.1)
            self.tilt += (target - self.tilt) * 0.25
            self.nod += (nod_target - self.nod) * 0.25

            # Mode transitions trigger one-shot gestures.
            if mode == "recording" and self._prev_mode != "recording":
                order = [n for n in ("alert", "shake", "nod") if n in self.clips]
                if order:  # alternate the listening loop each dictation
                    cur = self._rec_clip if self._rec_clip in order else order[0]
                    self._rec_clip = order[(order.index(cur) + 1) % len(order)]
                self.play_oneshot("wake")     # perk up when you start (if present)
            elif mode == "transcribing" and self._prev_mode == "recording":
                self.play_oneshot("spin")     # flourish when you finish
            self._prev_mode = mode

            # Occasional idle micro-gesture: pick a random gesture from the pool
            # now and then while resting, so Marvin reads as alive not frozen.
            if mode == "idle" and self._oneshot is None and self._idle_gestures:
                self._idle_gesture_t -= 1
                if self._idle_gesture_t <= 0:
                    self.play_oneshot(random.choice(self._idle_gestures))
                    self._idle_gesture_t = random.randint(300, 700)
            elif mode != "idle":
                self._idle_gesture_t = random.randint(300, 700)

            # Advance a one-shot gesture (plays once, then clears).
            if self._oneshot:
                frames = self.clips.get(self._oneshot)
                if frames and self._oneshot_f < len(frames) - 1:
                    self._oneshot_f += _ONESHOT_SPEED.get(self._oneshot, 1.0)
                else:
                    self._oneshot = None
                    if self._demo_queue:  # "play all": chain the next clip
                        self.play_oneshot(self._demo_queue.pop(0))

            # Advance the looping listening clip (ping-pong) unless a one-shot
            # is playing; hold still on centre when idle.
            clip = None if self._oneshot else self._active_clip(mode)
            if clip:
                n = len(clip)
                if mode == "sleep":
                    # doze off: play forward once, then hold on the closed frame
                    self._clip_f = min(n - 1, self._clip_f + 0.8)
                else:
                    self._clip_f += self._clip_dir * 0.8
                    if self._clip_f >= n - 1:
                        self._clip_f = n - 1
                        self._clip_dir = -1
                    elif self._clip_f <= 0:
                        self._clip_f = 0
                        self._clip_dir = 1
            else:
                self._clip_f = 0.0
                self._clip_dir = 1
            self._render()

        # -------- marvin_live: procedural drive ---------------------------

        def _tick_live(self, mode, level):
            """Advance the procedural eye systems once (called at ~20 Hz)."""
            # voice -> eye glow
            self.live_glow += ((level if mode == "recording" else 0.0)
                               - self.live_glow) * 0.3

            # expression: hold a one-shot / demo step, else follow the mode
            if self._expr_hold > 0:
                self._expr_hold -= 1
            elif self._demo_queue:
                self._expr = self._demo_queue.pop(0)
                self._expr_hold = 30
            else:
                self._expr = {"recording": "alert",
                              "transcribing": "neutral"}.get(mode, "neutral")
                # occasional idle micro-expression so he's never a blank stare
                if mode == "idle" and random.random() < 0.004:
                    self.play_oneshot(random.choice(
                        ("curious", "skeptic", "sad", "happy")))

            if self._expr in _EXPR_ASYM:
                tl, tr = _EXPR_ASYM[self._expr]
            else:
                tl = tr = _EXPR.get(self._expr, _EXPR["neutral"])
            for cur, tgt in ((self.eye_l, tl), (self.eye_r, tr)):
                for k in ("ap", "off", "bow", "outer"):
                    cur[k] += (tgt[k] - cur[k]) * 0.25

            # blink: play the sequence, else count down to the next one
            if self._blink_seq:
                self.blink = self._blink_seq.pop(0)
            else:
                self.blink = 1.0
                self._blink_t -= 1
                if self._blink_t <= 0:
                    self._blink_seq = list(_BLINK_SEQ)
                    base = random.randint(40, 160)
                    self._blink_t = base // 2 if mode == "recording" else base

            # gaze: snappy saccades toward a target (cursor / glance) + drift
            self._sacc_t -= 1
            if self._sacc_t <= 0:
                self._gaze_t = self._pick_gaze(mode)
                self._sacc_t = random.randint(24, 80)
            for i in (0, 1):
                self.gaze[i] += (self._gaze_t[i] - self.gaze[i]) * 0.55
                self.gaze[i] += random.uniform(-0.002, 0.002)
                self.gaze[i] = max(-0.09, min(0.09, self.gaze[i]))

            # head: gentle bob + tiny roll (eyes lead via gaze/blink above)
            self._anim += 0.05
            if mode in ("recording", "transcribing"):
                self.live_tilt += (2.2 * math.sin(self._anim * 1.6) - self.live_tilt) * 0.2
                self.live_bob += (1.4 * math.sin(self._anim * 2.0) - self.live_bob) * 0.2
            else:
                self.live_tilt += (1.0 * math.sin(self._anim * 0.7) - self.live_tilt) * 0.15
                self.live_bob += (1.2 * math.sin(self._anim * 0.9) - self.live_bob) * 0.15

            # mode-transition flourishes: perk-up blink on start, a brief happy
            # squint when a dictation finishes (recording -> transcribing).
            if mode == "recording" and self._prev_mode != "recording":
                self._blink_seq = list(_BLINK_SEQ)
            elif mode == "transcribing" and self._prev_mode == "recording":
                self.play_oneshot("happy")
            self._prev_mode = mode

        def _pick_gaze(self, mode):
            if mode == "transcribing":  # "thinking" — avert up and to a side
                return [random.choice((-1.0, 1.0)) * random.uniform(0.03, 0.06),
                        -random.uniform(0.03, 0.06)]
            # mostly follow the cursor, sometimes disengage with a random glance
            if random.random() < (0.7 if mode == "recording" else 0.55):
                g = self._cursor_gaze()
                if g is not None:
                    return g
            r = 0.075
            return [random.uniform(-r, r), random.uniform(-r * 0.6, r * 0.6)]

        def _cursor_gaze(self):
            """Gaze offset toward the mouse cursor (bounded), or None on failure."""
            try:
                p = NSEvent.mouseLocation()
                f = self.window.frame()
                dx = p.x - (f.origin.x + f.size.width / 2.0)
                dy = p.y - (f.origin.y + f.size.height / 2.0)  # screen y is up
                scale = 500.0
                gx = max(-1.0, min(1.0, dx / scale)) * 0.075
                gy = max(-1.0, min(1.0, -dy / scale)) * 0.06   # up -> eyes up
                return [gx, gy]
            except Exception:
                return None

        def play_oneshot(self, name):
            """Trigger a gesture clip to play through once (no-op if absent).
            In marvin_live, gesture names map to procedural expressions."""
            if self.style == "marvin_live":
                expr = {"wake": "alert", "spin": "happy", "nod": "happy",
                        "shake": "skeptic", "react": "curious",
                        "blink": "neutral"}.get(name, name)
                if expr in _EXPR or expr in _EXPR_ASYM:
                    self._expr = expr
                    self._expr_hold = 26      # ~1.3 s, then eases back to mode
                if name == "blink":
                    self._blink_seq = list(_BLINK_SEQ)
                return
            if name in self.clips:
                self._oneshot = name
                self._oneshot_f = 0.0

        def play_all(self):
            """Showcase: play every gesture once, back to back."""
            if self.style == "marvin_live":
                queue = ["alert", "curious", "skeptic", "happy", "sad", "neutral"]
                self._expr = queue[0]
                self._expr_hold = 30
                self._demo_queue = queue[1:]
                return
            order = ["wake", "alert", "happy", "sad", "angry", "skeptic", "blink",
                     "glance", "look_left", "look_right", "nod", "shake",
                     "emote", "super_saiyan", "spin"]
            queue = [n for n in order if n in self.clips]
            # append any other loaded clips (e.g. sleep) at the end, for completeness
            queue += [n for n in self.clips if n not in queue]
            if not queue:
                return
            self._demo_queue = queue[1:]
            self.play_oneshot(queue[0])

        def _active_clip_name(self, mode):
            if mode == "recording":
                return self._rec_clip if self._rec_clip in self.clips else (
                    "shake" if "shake" in self.clips else None)
            if mode == "transcribing":
                return "spin" if "spin" in self.clips else None
            if mode == "sleep":
                return "sleep" if "sleep" in self.clips else None
            return None

        def _active_clip(self, mode):
            name = self._active_clip_name(mode)
            return self.clips.get(name) if name else None

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform", on_double_click=None):
        """Build and show the pill. Returns a controller, or None on failure."""
        try:
            return _Pill(on_click=on_click, on_move=on_move,
                         on_menu=on_menu, pos=pos, style=style,
                         on_double_click=on_double_click)
        except Exception as exc:  # pragma: no cover - UI environment dependent
            print(f"[pill] disabled ({exc})", flush=True)
            return None

except Exception as _pill_import_err:  # pragma: no cover - AppKit unavailable
    import traceback as _tb
    print(f"[pill] UI unavailable: {_pill_import_err!r}", flush=True)
    _tb.print_exc()

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform", on_double_click=None):
        return None
