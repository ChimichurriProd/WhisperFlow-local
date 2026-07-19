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
_MARVIN_SIZE = 280.0  # big window so auras/halos can flare beyond the head without clipping.
# The extra window is transparent/invisible — only Marvin + his glow show.
# v3 clips are authored with ONE consistent framing: the head is ~70% of the
# 256px frame with a wide safe margin baked in for glow/motion, so every clip
# (including effect-heavy ones) draws with this single inset. Head ~58px.
_MARVIN_INSET = 0.352

# Two-faced Marvin: a second "oracle" persona lives on the BACK of his head
# (cyan lens eyes + halo, assets/marvin/back.png). While dictating he faces you
# as the green-eyed "scribe"; when he answers a question he whips around (motion
# -blurred spin clip: frame 0 = front scribe, frame _TURN_BACK = cyan oracle) to
# reveal the oracle, then whips back when he's done.
_TURN_BACK = 60        # spin frame index showing the full oracle face
_TURN_SPEED = 4.0      # spin frames advanced per 20 Hz tick (~0.75 s per turn)
# Cyan eye centres on the back face (normalized to the frame, y from top),
# measured from back.png — drives the oracle's voice-reactive glow.
_BACK_EYES = ((0.360, 0.525), (0.638, 0.525))
_BACK_EYE_RGB = (0.37, 0.88, 1.0)   # cyan bloom for the oracle's voice glow


def _ease_toward(cur, target, speed):
    """Step *cur* toward *target* by at most *speed* (never overshoots). Used to
    animate the front/back persona turn one 20 Hz tick at a time. Defined at
    module scope (outside the AppKit import) so it's testable headlessly."""
    if cur < target:
        return min(target, cur + speed)
    if cur > target:
        return max(target, cur - speed)
    return cur

# Per-STT-model idle loop clips: when assets/marvin/<clip>/ exists for the
# active model, Marvin's resting loop plays it. The v3 set ships ONE shared
# "idle" (the per-model folders fall back to it — see _active_clip_name).
_MODEL_IDLE_CLIP = {
    "base": "idle_base",
    "small": "idle_small",
    "medium": "idle_medium",
    "large-v3-turbo": "idle_large",
}

# Clip names eligible as idle micro-gestures: while resting, Marvin randomly
# plays one of these every so often to read as alive. A name only joins the
# pool if a matching assets/marvin/<name>/ clip exists, so dropping in a new
# folder (e.g. "blink") auto-enrolls it. Excludes nod/shake/wake/spin, which
# have their own event triggers.
_IDLE_GESTURE_CLIPS = ("skeptic", "curious", "glance", "blink", "yawn", "emote",
                       "angry", "love", "stressed", "glow", "super_saiyan",
                       "droop", "drift", "eyeroll",          # B personals
                       "focus", "gleam", "doubletake",       # A personals
                       "stretch", "nuzzle", "hearteyes")     # G personals
# Per-clip one-shot playback speed (frames advanced per 20 Hz tick; default 1.0).
# The 120-frame out-and-back gestures play at 1.6 (~3.75 s); spin stays a
# quick flourish.
_ONESHOT_SPEED = {"spin": 2.4, "angry": 1.6, "droop": 1.6, "drift": 1.6,
                  "eyeroll": 1.6, "focus": 1.6, "gleam": 1.6,
                  "doubletake": 1.6, "stretch": 1.6, "nuzzle": 1.6,
                  "hearteyes": 1.6, "super_saiyan": 1.6}

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

    def _marvin_dir(skin=None):
        """Asset dir for a Marvin skin. assets/marvin/ is the default (B);
        alternate skins live in assets/marvin/_skins/<name>/ with the same
        layout (center.png, back.png, meta.json, <clip>/frame_*.png)."""
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "assets" / "marvin"
        if skin:
            cand = base / "_skins" / str(skin)
            if cand.is_dir():
                return cand
        return base

    def _load_marvin_center(base):
        """Load the resting head image (center.png), shown when idle."""
        p = base / "center.png"
        return NSImage.alloc().initWithContentsOfFile_(str(p)) if p.exists() else None

    def _load_marvin_back(base):
        """Load the back-of-head 'oracle' face (back.png), shown while answering."""
        p = base / "back.png"
        return NSImage.alloc().initWithContentsOfFile_(str(p)) if p.exists() else None

    def _load_marvin_meta(base):
        """Optional per-skin meta.json (e.g. measured oracle eye centres)."""
        import json

        p = base / "meta.json"
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            return {}

    def _load_marvin_clips(base):
        """Load frame-sequence clips from <base>/<name>/frame_*.png plus
        per-frame eye positions from <name>/eyes.json.

        Returns (clips, eyes): clips={name:[NSImage,...]}, eyes={name:[[(lx,ly),
        (rx,ry)],...]} for the voice-reactive glow.
        """
        import json

        clips, eyes = {}, {}
        if base.is_dir():
            for sub in base.iterdir():
                if not sub.is_dir() or sub.name.startswith("_"):
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
            # Turned toward the back 'oracle' persona: draw the turnaround / back
            # face instead of the front clips.
            if c._turn_f > 0.5:
                self._draw_marvin_back(c, w, h)
                return
            # v3 clips share one framing (head ~70% + safe margin baked into the
            # frames), so every clip draws at the same inset — no per-clip boxes.
            m = w * _MARVIN_INSET
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
        def _draw_marvin_back(self, c, w, h):
            """Draw the back 'oracle' persona: the rotating turnaround frame while
            he's mid-turn, and the cyan-eyed back face once settled (with a
            voice-reactive cyan bloom while he's listening to a question)."""
            m = w * _MARVIN_INSET
            rx, ry, rw, rh = m, m, w - 2 * m, h - 2 * m
            rect = NSMakeRect(rx, ry, rw, rh)

            spin = c.clips.get("spin")
            settled = c._turn_f >= _TURN_BACK - 0.5
            if settled and c.back is not None:
                c.back.drawInRect_(rect)                 # back face + cyan eyes
            elif spin:
                fi = max(0, min(len(spin) - 1, int(round(c._turn_f))))
                spin[fi].drawInRect_(rect)               # mid-turn (no face yet)
            elif c.back is not None:
                c.back.drawInRect_(rect)                 # no turnaround: just snap
            else:
                return

            level = c.levels[-1] if c.levels else 0.0
            if settled and c.mode == "recording" and level > 0.05:
                NSGraphicsContext.saveGraphicsState()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(w * 0.02, h * 0.02, w * 0.96, h * 0.96)
                ).addClip()
                self._eye_glow(rx, ry, rw, rh, c.back_eyes, level,
                               rgb=_BACK_EYE_RGB)
                NSGraphicsContext.restoreGraphicsState()

        @objc.python_method
        def _eye_glow(self, rx, ry, rw, rh, eyes, glow, rgb=(0.60, 1.0, 0.45)):
            """Soft feathered bloom over each eye (stacked radial gradients so it
            fades gradually), scaled by loudness. Green for the front scribe,
            cyan for the back oracle (via *rgb*)."""
            r, g, b = rgb
            core = min(0.65, glow * 0.85)
            for nx, ny in eyes:
                gx, gy = rx + nx * rw, ry + ny * rh
                for scale, alpha in ((0.08, core), (0.15, core * 0.5),
                                     (0.24, core * 0.25)):
                    rad = rw * scale * (1.0 + glow * 0.4)
                    grad = NSGradient.alloc().initWithColors_([
                        _rgb(r, g, b, alpha),
                        _rgb(r, g, b, 0.0),
                    ])
                    path = NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(gx - rad, gy - rad, 2 * rad, 2 * rad)
                    )
                    grad.drawInBezierPath_relativeCenterPosition_(
                        path, NSMakePoint(0.0, 0.0)
                    )

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

    class _ShowcaseView(NSView):
        """Grid canvas for the showcase window: draws every clip's current
        frame with its name underneath. Frame advancement lives in _Showcase;
        this just renders."""

        def initWithGallery_(self, gallery):
            self = objc.super(_ShowcaseView, self).init()
            if self is None:
                return None
            self.gallery = gallery
            return self

        def drawRect_(self, rect):
            # An exception escaping drawRect_ takes the whole app down via
            # AppKit's _crashOnException — never let one out.
            try:
                self._draw()
            except Exception as exc:
                print(f"[showcase] draw failed: {exc!r}", flush=True)

        @objc.python_method
        def _draw(self):
            g = self.gallery
            _rgb(0.07, 0.07, 0.08).setFill()
            NSBezierPath.fillRect_(self.bounds())
            H = self.bounds().size.height
            attrs = NSMutableDictionary.dictionary()
            attrs[NSFontAttributeName] = NSFont.systemFontOfSize_(11.0)
            attrs[NSForegroundColorAttributeName] = _rgb(0.72, 0.78, 0.72)
            NSString = objc.lookUpClass("NSString")
            for i, name in enumerate(g.names):
                col, row = i % g.cols, i // g.cols
                x = 10.0 + col * g.CELL_W
                y = H - 10.0 - (row + 1) * g.CELL_H
                frames = g.clips[name]
                fi = int(g.pos[name]) % len(frames)
                img_rect = NSMakeRect(x + (g.CELL_W - g.IMG) / 2.0,
                                      y + g.CELL_H - g.IMG - 4.0, g.IMG, g.IMG)
                frames[fi].drawInRect_(img_rect)
                label = NSString.stringWithString_(name.replace("_", " "))
                lw = label.sizeWithAttributes_(attrs).width
                label.drawAtPoint_withAttributes_(
                    NSMakePoint(x + (g.CELL_W - lw) / 2.0, y + 6.0), attrs)

    class _Showcase:
        """'All animations' window: every clip of the active skin looping at
        once in a grid. Advanced from the pill's 20 Hz tick (main thread)."""

        ORDER = ["idle", "listening", "thinking", "angry",
                 "droop", "drift", "eyeroll", "focus", "gleam", "doubletake",
                 "stretch", "nuzzle", "hearteyes", "super_saiyan", "spin"]
        CELL_W, CELL_H, IMG = 152.0, 176.0, 140.0

        def __init__(self, pill):
            self.clips = dict(pill.clips)
            self.names = [n for n in self.ORDER if n in self.clips]
            self.names += [n for n in sorted(self.clips)
                           if n not in self.names]
            self.pos = {n: 0.0 for n in self.names}
            self.cols = min(4, max(1, len(self.names)))
            self.rows = (len(self.names) + self.cols - 1) // self.cols
            w = self.cols * self.CELL_W + 20.0
            h = self.rows * self.CELL_H + 20.0
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(220.0, 220.0, w, h),
                1 | 2,  # titled | closable
                NSBackingStoreBuffered,
                False,
            )
            self.window.setTitle_("Marvin — all animations")
            self.window.setReleasedWhenClosed_(False)
            self.view = _ShowcaseView.alloc().initWithGallery_(self)
            self.view.setFrame_(NSMakeRect(0, 0, w, h))
            self.window.setContentView_(self.view)
            self.window.makeKeyAndOrderFront_(None)

        def tick(self):
            """Advance all clips one frame; False once the window is closed."""
            if not self.window.isVisible():
                return False
            for n in self.names:
                self.pos[n] = (self.pos[n] + 1.0) % len(self.clips[n])
            self.view.setNeedsDisplay_(True)
            return True

    class _Pill:
        def __init__(self, on_click=None, on_move=None, on_menu=None, pos=None,
                     style="waveform", on_double_click=None, skin=None):
            self.on_click = on_click
            self.on_double_click = on_double_click
            self.on_move = on_move
            self.on_menu = on_menu
            self.style = style
            self.mode = "idle"
            self.model = "small"
            self.hover = False
            self.skin = skin
            self.center = None
            self.back = None
            self.clips, self.clip_eyes = {}, {}
            self.back_eyes = _BACK_EYES
            if style == "marvin":
                self._load_skin_assets(skin)
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
            # Front/back persona turn: _turn_f is the current facing in spin-frame
            # units (0 = front scribe, _TURN_BACK = back oracle); _face_target is
            # where he's turning to. Equal => settled.
            self._turn_f = 0.0
            self._face_target = 0.0
            # Idle micro-gestures: the subset of _IDLE_GESTURE_CLIPS actually
            # present, one played at random every ~15-35s (20 ticks/sec) so
            # Marvin reads as alive rather than frozen while resting.
            self._idle_gestures = [
                n for n in _IDLE_GESTURE_CLIPS if n in self.clips
            ]
            self._idle_gesture_t = random.randint(300, 700)
            self._demo_queue = []      # remaining clips in a "play all" showcase
            self._showcase = None      # the "all animations" grid window
            _marv = style == "marvin"
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

        def _load_skin_assets(self, skin):
            base = _marvin_dir(skin)
            self.center = _load_marvin_center(base)
            self.back = _load_marvin_back(base)
            self.clips, self.clip_eyes = _load_marvin_clips(base)
            meta = _load_marvin_meta(base)
            be = meta.get("back_eyes")
            self.back_eyes = (tuple(map(tuple, be)) if be else _BACK_EYES)

        def set_skin(self, skin):
            """Swap to another Marvin skin in place (main thread only). Resets
            any in-flight clip state so the new skin starts clean."""
            if self.style != "marvin":
                return
            self.skin = skin
            self._load_skin_assets(skin)
            self._oneshot = None
            self._demo_queue = []
            self._clip_f = 0.0
            self._clip_dir = 1
            self._turn_f = 0.0
            self._face_target = 0.0
            self._idle_gestures = [
                n for n in _IDLE_GESTURE_CLIPS if n in self.clips
            ]
            self._idle_gesture_t = random.randint(300, 700)
            if self._showcase is not None and self._showcase.window.isVisible():
                self._showcase.window.close()   # reopen with the new skin
                self._showcase = _Showcase(self)
            self._render()

        def set_model(self, model):
            self.model = model
            self._render()

        def set_hover(self, value):
            self.hover = bool(value)
            self._render()

        def _target_width(self):
            if self.style == "marvin":
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

        def show_showcase(self):
            """Open (or focus) the grid window with every clip animating."""
            if self.style != "marvin" or not self.clips:
                return
            if self._showcase is not None and self._showcase.window.isVisible():
                self._showcase.window.makeKeyAndOrderFront_(None)
                return
            self._showcase = _Showcase(self)

        def tick(self, mode, level):
            self.mode = mode
            if self._showcase is not None and not self._showcase.tick():
                self._showcase = None
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

            # Turn between the front (scribe) and back (oracle) personas: ease
            # _turn_f toward the target facing. >0.5 means his back face shows.
            self._turn_f = _ease_toward(self._turn_f, self._face_target, _TURN_SPEED)

            if self._turn_f > 0.5:
                # The oracle owns the whole display while he's turned around;
                # suppress the front scribe's gestures/clips so nothing is left
                # mid-play when he turns back to face you.
                self._oneshot = None
                self._clip_f = 0.0
                self._clip_dir = 1
                self._idle_gesture_t = random.randint(300, 700)
                self._prev_mode = mode
                self._render()
                return

            # ------- front (scribe) persona -------
            # Mode transitions trigger one-shot gestures.
            if mode == "recording" and self._prev_mode != "recording":
                order = [n for n in ("listening", "alert", "shake", "nod")
                         if n in self.clips]
                if order:  # alternate the listening loop each dictation
                    cur = self._rec_clip if self._rec_clip in order else order[0]
                    self._rec_clip = order[(order.index(cur) + 1) % len(order)]
                self.play_oneshot("wake")     # perk up when you start (if present)
            elif mode == "transcribing" and self._prev_mode == "recording":
                if "thinking" not in self.clips:
                    self.play_oneshot("spin")  # legacy flourish (no thinking clip)
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

        def play_oneshot(self, name):
            """Trigger a gesture clip to play through once (no-op if absent)."""
            if name in self.clips:
                self._oneshot = name
                self._oneshot_f = 0.0

        def face_back(self):
            """Turn to the back 'oracle' persona (while answering a question).
            No-op if there's no back face to show."""
            if self.back is None:
                return
            self._face_target = float(_TURN_BACK)
            if "spin" not in self.clips:   # no turnaround frames -> snap around
                self._turn_f = float(_TURN_BACK)

        def face_front(self):
            """Turn back to the front 'scribe' persona (dictation / resting)."""
            self._face_target = 0.0
            if "spin" not in self.clips:
                self._turn_f = 0.0

        def play_all(self):
            """Showcase: play every gesture once, back to back."""
            order = ["listening", "thinking", "angry", "droop", "drift",
                     "eyeroll", "focus", "gleam", "doubletake",
                     "stretch", "nuzzle", "hearteyes",
                     "wake", "alert", "happy", "sad", "love",
                     "stressed", "glow", "skeptic", "blink",
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
                if self._rec_clip in self.clips:
                    return self._rec_clip
                for n in ("listening", "shake"):
                    if n in self.clips:
                        return n
                return None
            if mode == "transcribing":
                # v3: a dedicated pondering loop; legacy sets twirled the spin.
                if "thinking" in self.clips:
                    return "thinking"
                return "spin" if "spin" in self.clips else None
            if mode == "sleep":
                return "sleep" if "sleep" in self.clips else None
            if mode == "idle":
                # per-model idle if that clip exists, else the shared idle
                name = _MODEL_IDLE_CLIP.get(self.model)
                if name not in self.clips:
                    name = "idle"
                return name if name in self.clips else None
            return None

        def _active_clip(self, mode):
            name = self._active_clip_name(mode)
            return self.clips.get(name) if name else None

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform", on_double_click=None, skin=None):
        """Build and show the pill. Returns a controller, or None on failure."""
        try:
            return _Pill(on_click=on_click, on_move=on_move,
                         on_menu=on_menu, pos=pos, style=style,
                         on_double_click=on_double_click, skin=skin)
        except Exception as exc:  # pragma: no cover - UI environment dependent
            print(f"[pill] disabled ({exc})", flush=True)
            return None

except Exception as _pill_import_err:  # pragma: no cover - AppKit unavailable
    import traceback as _tb
    print(f"[pill] UI unavailable: {_pill_import_err!r}", flush=True)
    _tb.print_exc()

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform", on_double_click=None, skin=None):
        return None
