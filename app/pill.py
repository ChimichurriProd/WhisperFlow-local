"""Floating desktop 'pill': the primary WhisperFlow UI.

A small borderless always-on-top window with a modern frosted-glass look.
Draggable anywhere; remembers where you put it. Click (without dragging)
cycles the model. Hover reveals the current model.

States:
- idle:         a small glowing round dot (color+size reflect the model)
- idle+hover:   expands to show the model ("Model: small")
- recording:    expands, gradient waveform driven by mic level
- transcribing: a gentle animated shimmer ("thinking")

Marvin (pill_style "marvin") shows the active STT model as a magical
"firefly": a small glowing orb orbiting his head in 3D — behind him it
shrinks, dims and is occluded by the head; in front it passes over his face,
its light faintly washing over him. Colour+size step blue/small (base) to
red/big (large), the same language as the waveform idle dot, so it works
across every skin without per-model clips. Cycling the model pops the orb
with a flare ring. While resting he occasionally notices it — and once in a
while swallows it, his head pulsing in its colour until it pops back out.

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


def _orb_orbit(t, theta, w):
    """Screen position + depth of the model firefly on its tilted 3D orbit
    around Marvin's head (treated as a ball at the canvas centre). Returns
    (x, y, z) with z in [-1, 1]: negative = behind the head (drawn first so
    the opaque head occludes it, smaller and dimmer), positive = in front
    (drawn over his face, slightly larger). The orbit plane is seen a little
    from above, so the front pass crosses below centre and the back pass
    above. Radius, tilt and height all breathe on slow incommensurate sines
    so the path never repeats exactly — organic, not mechanical. Module
    scope (outside the AppKit import) so it's testable headlessly."""
    cx = cy = w / 2.0
    rx = w * (0.150 + 0.018 * math.sin(t * 0.37))
    # steeper tilt + a small downward shift keep the front pass at his chin
    # instead of crossing his eyes (reads as an artefact when over the face)
    tilt = 0.48 + 0.10 * math.sin(t * 0.23 + 2.0)
    bob = w * 0.012 * math.sin(t * 0.53 + 1.0)
    z = math.sin(theta)
    x = cx + rx * math.cos(theta)
    y = cy + w * 0.012 + bob + rx * tilt * z
    return x, y, z

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
                       "stretch", "nuzzle", "hearteyes",     # G personals
                       "dizzy", "frazzle", "disco")          # G crazy set
# Per-clip one-shot playback speed (frames advanced per 20 Hz tick; default 1.0).
# The 120-frame out-and-back gestures play at 1.6 (~3.75 s); spin stays a
# quick flourish.
_ONESHOT_SPEED = {"spin": 2.4, "angry": 1.6, "droop": 1.6, "drift": 1.6,
                  "eyeroll": 1.6, "focus": 1.6, "gleam": 1.6,
                  "doubletake": 1.6, "stretch": 1.6, "nuzzle": 1.6,
                  "hearteyes": 1.6, "super_saiyan": 1.6, "dizzy": 1.6,
                  "frazzle": 1.6, "disco": 1.6}

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

    def _live_style(c):
        """The model dot/orb's colour+size right now: the model's own colour,
        or an unlit grey breathing pulse while the STT models are still
        loading — the visible answer to "is it broken, or just loading?".
        set_warming(False) pops the orb (the model-switch flare) as "ready"."""
        color, radius = _model_style(c.model)
        if getattr(c, "warming", False):
            color = (0.62, 0.64, 0.70)
            radius *= 0.70 + 0.30 * math.sin(c._anim * 3.0)
        return color, radius

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
                # The firefly orbits in 3D: its behind-the-head pass draws
                # first (so the opaque head occludes it), the head next, then
                # the in-front pass on top.
                if c.clips or c.center is not None:
                    self._draw_orb_layer(c, w, h, behind=True)
                    self._draw_marvin_image(c, w, h)  # video flipbook / still
                    self._draw_orb_layer(c, w, h, behind=False)
                    return
                # fallback (no assets): 2D fake tilt/nod on the vector face
                self._draw_orb_layer(c, w, h, behind=True)
                NSGraphicsContext.saveGraphicsState()
                t = NSAffineTransform.transform()
                t.translateXBy_yBy_(0.0, c.nod)
                t.translateXBy_yBy_(w / 2.0, h / 2.0)
                t.rotateByDegrees_(c.tilt)
                t.translateXBy_yBy_(-w / 2.0, -h / 2.0)
                t.concat()
                self._draw_marvin_vector(c, w, h)
                NSGraphicsContext.restoreGraphicsState()
                self._draw_orb_layer(c, w, h, behind=False)
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
                color, radius = _live_style(c)
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
        def _orb_dot(self, x, y, r, rgb, a):
            """A magical firefly point: layered soft halos around a hot,
            whitened core, all scaled by alpha *a* so depth/trail fading
            keeps the same look."""
            cr, cg, cb = rgb
            for scale, alpha in ((3.4, 0.08), (1.9, 0.20), (1.0, 0.90)):
                _rgb(cr, cg, cb, min(1.0, alpha * a)).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(x - r * scale, y - r * scale,
                               2 * r * scale, 2 * r * scale)
                ).fill()
            hot = (cr + (1 - cr) * 0.75, cg + (1 - cg) * 0.75,
                   cb + (1 - cb) * 0.75)
            hr = r * 0.45
            _rgb(*hot, min(1.0, 0.9 * a)).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(x - hr, y - hr, 2 * hr, 2 * hr)
            ).fill()

        @objc.python_method
        def _draw_orb_layer(self, c, w, h, behind):
            """One depth layer of the model firefly (path: _orb_orbit, state:
            _tick_orb). The behind pass runs before the head is drawn so the
            head occludes it; the front pass runs after, passing over his
            face, and also casts the orb's light onto the head and renders
            the post-swallow digest glow."""
            if c._turn_f > 0.5:
                return  # the oracle owns the stage while he's turned around
            color, radius = _live_style(c)
            ev = c._orb_event
            if ev is not None and ev[0] == "digest":
                if not behind:
                    self._digest_glow(c, w, h, color, ev[1])
                return
            pos = c._orb_pos
            if pos is None:
                return
            x, y, z, s = pos
            # comet trail: oldest faintest, each sample on its own depth side
            n = len(c._orb_trail)
            for i, (tx, ty, tz, ts) in enumerate(c._orb_trail):
                if (tz < 0.0) != behind:
                    continue
                fade = (i + 1) / (n + 1.0)
                tr = radius * 0.85 * ts * (1.0 + 0.28 * tz) * 0.6 * fade
                if tr > 0.3:
                    self._orb_dot(tx, ty, tr, color, 0.22 * fade)
            if (z < 0.0) == behind:
                a = 0.45 + 0.55 * (z + 1.0) / 2.0     # dim when behind him
                a *= 0.82 + 0.18 * math.sin(c._anim * 6.7)  # firefly twinkle
                r = radius * 0.85 * s * (1.0 + 0.28 * z)
                if c.mode == "recording" and c.levels:
                    r *= 1.0 + 0.25 * max(0.0, min(1.0, c.levels[-1]))
                r *= 1.0 + 0.8 * c._orb_pop
                if r > 0.3:
                    self._orb_dot(x, y, r, color, a)
                if c._orb_pop > 0.02:
                    # expanding ring acknowledging the model switch
                    cr, cg, cb = color
                    ring = r * (1.8 + (1.0 - c._orb_pop) * 3.0)
                    _rgb(cr, cg, cb, 0.5 * c._orb_pop).setStroke()
                    p = NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(x - ring, y - ring, 2 * ring, 2 * ring)
                    )
                    p.setLineWidth_(1.5)
                    p.stroke()
            if not behind:
                self._orb_light(c, w, h, color, pos)

        @objc.python_method
        def _orb_light(self, c, w, h, color, pos):
            """The orb's light falling on Marvin: a faint colour wash over
            the head ball, centred on the orb and clipped to the head, so the
            firefly reads as actually illuminating him. Subtle — ambience,
            not a spotlight."""
            x, y, z, s = pos
            if s < 0.3:
                return  # mostly swallowed: the digest glow takes over
            cr, cg, cb = color
            hr = w * 0.118                    # ~the head ball
            # behind him the wash doubles as a faint rim light, so he still
            # hints at the firefly while the head fully occludes it
            a = (0.08 + 0.10 * (z + 1.0) / 2.0) * s
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(w / 2.0 - hr, h / 2.0 - hr, 2 * hr, 2 * hr)
            ).addClip()
            rad = w * 0.16
            grad = NSGradient.alloc().initWithColors_([
                _rgb(cr, cg, cb, a), _rgb(cr, cg, cb, 0.0)])
            path = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(x - rad, y - rad, 2 * rad, 2 * rad))
            grad.drawInBezierPath_relativeCenterPosition_(
                path, NSMakePoint(0.0, 0.0))
            NSGraphicsContext.restoreGraphicsState()

        @objc.python_method
        def _digest_glow(self, c, w, h, color, k):
            """He swallowed the firefly: the whole head pulses softly in the
            model colour for a moment before the orb pops back out."""
            cr, cg, cb = color
            pulse = 0.5 - 0.5 * math.cos(k * (2.0 * math.pi / 12.0))
            a = 0.14 + 0.20 * pulse
            hr = w * 0.118
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(w / 2.0 - hr, h / 2.0 - hr, 2 * hr, 2 * hr)
            ).addClip()
            rad = w * 0.14
            grad = NSGradient.alloc().initWithColors_([
                _rgb(cr, cg, cb, a), _rgb(cr, cg, cb, 0.0)])
            path = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(w / 2.0 - rad, h / 2.0 - rad, 2 * rad, 2 * rad))
            grad.drawInBezierPath_relativeCenterPosition_(
                path, NSMakePoint(0.0, 0.0))
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
                 "stretch", "nuzzle", "hearteyes", "dizzy", "frazzle",
                 "disco", "super_saiyan", "spin"]
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
            self.warming = False  # STT models still loading (grey orb pulse)
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
            # Model "firefly" switch feedback: _orb_pop = 1->0 flare of the
            # orb, set on a model switch and decayed in tick().
            self._orb_pop = 0.0
            # Firefly orbit state (see _tick_orb / _orb_orbit): angle on the
            # 3D path, latest (x, y, z, scale) sample, a short comet trail,
            # and the rare idle-only events — Marvin noticing it, or
            # swallowing it (phases "swallow" -> "digest" -> pop back out).
            self._orb_theta = random.uniform(0.0, 2.0 * math.pi)
            self._orb_pos = None
            self._orb_trail = []
            self._orb_event = None
            self._orb_notice_t = random.randint(900, 2400)    # ~45-120 s idle
            self._orb_swallow_t = random.randint(3600, 7200)  # ~3-6 min idle
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
            changed = model != self.model
            self.model = model
            if changed and self.style == "marvin":
                self._orb_pop = 1.0      # firefly flare acknowledging the switch
            self._render()

        def set_warming(self, on):
            """Grey out the model orb while the STT models load (see
            _live_style); flare it when they're ready. Main thread only."""
            on = bool(on)
            if on == getattr(self, "warming", False):
                return
            self.warming = on
            if not on and self.style == "marvin":
                self._orb_pop = 1.0      # lights on: same flare as a switch
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
            # Decay the model-switch flare (~0.6 s). Runs before the oracle
            # early-return so a switch can't leave it stuck.
            if self._orb_pop > 0.0:
                self._orb_pop = max(0.0, self._orb_pop - 0.08)
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
            self._tick_orb(mode)
            self._render()

        def _tick_orb(self, mode):
            """Advance the model firefly one 20 Hz tick: the organically
            wobbling 3D orbit, the comet trail, and the rare idle-only events
            (Marvin noticing it, or swallowing it). Skipped while he's turned
            to the oracle (tick() returns before reaching this), so an event
            can't run unseen behind his back."""
            if self.style != "marvin":
                return
            t = self._anim
            ev = self._orb_event
            if ev is None:
                # angular speed breathes on incommensurate sines: the firefly
                # hurries, loiters, hurries — never a metronome. Always > 0.
                sp = 0.032 * (1.0 + 0.45 * math.sin(t * 0.53 + 1.7)
                              + 0.25 * math.sin(t * 1.31))
                self._orb_theta = (self._orb_theta + sp) % (2.0 * math.pi)
                x, y, z = _orb_orbit(t, self._orb_theta, self._w)
                self._orb_pos = (x, y, z, 1.0)
                if mode == "idle" and self._oneshot is None:
                    self._orb_notice_t -= 1
                    self._orb_swallow_t -= 1
                    # both events wait for the orb to be out front where the
                    # story reads: he reacts to something he can see.
                    if self._orb_swallow_t <= 0 and z > 0.45:
                        self._orb_event = ["swallow", 0, x, y]
                        self._orb_swallow_t = random.randint(3600, 7200)
                    elif self._orb_notice_t <= 0 and z > 0.3:
                        self._orb_notice()
                        self._orb_notice_t = random.randint(900, 2400)
            elif ev[0] == "swallow":
                # spiral from the orbit into his mouth over ~1.5 s
                ev[1] += 1
                p = min(1.0, ev[1] / 30.0)
                e = p * p                      # ease-in: the pull accelerates
                mx = self._w / 2.0
                my = self._w / 2.0 + self._w * 0.062   # his mouth area
                ang = self._orb_theta + p * 7.0        # tightening spiral
                wob = (1.0 - p) * self._w * 0.022
                self._orb_pos = (
                    ev[2] + (mx - ev[2]) * e + math.cos(ang) * wob,
                    ev[3] + (my - ev[3]) * e + math.sin(ang) * wob * 0.6,
                    1.0, 1.0 - p,
                )
                if p >= 1.0:
                    self._orb_event = ["digest", 0]
                    self._orb_pos = None
                    self._orb_trail = []
            elif ev[0] == "digest":
                # ~1.8 s of head pulsing in the orb's colour (_digest_glow),
                # then it pops back out onto its orbit with a flare.
                ev[1] += 1
                if ev[1] >= 36:
                    self._orb_event = None
                    self._orb_pop = 1.0
            if self._orb_pos is not None:
                self._orb_trail.append(self._orb_pos)
                if len(self._orb_trail) > 8:
                    self._orb_trail.pop(0)

        def _orb_notice(self):
            """Marvin clocks his firefly as it crosses in front of him: play
            the most watchful gesture this skin has (deadpan eyeroll for B,
            doubletake for A, ...), falling back to any idle gesture."""
            for name in ("doubletake", "glance", "skeptic", "curious",
                         "focus", "eyeroll"):
                if name in self.clips:
                    self.play_oneshot(name)
                    return
            if self._idle_gestures:
                self.play_oneshot(random.choice(self._idle_gestures))

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
                     "stretch", "nuzzle", "hearteyes", "dizzy", "frazzle", "disco",
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
