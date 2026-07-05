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

_HEIGHT = 32.0
_WIDTH_IDLE = 32.0  # equals height => a round dot when idle
_WIDTH_REC = 250.0  # hover width is measured from the label (see _target_width)
_BARS = 24
_MARVIN_SIZE = 100.0  # window (larger than the head so the eye-glow has room)
_MARVIN_INSET = 0.18  # head is ~64px inside the window; margin holds the glow


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
    from Foundation import NSMutableDictionary
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

    # eye centroids per pose (normalized, y from top), measured from the assets
    _POSE_EYES = {
        "center": [(0.226, 0.639), (0.678, 0.633)],
        "up":     [(0.243, 0.395), (0.669, 0.384)],
        "down":   [(0.267, 0.818), (0.589, 0.819)],
        "left":   [(0.334, 0.699), (0.646, 0.689)],
        "right":  [(0.128, 0.645), (0.513, 0.636)],
    }

    def _load_marvin_poses():
        """Load the named 3D head poses (center/up/down/left/right) as NSImages.

        Returns a dict {name: NSImage} for whichever exist. With 'center'
        present the pill does a real look-around by cross-fading poses.
        """
        from pathlib import Path

        d = Path(__file__).resolve().parent.parent / "assets" / "marvin"
        poses = {}
        if d.is_dir():
            for name in ("center", "up", "down", "left", "right"):
                p = d / f"{name}.png"
                if p.exists():
                    img = NSImage.alloc().initWithContentsOfFile_(str(p))
                    if img is not None:
                        poses[name] = img
        return poses

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
            if not self._dragged and self._c.on_click:
                try:
                    self._c.on_click()
                except Exception:
                    pass
            elif self._dragged and self._c.on_move:
                o = self.window().frame().origin
                try:
                    self._c.on_move(float(o.x), float(o.y))
                except Exception:
                    pass
            self._down = None

        # -------- drawing --------------------------------------------------

        def drawRect_(self, rect):
            c = self._c
            w = self.frame().size.width
            h = self.frame().size.height

            if c.style == "marvin":
                if c.poses.get("center") is not None:
                    self._draw_marvin_poses(c, w, h)  # real 3D look-around
                    return
                # fallback (no pose assets): 2D fake tilt/nod on the vector face
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
        def _draw_marvin_poses(self, c, w, h):
            """Cross-fade between 3D head poses for a real look-around, and glow
            the current pose's eyes with voice."""
            m = w * _MARVIN_INSET
            rx, ry, rw, rh = m, m, w - 2 * m, h - 2 * m
            rect = NSMakeRect(rx, ry, rw, rh)
            zero = NSMakeRect(0, 0, 0, 0)

            # HARD CUT (cartoon): show exactly one pose — center, or the glance
            # pose while a glance is active. No blending, no ghosting.
            gdir = c._glance_dir if c._glance_dir in c.poses else None
            img = c.poses[gdir] if gdir else c.poses["center"]

            # The fraction/alpha draw API renders upside-down in a flipped view,
            # so flip the context back around the head rect before drawing.
            NSGraphicsContext.saveGraphicsState()
            flip = NSAffineTransform.transform()
            flip.translateXBy_yBy_(0.0, 2 * ry + rh)
            flip.scaleXBy_yBy_(1.0, -1.0)
            flip.concat()
            img.drawInRect_fromRect_operation_fraction_(rect, zero, 2, 1.0)
            NSGraphicsContext.restoreGraphicsState()

            level = c.levels[-1] if c.levels else 0.0
            glow = level if c.mode in ("recording", "transcribing") else 0.0
            if glow > 0.04:
                eyes = _POSE_EYES.get(gdir or "center", _POSE_EYES["center"])
                # clip the bloom to a circle so it never shows a square edge
                NSGraphicsContext.saveGraphicsState()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(w * 0.02, h * 0.02, w * 0.96, h * 0.96)
                ).addClip()
                self._eye_glow(rx, ry, rw, rh, eyes, glow)
                NSGraphicsContext.restoreGraphicsState()

        @objc.python_method
        def _eye_glow(self, rx, ry, rw, rh, eyes, glow):
            """Soft, feathered green bloom over both eyes (3 stacked radial
            gradients so it fades gradually, not a hard disc)."""
            core = min(0.6, glow * 0.8)
            for nx, ny in eyes:
                gx, gy = rx + nx * rw, ry + ny * rh
                for scale, alpha in ((0.11, core), (0.20, core * 0.55),
                                     (0.34, core * 0.28)):
                    rad = rw * scale * (1.0 + glow * 0.4)
                    grad = NSGradient.alloc().initWithColors_([
                        _rgb(0.55, 1.0, 0.42, alpha),
                        _rgb(0.55, 1.0, 0.42, 0.0),
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

    class _Pill:
        def __init__(self, on_click=None, on_move=None, on_menu=None, pos=None,
                     style="waveform"):
            self.on_click = on_click
            self.on_move = on_move
            self.on_menu = on_menu
            self.style = style
            self.mode = "idle"
            self.model = "small"
            self.hover = False
            self.poses = _load_marvin_poses() if style == "marvin" else {}
            self.levels = [0.0] * _BARS
            self._phase = 0.0
            self._anim = 0.0
            self.tilt = 0.0  # head-roll degrees (fallback single-image only)
            self.nod = 0.0   # vertical nod offset px (fallback single-image only)
            # occasional glances while dictating; still (center) when idle
            self._dirs = ["down", "left", "up", "right"]
            self._dir_i = 0
            self._glance_dir = None   # None = looking straight ahead
            self._glance_t = 0.0      # progress through the current glance
            self._glance_cd = 50      # ticks until the next glance
            self._h = _MARVIN_SIZE if style == "marvin" else _HEIGHT
            self._w = _MARVIN_SIZE if style == "marvin" else _WIDTH_IDLE

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
            self.window.setHasShadow_(style != "marvin")
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
            if style == "marvin":
                fx.setHidden_(True)  # face is opaque; no glass needed behind it
            self.window.setContentView_(fx)

            self.view = _WaveView.alloc().initWithController_(self)
            self.view.setFrame_(NSMakeRect(0, 0, self._w, self._h))
            self.view.setAutoresizingMask_(1 << 1 | 1 << 4)  # width | height
            self.window.setContentView_(self.view) if style == "marvin" else fx.addSubview_(self.view)
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

        def tick(self, mode, level):
            self.mode = mode
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

            # Glances: only while dictating. Hold center, then every so often
            # flick to a direction and back. Perfectly still when idle.
            if mode in ("recording", "transcribing"):
                if self._glance_dir is None:
                    self._glance_cd -= 1
                    if self._glance_cd <= 0:
                        self._glance_dir = self._dirs[self._dir_i]
                        self._dir_i = (self._dir_i + 1) % len(self._dirs)
                        self._glance_t = 0.0
                else:
                    self._glance_t += 0.05          # ~1s per glance
                    if self._glance_t >= 1.0:
                        self._glance_dir = None
                        self._glance_cd = 55        # ~2.7s between glances
            else:
                self._glance_dir = None
                self._glance_cd = 30
            self._render()

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform"):
        """Build and show the pill. Returns a controller, or None on failure."""
        try:
            return _Pill(on_click=on_click, on_move=on_move,
                         on_menu=on_menu, pos=pos, style=style)
        except Exception as exc:  # pragma: no cover - UI environment dependent
            print(f"[pill] disabled ({exc})", flush=True)
            return None

except Exception as _pill_import_err:  # pragma: no cover - AppKit unavailable
    import traceback as _tb
    print(f"[pill] UI unavailable: {_pill_import_err!r}", flush=True)
    _tb.print_exc()

    def create_pill(on_click=None, on_move=None, on_menu=None, pos=None,
                    style="waveform"):
        return None
