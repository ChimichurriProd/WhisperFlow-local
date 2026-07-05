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


try:
    from AppKit import (
        NSAnimationContext,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSEvent,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSGradient,
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

        def mouseDown_(self, event):
            self._down = NSEvent.mouseLocation()
            self._win0 = self.window().frame().origin
            self._dragged = False

        def mouseDragged_(self, event):
            if self._down is None:
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
        def __init__(self, on_click=None, on_move=None, pos=None):
            self.on_click = on_click
            self.on_move = on_move
            self.mode = "idle"
            self.model = "small"
            self.hover = False
            self.levels = [0.0] * _BARS
            self._phase = 0.0
            self._w = _WIDTH_IDLE

            if pos and len(pos) == 2 and pos[0] is not None:
                x, y = float(pos[0]), float(pos[1])
            else:
                screen = _screen_with_mouse().frame()
                x = screen.origin.x + (screen.size.width - _WIDTH_IDLE) / 2.0
                y = screen.origin.y + screen.size.height - _HEIGHT - 64.0

            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, self._w, _HEIGHT),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setLevel_(NSStatusWindowLevel)
            self.window.setHasShadow_(True)
            self.window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
            )

            # frosted-glass content, rounded to a pill
            fx = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, 0, self._w, _HEIGHT)
            )
            fx.setMaterial_(NSVisualEffectMaterialHUDWindow)
            fx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
            fx.setState_(NSVisualEffectStateActive)
            fx.setWantsLayer_(True)
            fx.layer().setCornerRadius_(_HEIGHT / 2.0)
            fx.layer().setMasksToBounds_(True)
            self.window.setContentView_(fx)

            self.view = _WaveView.alloc().initWithController_(self)
            self.view.setFrame_(NSMakeRect(0, 0, self._w, _HEIGHT))
            self.view.setAutoresizingMask_(1 << 1 | 1 << 4)  # width | height
            fx.addSubview_(self.view)
            self.window.orderFrontRegardless()

        def _resize(self, width, animate=True):
            if abs(width - self._w) < 0.5:
                return
            self._w = width
            f = self.window.frame()
            new = NSMakeRect(f.origin.x, f.origin.y, width, _HEIGHT)  # left-anchored
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
            self._render()

    def create_pill(on_click=None, on_move=None, pos=None):
        """Build and show the pill. Returns a controller, or None on failure."""
        try:
            return _Pill(on_click=on_click, on_move=on_move, pos=pos)
        except Exception as exc:  # pragma: no cover - UI environment dependent
            print(f"[pill] disabled ({exc})", flush=True)
            return None

except Exception as _pill_import_err:  # pragma: no cover - AppKit unavailable
    import traceback as _tb
    print(f"[pill] UI unavailable: {_pill_import_err!r}", flush=True)
    _tb.print_exc()

    def create_pill(on_click=None, on_move=None, pos=None):
        return None
