"""A small floating speech bubble for Marvin's answers.

A borderless, non-activating panel that appears near the pill, shows Marvin's
answer as wrapped text, and auto-dismisses after a spell. It deliberately never
takes focus (you're working in another app while you ask him something) and
ignores mouse events, so it can float over anything without getting in the way.

All AppKit objects are created/mutated on the main thread only — the menu-bar
app drives show()/tick()/hide() from its main-thread pill timer. If PyObjC UI
isn't available, create_bubble() returns None and the app falls back to a
notification / stdout.
"""

import time

try:
    from AppKit import (
        NSBackingStoreBuffered,
        NSColor,
        NSFont,
        NSLineBreakByWordWrapping,
        NSMakeRect,
        NSScreen,
        NSTextField,
        NSVisualEffectBlendingModeBehindWindow,
        NSVisualEffectStateActive,
        NSVisualEffectView,
        NSWindow,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
    )

    NSWindowStyleMaskBorderless = 0
    NSStatusWindowLevel = 25
    NSVisualEffectMaterialHUDWindow = 13

    _MAXW = 320.0   # bubble width; text wraps within
    _PAD = 14.0     # inner padding around the text
    _FONT = 13.0
    _GAP = 40.0     # vertical gap between Marvin's head centre and the bubble

    def _screen_for_point(x, y):
        for s in NSScreen.screens():
            f = s.frame()
            if (f.origin.x <= x <= f.origin.x + f.size.width
                    and f.origin.y <= y <= f.origin.y + f.size.height):
                return s
        return NSScreen.mainScreen()

    class _Bubble:
        def __init__(self):
            self._deadline = 0.0
            self._visible = False

            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, _MAXW, 60),
                NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered,
                False,
            )
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(NSColor.clearColor())
            self.window.setLevel_(NSStatusWindowLevel)
            self.window.setHasShadow_(True)
            self.window.setIgnoresMouseEvents_(True)  # never steal clicks/focus
            self.window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
            )

            fx = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, 0, _MAXW, 60)
            )
            fx.setMaterial_(NSVisualEffectMaterialHUDWindow)
            fx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
            fx.setState_(NSVisualEffectStateActive)
            fx.setWantsLayer_(True)
            fx.layer().setCornerRadius_(14.0)
            fx.layer().setMasksToBounds_(True)
            self.window.setContentView_(fx)

            label = NSTextField.alloc().initWithFrame_(
                NSMakeRect(_PAD, _PAD, _MAXW - 2 * _PAD, 60 - 2 * _PAD)
            )
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setBordered_(False)
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setFont_(NSFont.systemFontOfSize_(_FONT))
            label.setTextColor_(NSColor.whiteColor())
            label.cell().setWraps_(True)
            label.cell().setLineBreakMode_(NSLineBreakByWordWrapping)
            fx.addSubview_(label)
            self._label = label

        def _fit_height(self, text):
            self._label.setStringValue_(text)
            w = _MAXW - 2 * _PAD
            size = self._label.cell().cellSizeForBounds_(
                NSMakeRect(0, 0, w, 1.0e6)
            )
            return float(size.height)

        def show(self, text, anchor, duration=None):
            """Show *text* near the pill. `anchor` = the pill window frame as
            (x, y, w, h) in screen coords (bottom-left origin)."""
            text = (text or "").strip()
            if not text:
                return
            th = self._fit_height(text)
            h = th + 2 * _PAD
            w = _MAXW

            ax, ay, aw, ah = anchor
            head_cx = ax + aw / 2.0
            head_cy = ay + ah / 2.0
            vf = _screen_for_point(head_cx, head_cy).visibleFrame()

            x = head_cx - w / 2.0
            # Prefer floating above Marvin's head; drop below if there's no room.
            above_y = head_cy + _GAP
            if above_y + h <= vf.origin.y + vf.size.height - 8:
                y = above_y
            else:
                y = head_cy - _GAP - h
            # Keep the whole bubble on screen.
            x = max(vf.origin.x + 8,
                    min(x, vf.origin.x + vf.size.width - w - 8))
            y = max(vf.origin.y + 8,
                    min(y, vf.origin.y + vf.size.height - h - 8))

            self.window.setFrame_display_(NSMakeRect(x, y, w, h), True)
            self._label.setFrame_(NSMakeRect(_PAD, _PAD, w - 2 * _PAD, h - 2 * _PAD))
            self.window.orderFrontRegardless()
            self._visible = True

            if duration is None:
                words = max(1, len(text.split()))
                duration = max(4.0, min(20.0, 1.5 + words * 0.4))
            self._deadline = time.monotonic() + duration

        def tick(self):
            """Main-thread heartbeat: auto-dismiss once the deadline passes."""
            if self._visible and time.monotonic() >= self._deadline:
                self.hide()

        def hide(self):
            if self._visible:
                self.window.orderOut_(None)
                self._visible = False

    def create_bubble():
        """Build a hidden speech bubble. Returns a controller, or None on failure."""
        try:
            return _Bubble()
        except Exception as exc:  # pragma: no cover - UI environment dependent
            print(f"[bubble] disabled ({exc})", flush=True)
            return None

except Exception as _bubble_import_err:  # pragma: no cover - AppKit unavailable
    print(f"[bubble] UI unavailable: {_bubble_import_err!r}", flush=True)

    def create_bubble():
        return None
