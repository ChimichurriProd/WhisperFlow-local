"""Inject text at the cursor of the focused app (macOS).

Two delivery methods, selected by config["injection"]["delivery_method"]:

  "clipboard"  — put text on the clipboard (pbcopy), synthesize Cmd+V via a
                 Quartz CGEvent, then optionally restore the previous clipboard.
  "type"       — type the text directly with synthesized per-character
                 unicode key events (works in fields that block paste).

Key events are posted with raw Quartz CGEvents (thread-safe); pynput's
Controller is deliberately NOT used here — see _post_key_event.

Both require the running process (your terminal / Python) to have
Accessibility permission: System Settings -> Privacy & Security -> Accessibility.
"""

import subprocess
import sys
import time


def _require_macos():
    if sys.platform != "darwin":
        raise RuntimeError(
            "Text injection requires macOS (pbcopy/Quartz). "
            "Use scripts/dry_run.py on other platforms."
        )


# ---------------------------------------------------------------- clipboard

def _get_clipboard_text():
    result = subprocess.run(["pbpaste"], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def _set_clipboard_text(text):
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


_KVK_ANSI_V = 9  # macOS virtual keycode for the V key


def _post_key_event(vk, down, flags=0, unicode_char=None):
    """Post one keyboard CGEvent. Pure Quartz — no pynput Controller.

    pynput's Controller queries the keyboard layout via Text Services Manager
    (TSMGetInputSourceProperty), which macOS 26 asserts must happen on the
    main thread; calling it from our transcription worker thread crashes the
    process (dispatch_assert_queue SIGTRAP) once the AppKit menu-bar loop is
    running. Raw CGEvents with fixed keycodes never touch TSM and are
    thread-safe.
    """
    import Quartz

    event = Quartz.CGEventCreateKeyboardEvent(None, vk, down)
    if unicode_char is not None:
        Quartz.CGEventKeyboardSetUnicodeString(
            event, len(unicode_char), unicode_char
        )
    Quartz.CGEventSetFlags(event, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def _send_cmd_v():
    import Quartz

    cmd = Quartz.kCGEventFlagMaskCommand
    _post_key_event(_KVK_ANSI_V, True, flags=cmd)
    _post_key_event(_KVK_ANSI_V, False, flags=cmd)


# ------------------------------------------------------------------ public

def inject_clipboard(text, restore_clipboard=True):
    """Clipboard + Cmd+V delivery."""
    _require_macos()
    previous = _get_clipboard_text() if restore_clipboard else None
    _set_clipboard_text(text)
    time.sleep(0.05)  # let the focused app observe the clipboard change
    _send_cmd_v()
    if restore_clipboard and previous is not None:
        time.sleep(0.15)  # let the paste land before swapping the clipboard back
        _set_clipboard_text(previous)


def inject_type(text, char_delay_ms=5):
    """Direct typing via synthesized per-character unicode key events."""
    _require_macos()
    delay = max(char_delay_ms, 0) / 1000.0
    for ch in text:
        # Unicode-string events type any character without a layout lookup.
        _post_key_event(0, True, unicode_char=ch)
        _post_key_event(0, False, unicode_char=ch)
        if delay:
            time.sleep(delay)


def inject_text(text, config):
    """Dispatch to the configured delivery method."""
    if not text:
        return
    cfg = config["injection"]
    method = cfg["delivery_method"]
    if method == "clipboard":
        inject_clipboard(text, restore_clipboard=cfg["restore_clipboard"])
    elif method == "type":
        inject_type(text, char_delay_ms=cfg["type_char_delay_ms"])
    else:
        raise ValueError(f"Unknown delivery_method: {method!r}")
