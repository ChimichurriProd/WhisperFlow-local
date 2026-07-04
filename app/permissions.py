"""macOS permission checks: Accessibility trust for global hotkeys + injection.

The hotkey listener (CGEvent tap) and text injection (CGEventPost) both require
the running process to be a trusted Accessibility client. This module reports
that state and can trigger the system prompt that adds the process to the
System Settings -> Privacy & Security -> Accessibility list, so the user just
has to flip the toggle.
"""

import sys


def is_trusted():
    """True if this process is a trusted Accessibility client."""
    if sys.platform != "darwin":
        return True
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:
        return True  # can't tell -> don't block startup


def prompt_for_trust():
    """Ask macOS to show the 'grant Accessibility' prompt for this process.

    Returns the current trust state. When untrusted, macOS registers the
    process in the Accessibility list (so it appears with a toggle) and shows
    a dialog directing the user to System Settings.
    """
    if sys.platform != "darwin":
        return True
    try:
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )

        return bool(
            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        )
    except Exception:
        return is_trusted()
