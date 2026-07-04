"""Push-to-talk main loop (macOS): hold the hotkey to record, release to transcribe+inject.

Uses a pynput global listener. The combo (e.g. "control + shift + space") is
tracked manually: recording starts when every key in the combo is down and
stops when any of them is released. Transcription runs on a worker thread so
the listener callback never blocks.

Requires Input Monitoring (listener) and Accessibility (keystroke synthesis)
permissions for your terminal: System Settings -> Privacy & Security.
"""

import sys
import threading
import time

from .audio import Recorder
from .cleanup import clean_transcript
from .injection import inject_text
from .stt import Transcriber


def parse_hotkey(spec):
    """Parse "control + shift + space" into a frozenset of canonical pynput keys."""
    from pynput.keyboard import Key, KeyCode

    aliases = {
        "control": Key.ctrl, "ctrl": Key.ctrl,
        "shift": Key.shift,
        "alt": Key.alt, "option": Key.alt,
        "cmd": Key.cmd, "command": Key.cmd, "win": Key.cmd,
        "space": Key.space, "tab": Key.tab, "enter": Key.enter,
        "esc": Key.esc, "escape": Key.esc,
    }
    keys = set()
    for part in spec.split("+"):
        name = part.strip().lower()
        if not name:
            continue
        if name in aliases:
            keys.add(aliases[name])
        elif name.startswith("f") and name[1:].isdigit():
            keys.add(getattr(Key, name))
        elif len(name) == 1:
            keys.add(KeyCode.from_char(name))
        else:
            raise ValueError(f"Unknown key in hotkey spec: {name!r}")
    if not keys:
        raise ValueError(f"Empty hotkey spec: {spec!r}")
    return frozenset(keys)


def canonicalize(key):
    """Fold left/right modifier variants into their generic key."""
    from pynput.keyboard import Key, KeyCode

    folds = {
        Key.ctrl_l: Key.ctrl, Key.ctrl_r: Key.ctrl,
        Key.shift_l: Key.shift, Key.shift_r: Key.shift,
        Key.alt_l: Key.alt, Key.alt_r: Key.alt, Key.alt_gr: Key.alt,
        Key.cmd_l: Key.cmd, Key.cmd_r: Key.cmd,
    }
    if key in folds:
        return folds[key]
    if isinstance(key, KeyCode) and key.char is not None:
        return KeyCode.from_char(key.char.lower())
    return key


def split_combo(required):
    """Split a hotkey set into (modifier keys, trigger keycode or None).

    The trigger's macOS virtual keycode lets us suppress exactly that key at
    the event-tap level so the focused app never sees it (no stray characters
    typed while dictating). Returns vk None when it can't be resolved — the
    hotkey still works then, just without suppression.
    """
    from pynput.keyboard import Key

    modifier_keys = {Key.ctrl, Key.shift, Key.alt, Key.cmd}
    modifiers = frozenset(k for k in required if k in modifier_keys)
    triggers = [k for k in required if k not in modifier_keys]
    if len(triggers) != 1:
        return modifiers, None
    trigger = triggers[0]
    vk = trigger.value.vk if isinstance(trigger, Key) else trigger.vk
    return modifiers, vk


class PushToTalkApp:
    def __init__(self, config, on_status=None):
        self.config = config
        self.recorder = Recorder(
            sample_rate=config["audio"]["sample_rate"],
            channels=config["audio"]["channels"],
            mic_gain=config["audio"].get("mic_gain", 4.5),
        )
        self.transcriber = Transcriber(**config["stt"])
        self._busy = threading.Lock()
        self._pressed = set()  # currently-held keys, maintained by the listener
        self._on_status = on_status  # callable(state: str), e.g. menu-bar icon

    def _status(self, state):
        if self._on_status is not None:
            self._on_status(state)

    def set_model(self, model_name):
        """Swap the STT model at runtime (loads lazily on next dictation)."""
        self.config["stt"]["model"] = model_name
        self.transcriber = Transcriber(**self.config["stt"])

    def _wait_hotkey_released(self, timeout=1.0):
        """Block until the user lets go of the hotkey keys (or timeout).

        Fast transcriptions can finish while Ctrl/Shift are still physically
        held; pasting then would synthesize Cmd+V mixed with those modifiers,
        which some apps interpret as a different shortcut.
        """
        deadline = time.monotonic() + timeout
        from pynput.keyboard import Key

        modifiers = {Key.ctrl, Key.shift, Key.alt, Key.cmd}
        while self._pressed & modifiers and time.monotonic() < deadline:
            time.sleep(0.02)

    def on_press(self):
        if self._busy.locked():
            return
        print("[rec] listening...", flush=True)
        self._status("recording")
        self.recorder.start()

    def on_release(self):
        with self._busy:
            audio = self.recorder.stop()
            self._status("transcribing")
            try:
                seconds = len(audio) / self.config["audio"]["sample_rate"]
                print(f"[rec] captured {seconds:.1f}s, transcribing...", flush=True)
                raw = self.transcriber.transcribe(audio)
                if not raw:
                    print("[stt] (nothing recognized)", flush=True)
                    return
                cleaned = clean_transcript(raw, self.config)
                print(f'[out] injecting into focused app: "{cleaned}"', flush=True)
                self._wait_hotkey_released()
                to_inject = cleaned
                if self.config["injection"].get("append_trailing_space", True):
                    to_inject += " "  # keep a gap before the next dictation
                inject_text(to_inject, self.config)
            finally:
                self._status("idle")

    def build_listener(self):
        """Create (but don't start) the global hotkey listener."""
        if sys.platform != "darwin":
            raise RuntimeError(
                "The push-to-talk loop requires macOS. "
                "On this platform, run scripts/dry_run.py instead."
            )
        import Quartz
        from pynput import keyboard

        binding = self.config["hotkey"]["push_to_talk"]
        required = parse_hotkey(binding)
        modifiers, trigger_vk = split_combo(required)
        pressed = self._pressed  # shared with _wait_hotkey_released
        active = False

        def start():
            nonlocal active
            active = True
            self.on_press()

        def stop():
            nonlocal active
            active = False
            # Off the listener thread so slow STT never delays key events.
            threading.Thread(target=self.on_release, daemon=True).start()

        def intercept(event_type, event):
            """Swallow the trigger key at the event tap so the focused app
            never receives it (otherwise held Ctrl+Shift+Space types stray
            characters like ^@ into whatever you're dictating into)."""
            if trigger_vk is None:
                return event
            if event_type not in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
                return event
            vk = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if vk != trigger_vk:
                return event
            if event_type == Quartz.kCGEventKeyDown:
                if active:
                    return None  # key-repeat while recording: just swallow
                if modifiers <= pressed:
                    start()
                    return None
                return event  # trigger key without the modifiers: normal typing
            if active:  # kCGEventKeyUp ending the dictation
                stop()
                return None
            return event

        def on_press(key):
            pressed.add(canonicalize(key))
            # Fallback activation when suppression is unavailable (multi-key
            # or unresolvable trigger): behave as a plain observer combo.
            if trigger_vk is None and not active and required <= pressed:
                start()

        def on_release(key):
            k = canonicalize(key)
            pressed.discard(k)
            # Releasing a modifier first also ends the dictation.
            if active and k in required:
                stop()

        print(f"Ready. Hold [{binding}] to dictate.", flush=True)
        print("(If nothing happens, grant Input Monitoring and Accessibility "
              "in System Settings -> Privacy & Security.)", flush=True)
        if trigger_vk is None:
            print("note: hotkey suppression unavailable for this combo; the "
                  "focused app may also see the keystrokes.", flush=True)
        return keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            darwin_intercept=intercept,
        )

    def run(self):
        """Blocking CLI mode: run the listener until Ctrl+C."""
        with self.build_listener() as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nBye.", flush=True)
