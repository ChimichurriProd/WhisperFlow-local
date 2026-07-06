"""Push-to-talk main loop (macOS): hold the hotkey to record, release to transcribe+inject.

Uses a pynput global listener. The combo (e.g. "control + shift + space") is
tracked manually: recording starts when every key in the combo is down and
stops when any of them is released. Transcription runs on a worker thread so
the listener callback never blocks.

Requires Input Monitoring (listener) and Accessibility (keystroke synthesis)
permissions for the running process: System Settings -> Privacy & Security.
A watchdog force-stops recording if the release event is missed (macOS can
disable the event tap under load) or a recording runs past a hard cap.
"""

import sys
import threading
import time

from .audio import Recorder
from .cleanup import clean_transcript
from .injection import inject_text
from .sound import play_done, play_start
from .stt import Transcriber, build_initial_prompt


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
        # single right-side modifiers (held alone as an easy trigger)
        "right control": Key.ctrl_r, "right ctrl": Key.ctrl_r,
        "right shift": Key.shift_r,
        "right option": Key.alt_r, "right alt": Key.alt_r,
        "right command": Key.cmd_r, "right cmd": Key.cmd_r,
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
        self._initial_prompt = build_initial_prompt(
            config.get("vocabulary", {}).get("terms", [])
        )
        self.transcriber = Transcriber(
            **config["stt"], initial_prompt=self._initial_prompt
        )
        self._busy = threading.Lock()
        self._pressed = set()  # currently-held keys, maintained by the listener
        self._on_status = on_status  # callable(state: str), e.g. menu-bar icon
        self._active = False           # currently recording
        self._trigger_vk = None        # set by build_listener
        self._rec_start = 0.0
        self._state_lock = threading.Lock()
        self._max_seconds = 120.0      # hard cap so it can never record forever
        self._event_tap = None         # pynput's CGEventTap, captured for re-enable
        self._toggle = config["hotkey"].get("mode", "hold") == "toggle"
        # live hotkey matching params (see _set_hotkey_params)
        self._required = frozenset()
        self._modifiers = frozenset()
        self._rmod = None
        self._intercept_vk = None
        self._rmod_flag = None
        self._paused = False
        self._set_hotkey_params(config["hotkey"]["push_to_talk"])
        threading.Thread(target=self._watchdog, daemon=True).start()

    def set_paused(self, paused):
        """Pause/resume dictation without touching the listener (restarting the
        event tap crashes). While paused the hotkey simply does nothing."""
        self._paused = bool(paused)
        if paused and self._active:
            self._stop_recording("paused")

    def set_toggle_mode(self, toggle):
        """Switch between hold-to-talk and tap-to-toggle. If we're mid-recording
        when switching, stop cleanly."""
        self._toggle = bool(toggle)
        self.config["hotkey"]["mode"] = "toggle" if toggle else "hold"
        if self._active:
            self._stop_recording("mode changed")

    def reload_vocabulary(self):
        """Rebuild the STT bias prompt from the current config vocabulary."""
        self._initial_prompt = build_initial_prompt(
            self.config.get("vocabulary", {}).get("terms", [])
        )
        self._rebuild_transcriber()

    def _set_hotkey_params(self, binding):
        """Parse the binding into live matching params the listener reads each
        event, so the hotkey can change WITHOUT rebuilding the listener (which
        would restart the event tap and crash)."""
        from pynput.keyboard import Key

        required = parse_hotkey(binding)
        modifiers, trigger_vk = split_combo(required)
        rmods = {Key.cmd_r, Key.alt_r, Key.ctrl_r, Key.shift_r}
        rmod = next(iter(required)) if (
            len(required) == 1 and next(iter(required)) in rmods) else None
        self._required = required
        self._modifiers = modifiers
        self._rmod = rmod
        # intercept can suppress a real keyDown trigger; a right-modifier emits
        # FlagsChanged, so it's handled via the observer path (intercept off).
        self._intercept_vk = None if rmod is not None else trigger_vk
        # Watchdog: for a real key, poll its keycode. For a modifier,
        # CGEventSourceKeyState is unreliable, so poll the modifier FLAG instead.
        flag = {Key.cmd_r: 0x100000, Key.alt_r: 0x80000,
                Key.ctrl_r: 0x40000, Key.shift_r: 0x20000}
        if rmod is not None:
            self._trigger_vk = None
            self._rmod_flag = flag.get(rmod)
        else:
            self._trigger_vk = trigger_vk
            self._rmod_flag = None

    def set_hotkey(self, binding):
        """Change the hotkey live (no listener rebuild)."""
        self.config["hotkey"]["push_to_talk"] = binding
        self._set_hotkey_params(binding)
        print(f"Hotkey changed to [{binding}]", flush=True)

    def _status(self, state):
        if self._on_status is not None:
            self._on_status(state)

    def _start_recording(self):
        with self._state_lock:
            if self._active or self._paused:
                return
            self._active = True
            self._rec_start = time.monotonic()
        self.on_press()

    def _stop_recording(self, reason=""):
        with self._state_lock:
            if not self._active:
                return
            self._active = False
        if reason:
            print(f"[rec] stop ({reason})", flush=True)
        threading.Thread(target=self.on_release, daemon=True).start()

    def _watchdog(self):
        """Safety net, twice over:

        1. Keep the keyboard event tap alive. macOS disables the tap under load
           (e.g. during a heavy transcription) and pynput never re-enables it,
           which silently kills the hotkey until restart. We re-enable it.
        2. Force-stop a recording if the key is physically up but we missed the
           release event, or if it runs past the hard cap — using the real HID
           key state, independent of the (possibly disabled) tap.
        """
        while True:
            time.sleep(0.15)
            try:
                import Quartz

                tap = self._event_tap
                if tap is not None and not Quartz.CGEventTapIsEnabled(tap):
                    Quartz.CGEventTapEnable(tap, True)
                    print("[hotkey] event tap was disabled — re-enabled", flush=True)

                if self._active and not self._toggle:
                    # Hold mode: a physically-released key/modifier means the
                    # release event was missed — stop. (Toggle mode: the key is
                    # up on purpose, so this whole block is skipped.)
                    released = False
                    if self._rmod_flag is not None:  # modifier: check the flag
                        flags = Quartz.CGEventSourceFlagsState(
                            Quartz.kCGEventSourceStateHIDSystemState)
                        released = not (int(flags) & self._rmod_flag)
                    elif self._trigger_vk is not None:  # real key: check keycode
                        released = not Quartz.CGEventSourceKeyState(
                            Quartz.kCGEventSourceStateHIDSystemState,
                            self._trigger_vk)
                    if released:
                        self._stop_recording("watchdog: key released")
                        continue
                if self._active and time.monotonic() - self._rec_start > self._max_seconds:
                    self._stop_recording("watchdog: max duration")
            except Exception:
                pass

    def _rebuild_transcriber(self):
        self.transcriber = Transcriber(
            **self.config["stt"], initial_prompt=self._initial_prompt
        )

    def set_model(self, model_name):
        """Swap the STT model at runtime (loads lazily on next dictation)."""
        self.config["stt"]["model"] = model_name
        self._rebuild_transcriber()

    def set_language(self, language):
        """Set forced language (None = auto-detect); rebuilds the transcriber."""
        self.config["stt"]["language"] = language
        self._rebuild_transcriber()

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
            # A previous transcription is still finishing; ignore this press
            # (don't leave _active set, or the watchdog would spin on it).
            self._active = False
            return
        print("[rec] listening...", flush=True)
        self._status("recording")
        play_start(self.config)
        try:
            self.recorder.start()
        except Exception as exc:
            # Mic unavailable / permission denied: recover to idle instead of
            # letting the exception break the listener callback.
            print(f"[rec] could not start mic: {exc}", flush=True)
            self._active = False
            self._status("idle")

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
                if not cleaned:
                    print("[out] (empty after cleanup, nothing to inject)", flush=True)
                    return
                print(f'[out] injecting into focused app: "{cleaned}"', flush=True)
                self._wait_hotkey_released()
                to_inject = cleaned
                if self.config["injection"].get("append_trailing_space", True):
                    to_inject += " "  # keep a gap before the next dictation
                inject_text(to_inject, self.config)
                play_done(self.config)
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

        pressed = self._pressed  # shared with _wait_hotkey_released

        def intercept(event_type, event):
            """Swallow the trigger key at the event tap so the focused app never
            receives it. Reads the live hotkey params so the hotkey can change
            without rebuilding the listener."""
            tvk = self._intercept_vk
            if tvk is None:
                return event
            if event_type not in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
                return event
            vk = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if vk != tvk:
                return event
            if event_type == Quartz.kCGEventKeyDown:
                if self._toggle:
                    if self._active:
                        self._stop_recording()
                        return None
                    if self._modifiers <= pressed:
                        self._start_recording()
                        return None
                    return event
                if self._active:
                    return None  # key-repeat while recording: just swallow
                if self._modifiers <= pressed:
                    self._start_recording()
                    return None
                return event  # trigger key without the modifiers: normal typing
            if not self._toggle and self._active:  # keyUp ends hold-mode dictation
                self._stop_recording()
                return None
            return event

        def on_press(key):
            rmod = self._rmod
            if rmod is not None:  # single right-modifier: match the raw key
                if key == rmod:
                    if self._toggle:
                        (self._stop_recording if self._active
                         else self._start_recording)()
                    elif not self._active:
                        self._start_recording()
                return
            pressed.add(canonicalize(key))
            if self._intercept_vk is None and self._required <= pressed:
                if self._toggle:
                    (self._stop_recording if self._active
                     else self._start_recording)()
                elif not self._active:
                    self._start_recording()

        def on_release(key):
            rmod = self._rmod
            if rmod is not None:
                if key == rmod and not self._toggle and self._active:
                    self._stop_recording()
                return
            k = canonicalize(key)
            pressed.discard(k)
            if not self._toggle and self._active and k in self._required:
                self._stop_recording()

        print(f"Ready. Hold [{self.config['hotkey']['push_to_talk']}] "
              "to dictate.", flush=True)
        print("(If nothing happens, grant Input Monitoring and Accessibility "
              "in System Settings -> Privacy & Security.)", flush=True)
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            darwin_intercept=intercept,
        )
        # Capture the CGEventTap pynput creates so the watchdog can re-enable it
        # if macOS disables it under load (pynput itself never does).
        orig_create = listener._create_event_tap

        def _capture_tap():
            tap = orig_create()
            self._event_tap = tap
            return tap

        listener._create_event_tap = _capture_tap
        return listener

    def run(self):
        """Blocking CLI mode: run the listener until Ctrl+C."""
        with self.build_listener() as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nBye.", flush=True)
