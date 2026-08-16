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

import queue
import sys
import threading
import time

from .audio import Recorder
from .cleanup import clean_transcript, collapse_repeats
from .injection import inject_text
from .sound import play_done, play_start
from .stt import Transcriber, build_initial_prompt

# What Marvin "says" when the ask pipeline can't produce an answer — in
# character, so a failure still feels like Marvin rather than an error dialog.
# Covers both a down server and an Ollama error body (e.g. model not pulled);
# the real cause is in the log, so the spoken line stays deliberately vague.
_ASK_UNREACHABLE = (
    "Something went wrong reaching my brain. Is Ollama running, with the "
    "model pulled? Predictably grim.")
_ASK_EMPTY = "I have nothing to say. For once, that's honest rather than rude."


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


# US / most-Latin-layout ANSI virtual keycodes for letters + digits. pynput's
# KeyCode.from_char(c).vk is None on macOS (it does no layout lookup), so
# split_combo can't suppress a letter trigger. These fill that gap: the letter
# *positions* (and thus keycodes) are identical on Swedish and most Latin
# layouts, so Ctrl+Shift+A resolves correctly here too.
_ANSI_VK = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26,
    "8": 28, "9": 25, "0": 29,
}


def resolve_trigger_vk(required):
    """Like split_combo, but with an ANSI fallback so a single letter/digit
    trigger still gets a virtual keycode (needed to suppress it at the event
    tap). Returns (modifiers, vk-or-None)."""
    from pynput.keyboard import KeyCode

    modifiers, vk = split_combo(required)
    if vk is None:
        chars = [k for k in required if isinstance(k, KeyCode) and k.char]
        if len(chars) == 1:
            vk = _ANSI_VK.get(chars[0].char.lower())
    return modifiers, vk


class PushToTalkApp:
    def __init__(self, config, on_status=None, on_answer=None, on_ask_start=None):
        self.config = config
        # on_answer(question, answer) — the UI presents Marvin's reply (speech
        # bubble + optional TTS). None -> just print it.
        self._on_answer = on_answer
        # on_ask_start() — fired the moment an ask recording begins, so the UI
        # can react early (Marvin turns to his 'oracle' back face for the whole
        # question, not just once the answer lands).
        self._on_ask_start = on_ask_start
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
        # One pipeline worker drains finished recordings FIFO (STT -> cleanup
        # or answer -> inject). Recording is DECOUPLED from it: the mic is free
        # the moment _stop_recording snapshots the audio, so a new dictation
        # can start while the previous one is still transcribing. (The old
        # design held a busy lock through the whole pipeline and silently
        # swallowed the next press — you spoke into a mic that wasn't
        # recording.)
        self._jobs = queue.Queue()
        self._pipeline_busy = False
        # Serializes recorder.start()/stop() across the event-tap, listener,
        # watchdog and wake threads, so a lightning-fast tap can't interleave
        # them (a start() after its own stop() would capture forever).
        self._mic_lock = threading.Lock()
        self._pressed = set()  # currently-held keys, maintained by the listener
        self._on_status = on_status  # callable(state: str), e.g. menu-bar icon
        self._active = False           # currently recording
        self._active_kind = "dictate"  # "dictate" | "ask" — which pipeline this rec feeds
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
        # The trigger of the CURRENTLY-active recording, so the watchdog polls the
        # right key whether it was started by the dictate or the ask hotkey.
        self._active_trigger_vk = None
        self._active_rmod_flag = None
        # Second hotkey: "ask Marvin". Same recorder + STT, different last stage
        # (LLM answer + speak) — see _process. Always a modifier+trigger combo
        # (no right-modifier support), so one keycode covers intercept + watchdog.
        self._ask_required = frozenset()
        self._ask_modifiers = frozenset()
        self._ask_vk = None
        self._ask_on = config.get("ask", {}).get("enabled", True)
        self._paused = False
        # Hands-free: this recording was opened by the wake word, so no key will
        # ever be released to end it (see _wake_endpoint).
        self._hands_free = False
        self.wake = None  # WakeWord detector, built by start_wakeword()
        # Barge-in state: who is speaking, so the wake word can cut him off.
        self._speaking = False
        self._speaker = None
        # Rolling conversation so follow-ups make sense ("and Denmark?").
        # Cleared after conversation_idle_seconds — an hour-old thread is not
        # context, it's confusion.
        self._turns = []
        self._last_turn = 0.0
        self._set_hotkey_params(config["hotkey"]["push_to_talk"])
        self._set_ask_hotkey_params(
            config.get("hotkey", {}).get("ask", "control + shift + a")
        )
        # File-transcription progress (0..1 while a drop job runs, else None)
        # — the pill timer reads it and draws the ring around Marvin.
        self.file_progress = None
        # One file job at a time: a second drop while one runs QUEUES behind
        # it instead of interleaving on the MLX pool (two parallel jobs fight
        # over file_progress and the ring jumps around; observed live when a
        # 5th file was dropped mid-batch).
        self._file_job_lock = threading.Lock()
        # Prewarm state: True while the lazy models are still loading in the
        # background. The UI reads it for the "loading, not crashed" cue.
        self.warming = False
        self._warm_thread = None
        threading.Thread(target=self._watchdog, daemon=True).start()
        threading.Thread(target=self._pipeline, daemon=True,
                         name="pipeline").start()

    def prewarm(self):
        """Load every lazy model in the background so the FIRST dictation and
        the FIRST answer are as fast as every later one: both Whisper models
        (through the same single MLX worker real dictations use, so a dictation
        started mid-warm simply queues behind the load instead of failing) and
        the Ollama answer model. While this runs, self.warming is True and the
        pill greys out Marvin's model orb — the visible difference between
        "still loading" and "crashed"."""
        if self._warm_thread is not None and self._warm_thread.is_alive():
            return self._warm_thread

        def _run():
            self.warming = True
            t0 = time.monotonic()
            try:
                try:
                    self.transcriber.warm()
                except Exception as exc:
                    print(f"[warm] stt warm failed: {exc!r}", flush=True)
                try:
                    from . import ollama

                    cfg = self.config.get("cleanup", {})
                    ollama.warm(
                        cfg.get("ollama_url", "http://localhost:11434"),
                        cfg.get("ollama_model", "gemma4:12b"),
                        keep_alive=cfg.get("keep_alive", "30m"),
                    )
                except Exception:
                    pass  # ollama down: the ask path already degrades politely
            finally:
                self.warming = False
                print(f"[warm] models ready ({time.monotonic() - t0:.1f}s)",
                      flush=True)

        self._warm_thread = threading.Thread(
            target=_run, daemon=True, name="prewarm"
        )
        self._warm_thread.start()
        return self._warm_thread

    def transcribe_files_async(self, paths, on_done=None):
        """Transcribe audio FILES (dropped on Marvin / picked from the menu)
        on a worker thread. Chunks share the single MLX worker with dictation,
        so dictating mid-job waits a couple of seconds, never the whole file.
        on_done(summary_dict_or_None) fires from the worker thread."""
        from .transcribe_file import transcribe_files

        def _run():
            with self._file_job_lock:  # a second drop queues, FIFO
                try:
                    res = transcribe_files(
                        paths, self.transcriber, self.config,
                        on_progress=lambda m: print(f"[file] {m}", flush=True),
                        # Drives the progress ring around Marvin (read by the
                        # pill timer on the main thread).
                        on_fraction=lambda f: setattr(self, "file_progress", f),
                    )
                    print(f"[file] klart: {res['out_path']}", flush=True)
                except Exception as exc:
                    print(f"[file] transcription failed: {exc!r}", flush=True)
                    res = None
                finally:
                    self.file_progress = None  # hide the ring, success or not
            if on_done is not None:
                try:
                    on_done(res)
                except Exception as exc:
                    print(f"[file] on_done failed: {exc!r}", flush=True)

        threading.Thread(target=_run, daemon=True,
                         name="file-transcribe").start()

    # ------------------------------------------------------------- wake word

    def start_wakeword(self):
        """Build and start the "Hey Marvin" detector if it's switched on. Safe
        to call repeatedly; returns the detector or None."""
        self.stop_wakeword()
        cfg = self.config.get("wakeword", {})
        if not cfg.get("enabled", False):
            return None
        from .wakeword import WakeWord, default_model_path

        path = cfg.get("model") or default_model_path()
        if not path:
            print("[wake] no wake-word model installed — see "
                  "scripts/train_wakeword.sh", flush=True)
            return None
        self.wake = WakeWord(
            path, on_detect=self._on_wake,
            threshold=cfg.get("threshold", 0.5),
            debounce=cfg.get("debounce", 2.0),
            sample_rate=self.config["audio"]["sample_rate"],
            debug=cfg.get("debug", False),
            dump_seconds=cfg.get("dump_seconds", 0),
        )
        if "barge_threshold" in cfg:
            self.wake.barge_threshold = float(cfg["barge_threshold"])
        # He may already be mid-quip when the wake word (re)starts.
        self.wake.set_barge_mode(self._speaking)
        # Share the mic stream the recorder already holds open rather than
        # opening a second one (audio.py explains why that matters).
        self.wake.overflow_source = lambda: self.recorder.overflows
        self.recorder.on_frame = self.wake.feed
        self.recorder._ensure_stream()
        self.wake.start()
        return self.wake

    def stop_wakeword(self):
        self.recorder.on_frame = None
        if self.wake is not None:
            self.wake.stop()
            self.wake = None

    def set_wakeword_enabled(self, enabled):
        self.config.setdefault("wakeword", {})["enabled"] = bool(enabled)
        if enabled:
            return self.start_wakeword()
        self.stop_wakeword()
        return None

    def set_wake_suppressed(self, suppressed):
        """Go deaf while Marvin is recording — his own voice saying his own
        name would otherwise start a fresh question."""
        if self.wake is not None:
            self.wake.set_suppressed(suppressed)
            if not suppressed:
                self.wake.hush()  # ignore the tail of his own sentence

    def stop_speaking(self):
        """Cut Marvin off mid-answer. Returns True if there was speech to cut.

        Every stop path funnels through here: clicking Marvin, saying "Hey
        Marvin", or starting any recording — a misheard question must never
        earn fifteen seconds of confidently wrong monologue."""
        speaker = self._speaker
        if not self._speaking or speaker is None:
            return False
        try:
            speaker.interrupt()
        except Exception as exc:
            print(f"[speak] interrupt failed: {exc!r}", flush=True)
        self._speaking = False
        return True

    def set_speaking(self, speaking, speaker):
        """Marvin started/finished talking. Unlike recording, he stays LISTENING
        while he speaks, so "Hey Marvin" can barge in and cut him off."""
        self._speaker = speaker if speaking else None
        self._speaking = bool(speaking)
        if self.wake is None:
            return
        # Barge-in mode: a lower wake threshold while his own voice is the
        # thing drowning out the user's (see WakeWord.set_barge_mode).
        self.wake.set_barge_mode(bool(speaking))
        if speaking:
            # Listen through his own voice. He never says the wake phrase, and
            # the model scores continuous unrelated speech near zero. BUT a
            # recording in flight keeps its suppression — a quip racing a
            # dictation start must not reopen the wake word mid-question.
            if not self._active:
                self.wake.set_suppressed(False)
        else:
            self.wake.hush()  # don't let the tail of his sentence retrigger

    def _on_wake(self, score):
        """Wake phrase heard: open an ask episode with no key held."""
        if self._paused:
            return
        # Barge-in: if he's mid-answer, cut him off and take the new question.
        # This has to happen before the pipeline-busy check — during playback
        # the ask pipeline may still be running.
        if self.stop_speaking():
            print("[wake] interrupted mid-answer", flush=True)
            # Give playback a moment to die so its tail isn't recorded.
            time.sleep(0.15)
        if self._active or self._pipeline_busy or not self._jobs.empty():
            return
        if not self._ask_on:
            print("[wake] heard, but Ask Marvin is switched off", flush=True)
            return
        self._start_recording("ask", hands_free=True)
        threading.Thread(target=self._wake_endpoint, daemon=True).start()

    def _wake_endpoint(self):
        """End a hands-free recording when the user stops talking. No key is
        held, so this is the only thing that can close it (bar the 120s cap).

        The silence countdown starts when SPEECH does, not when the recording
        does. Running it from the start meant a user who took a beat to think
        after the beep was cut off before saying anything — recordings came
        back 1.0s long and empty, and Whisper hallucinated on the silence.
        """
        cfg = self.config.get("wakeword", {})
        silence = float(cfg.get("silence_seconds", 0.9))
        max_s = float(cfg.get("max_seconds", 15.0))
        wait_s = float(cfg.get("start_timeout_seconds", 4.0))
        # silence_rms is a FLOOR, not the whole story: the live threshold sits
        # above the measured room noise, so a quiet mic still registers speech.
        floor = float(cfg.get("silence_rms", 0.004))
        ambient = getattr(self.wake, "ambient", 0.0) if self.wake else 0.0
        # Adaptive, but capped: in a NOISY room 3x ambient could climb above
        # speech itself, and then the endpoint would never see the user start
        # talking — the same "cut off before a word was said" bug from the
        # other direction. 0.02 sits well under measured speech on this mic.
        threshold = max(floor, min(ambient * 3.0, 0.02))
        start = last_voice = time.monotonic()
        speaking = False
        while self._active and self._hands_free:
            time.sleep(0.05)
            now = time.monotonic()
            if self.wake is not None and self.wake.rms > threshold:
                last_voice = now
                speaking = True
            if not speaking:
                # Still waiting for them to begin. Only give up if they never do.
                if now - start > wait_s:
                    self._stop_recording("wake: nothing said")
                    return
                continue
            if now - last_voice > silence:
                self._stop_recording("wake: silence")
                return
            if now - start > max_s:
                self._stop_recording("wake: max duration")
                return

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
        # resolve_trigger_vk (not split_combo) so a single-letter dictate
        # trigger also gets a keycode and is suppressed at the event tap
        # instead of leaking the letter into the focused app.
        modifiers, trigger_vk = resolve_trigger_vk(required)
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

    def _set_ask_hotkey_params(self, binding):
        """Parse the ask-Marvin binding into live matching params (a plain
        modifier+trigger combo — no right-modifier / toggle-flag polling)."""
        required = parse_hotkey(binding)
        modifiers, vk = resolve_trigger_vk(required)
        self._ask_required = required
        self._ask_modifiers = modifiers
        self._ask_vk = vk  # matches at the event tap AND feeds the watchdog
        if vk is None:
            # No resolvable keycode -> the intercept can't route it. Rather than
            # a silent no-op, say so (all letter/digit triggers resolve fine).
            print(f"[hotkey] ask hotkey [{binding}] has no keycode — pick a "
                  "letter/digit trigger so it can be captured.", flush=True)

    def set_ask_hotkey(self, binding):
        """Change the ask-Marvin hotkey live (no listener rebuild)."""
        self.config.setdefault("hotkey", {})["ask"] = binding
        self._set_ask_hotkey_params(binding)
        print(f"Ask hotkey changed to [{binding}]", flush=True)

    def set_ask_enabled(self, enabled):
        """Turn the ask-Marvin hotkey on/off. When off, the combo types
        normally (no suppression, no recording)."""
        self._ask_on = bool(enabled)
        self.config.setdefault("ask", {})["enabled"] = self._ask_on

    def _status(self, state):
        if self._on_status is not None:
            self._on_status(state)

    def _start_recording(self, kind="dictate", hands_free=False):
        # Any recording starting shuts him up: recording over his own playback
        # would put HIS voice in the user's dictation/question.
        if self.stop_speaking():
            print("[rec] cut Marvin off (new recording)", flush=True)
        with self._state_lock:
            if self._active or self._paused:
                return
            self._active = True
            self._active_kind = kind
            self._hands_free = hands_free
            # Tell the watchdog which key to poll for release (dictate vs ask).
            # A hands-free take has no key at all: leaving both as None keeps
            # the watchdog's "was it released?" check from instantly killing it.
            if hands_free:
                self._active_trigger_vk = None
                self._active_rmod_flag = None
            elif kind == "ask":
                self._active_trigger_vk = self._ask_vk
                self._active_rmod_flag = None
            else:
                self._active_trigger_vk = self._trigger_vk
                self._active_rmod_flag = self._rmod_flag
            self._rec_start = time.monotonic()
        # Don't let the mic hear him while he's listening to a question.
        if self.wake is not None:
            self.wake.set_suppressed(True)
        if kind == "ask":
            # Signal the UI that an ask episode has begun (Marvin turns around).
            if self._on_ask_start is not None:
                try:
                    self._on_ask_start()
                except Exception as exc:
                    print(f"[ask] on_ask_start failed: {exc!r}", flush=True)
            # Preload the LLM now, overlapping the model load with the seconds
            # the user spends actually speaking — so answering feels instant
            # even on the first ask after Ollama has idled the model out.
            threading.Thread(target=self._warm_ask_model, daemon=True).start()
        self.on_press()

    def _stop_recording(self, reason=""):
        with self._state_lock:
            if not self._active:
                return
            self._active = False
            self._hands_free = False
            kind = self._active_kind
        if reason:
            print(f"[rec] stop ({reason})", flush=True)
        # Snapshot the audio NOW — recorder.stop() only flips a flag and
        # concatenates buffers, so the mic is immediately free for the next
        # press. The heavy stages run on the pipeline worker, FIFO behind
        # whatever is still finishing.
        with self._mic_lock:
            try:
                audio = self.recorder.stop()
            except Exception as exc:
                print(f"[rec] recorder stop failed: {exc!r}", flush=True)
                audio = None
        self._jobs.put((kind, audio))
        # Thinking-face feedback the instant the key lifts (unless the user
        # already started the next take — then "recording" owns the display).
        if not self._active:
            self._status("transcribing")

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
                    if self._active_rmod_flag is not None:  # modifier: check flag
                        flags = Quartz.CGEventSourceFlagsState(
                            Quartz.kCGEventSourceStateHIDSystemState)
                        released = not (int(flags) & self._active_rmod_flag)
                    elif self._active_trigger_vk is not None:  # real key: keycode
                        released = not Quartz.CGEventSourceKeyState(
                            Quartz.kCGEventSourceStateHIDSystemState,
                            self._active_trigger_vk)
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

    def set_swedish_model(self, name):
        """Choose the KB-Whisper model Swedish is transcribed with (None = use
        the general model for every language). Loads lazily on first Swedish
        utterance, so switching here costs nothing until then."""
        self.config["stt"]["swedish_model"] = name
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
        print("[rec] listening...", flush=True)
        self._status("recording")
        play_start(self.config)
        try:
            with self._mic_lock:
                # A lightning-fast tap can already be over (its stop() ran and
                # captured nothing) — don't reopen the mic for a dead take, or
                # it would capture forever with no stop() ever coming.
                if self._active:
                    self.recorder.start()
        except Exception as exc:
            # Mic unavailable / permission denied: recover to idle instead of
            # letting the exception break the listener callback.
            print(f"[rec] could not start mic: {exc}", flush=True)
            self._active = False
            self.set_wake_suppressed(False)
            self._status("idle")

    def _pipeline(self):
        """Worker draining finished recordings FIFO, one at a time."""
        while True:
            kind, audio = self._jobs.get()
            self._pipeline_busy = True
            try:
                self._process(kind, audio)
            except Exception as exc:
                print(f"[rec] pipeline job failed: {exc!r}", flush=True)
            finally:
                self._pipeline_busy = False
                if not self._active:
                    # Listen again — the processing time is the natural beat
                    # that keeps the tail of the user's own speech from being
                    # heard as a fresh "Hey Marvin". A newer recording owns
                    # the suppression, so leave it alone then.
                    self.set_wake_suppressed(False)
                    if self._jobs.empty():
                        self._status("idle")

    def _process(self, kind, audio):
        # "transcribing" doubles as the busy/thinking indicator for both
        # pipelines (STT, and for ask, the LLM answer too) — but never
        # overwrite "recording" while a newer take is being spoken.
        if not self._active:
            self._status("transcribing")
        if audio is None or len(audio) == 0:
            print("[rec] no audio captured", flush=True)
            return
        seconds = len(audio) / self.config["audio"]["sample_rate"]
        print(f"[rec] captured {seconds:.1f}s, transcribing...", flush=True)
        raw = self.transcriber.transcribe(audio)
        if not raw:
            print("[stt] (nothing recognized)", flush=True)
            return
        if kind == "ask":
            # Same hallucination guard as dictation: don't send a
            # Whisper repetition loop to the LLM as a "question".
            question = collapse_repeats(raw)
            if not question:
                print("[stt] (hallucination filtered, ask dropped)",
                      flush=True)
                return
            self._handle_ask(question)
            return
        cleaned = clean_transcript(raw, self.config)
        if not cleaned:
            print("[out] (empty after cleanup, nothing to inject)", flush=True)
            return
        print(f'[out] injecting into focused app: "{cleaned}"', flush=True)
        while self._active:
            # A newer take is being dictated: injecting now would synthesize
            # Cmd+V with the hotkey's modifiers held down. Wait it out — the
            # watchdog caps every recording, so this always ends.
            time.sleep(0.1)
        self._wait_hotkey_released()
        to_inject = cleaned
        if self.config["injection"].get("append_trailing_space", True):
            to_inject += " "  # keep a gap before the next dictation
        inject_text(to_inject, self.config)
        play_done(self.config)

    def _warm_ask_model(self):
        """Best-effort preload of the answer model (called at ask-record start)."""
        from . import ollama

        ask = self.config.get("ask", {})
        clean = self.config.get("cleanup", {})
        url = ask.get("ollama_url") or clean.get("ollama_url",
                                                 "http://localhost:11434")
        model = ask.get("ollama_model") or clean.get("ollama_model",
                                                      "gemma4:12b")
        ollama.warm(url, model, keep_alive=clean.get("keep_alive", "30m"))

    def _conversation_history(self):
        """Prior turns to send with the next question, oldest first.

        Expires after ask.conversation_idle_seconds: reviving an hour-old
        thread isn't context, it's confusion — and every retained turn is
        tokens gemma4 has to re-read before it can start answering.
        """
        ask = self.config.get("ask", {})
        idle = float(ask.get("conversation_idle_seconds", 180))
        if not self._turns or time.monotonic() - self._last_turn > idle:
            self._turns = []
            return []
        keep = int(ask.get("conversation_turns", 3)) * 2
        return self._turns[-keep:] if keep > 0 else []

    def _remember_turn(self, question, answer):
        self._turns.extend(
            [("user", question), ("assistant", answer)]
        )
        keep = int(self.config.get("ask", {}).get("conversation_turns", 3)) * 2
        del self._turns[:-keep or None]
        self._last_turn = time.monotonic()

    def reset_conversation(self):
        """Forget the thread (menu action, and whenever the wake word starts a
        fresh episode after a long gap)."""
        self._turns = []

    def _handle_ask(self, question):
        """Ask-Marvin last stage: send the transcribed question to the local LLM
        and hand the answer to the UI (bubble + optional speech)."""
        from .answer import answer_question

        lang = getattr(self.transcriber, "last_language", None)
        history = self._conversation_history()
        print(f'[ask] question ({lang}, {len(history)//2} prior turns): '
              f'"{question}"', flush=True)
        try:
            answer = answer_question(question, self.config, history=history,
                                     question_lang=lang)
        except Exception as exc:
            print(f"[ask] answer failed: {exc!r}", flush=True)
            answer = _ASK_UNREACHABLE
        if not answer:
            answer = _ASK_EMPTY
        else:
            self._remember_turn(question, answer)
        print(f'[ask] answer: "{answer}"', flush=True)
        if self._on_answer is not None:
            try:
                self._on_answer(question, answer)
            except Exception as exc:
                print(f"[ask] present failed: {exc!r}", flush=True)

    def _intercept_trigger(self, event_type, event, kind, modifiers):
        """Event-tap handler for one hotkey's trigger key (dictate or ask).
        Returns the event to pass it through, or None to swallow it so the
        focused app never sees the keystroke."""
        import Quartz

        if kind == "ask" and not self._ask_on:
            return event  # feature off: let the combo type normally
        if event_type == Quartz.kCGEventKeyDown:
            if self._toggle:
                if self._active:
                    # Any hotkey press ends the current toggle recording. Same
                    # kind = the natural stop; the other kind stops it too (then
                    # a second press starts that mode) so the key is never
                    # swallowed with nothing happening.
                    self._stop_recording()
                    return None
                if modifiers <= self._pressed:
                    self._start_recording(kind)
                    return None
                return event
            if self._active:
                return None  # key-repeat (or the other combo) while recording
            if modifiers <= self._pressed:
                self._start_recording(kind)
                return None
            return event  # trigger without the modifiers: normal typing
        # keyUp: end a hold-mode recording only if THIS hotkey started it.
        if not self._toggle and self._active and self._active_kind == kind:
            self._stop_recording()
            return None
        return event

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
            """Swallow a hotkey's trigger key at the event tap so the focused
            app never receives it. Reads the live hotkey params so a hotkey can
            change without rebuilding the listener. Handles both the dictate and
            the ask trigger (matched by virtual keycode)."""
            if event_type not in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
                return event
            vk = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if self._intercept_vk is not None and vk == self._intercept_vk:
                return self._intercept_trigger(
                    event_type, event, "dictate", self._modifiers)
            if self._ask_vk is not None and vk == self._ask_vk:
                return self._intercept_trigger(
                    event_type, event, "ask", self._ask_modifiers)
            return event

        def on_press(key):
            # Always track held keys, even for a right-modifier dictate hotkey:
            # the ask combo's modifier check (in the intercept) reads this set.
            pressed.add(canonicalize(key))
            rmod = self._rmod
            if rmod is not None:  # single right-modifier dictate: match raw key
                if key == rmod:
                    if self._toggle:
                        (self._stop_recording if self._active
                         else self._start_recording)("dictate")
                    elif not self._active:
                        self._start_recording("dictate")
            elif self._intercept_vk is None and self._required <= pressed:
                # A dictate hotkey with no keycode (exotic trigger) can't be
                # suppressed, so match it here instead. Both hotkeys with a
                # keycode go through the event-tap intercept, not this path.
                if self._toggle:
                    (self._stop_recording if self._active
                     else self._start_recording)("dictate")
                elif not self._active:
                    self._start_recording("dictate")

        def on_release(key):
            k = canonicalize(key)
            pressed.discard(k)
            rmod = self._rmod
            if rmod is not None and key == rmod:
                if not self._toggle and self._active:
                    self._stop_recording()
                return
            # Releasing any key of the CURRENTLY-active combo ends a hold-mode
            # recording (idempotent: the intercept keyUp may have stopped it
            # already). Keyed to the active kind so the two hotkeys don't cross.
            if not self._toggle and self._active:
                combo = (self._ask_required if self._active_kind == "ask"
                         else self._required)
                if k in combo:
                    self._stop_recording()

        print(f"Ready. Hold [{self.config['hotkey']['push_to_talk']}] "
              "to dictate.", flush=True)
        if self._ask_on:
            print(f"Hold [{self.config.get('hotkey', {}).get('ask', 'control + shift + a')}] "
                  "to ask Marvin a question.", flush=True)
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
