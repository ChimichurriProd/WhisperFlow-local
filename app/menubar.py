"""Menu-bar mode: a status-bar title + the floating pill, wrapping the engine.

The pynput listener runs on its own thread; rumps (AppKit) owns the main
thread. The menu-bar title (see ICONS) and the pill both reflect pipeline
state: idle, recording, transcribing, paused, or blocked. The pill is the
primary UI; the menu-bar title is a compact text fallback.
"""

import json
import os

import rumps
from AppKit import NSMenu, NSMenuItem
from Foundation import NSObject
import objc

from .hotkey import PushToTalkApp
from .permissions import is_trusted, prompt_for_trust
from .pill import create_pill


class _MenuTarget(NSObject):
    """Objective-C action target that dispatches menu clicks to Python callables
    stored (by tag) on the owning app — used for the pill's right-click menu.
    """

    def initWithApp_(self, app):
        self = objc.super(_MenuTarget, self).init()
        if self is None:
            return None
        self._app = app
        return self

    def fire_(self, sender):
        cbs = getattr(self._app, "_menu_callbacks", [])
        idx = sender.tag()
        if 0 <= idx < len(cbs):
            try:
                cbs[idx]()
            except Exception as exc:  # never let a setting change crash the app
                print(f"[menu] action failed: {exc!r}", flush=True)

# Plain-text titles: emoji can render as an invisible glyph in the macOS menu
# bar on some systems, so we use short text labels that always show.
ICONS = {"idle": "Flow", "recording": "● Rec", "transcribing": "Flow…",
         "paused": "Flow ‖", "blocked": "Flow ⚠"}

# Model choices shown in the Model submenu: (config value, human label).
# On the GPU (mlx) all are fast; the trade is accuracy, not speed.
MODEL_CHOICES = [
    ("base", "base — quickest, basic"),
    ("small", "small — fast"),
    ("medium", "medium — accurate"),
    ("large-v3-turbo", "large — best accuracy"),
]


class MenuBarApp(rumps.App):
    def __init__(self, config, config_path=None):
        super().__init__(ICONS["idle"], quit_button=None)
        self.config = config
        self.config_path = config_path
        binding = config["hotkey"]["push_to_talk"]

        # Accessibility is required for the hotkey tap + text injection. If
        # this process isn't trusted, trigger the system prompt so it lands
        # in the Accessibility list, and reflect the blocked state.
        trusted = is_trusted()
        if not trusted:
            prompt_for_trust()

        self.status_item = rumps.MenuItem(f"Hold {binding} to dictate")
        self.status_item.set_callback(None)  # informational, not clickable
        self.perms_item = rumps.MenuItem(
            "Grant Accessibility permission…", callback=self.request_permission
        )
        self.pause_item = rumps.MenuItem("Pause listening", callback=self.toggle_pause)

        self.model_menu = rumps.MenuItem("Model")
        self._model_items = {}
        for value, label in MODEL_CHOICES:
            item = rumps.MenuItem(label, callback=self._make_model_cb(value))
            self._model_items[value] = item
            self.model_menu.add(item)
        self._mark_current_model()

        self.settings_menu = self._build_settings_menu()

        self.demo_item = rumps.MenuItem(
            "Play all animations", callback=self._play_all_anims
        )
        self.menu = [self.status_item, self.model_menu, self.settings_menu,
                     self.perms_item, self.pause_item, self.demo_item, None,
                     rumps.MenuItem("Quit", callback=rumps.quit_application)]

        # Ask-Marvin: the engine hands answers back via _present_answer, which
        # marshals them to the main-thread pill timer (AppKit is main-only).
        self._bubble = None
        self._pending_answer = None
        # Two-faced Marvin: an ask episode turns him around to his 'oracle' back
        # face for the whole question, then back to the front 'scribe' when done.
        # _ask_started is set (off-thread) the moment an ask recording begins;
        # _answering stays true for the whole episode. Both are driven on the
        # main-thread pill timer, the only safe place to move the pill.
        self._ask_started = False
        self._answering = False
        self.engine = PushToTalkApp(
            config, on_status=self.set_state, on_answer=self._present_answer,
            on_ask_start=self._on_ask_start,
        )
        self.listener = self.engine.build_listener()
        self.listener.start()
        self._paused = False  # dictation paused via engine flag (not by stopping)

        # Floating pill = the primary UI. Driven by a main-thread timer that
        # reads the shared mode + live mic level (both set from other threads).
        self._mode = "blocked" if not trusted else "idle"
        saved = config.get("pill", {})
        pos = (saved.get("x"), saved.get("y")) if "x" in saved else None
        self._menu_target = _MenuTarget.alloc().initWithApp_(self)
        self._menu_callbacks = []
        self.pill = create_pill(
            on_click=self.cycle_model, on_move=self._save_pill_pos,
            on_menu=self.show_pill_menu, pos=pos,
            style=config.get("ui", {}).get("pill_style", "waveform"),
            on_double_click=self._marvin_speak,
            skin=config.get("ui", {}).get("marvin_skin"),
        )
        if self.pill is not None:
            self.pill.set_model(config["stt"]["model"])
        self._pill_timer = rumps.Timer(self._drive_pill, 0.05)
        self._pill_timer.start()

        # Marvin's shared TTS voice. The Speaker itself is cheap; the ~310MB
        # Kokoro model is warmed on a background thread at launch so the first
        # double-click-to-talk is snappy instead of paying the load then.
        # _speak_lock serializes playback so two lines never overlap; quips drop
        # if it's busy, answers wait for it (see _speak_text).
        import threading
        self._speak_lock = threading.Lock()
        try:
            from .tts import Speaker
            self._speaker = Speaker()
        except Exception:
            self._speaker = None
        if self._speaker is not None and config.get("ui", {}).get(
                "double_click_talk", True):
            import threading
            from .tts import _QUIPS

            def _warm_and_cache():
                self._speaker.warm()
                self._speaker.prerender(_QUIPS)  # so clicks play instantly

            threading.Thread(target=_warm_and_cache, daemon=True).start()

        if not trusted:
            self.set_state("blocked")

    def _drive_pill(self, _timer):
        # Runs on the main thread (rumps timer): the only safe place to touch
        # AppKit. Sync the menu-bar title here from the thread-safe _mode flag
        # instead of from worker threads.

        # A worker thread may have parked an answer for us to present (creating
        # the bubble / speaking must happen on the main thread).
        pending = self._pending_answer
        if pending is not None:
            self._pending_answer = None
            try:
                self._show_answer(*pending)
            except Exception as exc:
                print(f"[ask] show failed: {exc!r}", flush=True)
        if self._bubble is not None:
            self._bubble.tick()  # auto-dismiss once its time is up

        # Two-faced Marvin: turn to the oracle when an ask begins, and back to
        # the scribe once the whole episode is over (not recording/transcribing,
        # no bubble showing, done speaking) — covers answered, spoken, empty and
        # failed asks alike.
        if self.pill is not None and getattr(self.pill, "style", None) == "marvin":
            if self._ask_started:
                self._ask_started = False
                self._answering = True
                self.pill.face_back()
            if self._answering:
                busy = self._mode in ("recording", "transcribing")
                bubble_up = (self._bubble is not None
                             and getattr(self._bubble, "_visible", False))
                speaking = self._speak_lock.locked()
                if not busy and not bubble_up and not speaking:
                    self._answering = False
                    self.pill.face_front()

        mode = self._mode
        title = ICONS.get(mode, ICONS["idle"])
        if self.title != title:
            self.title = title

        if self.pill is None:
            return
        # blocked shows as idle; paused = Marvin dozes off (sleep clip), else idle.
        if mode == "paused" and self.pill.style == "marvin":
            pill_mode = "sleep"
        elif mode in ("idle", "blocked", "paused"):
            pill_mode = "idle"
        else:
            pill_mode = mode
        level = self.engine.recorder.level if pill_mode == "recording" else 0.0
        # Waveform idle costs nothing once settled; Marvin keeps a subtle idle
        # bob so he always looks a little alive.
        if (pill_mode == "idle" and self.pill.style != "marvin"
                and not self.pill.hover
                and not any(v > 0.001 for v in self.pill.levels)):
            return
        self.pill.tick(pill_mode, level)

    def cycle_model(self):
        values = [v for v, _ in MODEL_CHOICES]
        cur = self.config["stt"]["model"]
        nxt = values[(values.index(cur) + 1) % len(values)] if cur in values else values[0]
        self.select_model(nxt)

    def _save_pill_pos(self, x, y):
        self.config["pill"] = {"x": x, "y": y}
        self._save_config()

    # -------------------------------------------------------------- settings

    _LANGUAGES = [("Auto-detect", None), ("Svenska", "sv"),
                  ("English", "en"), ("Español", "es")]

    _HOTKEYS = [
        ("right command", "Right ⌘  (hold, one thumb) — easiest"),
        ("right option", "Right ⌥  (hold, one thumb)"),
        ("option + space", "⌥ + Space  (one hand)"),
        ("control + space", "⌃ + Space  (one hand)"),
        ("control + shift + space", "⌃⇧Space  (default)"),
        ("f9", "F9  (needs the fn setting on laptops)"),
    ]

    _SOUND_NAMES = ["Tink", "Pop", "Glass", "Ping", "Bottle", "Frog",
                    "Funk", "Hero", "Morse", "Purr", "Sosumi", "Submarine"]

    def _build_settings_menu(self):
        menu = rumps.MenuItem("Settings")

        lang_menu = rumps.MenuItem("Language")
        self._lang_items = {}
        current_lang = self.config["stt"].get("language")
        for label, code in self._LANGUAGES:
            it = rumps.MenuItem(label, callback=self._make_lang_cb(code))
            it.state = 1 if code == current_lang else 0
            self._lang_items[code] = it
            lang_menu.add(it)
        menu.add(lang_menu)

        self._cleanup_item = rumps.MenuItem(
            "AI cleanup", callback=self._toggle_cleanup
        )
        self._cleanup_item.state = 1 if self.config["cleanup"].get("enabled", True) else 0
        menu.add(self._cleanup_item)

        self._sound_item = rumps.MenuItem(
            "Sound cues", callback=self._toggle_sound
        )
        self._sound_item.state = 1 if self.config.get("sound_cues", {}).get("enabled", True) else 0
        menu.add(self._sound_item)

        ask_binding = self.config.get("hotkey", {}).get("ask", "control + shift + a")
        self._ask_item = rumps.MenuItem(
            f"Ask Marvin ({ask_binding})", callback=self._toggle_ask
        )
        self._ask_item.state = 1 if self.config.get("ask", {}).get("enabled", True) else 0
        menu.add(self._ask_item)

        self._ask_voice_item = rumps.MenuItem(
            "Speak answers", callback=self._toggle_ask_voice
        )
        self._ask_voice_item.state = 1 if self.config.get("ask", {}).get("voice", True) else 0
        menu.add(self._ask_voice_item)

        skin_menu = rumps.MenuItem("Marvin skin")
        self._skin_items = {}
        current_skin = self.config.get("ui", {}).get("marvin_skin", "B")
        for label, key in (("B — Stoned", "B"), ("A — Clean", "A"),
                           ("G — Plush", "G")):
            it = rumps.MenuItem(label, callback=self._make_skin_cb(key))
            it.state = 1 if key == current_skin else 0
            self._skin_items[key] = it
            skin_menu.add(it)
        menu.add(skin_menu)
        return menu

    def _make_skin_cb(self, key):
        return lambda _sender: self._apply_skin(key)

    def _apply_skin(self, key):
        """Swap the Marvin skin live (rumps callbacks run on the main thread,
        which is where all pill/AppKit mutation must happen)."""
        self.config.setdefault("ui", {})["marvin_skin"] = key
        for k, item in self._skin_items.items():
            item.state = 1 if k == key else 0
        if self.pill is not None and hasattr(self.pill, "set_skin"):
            self.pill.set_skin(key)
        self._save_config()

    def _make_lang_cb(self, code):
        return lambda _sender: self._apply_language(code)

    # Setters: single source of truth so the menu-bar Settings items and the
    # pill's right-click menu stay in sync.

    def _apply_language(self, code):
        self.engine.set_language(code)
        for c, item in self._lang_items.items():
            item.state = 1 if c == code else 0
        self._save_config()

    def _apply_cleanup(self, enabled):
        self.config["cleanup"]["enabled"] = enabled
        self._cleanup_item.state = 1 if enabled else 0
        self._save_config()

    def _apply_sound(self, enabled):
        self.config.setdefault("sound_cues", {})["enabled"] = enabled
        self._sound_item.state = 1 if enabled else 0
        self._save_config()

    # -------- Marvin speaks (double-click / local TTS) ---------------------

    def _apply_talk(self, enabled):
        self.config.setdefault("ui", {})["double_click_talk"] = enabled
        self._save_config()

    def _marvin_speak(self, force=False):
        """Double-click Marvin -> a random deadpan quip. `force` bypasses the
        on/off setting (the 'Say something' menu item). Quips are ambient, so
        they're skipped when he's already speaking (drop_if_busy)."""
        if not force and not self.config.get("ui", {}).get("double_click_talk", True):
            return
        import random
        from .tts import _QUIPS

        if self.pill is not None:
            try:
                self.pill.play_oneshot("alert")  # perk up as he speaks
            except Exception:
                pass
        self._speak_text(random.choice(_QUIPS), drop_if_busy=True)

    # -------- Ask Marvin (Ctrl+Shift+A) ------------------------------------

    def _present_answer(self, question, answer):
        """Engine callback (worker thread): park the answer for the main-thread
        pill timer, which owns AppKit (bubble) and kicks off speech."""
        self._pending_answer = (question, answer)

    def _speakable(self, text):
        """Kokoro TTS is English-only. Gate on the ANSWER text (not the STT-
        detected question language, which is easy to mis-detect on short
        utterances): Swedish letters mean 'don't voice this', so it stays
        text-only in the bubble."""
        return not any(c in "åäöÅÄÖ" for c in (text or ""))

    def _on_ask_start(self):
        """Engine callback (off-thread): an ask recording has begun. Just set a
        flag; the main-thread pill timer does the actual turn."""
        self._ask_started = True

    def _show_answer(self, question, answer):
        """Main thread: float the answer in a bubble by Marvin, turn him to his
        oracle back-face, and speak it when voice is on and it's a language he
        can voice."""
        # Also covers the typed 'Ask Marvin…' tester, which never fires
        # on_ask_start; turning here is idempotent with the hotkey path.
        if self.pill is not None and getattr(self.pill, "style", None) == "marvin":
            self._answering = True
            self.pill.face_back()
        if self._bubble is None:
            from .bubble import create_bubble

            self._bubble = create_bubble()
        if self._bubble is not None and self.pill is not None:
            f = self.pill.window.frame()
            anchor = (float(f.origin.x), float(f.origin.y),
                      float(f.size.width), float(f.size.height))
            self._bubble.show(answer, anchor)
        else:
            # No AppKit bubble available: fall back to a notification.
            try:
                rumps.notification("Marvin", question, answer)
            except Exception:
                pass

        # (No front 'alert' gesture here — the turn to his back face IS the
        # reaction, and a front clip would be invisible while he's turned away.)

        if self.config.get("ask", {}).get("voice", True) and self._speakable(answer):
            self._speak_text(answer)

    def _speak_text(self, text, drop_if_busy=False):
        """Speak *text* in Marvin's voice on a daemon thread. Playback is
        serialized by _speak_lock so two lines never overlap: answers wait their
        turn; ambient quips pass drop_if_busy=True to skip while he's speaking.
        The Speaker is built lazily so a failed launch-time init still recovers."""
        text = (text or "").strip()
        if not text:
            return
        if drop_if_busy and self._speak_lock.locked():
            return
        if self._speaker is None:
            try:
                from .tts import Speaker
                self._speaker = Speaker()
            except Exception:
                self._speaker = None
                return

        def _run():
            with self._speak_lock:  # answers queue behind a quip instead of dropping
                try:
                    self._speaker.speak(text)
                except Exception as exc:  # never let TTS crash the app
                    print(f"[marvin] speak failed: {exc!r}", flush=True)

        import threading
        threading.Thread(target=_run, daemon=True).start()

    def _apply_ask_enabled(self, enabled):
        self.engine.set_ask_enabled(enabled)
        if getattr(self, "_ask_item", None) is not None:
            self._ask_item.state = 1 if enabled else 0
        self._save_config()

    def _apply_ask_voice(self, enabled):
        self.config.setdefault("ask", {})["voice"] = enabled
        if getattr(self, "_ask_voice_item", None) is not None:
            self._ask_voice_item.state = 1 if enabled else 0
        self._save_config()

    def _toggle_ask(self, _sender):
        self._apply_ask_enabled(not self.config.get("ask", {}).get("enabled", True))

    def _toggle_ask_voice(self, _sender):
        self._apply_ask_voice(not self.config.get("ask", {}).get("voice", True))

    def _ask_prompt(self):
        """Typed tester: ask Marvin a question without using the mic. Routes
        through the engine's ask handler so the answer/fallback/present logic
        is shared with the spoken path (no second copy to drift)."""
        resp = rumps.Window(
            message="Ask Marvin a question:",
            title="Ask Marvin", default_text="", ok="Ask", cancel="Cancel",
            dimensions=(320, 60),
        ).run()
        q = resp.text.strip()
        if not (resp.clicked and q):
            return
        import threading
        threading.Thread(
            target=lambda: self.engine._handle_ask(q), daemon=True
        ).start()

    def _toggle_cleanup(self, _sender):
        self._apply_cleanup(not self.config["cleanup"].get("enabled", True))

    def _toggle_sound(self, _sender):
        self._apply_sound(not self.config.get("sound_cues", {}).get("enabled", True))

    def _play_all_anims(self, _sender):
        """Showcase every Marvin gesture, back to back (no-op for waveform)."""
        pill = getattr(self, "pill", None)
        if pill is not None and getattr(pill, "style", None) == "marvin":
            pill.play_all()

    # -------- input mode / hotkey / vocabulary / sound pack ----------------

    def _apply_input_mode(self, toggle):
        self.engine.set_toggle_mode(toggle)
        self._save_config()

    def _apply_hotkey(self, binding):
        # Change live — no listener rebuild (restarting the event tap crashes).
        self.engine.set_hotkey(binding)
        self._save_config()

    def _add_vocab_word(self):
        resp = rumps.Window(
            message="Add a word or name Marvin should always transcribe "
                    "correctly:",
            title="Vocabulary", default_text="", ok="Add", cancel="Cancel",
            dimensions=(320, 22),
        ).run()
        term = resp.text.strip()
        if resp.clicked and term:
            self.config.setdefault("vocabulary", {}).setdefault(
                "terms", []).append(term)
            self.engine.reload_vocabulary()
            self._save_config()
            rumps.notification("WhisperFlow", "Vocabulary", f"Added: {term}")

    def _add_vocab_fix(self):
        resp = rumps.Window(
            message="Add a correction as  wrong = right   "
                    "(e.g.  olama = Ollama):",
            title="Correction", default_text="", ok="Add", cancel="Cancel",
            dimensions=(320, 22),
        ).run()
        if resp.clicked and "=" in resp.text:
            wrong, right = (s.strip() for s in resp.text.split("=", 1))
            if wrong and right:
                self.config.setdefault("vocabulary", {}).setdefault(
                    "fixes", {})[wrong] = right
                self._save_config()
                rumps.notification("WhisperFlow", "Correction",
                                   f"{wrong} → {right}")

    def _set_cue_sound(self, which, name):
        self.config.setdefault("sound_cues", {})[which] = name
        self._save_config()
        from .sound import _play

        _play(name)  # preview

    def set_pill_style(self, style):
        """Switch the on-screen indicator between the Marvin face and the
        waveform bar. Recreates the pill window (sizes differ)."""
        self.config.setdefault("ui", {})["pill_style"] = style
        self._save_config()
        saved = self.config.get("pill", {})
        pos = (saved.get("x"), saved.get("y")) if "x" in saved else None
        if self.pill is not None:
            try:
                self.pill.window.close()
            except Exception:
                pass
        self.pill = create_pill(
            on_click=self.cycle_model, on_move=self._save_pill_pos,
            on_menu=self.show_pill_menu, pos=pos, style=style,
            on_double_click=self._marvin_speak,
        )
        if self.pill is not None:
            self.pill.set_model(self.config["stt"]["model"])

    # ------------------------------------------------- pill right-click menu

    def show_pill_menu(self, view, event):
        """Build and pop up a native menu at the pill (right/control-click)."""
        if self.pill is not None:
            self.pill.play_oneshot("react")  # a little reaction (if present)
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        self._menu_callbacks = []

        def add(title, cb, parent, state=0, enabled=True):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, "fire:", ""
            )
            item.setTarget_(self._menu_target)
            item.setTag_(len(self._menu_callbacks))
            item.setState_(1 if state else 0)
            item.setEnabled_(enabled)
            self._menu_callbacks.append(cb)
            parent.addItem_(item)

        def submenu(title, parent=menu):
            parent_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, None, ""
            )
            sub = NSMenu.alloc().init()
            sub.setAutoenablesItems_(False)
            parent_item.setSubmenu_(sub)
            parent.addItem_(parent_item)
            return sub

        header = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "WhisperFlow", None, ""
        )
        header.setEnabled_(False)
        menu.addItem_(header)
        menu.addItem_(NSMenuItem.separatorItem())

        cur_model = self.config["stt"]["model"]
        model_sub = submenu("Model")
        for val, label in MODEL_CHOICES:
            add(label, (lambda v=val: self.select_model(v)), model_sub,
                state=(val == cur_model))

        cur_lang = self.config["stt"].get("language")
        lang_sub = submenu("Language")
        for label, code in self._LANGUAGES:
            add(label, (lambda c=code: self._apply_language(c)), lang_sub,
                state=(code == cur_lang))

        cur_style = self.config.get("ui", {}).get("pill_style", "waveform")
        look_sub = submenu("Appearance")
        for val, label in (("marvin", "Marvin"), ("waveform", "Waveform")):
            add(label, (lambda v=val: self.set_pill_style(v)), look_sub,
                state=(val == cur_style))

        # Preview all animations we've got.
        if self.pill is not None and getattr(self.pill, "clips", None):
            anim_sub = submenu("Animate")
            for cname in sorted(self.pill.clips):
                add(cname.replace("_", " ").title(),
                    (lambda n=cname: self.pill.play_oneshot(n)), anim_sub)

        menu.addItem_(NSMenuItem.separatorItem())

        # Input mode: hold-to-talk vs tap-to-toggle
        toggle_on = self.config["hotkey"].get("mode", "hold") == "toggle"
        in_sub = submenu("Input mode")
        add("Hold to talk", lambda: self._apply_input_mode(False), in_sub,
            state=not toggle_on)
        add("Tap to toggle", lambda: self._apply_input_mode(True), in_sub,
            state=toggle_on)

        # Hotkey presets
        cur_hk = self.config["hotkey"].get("push_to_talk")
        hk_sub = submenu("Hotkey")
        for binding, label in self._HOTKEYS:
            add(label, (lambda b=binding: self._apply_hotkey(b)), hk_sub,
                state=(binding == cur_hk))

        # Vocabulary editor
        vocab = self.config.get("vocabulary", {})
        voc_sub = submenu("Vocabulary")
        add(f"{len(vocab.get('terms', []))} words, "
            f"{len(vocab.get('fixes', {}))} fixes", lambda: None, voc_sub,
            enabled=False)
        add("Add word…", lambda: self._add_vocab_word(), voc_sub)
        add("Add correction…", lambda: self._add_vocab_fix(), voc_sub)

        menu.addItem_(NSMenuItem.separatorItem())
        talk_on = self.config.get("ui", {}).get("double_click_talk", True)
        add("Double-click to talk", lambda: self._apply_talk(not talk_on), menu,
            state=talk_on)
        add("Say something", lambda: self._marvin_speak(force=True), menu)

        # Ask Marvin (speak a question, he answers)
        ask_on = self.config.get("ask", {}).get("enabled", True)
        ask_voice = self.config.get("ask", {}).get("voice", True)
        ask_binding = self.config.get("hotkey", {}).get("ask", "control + shift + a")
        add(f"Ask Marvin  ({ask_binding})",
            lambda: self._apply_ask_enabled(not ask_on), menu, state=ask_on)
        add("Speak answers", lambda: self._apply_ask_voice(not ask_voice), menu,
            state=ask_voice)
        add("Ask Marvin…", lambda: self._ask_prompt(), menu)

        cleanup_on = self.config["cleanup"].get("enabled", True)
        add("AI cleanup", lambda: self._apply_cleanup(not cleanup_on), menu,
            state=cleanup_on)

        # Sounds: on/off + pick start/done from the system sounds
        sc = self.config.get("sound_cues", {})
        snd_sub = submenu("Sounds")
        add("Enabled", lambda: self._apply_sound(not sc.get("enabled", True)),
            snd_sub, state=sc.get("enabled", True))
        start_sub = submenu("Start sound", snd_sub)
        done_sub = submenu("Done sound", snd_sub)
        for nm in self._SOUND_NAMES:
            add(nm, (lambda n=nm: self._set_cue_sound("start", n)), start_sub,
                state=(nm == sc.get("start", "Tink")))
            add(nm, (lambda n=nm: self._set_cue_sound("done", n)), done_sub,
                state=(nm == sc.get("done", "Pop")))

        menu.addItem_(NSMenuItem.separatorItem())
        add("Resume listening" if self._paused else "Pause listening",
            lambda: self.toggle_pause(self.pause_item), menu)
        add("Quit WhisperFlow", lambda: rumps.quit_application(), menu)

        NSMenu.popUpContextMenu_withEvent_forView_(menu, event, view)

    # ---------------------------------------------------------------- model

    def _make_model_cb(self, value):
        return lambda _sender: self.select_model(value)

    def _mark_current_model(self):
        current = self.config["stt"]["model"]
        for value, item in self._model_items.items():
            item.state = 1 if value == current else 0

    def select_model(self, value):
        self.engine.set_model(value)  # rebuilds transcriber, updates config dict
        self._mark_current_model()
        if self.pill is not None:
            self.pill.set_model(value)
        self._save_config()
        rumps.notification("WhisperFlow", "Model changed", f"Now using: {value}")

    def _save_config(self):
        if not self.config_path:
            return
        try:
            # Atomic write (tmp + rename): a crash/kill mid-save can never
            # leave a half-written config.json that blocks the next launch.
            tmp = f"{self.config_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            os.replace(tmp, self.config_path)
        except OSError:
            pass  # non-fatal: choice still applies for this session

    # ---------------------------------------------------------- permissions

    def request_permission(self, _sender):
        if prompt_for_trust() or is_trusted():
            rumps.alert("WhisperFlow", "Accessibility is granted. You're ready "
                        "to dictate.")
            self.set_state("idle")
        else:
            rumps.alert(
                "WhisperFlow — permission needed",
                "Open System Settings → Privacy & Security → Accessibility and "
                "turn ON the Python entry (shown as 'python3.x'), then quit and "
                "reopen WhisperFlow.",
            )

    # ---------------------------------------------------------------- state

    def set_state(self, state):
        # Called from worker threads too, so only touch the plain flag here;
        # the main-thread pill timer applies it to the AppKit title.
        self._mode = state

    def toggle_pause(self, sender):
        # Pause via a flag — never stop/restart the listener (that crashes).
        self._paused = not self._paused
        self.engine.set_paused(self._paused)
        try:
            sender.title = "Resume listening" if self._paused else "Pause listening"
        except Exception:
            pass
        self.set_state("paused" if self._paused else "idle")
        # Wake-up animation when resuming (Marvin opens his eyes).
        if not self._paused and self.pill is not None and (
                "wake" in getattr(self.pill, "clips", {})):
            self.pill.play_oneshot("wake")


def _hide_dock_icon():
    """Force menu-bar-only mode (no Dock icon).

    The app's Info.plist sets LSUIElement, but we exec the framework Python,
    whose own bundle lacks it — so a Python 'rocket' Dock icon appears. Setting
    the activation policy to Accessory at runtime removes it.
    """
    try:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
        )

        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )
    except Exception:
        pass  # cosmetic only


def run_menubar(config, config_path=None):
    _hide_dock_icon()
    MenuBarApp(config, config_path=config_path).run()
