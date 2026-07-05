"""Menu-bar mode: a status-bar title + the floating pill, wrapping the engine.

The pynput listener runs on its own thread; rumps (AppKit) owns the main
thread. The menu-bar title (see ICONS) and the pill both reflect pipeline
state: idle, recording, transcribing, paused, or blocked. The pill is the
primary UI; the menu-bar title is a compact text fallback.
"""

import json

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
            cbs[idx]()

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

        self.menu = [self.status_item, self.model_menu, self.settings_menu,
                     self.perms_item, self.pause_item, None,
                     rumps.MenuItem("Quit", callback=rumps.quit_application)]

        self.engine = PushToTalkApp(config, on_status=self.set_state)
        self.listener = self.engine.build_listener()
        self.listener.start()

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
        )
        if self.pill is not None:
            self.pill.set_model(config["stt"]["model"])
        self._pill_timer = rumps.Timer(self._drive_pill, 0.05)
        self._pill_timer.start()

        if not trusted:
            self.set_state("blocked")

    def _drive_pill(self, _timer):
        # Runs on the main thread (rumps timer): the only safe place to touch
        # AppKit. Sync the menu-bar title here from the thread-safe _mode flag
        # instead of from worker threads.
        mode = self._mode
        title = ICONS.get(mode, ICONS["idle"])
        if self.title != title:
            self.title = title

        if self.pill is None:
            return
        # blocked/paused show as the idle dot (no active waveform).
        pill_mode = "idle" if mode in ("idle", "blocked", "paused") else mode
        level = self.engine.recorder.level if pill_mode == "recording" else 0.0
        # Idle costs nothing once the waveform has settled (skip unless hovered).
        if (pill_mode == "idle" and not self.pill.hover
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

    _LANGUAGES = [("Auto-detect", None), ("Svenska", "sv"), ("English", "en")]

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
        return menu

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

    def _toggle_cleanup(self, _sender):
        self._apply_cleanup(not self.config["cleanup"].get("enabled", True))

    def _toggle_sound(self, _sender):
        self._apply_sound(not self.config.get("sound_cues", {}).get("enabled", True))

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
        )
        if self.pill is not None:
            self.pill.set_model(self.config["stt"]["model"])

    # ------------------------------------------------- pill right-click menu

    def show_pill_menu(self, view, event):
        """Build and pop up a native menu at the pill (right/control-click)."""
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

        def submenu(title):
            parent_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, None, ""
            )
            sub = NSMenu.alloc().init()
            sub.setAutoenablesItems_(False)
            parent_item.setSubmenu_(sub)
            menu.addItem_(parent_item)
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
        for val, label in (("marvin", "Marvin face"), ("waveform", "Waveform")):
            add(label, (lambda v=val: self.set_pill_style(v)), look_sub,
                state=(val == cur_style))

        menu.addItem_(NSMenuItem.separatorItem())
        cleanup_on = self.config["cleanup"].get("enabled", True)
        add("AI cleanup", lambda: self._apply_cleanup(not cleanup_on), menu,
            state=cleanup_on)
        sound_on = self.config.get("sound_cues", {}).get("enabled", True)
        add("Sound cues", lambda: self._apply_sound(not sound_on), menu,
            state=sound_on)

        menu.addItem_(NSMenuItem.separatorItem())
        paused = self.listener is None
        add("Resume listening" if paused else "Pause listening",
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
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
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
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
            sender.title = "Resume listening"
            self.set_state("paused")
        else:
            self.listener = self.engine.build_listener()
            self.listener.start()
            sender.title = "Pause listening"
            self.set_state("idle")


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
