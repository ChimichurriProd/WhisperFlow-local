"""Smoke tests: filler stripping, the <10-word LLM skip, and injection dispatch.

OS keystroke/clipboard calls and Ollama HTTP calls are mocked throughout, so
these run on any platform.
"""

from unittest.mock import patch

import pytest

from app import cleanup, injection
from app.config import load_config


@pytest.fixture
def config():
    return load_config()  # built-in defaults


# ------------------------------------------------------------- cleanup rules

def test_cleanup_strips_filler_word():
    result = cleanup.strip_fillers("um, hello there, uh, this is a test")
    lowered = result.lower()
    assert "um" not in lowered.split()
    assert "uh" not in lowered.split()
    assert result == "Hello there, this is a test."


def test_cleanup_capitalizes_and_punctuates():
    assert cleanup.strip_fillers("send the report tomorrow") == "Send the report tomorrow."


def test_cleanup_strips_swedish_fillers():
    result = cleanup.strip_fillers("öh, kan du boka mötet, asså, till på torsdag")
    lowered = result.lower()
    assert "öh" not in lowered.split()
    assert "asså" not in lowered.split()
    assert result == "Kan du boka mötet, till på torsdag."


# ----------------------------------------------------- <10-word LLM skip rule

def test_short_utterance_skips_ollama(config):
    text = "um send the report tomorrow"  # 5 words < 10
    with patch.object(cleanup, "ollama_clean") as mock_llm:
        result = cleanup.clean_transcript(text, config)
    mock_llm.assert_not_called()
    assert result == "Send the report tomorrow."


def test_long_utterance_calls_ollama(config):
    text = " ".join(["word"] * 12)  # 12 words >= 10
    with patch.object(cleanup, "ollama_clean", return_value="Polished.") as mock_llm:
        result = cleanup.clean_transcript(text, config)
    mock_llm.assert_called_once()
    assert result == "Polished."


def test_vocabulary_fixes_applied(config):
    config["vocabulary"] = {"terms": [], "fixes": {"olama": "Ollama"}}
    text = "i use olama every day"  # 5 words -> skip LLM path
    result = cleanup.clean_transcript(text, config)
    assert "Ollama" in result
    assert "olama" not in result.lower().replace("ollama", "")


def test_vocabulary_fixes_whole_word_only():
    # should not touch substrings inside other words
    out = cleanup.apply_vocabulary_fixes("scala and cat", {"cat": "CAT"})
    assert out == "scala and CAT"


def test_build_initial_prompt():
    from app.stt import build_initial_prompt

    assert build_initial_prompt([]) is None
    p = build_initial_prompt(["Ollama", "WhisperFlow"])
    assert "Ollama" in p and "WhisperFlow" in p


def test_cleanup_disabled_is_verbatim(config):
    config["cleanup"]["enabled"] = False
    config["vocabulary"] = {"terms": [], "fixes": {"olama": "Ollama"}}
    raw = "um i use olama and it stays exactly like this"
    with patch.object(cleanup, "ollama_clean") as mock_llm:
        result = cleanup.clean_transcript(raw, config)
    mock_llm.assert_not_called()
    assert result == "um i use Ollama and it stays exactly like this"  # verbatim + fix


# --------------------------------------------------------------- ask Marvin

def test_answer_question_calls_ollama(config):
    from app import answer, ollama

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "  42, obviously.  "}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    with patch.object(ollama.requests, "post", side_effect=fake_post):
        out = answer.answer_question("what is the meaning of life", config)

    assert out == "42, obviously."  # stripped
    # ask.ollama_model defaults to None -> inherits the cleanup model.
    assert captured["json"]["model"] == config["cleanup"]["ollama_model"]
    assert captured["json"]["system"] == answer.ANSWER_SYSTEM
    assert captured["json"]["prompt"] == "what is the meaning of life"
    assert captured["json"]["stream"] is False
    assert captured["timeout"] == config["ask"]["timeout_seconds"]


def test_answer_question_empty_is_noop(config):
    from app import answer, ollama

    with patch.object(ollama.requests, "post") as post:
        assert answer.answer_question("   ", config) == ""
    post.assert_not_called()


def test_answer_falls_back_to_cleanup_endpoint():
    """With no explicit ask url/model, it reuses the cleanup Ollama settings."""
    from app import answer, ollama

    cfg = {
        "cleanup": {"ollama_url": "http://host:9/", "ollama_model": "m2",
                    "keep_alive": "30m"},
        "ask": {},  # no url/model override
    }
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok"}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    with patch.object(ollama.requests, "post", side_effect=fake_post):
        answer.answer_question("hi", cfg)
    assert captured["url"] == "http://host:9/api/generate"  # trailing slash trimmed
    assert captured["json"]["model"] == "m2"


def test_ollama_generate_raises_on_error_body():
    """A 200 response carrying an Ollama {"error": ...} body must raise (not
    silently return ''), so callers can tell 'server said no' from success."""
    from app import ollama

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "model 'nope' not found"}

    with patch.object(ollama.requests, "post", return_value=FakeResp()):
        with pytest.raises(RuntimeError):
            ollama.generate("hi", url="http://x", model="nope")


def test_ollama_generate_missing_response_is_empty():
    """A 200 body without 'response' returns '' rather than raising KeyError."""
    from app import ollama

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"done": True}

    with patch.object(ollama.requests, "post", return_value=FakeResp()):
        assert ollama.generate("hi", url="http://x", model="m") == ""


def test_resolve_trigger_vk_letter_uses_ansi_fallback():
    from app.hotkey import parse_hotkey, resolve_trigger_vk, split_combo

    # split_combo can't resolve a letter trigger to a keycode...
    _, plain = split_combo(parse_hotkey("control + shift + a"))
    assert plain is None
    # ...but resolve_trigger_vk fills it via the ANSI map (A = keycode 0).
    modifiers, vk = resolve_trigger_vk(parse_hotkey("control + shift + a"))
    assert vk == 0
    from pynput.keyboard import Key
    assert modifiers == frozenset({Key.ctrl, Key.shift})


def test_ollama_clean_sends_temperature_zero(config):
    import app.cleanup as cl

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok"}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResp()

    with patch.object(cl.requests, "post", side_effect=fake_post):
        cl.ollama_clean("hello there world", "http://x", "m")
    assert captured["options"]["temperature"] == 0.0


def test_setting_changes_dont_crash():
    """Every runtime setting change must be safe (no listener rebuild, etc.)."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp
    from app.menubar import MenuBarApp

    eng = PushToTalkApp(_lc())
    for binding, _label in MenuBarApp._HOTKEYS:
        eng.set_hotkey(binding)          # must not crash / rebuild the listener
    eng.set_toggle_mode(True)
    eng.set_toggle_mode(False)
    eng.set_paused(True)
    eng.set_paused(False)
    eng.set_language("es")
    eng.set_language(None)
    eng.set_model("small")
    eng.reload_vocabulary()
    eng.set_ask_enabled(False)        # ask-Marvin toggles: also no rebuild
    eng.set_ask_enabled(True)
    eng.set_ask_hotkey("control + shift + q")


def test_sound_cues_respect_toggle():
    from app import sound

    with patch.object(sound, "_play") as p:
        sound.play_start({"sound_cues": {"enabled": False}})
    p.assert_not_called()
    with patch.object(sound, "_play") as p:
        sound.play_start({"sound_cues": {"enabled": True, "start": "Tink"}})
    p.assert_called_once_with("Tink")


def test_ollama_failure_falls_back_to_rules(config):
    text = "um " + " ".join(["word"] * 11)
    with patch.object(
        cleanup, "ollama_clean",
        side_effect=cleanup.requests.ConnectionError("ollama down"),
    ):
        result = cleanup.clean_transcript(text, config)
    assert "um" not in result.lower().split()
    assert result.startswith("Word")


# ------------------------------------------------------- injection dispatch

def test_injection_dispatches_clipboard(config):
    config["injection"]["delivery_method"] = "clipboard"
    with patch.object(injection, "inject_clipboard") as clip, \
         patch.object(injection, "inject_type") as typ:
        injection.inject_text("Hello world.", config)
    clip.assert_called_once_with("Hello world.", restore_clipboard=True)
    typ.assert_not_called()


def test_injection_dispatches_type(config):
    config["injection"]["delivery_method"] = "type"
    with patch.object(injection, "inject_clipboard") as clip, \
         patch.object(injection, "inject_type") as typ:
        injection.inject_text("Hello world.", config)
    typ.assert_called_once_with("Hello world.", char_delay_ms=5)
    clip.assert_not_called()


def test_injection_clipboard_sets_pastes_and_restores():
    """Clipboard path: save old clipboard, set new text, Cmd+V, restore old."""
    calls = []
    with patch.object(injection, "_require_macos"), \
         patch.object(injection, "_get_clipboard_text", return_value="OLD"), \
         patch.object(injection, "_set_clipboard_text",
                      side_effect=lambda t: calls.append(("set", t))), \
         patch.object(injection, "_send_cmd_v",
                      side_effect=lambda: calls.append(("paste",))), \
         patch.object(injection.time, "sleep"):
        injection.inject_clipboard("Hello world.", restore_clipboard=True)
    assert calls == [("set", "Hello world."), ("paste",), ("set", "OLD")]


def test_injection_type_posts_unicode_events():
    """The type path posts per-char unicode down+up CGEvents (no pynput/TSM)."""
    events = []
    with patch.object(injection, "_require_macos"), \
         patch.object(injection, "_post_key_event",
                      side_effect=lambda vk, down, flags=0, unicode_char=None:
                      events.append((vk, down, unicode_char))):
        injection.inject_type("hi", char_delay_ms=0)
    assert events == [
        (0, True, "h"), (0, False, "h"),
        (0, True, "i"), (0, False, "i"),
    ]


def test_injection_empty_text_is_noop(config):
    with patch.object(injection, "inject_clipboard") as clip:
        injection.inject_text("", config)
    clip.assert_not_called()


def test_invalid_delivery_method_rejected():
    cfg = load_config()
    cfg["injection"]["delivery_method"] = "telepathy"
    with pytest.raises(ValueError):
        injection.inject_text("x", cfg)


# ------------------------------------------------------------ hotkey parsing

def test_parse_hotkey_combo():
    from pynput.keyboard import Key, KeyCode

    from app.hotkey import parse_hotkey

    assert parse_hotkey("control + shift + space") == frozenset(
        {Key.ctrl, Key.shift, Key.space}
    )
    assert parse_hotkey("cmd + d") == frozenset({Key.cmd, KeyCode.from_char("d")})
    assert parse_hotkey("f9") == frozenset({Key.f9})
    with pytest.raises(ValueError):
        parse_hotkey("control + bogus")


def test_split_combo_resolves_trigger_keycode():
    from pynput.keyboard import Key

    from app.hotkey import parse_hotkey, split_combo

    modifiers, vk = split_combo(parse_hotkey("control + shift + space"))
    assert modifiers == frozenset({Key.ctrl, Key.shift})
    assert vk == 49  # macOS virtual keycode for Space

    # Char triggers can't be resolved to a keycode -> suppression disabled.
    _, char_vk = split_combo(parse_hotkey("cmd + d"))
    assert char_vk is None


# --------------------------------------------- two-faced Marvin (front/back)

def test_ease_toward_converges_without_overshoot():
    from app.pill import _TURN_BACK, _TURN_SPEED, _ease_toward

    # front (0) -> back (_TURN_BACK): reaches it exactly, never overshoots,
    # in roughly one second at 20 Hz.
    f, ticks = 0.0, 0
    while f < _TURN_BACK:
        f = _ease_toward(f, float(_TURN_BACK), _TURN_SPEED)
        ticks += 1
        assert f <= _TURN_BACK
        assert ticks < 1000  # can't loop forever
    assert f == float(_TURN_BACK)
    assert ticks == pytest.approx(_TURN_BACK / _TURN_SPEED, abs=1)  # ~20 ticks

    # back -> front returns exactly to 0 (no undershoot past it).
    while f > 0.0:
        f = _ease_toward(f, 0.0, _TURN_SPEED)
        assert f >= 0.0
    assert f == 0.0


def _pill_turn_stub():
    """Bind the real _Pill.face_back/face_front to a bare stub so the pure turn
    logic runs without building an AppKit window. Skips if AppKit is absent."""
    from app import pill as P

    Pill = getattr(P, "_Pill", None)
    if Pill is None:
        pytest.skip("AppKit unavailable: _Pill not defined")

    class Stub:
        pass

    s = Stub()
    s.back = object()               # a back face is present
    s.clips = {"spin": [0] * 120}   # a turnaround exists -> animate the turn
    s._turn_f = 0.0
    s._face_target = 0.0
    s.face_back = Pill.face_back.__get__(s)
    s.face_front = Pill.face_front.__get__(s)
    return s, P


def test_face_back_and_front_set_target():
    s, P = _pill_turn_stub()
    s.face_back()
    assert s._face_target == float(P._TURN_BACK)  # turning to the oracle
    assert s._turn_f == 0.0                        # but not snapped (animates)
    s.face_front()
    assert s._face_target == 0.0                   # turning back to the scribe


def test_face_back_snaps_when_no_turnaround():
    s, P = _pill_turn_stub()
    s.clips = {}                     # no spin frames to animate with
    s.face_back()
    assert s._turn_f == float(P._TURN_BACK)  # snaps straight to the back face


def test_face_back_is_noop_without_back_face():
    s, _ = _pill_turn_stub()
    s.back = None                    # nothing to turn to
    s.face_back()
    assert s._face_target == 0.0     # stays facing front


def test_model_cycle_pops_marvins_firefly():
    """Switching model pops Marvin's firefly orb (flare ring); re-setting
    the same model (or a non-marvin style) does not."""
    from app import pill as P

    Pill = getattr(P, "_Pill", None)
    if Pill is None:
        pytest.skip("AppKit unavailable: _Pill not defined")

    class Stub:
        style = "marvin"
        model = "small"
        _orb_pop = 0.0

        def _render(self):
            pass

    s = Stub()
    s.set_model = Pill.set_model.__get__(s)
    s.set_model("medium")                 # a real switch -> flare
    assert s.model == "medium"
    assert s._orb_pop == 1.0

    s._orb_pop = 0.0
    s.set_model("medium")                 # same model again -> quiet
    assert s._orb_pop == 0.0

    s.style = "waveform"                  # waveform pill has its own dot
    s.set_model("base")
    assert s._orb_pop == 0.0


def test_orb_orbit_is_a_3d_path_around_the_head():
    """The firefly's orbit: z sweeps front (+1) to back (-1); the plane is
    seen slightly from above so the front pass crosses below centre and the
    back pass above; a full lap stays on the canvas."""
    import math

    from app.pill import _orb_orbit

    w = 280.0
    front = _orb_orbit(0.0, math.pi / 2.0, w)
    back = _orb_orbit(0.0, 3.0 * math.pi / 2.0, w)
    assert front[2] == pytest.approx(1.0)
    assert back[2] == pytest.approx(-1.0)
    assert front[1] > back[1]                 # below centre vs above centre
    for i in range(200):
        x, y, z = _orb_orbit(i * 0.31, i * math.pi / 25.0, w)
        assert 0.0 <= x <= w and 0.0 <= y <= w
        assert -1.0 <= z <= 1.0


def _orb_stub():
    """Bind the real _Pill orb logic to a bare stub (no AppKit window)."""
    import math

    from app import pill as P

    Pill = getattr(P, "_Pill", None)
    if Pill is None:
        pytest.skip("AppKit unavailable: _Pill not defined")

    class Stub:
        pass

    s = Stub()
    s.style = "marvin"
    s._anim = 0.0
    s._w = 280.0
    s._orb_theta = math.pi / 2.0      # right out front (z = +1)
    s._orb_pos = None
    s._orb_trail = []
    s._orb_event = None
    s._orb_notice_t = 10**9
    s._orb_swallow_t = 10**9
    s._orb_pop = 0.0
    s._oneshot = None
    s.clips = {}
    s._idle_gestures = []
    s.played = []
    s.play_oneshot = s.played.append
    s._tick_orb = Pill._tick_orb.__get__(s)
    s._orb_notice = Pill._orb_notice.__get__(s)
    return s


def test_orb_swallow_digest_respawn_cycle():
    """The swallow event: orb spirals in (scale -> 0), Marvin digests it
    (orb gone), then it pops back out onto its orbit with a flare."""
    s = _orb_stub()
    s._orb_swallow_t = 1              # about to fire, orb already out front
    s._tick_orb("idle")
    assert s._orb_event and s._orb_event[0] == "swallow"
    assert s._orb_swallow_t > 1       # re-armed for next time
    for _ in range(30):
        s._tick_orb("idle")
    assert s._orb_event[0] == "digest"
    assert s._orb_pos is None         # the orb is inside him
    assert s._orb_trail == []
    for _ in range(36):
        s._tick_orb("idle")
    assert s._orb_event is None       # popped back out...
    assert s._orb_pop == 1.0          # ...with a flare
    s._tick_orb("idle")
    assert s._orb_pos is not None     # orbiting again


def test_orb_notice_prefers_watchful_gesture_and_only_fires_idle():
    s = _orb_stub()
    s.clips = {"eyeroll": [0], "angry": [0]}
    s._orb_notice_t = 1
    s._tick_orb("recording")          # busy: the timer must not fire
    assert s.played == []
    s._tick_orb("idle")
    assert s.played == ["eyeroll"]    # B's deadpan look-at-it gesture


def test_ask_start_signal_fires_for_ask_only():
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    fired = []
    eng = PushToTalkApp(_lc(), on_ask_start=lambda: fired.append("ask"))
    with patch.object(eng, "on_press"):   # don't actually open the mic
        eng._start_recording("ask")
        assert fired == ["ask"]           # ask begins -> Marvin should turn
        eng._active = False               # let another recording start
        eng._start_recording("dictate")
        assert fired == ["ask"]           # dictation must NOT fire the turn
