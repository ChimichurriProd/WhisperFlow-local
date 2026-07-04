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
