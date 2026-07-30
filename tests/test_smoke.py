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


# ------------------------------------------- Whisper hallucination-loop guard


def test_collapse_repeats_drops_pure_hallucination():
    # The real-world case: "myślę, " × 94 on a misdetected short utterance.
    assert cleanup.collapse_repeats("myślę, " * 94) == ""


def test_collapse_repeats_collapses_phrase_loop():
    text = "skicka rapporten " * 5 + "imorgon"
    assert cleanup.collapse_repeats(text) == "skicka rapporten imorgon"


def test_collapse_repeats_keeps_triple_word():
    # Dictated "nej, nej, nej" is legitimate — single words collapse at >=4,
    # and an utterance that is NOTHING but one repeated token drops entirely.
    assert cleanup.collapse_repeats("nej, nej, nej.") == "nej, nej, nej."
    assert cleanup.collapse_repeats("ja ja ja ja tack") == "ja tack"
    assert cleanup.collapse_repeats("ja ja ja ja ja ja") == ""


def test_collapse_repeats_catches_fused_loops():
    """Whisper can weld a repeat loop into ONE token (seen injected into a
    real document: "useertasertasertas…"), which token-level collapse can't
    touch. The in-token guard must catch it."""
    fused = "For more information, use" + "ertas" * 60 + "."
    out = cleanup.collapse_repeats(fused)
    assert "ertasertas" not in out
    assert len(out) < 60
    # Legitimate words with short internal repeats survive.
    for ok in ("barnbarnsbarn är underbara",
               "banana bandana", "Mississippi is a river"):
        assert cleanup.collapse_repeats(ok) == ok


def test_collapse_repeats_drops_punctuation_only():
    assert cleanup.collapse_repeats("!") == ""
    assert cleanup.collapse_repeats("...") == ""


def test_collapse_repeats_passes_normal_text():
    text = "Det här är en helt vanlig mening utan upprepningar."
    assert cleanup.collapse_repeats(text) == text


def test_clean_transcript_filters_hallucination(config):
    # Runs even in verbatim mode (cleanup disabled) — it's an STT artifact.
    config["cleanup"]["enabled"] = False
    assert cleanup.clean_transcript("myślę, " * 94, config) == ""


# ----------------------------------------------------- <10-word LLM skip rule

def test_short_utterance_skips_ollama(config):
    text = "um send the report tomorrow"  # 5 words < 10
    with patch.object(cleanup, "ollama_clean") as mock_llm:
        result = cleanup.clean_transcript(text, config)
    mock_llm.assert_not_called()
    assert result == "Send the report tomorrow."


def test_long_utterance_calls_ollama(config):
    # 12 distinct words >= 10 (identical words would trip the loop guard)
    text = "please send the quarterly report to the whole team before friday morning"
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
            return {"message": {"content": "  42, obviously.  "}}

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
    assert captured["json"]["messages"] == [
        {"role": "system", "content": answer.ANSWER_SYSTEM},
        {"role": "user", "content": "what is the meaning of life"},
    ]
    assert captured["json"]["stream"] is False
    # Reasoning models must not spend the token budget on hidden thinking
    # (gemma4 returns an EMPTY answer otherwise).
    assert captured["json"]["think"] is False
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
            return {"message": {"content": "ok"}}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    with patch.object(ollama.requests, "post", side_effect=fake_post):
        answer.answer_question("hi", cfg)
    assert captured["url"] == "http://host:9/api/chat"  # trailing slash trimmed
    assert captured["json"]["model"] == "m2"


def test_ollama_generate_raises_on_error_body():
    """An Ollama {"error": ...} body must raise RuntimeError (not silently
    return ''), so callers can tell 'server said no' from success. /api/chat
    sends it with a 4xx status, so it has to be read BEFORE raise_for_status
    or the caller only ever sees a bare '404 Client Error'."""
    from app import ollama

    class FakeResp:
        def raise_for_status(self):
            raise ollama.requests.HTTPError("404 Client Error")

        def json(self):
            return {"error": "model 'nope' not found"}

    with patch.object(ollama.requests, "post", return_value=FakeResp()):
        with pytest.raises(RuntimeError, match="not found"):
            ollama.generate("hi", url="http://x", model="nope")


def test_ollama_generate_http_error_without_body_still_raises():
    """A failure with no JSON error body stays a requests exception ('server
    down'), which cleanup.py catches to fall back to rule-based cleanup."""
    from app import ollama

    class FakeResp:
        def raise_for_status(self):
            raise ollama.requests.HTTPError("500 Server Error")

        def json(self):
            raise ValueError("not json")

    with patch.object(ollama.requests, "post", return_value=FakeResp()):
        with pytest.raises(ollama.requests.HTTPError):
            ollama.generate("hi", url="http://x", model="m")


def test_ollama_generate_missing_content_is_empty():
    """A 200 body without a message returns '' rather than raising KeyError."""
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


# -------------------------------------------- hands-free wake word ("Hey Marvin")

def _wake(on_detect=None, **kw):
    """A WakeWord whose ONNX model is replaced by a scripted scorer, so the
    gating logic can be tested without a trained model."""
    from app.wakeword import WakeWord

    w = WakeWord("unused.onnx", on_detect or (lambda score: None), **kw)
    return w


class _FakeModel:
    """Stands in for WakeWordModel: returns queued scores, then 0, and records
    the window length it was handed."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.windows = []

    def predict(self, chunk):
        self.windows.append(len(chunk))
        return {"hey_marvin": self.scores.pop(0) if self.scores else 0.0}


def _pump(w, model, hops):
    """Run the worker's gating logic synchronously over *hops* of audio,
    mirroring WakeWord._run without starting a thread. Returns the scores that
    would have fired on_detect.

    Note the model only ever sees FULL 2s windows — the real one scores zero on
    anything shorter, so the worker must not call it before then."""
    import time

    import numpy as np

    import app.wakeword as W

    w._model = model
    fired = []
    # Feed and drain one hop at a time, the way the worker keeps up with the
    # mic. Queueing every hop first would trip the drop-oldest backpressure.
    for _ in range(hops):
        w.feed(np.zeros(W.HOP, dtype="float32"))
        hop = w._take(W.HOP)
        if hop is None:
            continue
        w.rms = float(np.sqrt(np.mean(np.square(hop))))
        if w._suppressed:
            w._window.clear()
            continue
        w._window.append(hop)
        if len(w._window) < W.WINDOW_FRAMES:
            continue
        score = max(model.predict(np.concatenate(w._window)).values())
        w.last_score = score
        if score < w.threshold or time.monotonic() < w._quiet_until:
            continue
        w.hush()
        w._window.clear()
        fired.append(score)
    return fired


def test_wake_feed_never_blocks_or_infers():
    """The audio callback must only enqueue — inference there would stall the
    realtime thread and drop microphone frames."""
    import numpy as np

    import app.wakeword as W

    w = _wake()
    w.feed(np.zeros(W.HOP, dtype="float32"))
    assert w._model is None          # nothing loaded, nothing scored
    assert w._take(W.HOP) is not None


def _endpoint_run(eng, rms_over_time, tick=0.05, ambient=0.001):
    """Drive _wake_endpoint against a scripted loudness trace. Returns
    (stop_reason, seconds_recorded)."""
    import itertools
    import time as _t

    from unittest.mock import patch

    clock = itertools.count(0.0, tick)
    now = [0.0]

    def fake_monotonic():
        now[0] = next(clock)
        return now[0]

    stopped = {}
    eng._active = True
    eng._hands_free = True

    def fake_stop(reason=""):
        stopped["reason"] = reason
        stopped["at"] = now[0]
        eng._active = False

    class FakeWake:
        rms = 0.0

    FakeWake.ambient = ambient
    eng.wake = FakeWake()
    trace = iter(rms_over_time)

    def fake_sleep(_s):
        eng.wake.rms = next(trace, 0.0)

    with patch.object(_t, "monotonic", fake_monotonic), \
         patch.object(_t, "sleep", fake_sleep), \
         patch.object(eng, "_stop_recording", fake_stop):
        eng._wake_endpoint()
    return stopped.get("reason"), stopped.get("at")


def test_wake_endpoint_waits_for_speech_before_counting_silence():
    """The bug that made him 'listen but not respond': the silence countdown
    ran from the beep, so a beat of thought ended the recording at 1.0s with
    nothing recorded. It must not close until speech has actually started."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    # 2s of thinking (quiet), then 2s of speech, then silence.
    trace = [0.0005] * 40 + [0.02] * 40 + [0.0005] * 60
    reason, at = _endpoint_run(eng, trace)
    assert reason == "wake: silence"
    assert at > 4.0, f"cut off after only {at:.1f}s — thinking time was eaten"


def test_wake_endpoint_gives_up_if_nothing_is_ever_said():
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    reason, at = _endpoint_run(eng, [0.0005] * 400)
    assert reason == "wake: nothing said"
    assert at == pytest.approx(
        eng.config["wakeword"]["start_timeout_seconds"], abs=0.3)


def test_wake_endpoint_threshold_adapts_to_a_quiet_mic():
    """A fixed floor of 0.01 sat ABOVE this user's speaking level, so their
    voice never counted as speech. The threshold tracks room noise instead."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    # Quiet mic: ambient 0.0008, speech only 0.006 — under the old 0.01 floor.
    trace = [0.0008] * 20 + [0.006] * 40 + [0.0008] * 60
    reason, at = _endpoint_run(eng, trace)
    assert reason == "wake: silence"
    assert at > 3.0, "quiet speech was not recognised as speech"


def test_single_instance_lock_rejects_a_second_engine(tmp_path):
    """Two engines at once grab the same mic/hotkeys and clobber each other's
    config saves (observed, not hypothetical — a stale duplicate bundle did
    exactly this). The second must refuse to start."""
    from app.__main__ import hold_instance_lock

    lock_path = tmp_path / "engine.lock"
    first = hold_instance_lock(lock_path)
    assert first is not None
    assert hold_instance_lock(lock_path) is None   # second engine: refused
    first.close()                                   # release
    third = hold_instance_lock(lock_path)
    assert third is not None                        # and it's not sticky
    third.close()


def test_transcriber_warm_loads_both_models():
    """Prewarm must touch the general AND the Swedish model — each one's
    first load is what the first dictation would otherwise pay for."""
    from app.stt import Transcriber

    t = Transcriber(model="large-v3-turbo", engine="mlx",
                    swedish_model="kb-large")
    loaded = []
    t._run_mlx = lambda audio, model, lang=None: loaded.append((model, lang))
    t.warm()
    assert loaded == [("large-v3-turbo", "en"), ("kb-large", "sv")]

    t2 = Transcriber(model="large-v3-turbo", engine="faster-whisper")
    t2._run_mlx = lambda *a, **k: loaded.append("wrong")
    t2.warm()                                   # CPU fallback: no-op
    assert "wrong" not in loaded


def test_prewarm_sets_warming_flag_for_the_ui():
    """The pill greys the model orb while engine.warming is True — that flag
    is the whole 'loading, not crashed' cue, so its lifecycle matters."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp
    from app import ollama

    eng = PushToTalkApp(_lc())
    seen = []
    with patch.object(eng.transcriber, "warm",
                      side_effect=lambda: seen.append(eng.warming)), \
         patch.object(ollama, "warm") as ow:
        thread = eng.prewarm()
        thread.join(timeout=5)
    assert seen == [True]          # flag was up while models loaded
    assert eng.warming is False    # and down when done
    ow.assert_called_once()


def test_wake_endpoint_caps_the_adaptive_threshold_in_noise():
    """A noisy room must not push the speech threshold above speech itself —
    that would recreate the cut-off-before-speaking bug from the other side."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    # Loud ambience (0.05): uncapped, 3x ambient = 0.15 and speech at 0.03
    # would never register. Capped at 0.02, it must.
    trace = [0.01] * 20 + [0.03] * 40 + [0.01] * 60
    reason, at = _endpoint_run(eng, trace, ambient=0.05)
    assert reason == "wake: silence"
    assert at > 3.0, "speech quieter than 3x ambient was missed"


def test_pill_warming_flag_flares_when_ready():
    from app import pill as P

    Pill = getattr(P, "_Pill", None)
    if Pill is None:
        pytest.skip("AppKit unavailable: _Pill not defined")

    class Stub:
        style = "marvin"
        model = "large-v3-turbo"
        warming = False
        _orb_pop = 0.0

        def _render(self):
            pass

    s = Stub()
    s.set_warming = Pill.set_warming.__get__(s)
    s.set_warming(True)
    assert s.warming is True and s._orb_pop == 0.0   # grey, no flare yet
    s.set_warming(False)
    assert s.warming is False and s._orb_pop == 1.0  # ready: flare
    s._orb_pop = 0.0
    s.set_warming(False)                             # idempotent: no re-flare
    assert s._orb_pop == 0.0


def test_conversation_history_is_replayed_and_expires():
    """Follow-ups need prior turns; stale ones must not linger."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    assert eng._conversation_history() == []          # nothing yet
    eng._remember_turn("capital of Norway?", "Oslo.")
    hist = eng._conversation_history()
    assert hist == [("user", "capital of Norway?"), ("assistant", "Oslo.")]

    # Only the last N turns ride along (tokens are latency).
    for i in range(6):
        eng._remember_turn(f"q{i}", f"a{i}")
    keep = eng.config["ask"]["conversation_turns"]
    assert len(eng._conversation_history()) == keep * 2
    assert eng._conversation_history()[-1] == ("assistant", "a5")

    # An idle gap wipes it — reviving an old thread is confusion, not context.
    eng._last_turn -= eng.config["ask"]["conversation_idle_seconds"] + 1
    assert eng._conversation_history() == []
    eng._remember_turn("q", "a")
    eng.reset_conversation()
    assert eng._conversation_history() == []


def test_answer_sends_history_as_chat_messages(config):
    """History must go as real chat turns, not glued into the prompt."""
    from app import answer, ollama

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "Copenhagen."}}

    with patch.object(ollama.requests, "post",
                      side_effect=lambda url, json=None, timeout=None:
                      (captured.update(json), FakeResp())[1]):
        answer.answer_question(
            "and Denmark?", config,
            history=[("user", "capital of Norway?"), ("assistant", "Oslo.")])

    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["messages"][-1]["content"] == "and Denmark?"


def test_speaker_interrupt_is_safe_when_idle():
    """Barge-in can fire at any moment, including when nothing is playing."""
    from app.tts import Speaker

    sp = Speaker.from_config({"engine": "kokoro"})
    sp.interrupt()                      # must not raise with no process
    assert sp._interrupted.is_set()

    class FakeProc:
        def __init__(self):
            self.killed = False

        def terminate(self):
            self.killed = True

    sp._proc = FakeProc()
    sp.interrupt()
    assert sp._proc.killed


def test_wake_stays_listening_while_marvin_speaks():
    """Barge-in only works if he is NOT deaf during playback — unlike while
    recording, where his own voice would start a fresh question."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    eng.wake = _wake()

    eng.set_speaking(True, object())
    assert eng._speaking is True
    assert eng.wake._suppressed is False        # listening THROUGH his voice

    eng.set_wake_suppressed(True)               # recording is different
    assert eng.wake._suppressed is True

    eng.set_speaking(False, None)
    assert eng._speaking is False
    assert eng._speaker is None

    # A quip starting while a RECORDING is active must not reopen the wake
    # word mid-question — the recording's suppression wins.
    eng.wake.set_suppressed(True)
    eng._active = True
    eng.set_speaking(True, object())
    assert eng.wake._suppressed is True
    eng._active = False
    eng.set_speaking(False, None)


class _FakeSpeaker:
    def __init__(self):
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


def test_stop_speaking_cuts_playback_and_is_safe_when_quiet():
    """Every 'shut up' path funnels through stop_speaking(): it must kill the
    playback when he's talking and be a calm no-op when he isn't."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    assert eng.stop_speaking() is False        # quiet: nothing to stop

    sp = _FakeSpeaker()
    eng.set_speaking(True, sp)
    assert eng.stop_speaking() is True
    assert sp.interrupted
    assert eng._speaking is False
    assert eng.stop_speaking() is False        # already stopped: no double-kill


def test_starting_any_recording_cuts_marvin_off():
    """Recording over his own playback would put HIS voice in the user's
    dictation — a new recording of either kind must silence him first."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    for kind in ("dictate", "ask"):
        eng = PushToTalkApp(_lc())
        sp = _FakeSpeaker()
        eng.set_speaking(True, sp)
        with patch.object(eng, "on_press"):
            eng._start_recording(kind)
        assert sp.interrupted, f"{kind} recording left him talking"
        assert eng._speaking is False


def test_pill_click_stops_speech_instead_of_cycling_the_model():
    """Poking Marvin mid-monologue means 'shut up', and must NOT also switch
    the STT model out from under the user."""
    from app.menubar import MenuBarApp

    class Eng:
        def __init__(self, talking):
            self._talking = talking

        def stop_speaking(self):
            was = self._talking
            self._talking = False
            return was

    class Stub:
        cycled = 0

        def cycle_model(self):
            self.cycled += 1

    s = Stub()
    s._pill_clicked = MenuBarApp._pill_clicked.__get__(s)

    s.engine = Eng(talking=True)
    s._pill_clicked()
    assert s.cycled == 0            # he was talking: stop only

    s.engine = Eng(talking=False)
    s._pill_clicked()
    assert s.cycled == 1            # he was quiet: normal model cycle


def test_wake_uses_lower_threshold_while_marvin_speaks():
    """His own voice on the speakers is exactly when the user's 'Hey Marvin'
    is hardest to hear — and a false positive then only cuts his own answer
    short. Barge mode must accept a lower score, and switch back cleanly."""
    import app.wakeword as W

    def fired_with(barge):
        w = _wake(threshold=0.7)
        w.set_barge_mode(barge)
        # A shout scoring 0.6: below the normal 0.7, above barge's 0.55. The
        # _pump helper mirrors the worker, so pick the threshold as it does.
        model = _FakeModel([0.6])
        w._model = model
        import numpy as np
        fired = []
        for _ in range(W.WINDOW_FRAMES):
            w.feed(np.zeros(W.HOP, dtype="float32"))
            hop = w._take(W.HOP)
            if hop is None:
                continue
            w._window.append(hop)
            if len(w._window) < W.WINDOW_FRAMES:
                continue
            score = max(model.predict(np.concatenate(w._window)).values())
            thr = w.barge_threshold if w._barge else w.threshold
            if score >= thr:
                fired.append(score)
        return fired

    assert fired_with(barge=False) == []       # normal: 0.6 is a near miss
    assert fired_with(barge=True) == [0.6]     # speaking: 0.6 cuts him off


def test_tts_splits_multi_sentence_lines_for_streaming():
    """Answers stream sentence by sentence so playback starts ~1s sooner. Short
    fragments merge forward — synthesizing "Oslo." alone costs nearly as much
    as a full sentence and makes delivery choppy for no latency gain."""
    from app.tts import _split_sentences

    assert _split_sentences("") == []
    assert _split_sentences("One sentence only.") == ["One sentence only."]
    # A short opener merges into the next sentence rather than standing alone.
    assert _split_sentences("Oslo. It hardly matters.") == [
        "Oslo. It hardly matters."]
    # A long enough first sentence streams on its own.
    assert _split_sentences(
        "The answer is 51, a trivial calculation. Ask me something harder."
    ) == ["The answer is 51, a trivial calculation.", "Ask me something harder."]


def test_wake_feed_copies_the_callback_buffer():
    """PortAudio reuses the buffer it passes the callback, so feed() MUST copy.
    Keeping a view meant the worker scored recycled memory: full windows,
    near-zero rms, every score 0.00, and the wake word never fired."""
    import numpy as np

    import app.wakeword as W

    w = _wake()
    scratch = np.ones(W.HOP, dtype="float32")   # the reused callback buffer
    w.feed(scratch)
    scratch[:] = 0.0                            # PortAudio overwrites it
    got = w._take(W.HOP)
    assert got is not None
    assert got.max() == 1.0, "feed() kept a view instead of copying"


def test_wake_drops_oldest_audio_when_worker_falls_behind():
    """A busy machine must not grow the queue without bound."""
    import numpy as np

    import app.wakeword as W

    w = _wake()
    for _ in range(W._MAX_CHUNKS * 3):   # far more than the cap
        w.feed(np.zeros(W.HOP, dtype="float32"))
    assert len(w._q) <= W._MAX_CHUNKS
    assert w._dropped > 0                # and it noticed, rather than silently


def test_wake_feed_never_blocks_the_audio_thread():
    """feed() runs on the realtime audio callback. If it can block on anything
    the worker holds, PortAudio drops input — measured at ~68% of realtime,
    which fragments the window into gibberish. So it must take no lock."""
    import threading

    import numpy as np

    import app.wakeword as W

    w = _wake()
    assert not hasattr(w, "_lock"), "feed()/_take() must not share a lock"

    # Hammer feed() from one thread while another drains, and assert every
    # sample survives in order — the property the lock used to provide.
    total = 300
    done = threading.Event()
    drained = []

    def drain():
        while not done.is_set() or len(w._q):
            hop = w._take(W.HOP)
            if hop is not None:
                drained.append(hop[0])

    t = threading.Thread(target=drain)
    t.start()
    for i in range(total):
        w.feed(np.full(W.HOP, float(i), dtype="float32"))
    done.set()
    t.join(timeout=5)
    assert drained == sorted(drained), "audio arrived out of order"


def test_wake_only_scores_complete_two_second_windows():
    """WakeWordModel is stateless and returns 0 for anything under ~2s, so the
    worker must buffer a full window before its first predict()."""
    import app.wakeword as W

    w = _wake()
    model = _FakeModel([])
    _pump(w, model, hops=W.WINDOW_FRAMES - 1)
    assert model.windows == []                    # not enough audio yet
    _pump(w, model, hops=1)
    assert model.windows == [W.HOP * W.WINDOW_FRAMES]   # exactly 2s


def test_wake_fires_above_threshold_and_debounces():
    import app.wakeword as W

    w = _wake(threshold=0.5, debounce=99.0)
    # One score per full window: below, then above, then above again.
    fired = _pump(w, _FakeModel([0.1, 0.9, 0.95]),
                  hops=W.WINDOW_FRAMES * 3)
    assert fired == [0.9]            # the later 0.95 is inside the debounce


def test_wake_is_deaf_while_suppressed():
    """While Marvin records or speaks, his own voice must not wake him."""
    import app.wakeword as W

    w = _wake(threshold=0.5)
    w.set_suppressed(True)
    assert _pump(w, _FakeModel([0.99]), hops=W.WINDOW_FRAMES * 2) == []
    w.set_suppressed(False)
    assert _pump(w, _FakeModel([0.99]), hops=W.WINDOW_FRAMES) == [0.99]


def test_wake_tracks_loudness_for_the_silence_endpoint():
    """rms drives the hands-free endpoint, so it must follow the audio even
    while detection is suppressed."""
    import numpy as np

    import app.wakeword as W

    w = _wake()
    w.set_suppressed(True)
    w._model = _FakeModel([])
    for _ in range(W.WINDOW_FRAMES):
        w.feed(np.full(W.HOP, 0.5, dtype="float32"))
        hop = w._take(W.HOP)
        w.rms = float(np.sqrt(np.mean(np.square(hop))))
    assert w.rms == pytest.approx(0.5, abs=1e-3)


def test_wake_refuses_broken_onnxruntime():
    """onnxruntime < 1.28 computes the speech embeddings WRONG (mel and
    classifier are fine, so nothing looks broken) and every score collapses to
    ~0.002 — the wake word silently never fires. Refuse rather than pretend."""
    from app.wakeword import check_onnxruntime

    for bad in ("1.27.0", "1.17.3", "0.9"):
        with pytest.raises(RuntimeError, match="1.28"):
            check_onnxruntime(bad)
    for ok in ("1.28.0", "1.29.1", "2.0.0"):
        check_onnxruntime(ok)          # must not raise
    check_onnxruntime("weird-build")   # unparseable: don't block the user


def test_wake_missing_model_disables_itself_quietly():
    """A missing .onnx must switch the feature off, not crash the app."""
    w = _wake()
    w._run()                          # loads "unused.onnx" -> fails
    assert w.available is False


def test_recorder_frame_tap_sees_audio_when_not_capturing():
    """The wake word shares the recorder's always-open stream, so the tap has
    to fire for frames that dictation itself discards."""
    import numpy as np

    from app.audio import Recorder

    r = Recorder()
    seen = []
    r.on_frame = seen.append
    r._dispatch(np.zeros((512, 1), dtype="float32"))
    assert len(seen) == 1             # not capturing, but the tap still ran
    assert r._frames == []            # ...and nothing was retained


def test_recorder_broken_tap_does_not_kill_the_mic():
    import numpy as np

    from app.audio import Recorder

    r = Recorder()
    r.on_frame = lambda frame: 1 / 0
    r._dispatch(np.zeros((512, 1), dtype="float32"))
    assert r.on_frame is None         # detached, mic callback survived


def test_hands_free_recording_has_no_key_for_the_watchdog():
    """A wake-word take has no key held. If it advertised the ask hotkey, the
    watchdog would see that key up and kill the recording instantly."""
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    with patch.object(eng, "on_press"):
        eng._start_recording("ask", hands_free=True)
    assert eng._hands_free is True
    assert eng._active_trigger_vk is None
    assert eng._active_rmod_flag is None


def test_wakeword_off_by_default_and_toggleable():
    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    cfg = _lc()
    assert cfg["wakeword"]["enabled"] is False   # opt-in, always
    eng = PushToTalkApp(cfg)
    assert eng.start_wakeword() is None          # disabled -> nothing runs
    assert eng.recorder.on_frame is None
    # Turning it on with no model installed must fail soft, not raise.
    with patch("app.wakeword.default_model_path", return_value=None):
        assert eng.set_wakeword_enabled(True) is None
    eng.set_wakeword_enabled(False)
    assert eng.wake is None


# ---------------------------------------- big-stack workers (native crash fix)

def test_worker_pools_get_big_stacks_and_restore_the_global():
    """MLX's compiler recursion blew the 512KB default worker stack (SIGBUS,
    app dead, no traceback). Workers must spawn under the 16MB setting, and
    the process-wide stack_size must be restored afterwards."""
    import threading

    from app import threads as T

    calls = []
    real = threading.stack_size

    def recorder(size=None):
        calls.append(size)
        return real(size) if size is not None else real()

    before = real()
    with patch.object(T.threading, "stack_size", side_effect=recorder):
        pool = T.single_worker_pool("test-pool")
        assert pool.submit(lambda: 42).result() == 42   # worker actually works
    pool.shutdown(wait=True)
    assert calls[0] == T.STACK_BYTES        # set big before spawning...
    assert calls[-1] == before              # ...restored after
    assert threading.stack_size() == before

    ran = []
    t = T.start_thread(lambda: ran.append(1), name="test-thread")
    t.join(timeout=5)
    assert ran == [1]
    assert threading.stack_size() == before


# ------------------------------------------- file transcription (drop on Marvin)

def test_filter_audio_paths_keeps_only_audio():
    from app.transcribe_file import filter_audio_paths

    got = filter_audio_paths(
        ["/a/take1.m4a", "/a/notes.txt", "/a/b.WAV", "/a/clip.mov",
         None, ""])                  # NSURL.path() can hand back None
    assert got == ["/a/take1.m4a", "/a/b.WAV"]
    assert filter_audio_paths([]) == []


def test_seam_chunks_cut_in_silence_and_lose_nothing():
    """Long files split near 60s, at the quietest 20ms — a fixed cut lands
    mid-word. And concatenating the chunks must reproduce the input exactly."""
    import numpy as np

    from app.transcribe_file import seam_chunks

    rate = 16000
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(rate * 130) * 0.1).astype("float32")
    silent = slice(int(58.5 * rate), int(59.5 * rate))
    audio[silent] = 0.0                      # the obvious place to cut

    chunks = seam_chunks(audio, rate)
    assert len(chunks) == 3                  # ~59s + ~60s + rest
    cut = len(chunks[0])
    assert silent.start <= cut <= silent.stop, "cut missed the silence"
    assert np.array_equal(np.concatenate(chunks), audio)

    short = audio[: rate * 30]
    assert [len(c) for c in seam_chunks(short, rate)] == [len(short)]
    assert seam_chunks(audio[:0], rate) == []


def test_write_output_single_multi_and_no_overwrite(tmp_path):
    from app.transcribe_file import write_output

    a = tmp_path / "intervju del1.m4a"
    b = tmp_path / "intervju del2.m4a"

    # Single file: '<stem>.txt', body is just the text.
    out, body = write_output([(str(a), "Hej världen.", 61.0)])
    assert out == tmp_path / "intervju del1.txt"
    assert out.read_text() == "Hej världen.\n" == body

    # Same name again: never silently overwrite.
    out2, _ = write_output([(str(a), "Andra tagningen.", 5.0)])
    assert out2 == tmp_path / "intervju del1 2.txt"
    assert out.read_text() == "Hej världen.\n"      # first is untouched

    # Dropped together: ONE combined file with a header per recording.
    out3, body3 = write_output(
        [(str(a), "Första delen.", 61.0), (str(b), "Andra delen.", 90.0)])
    assert out3 == tmp_path / "intervju del1 +1 filer.txt"
    assert "## intervju del1.m4a  (1:01)" in body3
    assert "## intervju del2.m4a  (1:30)" in body3
    assert body3.index("Första delen.") < body3.index("Andra delen.")


def test_transcribe_files_end_to_end_with_wavs(tmp_path):
    """Real decode path (WAV), fake transcriber: combined output, artifact
    guard, vocab fixes — and NO LLM call (recordings stay faithful)."""
    import numpy as np

    from app import cleanup as cl
    from app.config import load_config as _lc
    from app.transcribe_file import transcribe_files
    from app.tts import write_wav

    rate = 16000
    paths = []
    for name in ("del1.wav", "del2.wav"):
        p = tmp_path / name
        write_wav(str(p), np.zeros(rate, dtype="float32"), rate)
        paths.append(str(p))

    class FakeT:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio):
            self.calls += 1
            # An in-token loop + a misheard word, per chunk.
            return "möte med olama " + "ertas" * 10

    cfg = _lc()
    cfg["vocabulary"]["fixes"] = {"olama": "Ollama"}
    with patch.object(cl, "ollama_clean") as llm:
        res = transcribe_files(paths, FakeT(), cfg)
    llm.assert_not_called()                      # faithful: no LLM rewrite
    out = tmp_path / "del1 +1 filer.txt"
    assert res["out_path"] == str(out)
    text = out.read_text()
    assert "Ollama" in text and "olama" not in text.replace("Ollama", "")
    assert "ertasertas" not in text              # fused-loop guard applied
    assert res["n"] == 2 and abs(res["seconds"] - 2.0) < 0.01


def test_transcribe_files_reports_size_weighted_progress(tmp_path):
    """The ring must move smoothly: fractions monotonic 0->1, weighted by
    file size so a long file among short ones doesn't sprint-then-stall."""
    import numpy as np

    from app.config import load_config as _lc
    from app.transcribe_file import transcribe_files
    from app.tts import write_wav

    rate = 16000
    small = tmp_path / "kort.wav"
    big = tmp_path / "lang.wav"
    write_wav(str(small), np.zeros(rate, dtype="float32"), rate)      # 1s
    write_wav(str(big), np.zeros(rate * 3, dtype="float32"), rate)    # 3s

    class FakeT:
        def transcribe(self, audio):
            return "text"

    fracs = []
    transcribe_files([str(small), str(big)], FakeT(), _lc(),
                     on_fraction=fracs.append)
    assert fracs == sorted(fracs), "progress went backwards"
    assert fracs[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fracs)
    # After the small file (1s of 4s total bytes) the ring sits near 1/4,
    # NOT at 1/2 — that's the size weighting.
    after_small = fracs[len(fracs) // 2 - 1]  # last frac of file 1's chunks
    assert 0.2 <= 0.25 <= 0.35 or any(abs(f - 0.25) < 0.05 for f in fracs)


def test_pill_progress_ring_state():
    from app import pill as P

    Pill = getattr(P, "_Pill", None)
    if Pill is None:
        pytest.skip("AppKit unavailable: _Pill not defined")

    class Stub:
        progress = None
        rendered = 0

        def _render(self):
            self.rendered += 1

    s = Stub()
    s.set_progress = Pill.set_progress.__get__(s)
    s.set_progress(0.4)
    assert s.progress == 0.4 and s.rendered == 1
    s.set_progress(0.4)                 # unchanged: no re-render
    assert s.rendered == 1
    s.set_progress(None)                # job done: ring hidden
    assert s.progress is None and s.rendered == 2


def test_file_jobs_never_run_concurrently():
    """A drop during a running job must QUEUE, not interleave — two parallel
    jobs fight over file_progress and the ring jumps around (observed live
    when a 5th file was dropped mid-batch)."""
    import time as _t

    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp
    from app import transcribe_file as tf

    eng = PushToTalkApp(_lc())
    running = [0]
    overlap = []

    def slow_job(paths, *a, **k):
        running[0] += 1
        overlap.append(running[0])
        _t.sleep(0.15)
        running[0] -= 1
        return {"out_path": "x", "text": "", "n": 1, "seconds": 1.0}

    done = []
    with patch.object(tf, "transcribe_files", side_effect=slow_job):
        eng.transcribe_files_async(["a.wav"], on_done=done.append)
        eng.transcribe_files_async(["b.wav"], on_done=done.append)
        for _ in range(100):
            if len(done) == 2:
                break
            _t.sleep(0.05)
    assert len(done) == 2
    assert max(overlap) == 1, "two file jobs ran at the same time"


def test_engine_clears_file_progress_even_on_failure():
    """A failed job must not leave a frozen ring on screen forever."""
    import time as _t

    from app.config import load_config as _lc
    from app.hotkey import PushToTalkApp

    eng = PushToTalkApp(_lc())
    done = []
    eng.transcribe_files_async(["/does/not/exist.m4a"], on_done=done.append)
    for _ in range(100):
        if done:
            break
        _t.sleep(0.05)
    assert done == [None]               # failure reported...
    assert eng.file_progress is None    # ...and the ring is gone


def test_format_paragraphs_breaks_on_gaps_and_stamps():
    from app.transcribe_file import format_paragraphs

    segs = [(0.0, 4.0, "Första meningen."), (4.3, 8.0, "Fortsätter direkt."),
            (11.0, 14.0, "Nytt ämne efter paus."), (14.5, 15.0, "Mer.")]
    out = format_paragraphs(segs, gap_seconds=1.2, timestamps=True)
    paras = out.split("\n\n")
    assert len(paras) == 2                       # bruten vid 8.0 -> 11.0
    assert paras[0].startswith("[0:00] Första meningen. Fortsätter direkt.")
    assert paras[1].startswith("[0:11] Nytt ämne")
    plain = format_paragraphs(segs, gap_seconds=1.2, timestamps=False)
    assert "[0:" not in plain and plain.count("\n\n") == 1
    assert format_paragraphs([]) == ""


def test_transcribe_files_uses_segments_for_paragraphs(tmp_path):
    """Segment gaps from Whisper drive paragraph breaks, with times shifted
    to ABSOLUTE file time across the 60s chunk seams."""
    import numpy as np

    from app.config import load_config as _lc
    from app.transcribe_file import transcribe_files
    from app.tts import write_wav

    rate = 16000
    p = tmp_path / "möte.wav"
    write_wav(str(p), np.zeros(rate * 150, dtype="float32"), rate)  # 2.5 min

    class SegT:
        """Two chunks; each reports segments with a mid-chunk pause."""

        def __init__(self):
            self.n = 0

        def transcribe(self, audio):
            self.n += 1
            self.last_segments = [(0.0, 3.0, f"Block {self.n}A."),
                                  (10.0, 13.0, f"Block {self.n}B.")]
            return f"Block {self.n}A. Block {self.n}B."

    res = transcribe_files([str(p)], SegT(), _lc())
    text = (tmp_path / "möte.txt").read_text()
    assert "\n\n" in text                        # paragraphs, not a wall
    assert text.startswith("[0:00] Block 1A.")   # stamped (>=2 min)
    # Chunk 2's segments are shifted past the first seam (~60s), so its
    # paragraph stamp is in minute territory, not a duplicate [0:00].
    assert "[0:00] Block 2A." not in text
    assert res["n"] == 1


def test_answer_language_pinned_to_detected_question_language(config):
    """Observed live: en question -> sv answer and vice versa. In auto mode
    the STT-detected question language must become a HARD override."""
    from app.answer import ANSWER_SYSTEM, _system_prompt

    config["tts"]["language"] = "auto"
    assert "entire reply in Swedish" in _system_prompt(config, "sv")
    assert "entire reply in English" in _system_prompt(config, "en")
    # A fixed menu choice beats the detection.
    config["tts"]["language"] = "en"
    assert "entire reply in English" in _system_prompt(config, "sv")
    # Unknown detection: no override, keep the polite instruction only.
    config["tts"]["language"] = "auto"
    assert _system_prompt(config, "zh") == ANSWER_SYSTEM
    assert _system_prompt(config, None) == ANSWER_SYSTEM


def test_maybe_go_offline_requires_every_model(tmp_path, monkeypatch):
    """Offline mode only when ALL needed repos are cached — a half-cached
    install must stay online so first-run downloads still work."""
    import os

    from app.__main__ import maybe_go_offline, required_hf_repos
    from app.config import load_config as _lc

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    cfg = _lc()
    repos = required_hf_repos(cfg)
    assert len(repos) >= 3                       # whisper + kb + chatterbox…

    assert maybe_go_offline(cfg, hub_dir=tmp_path) is False   # nothing cached
    assert "HF_HUB_OFFLINE" not in os.environ

    for r in repos:                              # fake a full cache
        d = tmp_path / ("models--" + r.replace("/", "--")) / "snapshots" / "x"
        d.mkdir(parents=True)
    assert maybe_go_offline(cfg, hub_dir=tmp_path) is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


def test_decode_audio_rejects_garbage(tmp_path):
    from app.transcribe_file import decode_audio

    bad = tmp_path / "fake.m4a"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(ValueError, match="fake.m4a"):
        decode_audio(bad)


# ------------------------------------------------ Swedish STT (KB-Whisper)

def _routed_transcriber(**kw):
    """A Transcriber whose MLX decode is replaced by a recorder, so the routing
    can be tested without loading a model. The fake reports Swedish unless the
    caller forced another language, mimicking Whisper's auto-detect."""
    from app.stt import Transcriber

    opts = {"model": "large-v3-turbo", "engine": "mlx", "language": None,
            "swedish_model": "kb-large"}
    opts.update(kw)
    t = Transcriber(**opts)
    calls = []

    def fake_run(audio, model_name, language=None):
        calls.append((model_name, language))
        t.last_language = language or t.language or "sv"
        t.last_model = model_name
        return f"text from {model_name}"

    t._run_mlx = fake_run
    return t, calls


def test_swedish_model_reroutes_detected_swedish():
    """Auto-detect: the general model transcribes and detects, then Swedish
    earns a second pass on KB-Whisper (whose text is what we return)."""
    t, calls = _routed_transcriber()
    assert t.transcribe([0.0] * 16) == "text from kb-large"
    assert calls == [("large-v3-turbo", None), ("kb-large", "sv")]


def test_swedish_model_leaves_other_languages_alone():
    """A non-Swedish detection must not pay for a second decode."""
    t, calls = _routed_transcriber()
    t._run_mlx = lambda audio, model_name, language=None: (
        calls.append((model_name, language))
        or setattr(t, "last_language", "en") or "hello"
    )
    assert t.transcribe([0.0] * 16) == "hello"
    assert calls == [("large-v3-turbo", None)]


def test_forced_swedish_is_a_single_pass():
    """Svenska in the Language menu is known up front — go straight to the
    specialist rather than decoding twice."""
    t, calls = _routed_transcriber(language="sv")
    assert t.transcribe([0.0] * 16) == "text from kb-large"
    assert calls == [("kb-large", "sv")]


def test_forced_non_swedish_never_reroutes():
    t, calls = _routed_transcriber(language="en")
    t.transcribe([0.0] * 16)
    assert calls == [("large-v3-turbo", None)]


def test_swedish_model_off_uses_one_model_for_everything():
    t, calls = _routed_transcriber(swedish_model=None)
    assert t.transcribe([0.0] * 16) == "text from large-v3-turbo"
    assert calls == [("large-v3-turbo", None)]


def test_broken_swedish_model_keeps_the_general_transcript():
    """A missing/broken KB-Whisper must never cost the user the transcript the
    general model already produced — and stops being retried after two."""
    t, calls = _routed_transcriber()

    def fake_run(audio, model_name, language=None):
        calls.append((model_name, language))
        if model_name == "kb-large":
            raise OSError("repo not found")
        t.last_language = "sv"
        return "the general transcript"

    t._run_mlx = fake_run
    assert t.transcribe([0.0] * 16) == "the general transcript"
    assert t.swedish_model == "kb-large"     # one failure: still armed
    assert t.transcribe([0.0] * 16) == "the general transcript"
    assert t.swedish_model is None           # two: disabled for the session
    t.transcribe([0.0] * 16)
    assert calls[-1] == ("large-v3-turbo", None)  # no further retries


def test_swedish_models_are_reachable_repos():
    from app.stt import SWEDISH_MODELS, _MLX_REPOS

    for name in SWEDISH_MODELS:
        assert "/" in _MLX_REPOS[name]


def test_ollama_clean_sends_temperature_zero(config):
    import app.cleanup as cl

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

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
    for _label, key in MenuBarApp._SWEDISH_MODELS:
        eng.set_swedish_model(key)
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
