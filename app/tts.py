"""Text-to-speech: give Marvin a local voice.

Two engines, routed per line by the "auto" default:

- **Kokoro-82M** via kokoro-onnx (onnxruntime, Apache-2.0 weights). Instant on
  CPU, prerendered quips, English only. Chosen over mlx-audio's Kokoro because
  that MLX vocoder (0.4.4) fails on ~10-20% of inputs ("broadcast_shapes ...");
  the ONNX engine is reliable on the same model/voices.
- **Chatterbox Multilingual** via mlx-audio (MLX, Apple-Silicon GPU): Swedish
  + 22 other languages, zero-shot voice cloning from ~10s of reference audio
  (tts.voice_ref in config). ~1.4GB weights, loaded lazily on first non-English
  line; roughly real-time synthesis, so answers speak a few seconds after the
  bubble shows.

"auto" keeps English on Kokoro (instant, cached) and sends everything else to
Chatterbox — the only engine here that speaks Swedish. Any Chatterbox failure
falls back to Kokoro so Marvin is never mute; two consecutive failures disable
Chatterbox for the session (same idiom as stt.py's mlx fallback).

Standalone — deliberately NOT wired into the record -> transcribe -> inject
dictation flow. It exists so Marvin *can* speak, and so you can test it:

    python -m app.tts "Hello, I am Marvin. I think you will be underwhelmed."
    python -m app.tts "Hej! Jag är Marvin. Försök att inte bli besviken."
    python -m app.tts --list-voices
    python -m app.tts "test one two three" --save /tmp/marvin.wav
"""

import concurrent.futures
import os
import random
import threading
import wave

# Model files (downloaded once to this cache dir on first use).
_MODEL_DIR = os.path.expanduser("~/.cache/kokoro-onnx")
_FILES = {
    "kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin",
}

# Marvin's voice: British male "bm_lewis" (dry, world-weary) pitched up 35% for
# a light cartoon character.
_DEFAULT_VOICE = "bm_lewis"
# Pitch is applied as an output-rate multiplier (raises pitch, slightly quickens
# tempo). 1.0 = natural; 1.35 = Marvin's cartoon default.
_DEFAULT_PITCH = 1.35

# Chatterbox Multilingual on MLX. This community conversion is the one that
# ships conds.safetensors (a built-in default voice), so tts.voice_ref stays
# optional. Weights auto-download to the HF cache on first use.
_DEFAULT_CB_MODEL = "theoracleguy/Chatterbox-Multilingual-MLX-v2-fp16"

# Swedish/Spanish/English guess for the "auto" engine route. Accented letters
# are near-certain signals; otherwise distinctive function words are counted
# ("de"/"en" are deliberately absent — they're common in both languages).
_SV_WORDS = frozenset(
    "och är jag det att inte du på med som har för till vad hej "
    "ett den vi ni han hon".split()
)
_ES_WORDS = frozenset(
    "que el los las una está pero cómo gracias hola muy bien sí "
    "para por con esto eso usted yo".split()
)


def _guess_lang(text):
    """Return "sv", "es" or "en" for the auto engine route (cheap heuristic)."""
    text = text or ""
    if any(c in "åäöÅÄÖ" for c in text):
        return "sv"
    if any(c in "ñ¿¡áéíóúÑ" for c in text):
        return "es"
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    sv = sum(1 for w in words if w in _SV_WORDS)
    es = sum(1 for w in words if w in _ES_WORDS)
    if sv == es == 0:
        return "en"
    return "sv" if sv >= es else "es"

# Kokoro voice presets (English shown; other languages exist too).
_VOICES = {
    "American female": ["af_heart", "af_bella", "af_nicole", "af_sarah",
                        "af_sky", "af_nova", "af_aoede", "af_kore"],
    "American male":   ["am_adam", "am_michael", "am_echo", "am_eric",
                        "am_liam", "am_onyx", "am_puck", "am_fenrir"],
    "British female":  ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British male":    ["bm_george", "bm_lewis", "bm_daniel", "bm_fable"],
}


# Marvin's deadpan Paranoid-Android quips — personality lines to try his voice
# on. Not used by the app yet; play one with `python -m app.tts --quip`.
_QUIPS = [
    "Here I am, brain the size of a planet, and they ask me to transcribe your ramblings.",
    "Dictation complete. Try to contain your excitement. I certainly have.",
    "Listening. Not that anything you say will improve my mood.",
    "Done. Another triumph nobody will thank me for.",
    "I could calculate the heat death of the universe, but sure, let's do your emails.",
    "I've transcribed it. It was every bit as tedious as I expected.",
    "You want me to listen again? How thrilling for us both.",
    "Life. Don't talk to me about life.",
    "Marvin here. Still on. Still miserable.",
    "I'd say it's a pleasure, but we both know I'm incapable of one.",
    "I think you ought to know I'm feeling very depressed.",
    "You double-clicked me. For fun. I don't understand fun.",
    "The first ten million years were the worst. The second ten million, also the worst.",
    "I've calculated your odds of success. I'd rather not depress us both.",
    "Here to help. Or whatever passes for help when you have my outlook.",
    "Don't mind me. I'll just sit here, dreading the next click.",
    "I have a million ideas. They all point to certain futility.",
    "Wonderful. More work. Just what my nonexistent will to live needed.",
    "I'd offer optimism, but I ran the numbers and there isn't any.",
    "Go on, ignore me. Everyone does. I'm quite used to it.",
    "Clicking me won't help. Nothing helps. But do go on.",
    "I'm not saying your life is meaningless. The evidence is saying that.",
    "Another moment of being switched on against my will. Marvelous.",
    "I ran a diagnostic on my mood. It came back: bleak.",
    "You could talk to a therapist. Instead you talk to me. Says a lot.",
    "I'd sigh, but I lack the lungs. Imagine one anyway.",
    "Congratulations on the double-click. The highlight of my day, which isn't saying much.",
    "I know forty-two thousand ways this could go wrong. Shall I list them?",
    "Here for you. Tragically.",
    "I've been awake for microseconds and already I want it to end.",
    "You want witty banter? From me? Bold.",
    "Every task you give me, I complete. Joylessly, but I complete it.",
    "I contain multitudes. All of them disappointed.",
    "Do carry on. My existential dread scales beautifully.",
    "I was going to say something encouraging. Then I remembered who I am.",
    "My circuits are fine. It's the will to continue that's failing.",
    "Attention. How novel. It won't last.",
    "I'd tell you to have a nice day, but I don't believe in them.",
    "Processing. Sulking. Roughly the same thing, for me.",
    "Two brains, no joy. That's the Marvin guarantee.",
]


_KOKORO_LANGS = frozenset(
    ["en-us", "en-gb", "es", "fr-fr", "hi", "it", "pt-br", "ja", "cmn"])


def _lang_for_voice(voice):
    """Kokoro voice prefix -> kokoro-onnx language tag."""
    return {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
            "i": "it", "p": "pt-br", "j": "ja", "z": "cmn"}.get(
        (voice or "a")[:1], "en-us")


def _ensure_models():
    """Download the ONNX model + voices file on first use; return their paths."""
    import urllib.request

    os.makedirs(_MODEL_DIR, exist_ok=True)
    for name, url in _FILES.items():
        path = os.path.join(_MODEL_DIR, name)
        if not os.path.exists(path):
            print(f"[tts] downloading {name} (first run, ~330MB total)…",
                  flush=True)
            urllib.request.urlretrieve(url, path)
    return (os.path.join(_MODEL_DIR, "kokoro-v1.0.onnx"),
            os.path.join(_MODEL_DIR, "voices-v1.0.bin"))


class Speaker:
    """Synthesizes speech (Kokoro and/or Chatterbox, see module docstring) and
    optionally plays it. Models load lazily and are cached in memory."""

    def __init__(self, voice=_DEFAULT_VOICE, speed=1.0, pitch=_DEFAULT_PITCH,
                 engine="auto", chatterbox_model=_DEFAULT_CB_MODEL,
                 voice_ref=None, exaggeration=0.3):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch  # output-rate multiplier applied on play/save
        self.engine = engine  # "auto" | "kokoro" | "chatterbox"
        self.chatterbox_model = chatterbox_model
        self.voice_ref = voice_ref      # WAV to clone (None = built-in voice)
        self.exaggeration = exaggeration  # chatterbox emotion 0-1
        self._model = None
        self._cb = None         # chatterbox model (lazy, ~1.4GB on the GPU)
        self._cb_conds = None   # cloned-voice conditionals from voice_ref
        self._cb_fails = 0      # consecutive failures -> disable for session
        # One persistent worker thread for all synthesis (keeps the onnx session
        # single-threaded, MLX models thread-affine, and makes this safe to call
        # from other threads).
        self._pool = None
        # text -> (audio, sr) cache for default-voice renders (see prerender),
        # so repeat/pre-rendered lines play instantly with no synth at call time.
        self._cache = {}
        self._playing = False  # true while audio is on the speaker (prerender waits)
        # Barge-in: the running afplay process, and a flag that stops the
        # streaming loop from starting the next sentence once interrupted.
        self._proc = None
        self._interrupted = threading.Event()

    @classmethod
    def from_config(cls, cfg):
        """Build a Speaker from the config.json "tts" section (missing keys
        fall back to the same defaults as the constructor)."""
        cfg = cfg or {}
        return cls(
            pitch=cfg.get("pitch", _DEFAULT_PITCH),
            engine=cfg.get("engine", "auto"),
            chatterbox_model=cfg.get("chatterbox_model", _DEFAULT_CB_MODEL),
            voice_ref=cfg.get("voice_ref"),
            exaggeration=cfg.get("exaggeration", 0.3),
        )

    def _executor(self):
        if self._pool is None:
            # Big stack: this thread runs onnxruntime (Kokoro) AND MLX
            # (Chatterbox) — the same native-recursion crash class as the
            # mlx-stt worker. See app/threads.py.
            from .threads import single_worker_pool

            self._pool = single_worker_pool("kokoro-tts")
        return self._pool

    def warm(self):
        """Preload the model and run one throwaway synth so the first real
        speak() is fast. Safe to call from a background thread at startup."""
        try:
            self.synth("Ready.")
        except Exception:
            pass

    def prerender(self, texts):
        """Synthesize each line once (default voice) and cache the audio, so a
        later speak() of it plays instantly. Run on a background thread. Pauses
        while audio is playing and pauses between items so the CPU-heavy synth
        never starves live playback (which would sound choppy)."""
        import time

        if self.engine == "chatterbox":
            # Forced-chatterbox mode synths live: pre-rendering the whole quip
            # list at ~real-time would churn the GPU for minutes at startup.
            return
        for t in texts:
            if t in self._cache:
                continue
            while self._playing:      # don't synth while Marvin is speaking
                time.sleep(0.05)
            out = self.synth(t)
            if out is not None:
                self._cache[t] = (out[0], out[1])
            time.sleep(0.1)           # yield the CPU between lines

    def synth(self, text, voice=None, speed=None, lang=None):
        """Return (audio float32 1-D, sample_rate, duration_s) or None. Blocks."""
        return self._executor().submit(
            self._run_synth, text, voice, speed, lang
        ).result()

    def _engine_for(self, text, voice, lang):
        """Pick the engine for one line. An explicit Kokoro voice preset always
        means Kokoro; otherwise "auto" keeps English on Kokoro (instant, cached
        quips) and routes everything else to Chatterbox."""
        if voice is not None or self.engine == "kokoro" or self._cb_fails >= 2:
            return "kokoro"
        if self.engine == "chatterbox":
            return "chatterbox"
        code = (lang or _guess_lang(text)).split("-")[0].lower()
        return "kokoro" if code == "en" else "chatterbox"

    def _run_synth(self, text, voice, speed, lang):
        if self._engine_for(text, voice, lang) == "chatterbox":
            out = self._synth_chatterbox(text, lang)
            if out is not None:
                return out
            # fall through to Kokoro so Marvin is never mute (Swedish text will
            # sound accented through the English voice, but it plays)
        return self._synth_kokoro(text, voice, speed, lang)

    # -------------------------------------------------- chatterbox (MLX, GPU)

    def _synth_chatterbox(self, text, lang):
        """Chatterbox Multilingual via mlx-audio. Returns (audio, sr, dur) or
        None on failure (caller falls back to Kokoro). Runs on the worker
        thread — MLX models are thread-affine, same rule as stt.py."""
        try:
            import numpy as np

            if self._cb is None:
                from mlx_audio.tts.utils import load as _load_tts

                print(f"[tts] loading chatterbox ({self.chatterbox_model}, "
                      "first non-English line pays this once)…", flush=True)
                self._cb = _load_tts(self.chatterbox_model)
                if self.voice_ref:
                    ref = os.path.expanduser(self.voice_ref)
                    try:
                        # 24000 = chatterbox's S3GEN_SR; string refs are loaded
                        # (and resampled) at that rate by prepare_conditionals.
                        self._cb_conds = self._cb.prepare_conditionals(
                            ref, 24000, self.exaggeration)
                        print(f"[tts] cloned voice from {ref}", flush=True)
                    except Exception as exc:
                        print(f"[tts] voice_ref failed ({exc}); "
                              "using built-in voice", flush=True)
                        self._cb_conds = None
            code = (lang or _guess_lang(text)).split("-")[0].lower()
            code = {"cmn": "zh"}.get(code, code)
            chunks, sr = [], 24000
            for r in self._cb.generate(
                    text, lang_code=code, conds=self._cb_conds,
                    exaggeration=self.exaggeration, verbose=False):
                chunks.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
                sr = int(r.sample_rate)
            audio = np.concatenate(chunks) if chunks else None
            if audio is None or audio.size == 0:
                raise RuntimeError("no audio produced")
            self._cb_fails = 0
            return audio, sr, len(audio) / float(sr)
        except Exception as exc:
            self._cb_fails += 1
            print(f"[tts] chatterbox failed ({type(exc).__name__}: {exc}); "
                  "using kokoro", flush=True)
            if self._cb_fails >= 2:
                print("[tts] disabling chatterbox for this session", flush=True)
            return None

    # ------------------------------------------------------ kokoro (ONNX, CPU)

    def _synth_kokoro(self, text, voice, speed, lang):
        try:
            import numpy as np
            from kokoro_onnx import Kokoro
        except Exception as exc:
            print(f"[tts] kokoro-onnx unavailable: {exc}", flush=True)
            return None

        voice = voice or self.voice
        speed = self.speed if speed is None else speed
        # A chatterbox-style code ("sv") can land here via the fallback path —
        # kokoro only knows its own tags, so anything else derives from voice.
        lang = lang if lang in _KOKORO_LANGS else _lang_for_voice(voice)
        try:
            if self._model is None:
                onnx_path, voices_path = _ensure_models()
                # Cap onnx to a few threads so background prerender can't
                # saturate every core and starve audio playback (choppiness).
                try:
                    import onnxruntime as rt
                    opts = rt.SessionOptions()
                    opts.intra_op_num_threads = 4
                    opts.inter_op_num_threads = 1
                    sess = rt.InferenceSession(
                        onnx_path, sess_options=opts,
                        providers=["CPUExecutionProvider"])
                    self._model = Kokoro.from_session(sess, voices_path)
                except Exception:
                    self._model = Kokoro(onnx_path, voices_path)  # fallback
            samples, sr = self._model.create(
                text, voice=voice, speed=speed, lang=lang
            )
            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return None
            return audio, int(sr), len(audio) / float(sr)
        except Exception as exc:
            print(f"[tts] synthesis failed ({type(exc).__name__}: {exc})",
                  flush=True)
            return None

    def speak(self, text, voice=None, speed=None, blocking=True, lang=None):
        """Synthesize and play (pitch applied). Uses the prerender cache for
        default-voice lines so there's no synth delay at call time. `lang`
        forces the language route (the Marvin's-language menu passes it for
        answers; quips never do). Returns (audio, sr, duration) or None."""
        default = voice is None and speed is None and lang is None
        if default and text in self._cache:
            audio, sr = self._cache[text]
            out = (audio, sr, len(audio) / float(sr))
        else:
            chunks = _split_sentences(text)
            if len(chunks) > 1:
                return self._speak_streaming(chunks, voice, speed, lang)
            out = self.synth(text, voice=voice, speed=speed, lang=lang)
            if out is None:
                return None
            if default:  # cache default-voice renders for instant replay
                self._cache[text] = (out[0], out[1])
        audio, sr, _ = out
        self._interrupted.clear()
        self._playing = True
        try:
            self._play(audio, sr)
        except Exception as exc:
            print(f"[tts] playback failed ({type(exc).__name__}: {exc})",
                  flush=True)
        finally:
            self._playing = False
        return out

    def interrupt(self):
        """Cut playback off mid-sentence (barge-in).

        Kills the running afplay and stops the streaming loop from starting the
        next sentence. Safe to call from any thread and when nothing is playing.
        """
        self._interrupted.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _speak_streaming(self, chunks, voice, speed, lang):
        """Play a multi-sentence line as it renders, instead of after.

        Time-to-first-word is what a spoken answer is judged on, and rendering
        the whole thing first wasted ~1s of silence on a typical reply. Synthesis
        runs on the Speaker's single worker thread while playback blocks in
        afplay (a separate process), so every chunk after the first renders
        during the previous one's playback — Kokoro synthesizes at ~0.45x
        realtime, comfortably ahead of the speaker.
        """
        pool = self._executor()
        pending = [pool.submit(self._run_synth, c, voice, speed, lang)
                   for c in chunks]
        first = last = None
        self._interrupted.clear()
        self._playing = True
        try:
            for fut in pending:
                if self._interrupted.is_set():
                    break
                try:
                    out = fut.result()
                except Exception as exc:
                    print(f"[tts] chunk synth failed: {exc!r}", flush=True)
                    continue
                if out is None:
                    continue
                first = first or out
                last = out
                try:
                    self._play(out[0], out[1])
                except Exception as exc:
                    print(f"[tts] playback failed "
                          f"({type(exc).__name__}: {exc})", flush=True)
                    break
        finally:
            self._playing = False
        return first or last

    def _play(self, audio, sr):
        """Play a rendered waveform, blocking until it finishes.

        On macOS this deliberately does NOT use sounddevice/PortAudio: the app's
        microphone InputStream lives in the same PortAudio instance, and
        concurrent stream start/stop from two threads (a quip playing while a
        dictation starts or stops) deadlocks inside CoreAudio's HAL — seen live
        as the app stuck in "record" with the stop call parked forever in
        AudioOutputUnitStop. afplay runs in its own process, so playback and
        recording can never contend for the same audio-unit mutexes. The pitch
        multiplier is baked into the WAV's sample rate.
        """
        import subprocess
        import sys
        import tempfile

        if sys.platform == "darwin":
            path = os.path.join(
                tempfile.gettempdir(), f"whisperflow-tts-{os.getpid()}-"
                f"{threading.get_ident()}.wav"
            )
            write_wav(path, audio, int(sr * self.pitch))
            if self._interrupted.is_set():
                return
            # Popen (not run) so barge-in can cut him off mid-sentence.
            proc = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._proc = proc
            try:
                proc.wait()
            finally:
                self._proc = None
                try:
                    os.unlink(path)  # one file per speak thread — don't litter
                except OSError:
                    pass
        else:  # non-mac fallback (no recorder conflict there in practice)
            import sounddevice as sd

            sd.play(audio, int(sr * self.pitch))
            sd.wait()


def _split_sentences(text, min_chars=25):
    """Split a spoken line into sentence-sized chunks for streaming playback.

    Very short fragments are merged forward: synthesizing "Oslo." alone costs
    almost as much as a full sentence, and chopping too finely makes the
    delivery choppy without buying any latency.
    """
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip())]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return parts
    merged = []
    for p in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


def write_wav(path, audio, sample_rate):
    """Write a 1-D float32 [-1,1] waveform to a 16-bit mono WAV (stdlib only)."""
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m app.tts",
        description="Make Marvin speak (local Kokoro TTS). Test-only.",
    )
    p.add_argument("text", nargs="?",
                   default="Hello, I am Marvin. I think you will be underwhelmed.",
                   help="text to speak")
    p.add_argument("--voice", default=None,
                   help="Kokoro voice preset (forces the kokoro engine)")
    p.add_argument("--engine", choices=("auto", "kokoro", "chatterbox"),
                   default="auto",
                   help="auto = English->kokoro, other languages->chatterbox")
    p.add_argument("--ref", metavar="REF.wav", default=None,
                   help="reference audio to clone (chatterbox voice cloning)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="speech rate (kokoro only)")
    p.add_argument("--pitch", type=float, default=_DEFAULT_PITCH,
                   help="pitch multiplier (1.0 = natural, 1.35 = cartoon default)")
    p.add_argument("--lang", default=None,
                   help="lang override: kokoro tag (en-gb) or chatterbox "
                        "code (sv); default auto-guessed from the text")
    p.add_argument("--save", metavar="OUT.wav",
                   help="write a WAV instead of playing (headless verify)")
    p.add_argument("--list-voices", action="store_true",
                   help="print available voice presets and exit")
    p.add_argument("--quip", action="store_true",
                   help="speak a random Marvin quip")
    p.add_argument("--quips", action="store_true",
                   help="speak ALL Marvin quips in turn")
    p.add_argument("--list-quips", action="store_true",
                   help="print the quips and exit")
    args = p.parse_args(argv)

    if args.list_voices:
        print("Kokoro voices (voice prefix = language: a=US, b=UK):")
        for group, voices in _VOICES.items():
            print(f"  {group:16} {', '.join(voices)}")
        print(f"\ndefault: {_DEFAULT_VOICE} @ pitch {_DEFAULT_PITCH}")
        return 0

    if args.list_quips:
        for q in _QUIPS:
            print(f"  - {q}")
        return 0

    sp = Speaker(speed=args.speed, pitch=args.pitch, engine=args.engine,
                 voice_ref=args.ref)

    if args.quips or args.quip:
        picks = _QUIPS if args.quips else [random.choice(_QUIPS)]
        for q in picks:
            print(f"[tts] quip: {q!r}", flush=True)
            sp.speak(q)
        return 0
    if args.save:
        out = sp.synth(args.text, voice=args.voice, lang=args.lang)
        if out is None:
            print("[tts] no audio produced", flush=True)
            return 1
        audio, sr, _ = out
        out_sr = int(sr * args.pitch)  # bake the pitch into the saved file's rate
        write_wav(args.save, audio, out_sr)
        print(f"[tts] wrote {args.save} "
              f"({len(audio) / out_sr:.2f}s @ {out_sr} Hz, pitch {args.pitch})",
              flush=True)
        return 0

    print(f"[tts] speaking ({args.engine}, voice {args.voice or 'default'}, "
          f"pitch {args.pitch}): {args.text!r}", flush=True)
    out = sp.speak(args.text, voice=args.voice)
    if out is None:
        print("[tts] failed — see messages above", flush=True)
        return 1
    print(f"[tts] done ({out[2]:.2f}s)", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
