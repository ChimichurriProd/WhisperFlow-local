"""Text-to-speech: give Marvin a local voice.

Kokoro-82M via **kokoro-onnx** (onnxruntime, Apache-2.0 weights). Fully local
and offline once the model files are cached. Chosen over mlx-audio because the
MLX Kokoro vocoder in mlx-audio 0.4.4 fails on ~10-20% of inputs
("broadcast_shapes ..."); the ONNX engine is reliable on the same model/voices.

Standalone — deliberately NOT wired into the record -> transcribe -> inject
dictation flow. It exists so Marvin *can* speak, and so you can test it:

    python -m app.tts "Hello, I am Marvin. I think you will be underwhelmed."
    python -m app.tts --list-voices
    python -m app.tts "test one two three" --save /tmp/marvin.wav

Kokoro speaks English (US/UK) + a few other languages — NOT Swedish.
"""

import concurrent.futures
import os
import random
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
    """Synthesizes speech with Kokoro (ONNX) and optionally plays it. The model
    loads lazily and is cached in memory after the first call."""

    def __init__(self, voice=_DEFAULT_VOICE, speed=1.0, pitch=_DEFAULT_PITCH):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch  # output-rate multiplier applied on play/save
        self._model = None
        # One persistent worker thread for all synthesis (keeps the onnx session
        # single-threaded and makes this safe to call from other threads later).
        self._pool = None
        # text -> (audio, sr) cache for default-voice renders (see prerender),
        # so repeat/pre-rendered lines play instantly with no synth at call time.
        self._cache = {}
        self._playing = False  # true while audio is on the speaker (prerender waits)

    def _executor(self):
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kokoro-tts"
            )
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

    def _run_synth(self, text, voice, speed, lang):
        try:
            import numpy as np
            from kokoro_onnx import Kokoro
        except Exception as exc:
            print(f"[tts] kokoro-onnx unavailable: {exc}", flush=True)
            return None

        voice = voice or self.voice
        speed = self.speed if speed is None else speed
        lang = lang or _lang_for_voice(voice)
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

    def speak(self, text, voice=None, speed=None, blocking=True):
        """Synthesize and play (pitch applied). Uses the prerender cache for
        default-voice lines so there's no synth delay at call time. Returns the
        raw (audio, sr, duration) tuple or None."""
        default = voice is None and speed is None
        if default and text in self._cache:
            audio, sr = self._cache[text]
            out = (audio, sr, len(audio) / float(sr))
        else:
            out = self.synth(text, voice=voice, speed=speed)
            if out is None:
                return None
            if default:  # cache default-voice renders for instant replay
                self._cache[text] = (out[0], out[1])
        audio, sr, _ = out
        self._playing = True
        try:
            self._play(audio, sr)
        except Exception as exc:
            print(f"[tts] playback failed ({type(exc).__name__}: {exc})",
                  flush=True)
        finally:
            self._playing = False
        return out

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
                tempfile.gettempdir(), f"whisperflow-tts-{os.getpid()}.wav"
            )
            write_wav(path, audio, int(sr * self.pitch))
            subprocess.run(
                ["afplay", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        else:  # non-mac fallback (no recorder conflict there in practice)
            import sounddevice as sd

            sd.play(audio, int(sr * self.pitch))
            sd.wait()


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
    p.add_argument("--voice", default=_DEFAULT_VOICE, help="Kokoro voice preset")
    p.add_argument("--speed", type=float, default=1.0, help="speech rate")
    p.add_argument("--pitch", type=float, default=_DEFAULT_PITCH,
                   help="pitch multiplier (1.0 = natural, 1.35 = cartoon default)")
    p.add_argument("--lang", default=None,
                   help="lang override (default derived from voice prefix)")
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

    sp = Speaker(voice=args.voice, speed=args.speed, pitch=args.pitch)

    if args.quips or args.quip:
        picks = _QUIPS if args.quips else [random.choice(_QUIPS)]
        for q in picks:
            print(f"[tts] quip: {q!r}", flush=True)
            sp.speak(q)
        return 0
    if args.save:
        out = sp.synth(args.text, lang=args.lang)
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

    print(f"[tts] speaking as {args.voice} (pitch {args.pitch}): {args.text!r}",
          flush=True)
    out = sp.speak(args.text, voice=args.voice, speed=args.speed)
    if out is None:
        print("[tts] failed — see messages above", flush=True)
        return 1
    print(f"[tts] done ({out[2]:.2f}s)", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
