"""Hands-free "Hey Marvin": a wake word that opens an ask episode.

Runs on the microphone stream the Recorder ALREADY keeps open (see audio.py —
one persistent InputStream for the app's lifetime, because tearing PortAudio
streams up and down deadlocks CoreAudio's HAL). This is a tap on that stream,
not a second one, so enabling the wake word adds no new audio device handling
and no new permission: it costs one small ONNX inference per 80 ms of audio.

Threading: Recorder's callback runs on the realtime audio thread, so feed()
only enqueues. All inference happens on our own worker thread, and the
detection callback is handed to yet another thread so a slow handler can never
stall audio. Frames are scored and dropped; nothing is retained.

The detector is deliberately deaf while Marvin is RECORDING (the user's own
question must not retrigger him) — but it stays listening while he SPEAKS, so
"Hey Marvin" can barge in and cut an answer short. He never says the wake
phrase himself, and the model scores continuous unrelated speech near zero.
"""

import collections
import threading
import time

import numpy as np

# WakeWordModel is STATELESS: every predict() takes a complete ~2s window
# (exactly 16 embeddings for the classifier), and anything shorter scores zero.
# So we keep a sliding window and re-score it every hop, matching the reference
# listener in livekit.wakeword.inference.listener.
HOP = 1280                    # 80 ms at 16 kHz
WINDOW_FRAMES = 25            # 25 × 80 ms = the 2 s the model expects
# Callback blocks are small and variable (PortAudio picks the size), so bound
# the queue by CHUNK COUNT rather than samples — deque(maxlen=...) then evicts
# the oldest with no scan and no lock on the realtime thread.
_MAX_CHUNKS = 256


MIN_ONNXRUNTIME = (1, 28)


def check_onnxruntime(version):
    """Raise if *version* is one that computes the embeddings wrong.

    onnxruntime < 1.28 executes the bundled speech-embedding model INCORRECTLY.
    The mel frontend and the classifier are both fine on it, so nothing looks
    broken — but the embeddings come out wrong (measured: max abs error 36.8
    against a known-good run) and every score collapses to ~0.002, so the wake
    word silently never fires. Undetectable without a reference clip, hence
    this guard.
    """
    try:
        parsed = tuple(int(p) for p in version.split(".")[:2])
    except (AttributeError, ValueError):
        return  # unparseable version: assume the user knows what they're doing
    if parsed < MIN_ONNXRUNTIME:
        want = ".".join(str(p) for p in MIN_ONNXRUNTIME)
        raise RuntimeError(
            f"onnxruntime {version} computes the wake-word embeddings wrong "
            f"(needs >= {want}); detection would never fire. Fix: "
            f"<venv>/bin/python -m pip install 'onnxruntime>={want}'"
        )


class WakeWord:
    """Scores the mic stream for a wake phrase and fires on_detect().

    model_path: a livekit-wakeword / openWakeWord-compatible ONNX classifier.
    threshold:  score above which the phrase counts as spoken (0-1).
    debounce:   seconds to stay quiet after a detection, so one utterance
                cannot fire twice.
    """

    def __init__(self, model_path, on_detect, threshold=0.5, debounce=2.0,
                 sample_rate=16000, debug=False, dump_seconds=0):
        self.model_path = str(model_path)
        self.on_detect = on_detect
        self.threshold = float(threshold)
        self.debounce = float(debounce)
        self.sample_rate = sample_rate
        self.debug = bool(debug)  # log a heartbeat + near-misses (wakeword.debug)
        self.dump_seconds = dump_seconds  # save what it scored, for diagnosis
        self.overflow_source = None  # callable -> PortAudio overflow count
        self.rms = 0.0            # rolling loudness, drives the silence endpoint
        # Slow estimate of the room's noise floor, so "is he talking?" adapts to
        # the mic instead of trusting one hard-coded number. A quiet mic (low
        # macOS input volume) can put normal speech below a fixed threshold.
        self.ambient = 0.0
        self.last_score = 0.0     # most recent score (for the debug listener)
        self.available = None     # None = not started yet, False = model failed
        # Bounded queue between the audio callback and the worker. maxlen
        # gives drop-oldest backpressure for free, with no scan and no lock.
        self._q = collections.deque(maxlen=_MAX_CHUNKS)
        self._pending = []      # worker-side leftovers (worker thread only)
        self._pending_len = 0
        self._window = collections.deque(maxlen=WINDOW_FRAMES)
        self._stop = threading.Event()
        self._worker = None
        self._suppressed = False
        self._quiet_until = 0.0
        # Barge-in mode: while Marvin is SPEAKING, accept a lower score. His
        # voice on the speakers is competing with the user's shout at the mic
        # (worst SNR of any moment), and a false positive here only cuts his
        # own answer short — the cheapest false positive there is.
        self.barge_threshold = min(0.55, self.threshold)
        self._barge = False
        self._fed = 0        # samples handed in by the audio callback
        self._dropped = 0    # samples binned by the backpressure cap
        self._model = None

    # ---------------------------------------------------------------- control

    def start(self):
        """Begin scoring. The model loads on the worker thread, so a missing or
        broken ONNX file disables the feature instead of delaying startup."""
        if self._worker is not None:
            return
        self._stop.clear()
        # Big stack: onnxruntime inference runs here 12.5x/s; a native crash
        # on this thread has killed the whole app before (see app/threads.py
        # and the 2026-07-29 DiagnosticReports).
        from .threads import start_thread

        self._worker = start_thread(self._run, name="wakeword")

    def stop(self):
        self._stop.set()
        self._worker = None
        self._q.clear()
        self._pending = []
        self._pending_len = 0
        self._window.clear()

    def set_barge_mode(self, on):
        """While Marvin speaks, score against barge_threshold instead of
        threshold (see __init__ — interrupting him is cheap, missing the
        user shouting over his own voice is not)."""
        self._barge = bool(on)

    def set_suppressed(self, suppressed):
        """Mute detection (while recording or while Marvin speaks) without
        tearing anything down. Audio still flows so rms stays current."""
        self._suppressed = bool(suppressed)

    def hush(self, seconds=None):
        """Ignore detections for a while — used after a detection fires and
        after Marvin speaks, so his own voice can't retrigger him."""
        self._quiet_until = time.monotonic() + (
            self.debounce if seconds is None else float(seconds)
        )

    # ------------------------------------------------------------ audio input

    def feed(self, frame):
        """Called from the realtime audio callback: enqueue and return. No
        inference, no blocking.

        NOTHING here may block. This runs on the realtime audio thread, and
        PortAudio DROPS INPUT whenever the callback stalls — an earlier version
        took a lock that the worker held during np.concatenate, and the stream
        silently delivered only ~68% of realtime. The window then spanned ~3s
        of speech with samples missing, which is unintelligible to the model
        (and to Whisper). deque.append/popleft are atomic under the GIL, so
        producer and consumer never contend.

        The copy is REQUIRED, not an optimization to skip: PortAudio hands the
        callback a view onto a buffer it reuses the moment the callback
        returns, so anything kept without copying is overwritten before the
        worker thread reads it. Keeping a view here scored recycled memory —
        full windows, near-zero rms, every score 0.00. (np.asarray does NOT
        copy when the input is already float32; np.array does. The dictation
        path in audio.py copies for the same reason.)
        """
        arr = np.array(frame, dtype=np.float32).reshape(-1)
        if len(self._q) == self._q.maxlen:
            self._dropped += len(arr)   # deque drops the oldest for us
        self._q.append(arr)
        self._fed += len(arr)

    def _take(self, n):
        """Pop exactly n samples, or None if not enough have arrived yet.

        Runs on the worker thread only. Draining is deliberately split from
        feed(): all the concatenation cost lands here, never in the callback.
        """
        while self._pending_len < n:
            try:
                chunk = self._q.popleft()
            except IndexError:
                return None
            self._pending.append(chunk)
            self._pending_len += len(chunk)
        out = np.concatenate(self._pending)
        rest = out[n:]
        self._pending = [rest] if len(rest) else []
        self._pending_len = len(rest)
        return out[:n]

    # ----------------------------------------------------------------- worker

    def _load(self):
        import onnxruntime
        from livekit.wakeword import WakeWordModel

        # onnxruntime < 1.28 executes the bundled speech-embedding model
        # INCORRECTLY — the mel and the classifier are both fine, but the
        # embeddings come out wrong (measured: max abs error 36.8), so every
        # score collapses to ~0.002 and the wake word silently never fires.
        # This failure is invisible without a known-good clip to test against,
        # so refuse to run rather than pretend to listen.
        check_onnxruntime(onnxruntime.__version__)
        if self.sample_rate != 16000:
            # HOP/WINDOW_FRAMES are sample counts, and the model was trained at
            # 16 kHz — at any other rate the window is the wrong duration.
            print(f"[wake] audio.sample_rate is {self.sample_rate}, but the "
                  f"wake word needs 16000 — detection will be unreliable",
                  flush=True)
        model = WakeWordModel(models=[self.model_path])
        print(f"[wake] listening with {self.model_path}", flush=True)
        return model

    def _run(self):
        try:
            self._model = self._load()
            self.available = True
        except Exception as exc:
            self.available = False
            print(f"[wake] disabled — could not load the model "
                  f"({type(exc).__name__}: {exc})", flush=True)
            return
        hops = starved = 0
        peak = 0.0
        next_beat = time.monotonic() + 5.0
        # Debug capture: keep the exact audio the detector scored, so a failure
        # to fire can be checked against what the microphone really delivered
        # (transcribe it, re-score it offline) instead of guessed at.
        dump_secs = float(self.dump_seconds or 0)
        dumped = [] if dump_secs else None
        while not self._stop.is_set():
            hop = self._take(HOP)
            if self.debug and time.monotonic() >= next_beat:
                # Heartbeat: distinguishes "no audio reaching us" from "audio
                # fine, scores low" from "stuck suppressed" — the three ways
                # this silently does nothing.
                next_beat = time.monotonic() + 5.0
                fed, dropped = self._fed, self._dropped
                self._fed = self._dropped = 0
                print(f"[wake] {hops} hops, {starved} starved, rms {self.rms:.4f}, "
                      f"peak {peak:.2f}, suppressed={self._suppressed}, "
                      f"window {len(self._window)}/{WINDOW_FRAMES}, "
                      f"fed {fed/16000:.2f}s/5s DROPPED {dropped/16000:.2f}s, "
                      f"overflows {self.overflow_source() if self.overflow_source else -1}",
                      flush=True)
                hops = starved = 0
                peak = 0.0
            if hop is None:
                starved += 1
                time.sleep(0.01)
                continue
            hops += 1
            self.rms = r = float(np.sqrt(np.mean(np.square(hop))))
            # Track the noise floor from quiet hops only, so speech doesn't
            # drag the estimate up and desensitise the endpoint.
            if self.ambient <= 0.0:
                self.ambient = r
            elif r < self.ambient * 2.0:
                self.ambient = self.ambient * 0.98 + r * 0.02
            if dumped is not None:
                dumped.append(hop)
                if sum(len(h) for h in dumped) >= dump_secs * 16000:
                    self._write_dump(np.concatenate(dumped))
                    dumped = None
            if self._suppressed:
                # Stay deaf, and don't keep his own voice in the window either.
                self._window.clear()
                continue
            self._window.append(hop)
            if len(self._window) < WINDOW_FRAMES:
                continue  # not yet a full 2s window — the model would score 0
            try:
                scores = self._model.predict(np.concatenate(self._window))
            except Exception as exc:
                print(f"[wake] scoring failed, stopping: {exc!r}", flush=True)
                self.available = False
                return
            score = max(scores.values()) if scores else 0.0
            self.last_score = float(score)
            peak = max(peak, score)
            thr = self.barge_threshold if self._barge else self.threshold
            if score < thr or time.monotonic() < self._quiet_until:
                if self.debug and score >= 0.3:
                    print(f"[wake] near miss {score:.2f} "
                          f"(threshold {thr})", flush=True)
                continue
            self.hush()
            # Drop the window so the same utterance can't score twice as it
            # slides out of view.
            self._window.clear()
            print(f"[wake] heard it ({score:.2f})", flush=True)
            threading.Thread(
                target=self._fire, args=(float(score),), daemon=True
            ).start()

    def _write_dump(self, audio):
        """Save the audio the detector actually scored (wakeword.dump_seconds)."""
        try:
            from pathlib import Path

            from .tts import write_wav

            p = Path.home() / "Library" / "Logs" / "whisperflow-local" / "wake_dump.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            write_wav(str(p), audio, 16000)
            print(f"[wake] dumped {len(audio)/16000:.1f}s of scored audio to {p}",
                  flush=True)
        except Exception as exc:
            print(f"[wake] dump failed: {exc!r}", flush=True)

    def _fire(self, score):
        try:
            self.on_detect(score)
        except Exception as exc:
            print(f"[wake] handler failed: {exc!r}", flush=True)


def default_model_path():
    """The bundled 'Hey Marvin' classifier, or None when it isn't installed."""
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "assets" / "wakeword" / "hey_marvin.onnx"
    return p if p.exists() else None
