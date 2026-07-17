"""Microphone capture: hold the hotkey to record, release to get 16 kHz mono float32.

The InputStream is opened ONCE and kept running for the app's lifetime; each
dictation only toggles a capture flag. We deliberately never stop/close the
stream between utterances: stopping a live PortAudio stream tears down its
CoreAudio AudioUnit while the HAL's IO-proxy thread is mid-callback, and the two
deadlock on an inverted lock order (`AudioOutputUnitStop` parks forever in
`HALB_Mutex::Lock`, the app freezes stuck in "record", and the `on_release`
worker holds the busy lock for good so no later dictation can run). Keeping one
persistent stream removes that teardown from the hot path entirely.

Trade-off: the mic stays live (the macOS orange indicator stays on) the whole
time the app runs. Frames are discarded whenever we're not actively capturing,
so nothing is retained between dictations.
"""

import threading

import numpy as np

# One lock around every PortAudio stream *transition* in this process. CoreAudio's
# HAL deadlocks (lock-order inversion between the AudioUnit and the IO-proc mutex)
# when two threads open/close streams at once. We now open at most one stream and
# never close it on the hot path, but this still guards the lazy open and the
# shutdown-only close.
_PA_LOCK = threading.RLock()


class Recorder:
    """Push-to-talk recorder backed by a single persistent sounddevice.InputStream.

    The stream is opened lazily on the first start() and left running. The
    callback appends frames only while _capturing is set, so between dictations
    the mic stays live but nothing is retained. stop() just flips the flag off
    and returns whatever was captured — it never touches PortAudio, so it cannot
    deadlock CoreAudio's HAL.
    """

    def __init__(self, sample_rate=16000, channels=1, mic_gain=4.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.mic_gain = mic_gain
        self._frames = []
        self._lock = threading.Lock()
        self._stream = None
        self._capturing = False
        self.level = 0.0  # live 0..1 loudness, read by the waveform UI

    def _callback(self, indata, frames, time_info, status):
        # The stream runs continuously; drop frames unless a dictation is active.
        if not self._capturing:
            return
        with self._lock:
            self._frames.append(indata.copy())
        # Perceptual loudness for the waveform: sqrt curve makes normal speech
        # fill the bars, minus a small floor so ambient noise stays flat.
        rms = float(np.sqrt(np.mean(np.square(indata))))
        self.level = max(0.0, min(1.0, (rms ** 0.5) * self.mic_gain - 0.06))

    def _ensure_stream(self):
        """Open and start the persistent input stream if it isn't already running.

        Reopens if a previous stream went inactive (e.g. the audio device
        changed): an inactive stream isn't delivering callbacks, so tearing it
        down here won't race a live IO proc. This is the one place that still
        opens a PortAudio stream, so it runs under _PA_LOCK.
        """
        import sounddevice as sd  # lazy: keeps module importable without PortAudio

        with _PA_LOCK:
            if self._stream is not None and self._stream.active:
                return
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as exc:
                    print(f"[rec] stale stream close failed: {exc}", flush=True)
                self._stream = None
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def start(self):
        with self._lock:
            self._frames = []
        self._ensure_stream()
        self._capturing = True

    def stop(self):
        """Stop capturing and return the captured audio as 1-D float32.

        Only flips the capture flag and snapshots the buffered frames — the
        stream keeps running. No PortAudio teardown here, by design (see the
        module docstring): that teardown is what used to deadlock and freeze the
        app stuck in dictation.
        """
        self._capturing = False
        self.level = 0.0
        with self._lock:
            frames = self._frames
            self._frames = []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(frames, axis=0)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32)

    def close(self):
        """Tear the stream down. Shutdown-only — never call on the dictation hot
        path (that teardown is the CoreAudio deadlock this whole design avoids)."""
        self._capturing = False
        with _PA_LOCK:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:
                    print(f"[rec] stream close failed: {exc}", flush=True)


def load_wav(path, target_rate=16000):
    """Read a WAV file into 16 kHz mono float32 (for the dry-run path)."""
    import wave

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if rate != target_rate:
        # Linear resample; fine for speech fed to Whisper.
        duration = len(audio) / rate
        n_target = int(duration * target_rate)
        audio = np.interp(
            np.linspace(0.0, len(audio) - 1, n_target),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio
