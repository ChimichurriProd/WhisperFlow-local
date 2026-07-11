"""Microphone capture: start on hotkey press, stop on release, return 16 kHz mono float32."""

import threading
import time

import numpy as np

# One lock around every PortAudio stream transition in this process. CoreAudio's
# HAL deadlocks (lock-order inversion between the AudioUnit and IO-proc mutexes)
# when two threads start/stop streams at the same time — observed live as the
# app stuck in "record" with the stop call parked forever in AudioOutputUnitStop.
_PA_LOCK = threading.RLock()

# Stopping a stream inside its start-notification window races the same
# CoreAudio callback; a very quick hotkey tap can hit it. Enforce a minimum age.
_MIN_STREAM_AGE = 0.25  # seconds


class Recorder:
    """Push-to-talk recorder backed by sounddevice.InputStream.

    Frames accumulate in a list while the stream runs; stop() concatenates
    them into a single float32 mono array at the configured sample rate.
    """

    def __init__(self, sample_rate=16000, channels=1, mic_gain=4.5):
        self.sample_rate = sample_rate
        self.channels = channels
        self.mic_gain = mic_gain
        self._frames = []
        self._lock = threading.Lock()
        self._stream = None
        self._started_at = 0.0
        self.level = 0.0  # live 0..1 loudness, read by the waveform UI

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._frames.append(indata.copy())
        # Perceptual loudness for the waveform: sqrt curve makes normal speech
        # fill the bars, minus a small floor so ambient noise stays flat.
        rms = float(np.sqrt(np.mean(np.square(indata))))
        self.level = max(0.0, min(1.0, (rms ** 0.5) * self.mic_gain - 0.06))

    def start(self):
        import sounddevice as sd  # lazy: keeps module importable without PortAudio

        with self._lock:
            self._frames = []
        with _PA_LOCK:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        self._started_at = time.monotonic()

    def stop(self):
        """Stop the stream and return the captured audio as 1-D float32."""
        # Swap-then-stop so a second concurrent stop() (release event racing the
        # watchdog) finds None instead of stopping the same stream twice.
        stream, self._stream = self._stream, None
        if stream is not None:
            age = time.monotonic() - self._started_at
            if age < _MIN_STREAM_AGE:  # let a just-started stream finish starting
                time.sleep(_MIN_STREAM_AGE - age)
            with _PA_LOCK:
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:  # a broken stream must not wedge dictation
                    print(f"[rec] stream stop failed: {exc}", flush=True)
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
