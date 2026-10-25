"""Microphone capture: hold the hotkey to record, release to get 16 kHz mono float32.

The InputStream is opened ONCE and kept running for the app's lifetime; each
dictation only toggles a capture flag. We deliberately never stop/close the
stream between utterances: stopping a live PortAudio stream tears down its
CoreAudio AudioUnit while the HAL's IO-proxy thread is mid-callback, and the two
deadlock on an inverted lock order (`AudioOutputUnitStop` parks forever in
`HALB_Mutex::Lock`, the app freezes stuck in "record", and the `on_release`
worker holds the busy lock for good so no later dictation can run). Keeping one
persistent stream removes that teardown from the hot path entirely.

Audio is pulled by a dedicated reader thread doing blocking reads, NOT by a
PortAudio callback: a Python callback needs the GIL, and this app's AppKit main
thread holds it often enough to starve one (measured: 16% of realtime delivered,
with PortAudio reporting no overflow because the samples never reached Python).
See the Recorder docstring.

Trade-off: the mic stays live (the macOS orange indicator stays on) the whole
time the app runs. Frames are discarded whenever we're not actively capturing,
so nothing is retained between dictations.
"""

import re
import subprocess
import sys
import threading

import numpy as np

# One lock around every PortAudio stream *transition* in this process. CoreAudio's
# HAL deadlocks (lock-order inversion between the AudioUnit and the IO-proc mutex)
# when two threads open/close streams at once. We now open at most one stream and
# never close it on the hot path, but this still guards the lazy open and the
# shutdown-only close.
_PA_LOCK = threading.RLock()

# Samples per blocking read (~80 ms at 16 kHz). Matches the wake word's hop so
# its sliding window advances one read at a time.
READ_BLOCK = 1280

# audio.input_device values besides an exact device name.
BUILTIN = "builtin"   # the Mac's own microphone, whatever else is plugged in
DEFAULT = "default"   # follow macOS's input setting

# The Mac's own mic across models ("MacBook Pro Microphone", "MacBook Air
# Microphone", "iMac Microphone", older "Built-in Microphone").
_BUILTIN_RE = re.compile(r"^(macbook( pro| air)?|imac|built-in) microphone",
                         re.IGNORECASE)


def is_builtin(name):
    return bool(name) and bool(_BUILTIN_RE.match(name))


def lid_closed():
    """True when a MacBook's lid is shut. The built-in mic is then cut off in
    hardware but CoreAudio still lists it, so "builtin" must not pick it."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return False
    return '"AppleClamshellState" = Yes' in out


def input_devices():
    """[(index, name)] of every input device PortAudio knows about.

    PortAudio snapshots the device list when it initializes, so a mic plugged
    in after launch only shows up after rescan_devices(). Under _PA_LOCK so a
    menu listing never queries PortAudio while a rescan tears it down."""
    import sounddevice as sd

    with _PA_LOCK:
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]


def resolve_input_device(spec, devices, lid_is_closed=False):
    """Pick the device index for an audio.input_device setting, or None for
    the system default.

    Why "builtin" is the default: macOS makes any newly connected audio
    interface the default input, so a Scarlett with nothing plugged into it
    silently hands Marvin pure silence (and Whisper hallucinates words out of
    it). Pinning the built-in mic keeps dictation working whatever is
    connected; a missing device (or a shut lid) falls back to the system
    default.
    """
    if not spec or spec == DEFAULT:
        return None
    if spec == BUILTIN:
        if lid_is_closed:
            return None
        return next((i for i, n in devices if is_builtin(n)), None)
    return next((i for i, n in devices if n == spec), None)


def rescan_devices():
    """Re-read the device list (hot-plugged mics). Only safe while this
    process has NO open PortAudio stream — callers hold _PA_LOCK with the
    recorder's stream closed."""
    import sounddevice as sd

    sd._terminate()
    sd._initialize()


class Recorder:
    """Push-to-talk recorder backed by a single persistent sounddevice.InputStream.

    The stream is opened lazily on the first start() and left running, and a
    dedicated reader thread pulls from it. Frames are kept only while
    _capturing is set, so between dictations the mic stays live but nothing is
    retained. stop() just flips the flag off and returns whatever was captured
    — it never touches PortAudio, so it cannot deadlock CoreAudio's HAL.

    Why a reader thread and NOT a PortAudio callback: sounddevice's callback is
    Python, so it needs the GIL. This app renders Marvin on the AppKit main
    thread at 20 Hz, and while that holds the GIL the callback simply is not
    scheduled — measured delivery fell to 16% of realtime, with PortAudio
    reporting NO overflow because the samples never reached Python at all. The
    audio kept its nominal rate while losing a third of its samples, which
    silently corrupts anything assuming continuity (the wake word could not
    recognise a phrase, and Whisper transcribed the gaps as gibberish).
    Blocking reads move the buffering into PortAudio's C ring buffer, where a
    stalled GIL costs latency instead of data: measured 99%.
    """

    def __init__(self, sample_rate=16000, channels=1, mic_gain=4.5,
                 device=BUILTIN):
        self.sample_rate = sample_rate
        self.channels = channels
        self.mic_gain = mic_gain
        self.device = device      # audio.input_device setting
        self.device_name = None   # what the open stream actually uses
        self._frames = []
        self._lock = threading.Lock()
        self._stream = None
        self._reader_thread = None
        self._reader_stop = threading.Event()
        self._capturing = False
        self.level = 0.0  # live 0..1 loudness, read by the waveform UI
        # Optional tap: called with EVERY frame, capturing or not, so a
        # always-listening consumer (the wake word) can share this one stream
        # instead of opening a second one — see the module docstring for why a
        # second PortAudio stream is a bad idea. It runs on the reader thread,
        # so it should stay cheap: slow work here delays the next read.
        self.on_frame = None
        self.overflows = 0      # PortAudio input-overflow events (dropped audio)
        self.last_status = ""

    def _reader(self):
        """Pull audio off the stream on our own thread (see the class docstring
        for why this is not a PortAudio callback)."""
        while not self._reader_stop.is_set():
            stream = self._stream
            if stream is None:
                return
            try:
                data, overflowed = stream.read(READ_BLOCK)
            except Exception as exc:
                if not self._reader_stop.is_set():
                    print(f"[rec] reader stopped: {exc!r}", flush=True)
                return
            if overflowed:
                self.overflows += 1
            self._dispatch(data)

    def _dispatch(self, indata):
        tap = self.on_frame
        if tap is not None:
            try:
                tap(indata)
            except Exception:
                # A broken tap must never take the mic down with it.
                self.on_frame = None
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
            self._close_stream_locked()
            try:
                self._stream = self._open_stream(sd)
            except Exception as exc:
                # PortAudio's device list is a snapshot from launch: a mic
                # unplugged since then is still listed and fails to open.
                # The stream is closed here, so a rescan is safe; retry once.
                print(f"[rec] opening the mic failed ({exc}); rescanning "
                      f"devices", flush=True)
                rescan_devices()
                self._stream = self._open_stream(sd)
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._reader, name="mic-reader", daemon=True
            )
            self._reader_thread.start()
            actual = float(self._stream.samplerate)
            self.device_name = sd.query_devices(self._stream.device)["name"]
            print(f"[rec] mic stream: {actual:.0f} Hz, {self._stream.channels}ch, "
                  f"blocksize {self._stream.blocksize}, device "
                  f"{self.device_name!r} (setting: {self.device})", flush=True)
            if abs(actual - self.sample_rate) > 1:
                # Everything downstream assumes sample COUNTS map to time at
                # self.sample_rate (Whisper's 30s windows, the wake word's 2s
                # window). A stream at another rate silently distorts both.
                print(f"[rec] WARNING: asked for {self.sample_rate} Hz but got "
                      f"{actual:.0f} Hz — audio will be time-distorted",
                      flush=True)

    def _open_stream(self, sd):
        """Create and start the input stream for the current setting."""
        shut = self.device == BUILTIN and lid_closed()
        index = resolve_input_device(self.device, input_devices(), shut)
        if shut:
            print("[rec] lid closed — built-in mic is off, using the system "
                  "default", flush=True)
        elif index is None and self.device not in (None, DEFAULT):
            print(f"[rec] input device {self.device!r} not found — "
                  f"using the system default", flush=True)
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=index,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        return stream

    def _close_stream_locked(self):
        """Stop the reader and close the stream. Caller holds _PA_LOCK and
        guarantees no dictation is capturing (never on the hot path).

        Returns False if the reader did not exit (stuck in a read on a device
        that vanished) — PortAudio must then not be re-initialized under it."""
        self._reader_stop.set()
        thread, self._reader_thread = self._reader_thread, None
        clean = True
        if thread is not None:
            thread.join(timeout=1.0)
            clean = not thread.is_alive()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                print(f"[rec] stream close failed: {exc}", flush=True)
        return clean

    def set_device(self, spec):
        """Switch microphone (audio.input_device). Closes the stream, rescans
        so a mic plugged in after launch is found, and reopens it if it was
        running (the wake word keeps its tap). Callers make sure no dictation
        is capturing."""
        self.device = spec
        with _PA_LOCK:
            was_open = self._stream is not None
            clean = self._close_stream_locked()
            self.device_name = None
            if not clean:
                print("[rec] mic reader still stuck — skipping the device "
                      "rescan", flush=True)
            else:
                try:
                    rescan_devices()
                except Exception as exc:
                    print(f"[rec] device rescan failed: {exc!r}", flush=True)
        if was_open:
            self._ensure_stream()

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
            self._close_stream_locked()


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
