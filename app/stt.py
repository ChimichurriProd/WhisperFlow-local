"""Speech-to-text: faster-whisper (CTranslate2) with Silero VAD gating silence."""


class Transcriber:
    """Wraps faster_whisper.WhisperModel; the model loads lazily on first use."""

    def __init__(self, model="tiny.en", device="cpu", compute_type="int8",
                 language=None, vad_filter=True):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language or None  # None/"" -> auto-detect per utterance
        self.vad_filter = vad_filter
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: slow import, big download

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio):
        """audio: 1-D float32 at 16 kHz. Returns the joined transcript string."""
        if audio is None or len(audio) == 0:
            return ""
        model = self._ensure_model()
        segments, info = model.transcribe(
            audio,
            language=self.language,
            vad_filter=self.vad_filter,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if self.language is None and text:
            print(f"[stt] detected language: {info.language} "
                  f"(p={info.language_probability:.2f})", flush=True)
        return text
