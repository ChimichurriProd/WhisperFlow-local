"""Speech-to-text.

Two engines:
- "mlx" (default): mlx-whisper, which runs on the Apple-Silicon GPU. On an
  M-series Mac this is ~25-50x faster than the CPU path — large-v3-turbo
  transcribes a short utterance in ~0.2s — so we default to the most accurate
  model and still feel instant.
- "faster-whisper": CTranslate2 on CPU. Portable fallback; much slower on Mac
  because it can't use the GPU.

Both bias toward custom-vocabulary terms via initial_prompt.
"""

# config model name -> mlx-community HF repo
_MLX_REPOS = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def build_initial_prompt(terms):
    """Turn a vocabulary term list into a Whisper initial_prompt string.

    Seeding the decoder with the terms biases it toward those exact spellings,
    so names/jargon/brands transcribe correctly. Returns None when empty.
    """
    terms = [t.strip() for t in (terms or []) if t and t.strip()]
    return ("Vocabulary: " + ", ".join(terms) + ".") if terms else None


class Transcriber:
    """Transcribes 16 kHz mono float32 audio to text. Models load lazily and
    are cached in memory, so only the first use per model pays the load cost.
    """

    def __init__(self, model="large-v3-turbo", engine="mlx", language=None,
                 vad_filter=True, device="cpu", compute_type="int8",
                 initial_prompt=None):
        self.model_name = model
        self.engine = engine
        self.language = language or None  # None/"" -> auto-detect per utterance
        self.vad_filter = vad_filter
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self._fw_model = None  # faster-whisper instance (lazy)

    def transcribe(self, audio):
        """audio: 1-D float32 at 16 kHz. Returns the joined transcript string."""
        if audio is None or len(audio) == 0:
            return ""
        if self.engine == "mlx":
            try:
                return self._transcribe_mlx(audio)
            except ImportError:
                print("[stt] mlx-whisper unavailable; using faster-whisper",
                      flush=True)
                self.engine = "faster-whisper"
        return self._transcribe_faster_whisper(audio)

    # ------------------------------------------------------------- mlx (GPU)

    def _transcribe_mlx(self, audio):
        import mlx_whisper  # lazy: pulls in MLX

        repo = _MLX_REPOS.get(self.model_name, _MLX_REPOS["large-v3-turbo"])
        opts = {"path_or_hf_repo": repo}
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        if self.language:
            opts["language"] = self.language
        result = mlx_whisper.transcribe(audio, **opts)
        text = (result.get("text") or "").strip()
        if self.language is None and text:
            print(f"[stt] detected language: {result.get('language')}", flush=True)
        return text

    # ------------------------------------------------------ faster-whisper (CPU)

    def _ensure_fw(self):
        if self._fw_model is None:
            from faster_whisper import WhisperModel  # lazy: slow import

            self._fw_model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._fw_model

    def _transcribe_faster_whisper(self, audio):
        model = self._ensure_fw()
        segments, info = model.transcribe(
            audio,
            language=self.language,
            vad_filter=self.vad_filter,
            initial_prompt=self.initial_prompt,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if self.language is None and text:
            print(f"[stt] detected language: {info.language} "
                  f"(p={info.language_probability:.2f})", flush=True)
        return text
