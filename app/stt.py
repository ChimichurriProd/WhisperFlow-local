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

import concurrent.futures

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
        self._mlx_pool = None  # single-thread executor for MLX (lazy)
        self._mlx_fails = 0    # consecutive MLX failures -> auto-disable
        self.last_language = None  # language of the most recent transcription
                                   # (forced language, else what STT detected) —
                                   # lets the ask flow skip TTS for non-English.

    def transcribe(self, audio):
        """audio: 1-D float32 at 16 kHz. Returns the joined transcript string."""
        if audio is None or len(audio) == 0:
            return ""
        if self.engine == "mlx":
            try:
                text = self._transcribe_mlx(audio)
                self._mlx_fails = 0
                return text
            except Exception as exc:
                # Never let an MLX failure (e.g. the GPU-stream threading error
                # "There is no Stream(gpu, N) in current thread") kill dictation:
                # fall back to the CPU engine for this utterance, and disable
                # MLX after repeated failures so we stop paying its retry cost.
                self._mlx_fails += 1
                print(f"[stt] mlx-whisper failed "
                      f"({type(exc).__name__}: {exc}); using faster-whisper",
                      flush=True)
                if isinstance(exc, ImportError) or self._mlx_fails >= 2:
                    print("[stt] disabling mlx for this session", flush=True)
                    self.engine = "faster-whisper"
        return self._transcribe_faster_whisper(audio)

    # ------------------------------------------------------------- mlx (GPU)

    def _transcribe_mlx(self, audio):
        # MLX arrays and GPU streams are thread-affine: a model cached on the
        # thread that first loaded it cannot be evaluated from another thread
        # ("There is no Stream(gpu, N) in current thread"). on_release spawns a
        # fresh thread per dictation, so we pin ALL MLX work to one persistent
        # worker thread and block on its result.
        if self._mlx_pool is None:
            self._mlx_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx-stt"
            )
        return self._mlx_pool.submit(self._run_mlx, audio).result()

    def _run_mlx(self, audio):
        import mlx_whisper  # lazy: pulls in MLX

        repo = _MLX_REPOS.get(self.model_name, _MLX_REPOS["large-v3-turbo"])
        opts = {"path_or_hf_repo": repo}
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        if self.language:
            opts["language"] = self.language
        result = mlx_whisper.transcribe(audio, **opts)
        text = (result.get("text") or "").strip()
        self.last_language = self.language or result.get("language")
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
        self.last_language = self.language or info.language
        if self.language is None and text:
            print(f"[stt] detected language: {info.language} "
                  f"(p={info.language_probability:.2f})", flush=True)
        return text
