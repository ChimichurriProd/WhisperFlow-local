"""Speech-to-text.

Two engines:
- "mlx" (default): mlx-whisper, which runs on the Apple-Silicon GPU. On an
  M-series Mac this is ~25-50x faster than the CPU path — large-v3-turbo
  transcribes a short utterance in ~0.2s — so we default to the most accurate
  model and still feel instant.
- "faster-whisper": CTranslate2 on CPU. Portable fallback; much slower on Mac
  because it can't use the GPU.

Both bias toward custom-vocabulary terms via initial_prompt.

Swedish gets a specialist: see SWEDISH_MODELS and Transcriber.swedish_model.
"""

import concurrent.futures

# config model name -> mlx-community HF repo
_MLX_REPOS = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    # KB-Whisper (KBLab, fine-tuned on 50 000 h of Swedish): roughly a third
    # fewer Swedish errors than whisper-large-v3, and even the small one beats
    # it. Swedish-only — reached via swedish_model, never via the Model menu.
    # Community MLX conversions (no official mlx-community build exists).
    "kb-small": "Leonidng/kb-whisper-small-mlx",
    "kb-large": "jegeblad/kb-whisper-large-mlx-q8",
}

# Values accepted by stt.swedish_model ("Swedish accuracy" menu).
SWEDISH_MODELS = ("kb-small", "kb-large")


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
                 initial_prompt=None, swedish_model=None):
        self.model_name = model
        self.engine = engine
        self.language = language or None  # None/"" -> auto-detect per utterance
        self.vad_filter = vad_filter
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        # KB-Whisper name (see SWEDISH_MODELS) to handle Swedish with, or None
        # for "one model for every language". MLX engine only.
        self.swedish_model = swedish_model or None
        self._fw_model = None  # faster-whisper instance (lazy)
        self._mlx_pool = None  # single-thread executor for MLX (lazy)
        self._mlx_fails = 0    # consecutive MLX failures -> auto-disable
        self._sv_fails = 0     # consecutive Swedish-model failures -> give up
        self.last_language = None  # language of the most recent transcription
                                   # (forced language, else what STT detected) —
                                   # lets the ask flow skip TTS for non-English.
        self.last_model = None     # model name that produced last_language's text
                                   # (a kb-* name when Swedish routing kicked in)

    def warm(self):
        """Preload the MLX model(s) by pushing half a second of silence through
        each, so the first real dictation doesn't pay the model-load cost.
        Blocks until loaded — run it on a background thread (see
        PushToTalkApp.prewarm). mlx-whisper caches models process-wide, so the
        warmth survives _rebuild_transcriber (model/language switches).
        """
        if self.engine != "mlx":
            return  # faster-whisper is the rarely-used CPU fallback; skip
        import numpy as np

        silence = np.zeros(8000, dtype=np.float32)
        pool = self._ensure_pool()
        # Forced language: skips detection, and keeps warm-up out of the log.
        pool.submit(self._run_mlx, silence, self.model_name,
                    self.language or "en").result()
        if self.swedish_model:
            pool.submit(self._run_mlx, silence, self.swedish_model,
                        "sv").result()

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

    def _ensure_pool(self):
        # MLX arrays and GPU streams are thread-affine: a model cached on the
        # thread that first loaded it cannot be evaluated from another thread
        # ("There is no Stream(gpu, N) in current thread"). on_release spawns a
        # fresh thread per dictation, so we pin ALL MLX work to one persistent
        # worker thread and block on its result.
        if self._mlx_pool is None:
            self._mlx_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx-stt"
            )
        return self._mlx_pool

    def _transcribe_mlx(self, audio):
        pool = self._ensure_pool()
        run = lambda model, lang=None: pool.submit(  # noqa: E731
            self._run_mlx, audio, model, lang
        ).result()

        # Swedish routing. The language is either known up front (the user
        # forced Svenska -> straight to the specialist, one pass) or not (auto
        # -detect -> the general model transcribes AND detects, and a Swedish
        # verdict earns a second pass on KB-Whisper). Only Swedish utterances
        # pay for the second decode; everything else is unchanged.
        if self._use_swedish_model():
            if self.language == "sv":
                return run(self.swedish_model, "sv")
            text = run(self.model_name)
            if text and self.last_language == "sv":
                try:
                    return run(self.swedish_model, "sv")
                except Exception as exc:
                    # A broken/missing Swedish model must never lose the
                    # transcript we already have — keep the general one.
                    self._sv_fails += 1
                    print(f"[stt] {self.swedish_model} failed "
                          f"({type(exc).__name__}: {exc}); keeping "
                          f"{self.model_name}", flush=True)
                    if self._sv_fails >= 2:
                        print("[stt] disabling the Swedish model for this "
                              "session", flush=True)
                        self.swedish_model = None
            return text
        return run(self.model_name)

    def _use_swedish_model(self):
        """Swedish routing is on when a KB-Whisper model is configured and the
        language could still be Swedish (a forced en/es never reroutes)."""
        return bool(self.swedish_model) and self.language in (None, "sv")

    def _run_mlx(self, audio, model_name, language=None):
        import mlx_whisper  # lazy: pulls in MLX

        repo = _MLX_REPOS.get(model_name, _MLX_REPOS["large-v3-turbo"])
        language = language or self.language
        # Dictation utterances are independent — never seed a window with the
        # previous window's text. Conditioning lets a repetition loop in one
        # 30s window poison every following window (Whisper keeps the last
        # decode even when all temperature fallbacks fail its quality checks;
        # cleanup.collapse_repeats is the net for loops within a window).
        opts = {"path_or_hf_repo": repo, "condition_on_previous_text": False}
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        if language:
            opts["language"] = language
        result = mlx_whisper.transcribe(audio, **opts)
        text = (result.get("text") or "").strip()
        self.last_language = language or result.get("language")
        self.last_model = model_name
        if language is None and text:
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
        self.last_model = self.model_name  # CPU path has no Swedish specialist
        if self.language is None and text:
            print(f"[stt] detected language: {info.language} "
                  f"(p={info.language_probability:.2f})", flush=True)
        return text
