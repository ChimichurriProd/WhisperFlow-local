# WhisperFlow Local

A fully local, offline [Wispr Flow](https://wisprflow.ai) clone for macOS.
Hold a global hotkey, speak, release — your words are transcribed by a local
Whisper model, cleaned up by a local LLM, and typed into whatever app has focus.
No audio or text ever leaves your machine.

```
hold hotkey ──► mic capture ──► Silero VAD + faster-whisper ──► cleanup ──► inject at cursor
                                                                  │
                                              <10 words: rule-based only (fast path)
                                              ≥10 words: Ollama LLM polish
```

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10+
- [Ollama](https://ollama.com/download) for LLM cleanup (optional — without it,
  cleanup falls back to fast rule-based filler stripping)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pull the Ollama cleanup model

```bash
brew install ollama            # or download from ollama.com
ollama serve &                 # local server on http://localhost:11434
ollama pull gemma4:12b         # default — holds non-English better than llama3.1
# lighter alternatives (set cleanup.ollama_model in config.json):
ollama pull llama3.1:8b
ollama pull phi3:mini
```

WhisperFlow talks to Ollama's `/api/chat` with `think: false`. Reasoning-capable
models (gemma4, qwen3) spend their whole token budget on hidden thinking under
the older `/api/generate` and hand back an empty answer.

Everything stays on-device: Ollama is a local server, not a cloud API.

## macOS permissions (one-time)

Grant these to **the app you launch Python from** (Terminal, iTerm2, etc.) in
**System Settings → Privacy & Security**:

| Permission | Why |
|---|---|
| **Microphone** | record your dictation (macOS prompts on first run) |
| **Input Monitoring** | the global push-to-talk hotkey listener (pynput) |
| **Accessibility** | synthesizing the Cmd+V / typed keystrokes |

After granting Input Monitoring or Accessibility, restart the terminal app.

## Run

```bash
python -m app                   # terminal mode; uses ./config.json if present
python -m app --menubar         # menu-bar mode: 🎤 status icon, no terminal needed
python -m app --config my.json
python -m app --help
```

Menu-bar icon states: 🎤 idle · 🔴 recording · ✍️ transcribing · ⏸ paused.
The menu has Pause/Resume listening and Quit.

## Install as a desktop app (start at login)

```bash
bash scripts/install_login_item.sh
```

This registers a launchd agent that starts the menu-bar app at login and
restarts it if it crashes. Logs go to `~/Library/Logs/whisperflow-local/`.
To remove it:

```bash
bash scripts/uninstall_login_item.sh
```

**Permissions note:** when launched by launchd, the process is "Python"
rather than your terminal, so macOS asks for permissions again on first use —
grant **Microphone** when prompted, and add **Python** under **Input
Monitoring** and **Accessibility** (System Settings → Privacy & Security).
If the hotkey does nothing after install, that's why; check
`~/Library/Logs/whisperflow-local/stderr.log`.

## Hotkey setup

The push-to-talk key is set in `config.json`:

```json
"hotkey": { "push_to_talk": "control + shift + space" }
```

Hold it to record, release to transcribe and inject. Supported key names:
`control`/`ctrl`, `shift`, `alt`/`option`, `cmd`/`command`, `space`, `tab`,
`enter`, `esc`, `f1`–`f20`, and single characters — combined with `+`, e.g.
`"cmd + shift + d"` or `"f9"`. Avoid combos macOS already owns
(`cmd + space` is Spotlight).

## Text injection: the two delivery methods

Set `injection.delivery_method` in `config.json`:

| Method | How it works | When to use |
|---|---|---|
| `"clipboard"` (default) | Puts the text on the clipboard (`pbcopy`), synthesizes **Cmd+V**, then restores your previous clipboard (`restore_clipboard: true`). | Fastest for long text; works in almost every app. |
| `"type"` | Types the text character-by-character with synthesized keystrokes (`type_char_delay_ms` between chars). | Fields that block paste, terminals, remote-desktop windows, apps with clipboard managers. |

## Configuration reference (`config.json`)

```json
{
  "stt": {
    "engine": "mlx",           // "mlx" = Apple-Silicon GPU (fast); "faster-whisper" = CPU
    "model": "large-v3-turbo", // base | small | medium | large-v3-turbo
    "language": null,          // null = auto-detect (Swedish, English, ...); or "sv"/"en"
    "swedish_model": "kb-large", // KB-Whisper for Swedish: kb-large | kb-small | null
    "vad_filter": true,        // faster-whisper only
    "device": "cpu",           // faster-whisper only
    "compute_type": "int8"     // faster-whisper only
  },
  "cleanup": {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "gemma4:12b",
    "skip_llm_under_words": 10,   // latency rule: short utterances skip the LLM
    "timeout_seconds": 30
  },
  "hotkey": { "push_to_talk": "control + shift + space" },
  "injection": {
    "delivery_method": "clipboard",   // "clipboard" | "type"
    "restore_clipboard": true,
    "type_char_delay_ms": 5
  },
  "audio": { "sample_rate": 16000, "channels": 1, "mic_gain": 4.5 },
  "vocabulary": {
    "terms": ["Ollama", "WhisperFlow"],  // bias Whisper toward these spellings
    "fixes": { "olama": "Ollama" }        // force exact corrections (wrong->right)
  }
}
```

### Custom vocabulary

Teach WhisperFlow *your* words — names, jargon, brands, acronyms — so they
transcribe correctly instead of being mangled:

- **`terms`**: a list of words/phrases. These are fed to Whisper as context so
  it *prefers* those spellings at the source. Add colleague/company names,
  product names, technical terms you dictate often.
- **`fixes`**: an exact `"wrong": "right"` map applied after transcription
  (whole-word, case-insensitive). Use this for stubborn mishears — e.g.
  `"olama": "Ollama"`. Fixes are the final say, overriding STT and the LLM.

**Speed:** with the default `mlx` engine (Apple-Silicon GPU), even
`large-v3-turbo` transcribes a short utterance in ~0.2s — so the default is the
most *accurate* model and it still feels instant. On the GPU, model choice is
about accuracy, not speed. The `faster-whisper` (CPU) engine is the portable
fallback and is much slower on Mac. First use of each model downloads it once.

### Swedish accuracy (KB-Whisper)

Swedish is transcribed by [KB-Whisper](https://huggingface.co/KBLab) — the
National Library of Sweden's Whisper, fine-tuned on 50 000 hours of Swedish —
instead of the general model. Measured here on 60 FLEURS `sv_se` clips
(10.7 min of real human Swedish, 10.7s average clip):

| model | Swedish WER | time per clip |
|---|---|---|
| `large-v3-turbo` (general) | 12.0% | 0.13s |
| `kb-small` | 8.8% | 0.10s |
| `kb-large` (**default**) | **7.4%** | 0.34s |

Set it in **Settings → Swedish model** (or `stt.swedish_model`): `kb-large`,
`kb-small`, or `null` to use one model for every language.

How the routing works — with the language on **auto-detect**, the general model
transcribes and detects as usual, and only a *Swedish* result earns a second
pass on KB-Whisper. English and Spanish never pay for it. Pick **Svenska** in
the Language menu and it goes straight to KB-Whisper in a single pass. If the
Swedish model can't load, the general transcript is kept rather than lost.

## Marvin's voice (TTS)

Marvin speaks through two local engines, routed automatically per line
(`tts` in `config.json`):

- **Kokoro-82M** (CPU, instant) — English quips and answers, prerendered.
- **Chatterbox Multilingual** (Apple-Silicon GPU, ~1.4GB, lazy-loaded on the
  first non-English line) — Swedish + 22 other languages, speaking with
  Marvin's own voice cloned from `assets/marvin/voice_ref.wav`.

```json
"tts": {
  "engine": "auto",     // "auto" | "kokoro" (old English-only) | "chatterbox" (everything)
  "voice_ref": "…/assets/marvin/voice_ref.wav",  // WAV to clone (~10s); null = built-in voice
  "exaggeration": 0.3,  // Chatterbox emotion 0-1
  "pitch": 1.35         // cartoon pitch multiplier (both engines)
}
```

**Marvin's language** (Settings menu): "Match the question" or forced
Svenska/English/Español — forces both the Ollama answer language (small models
otherwise drift into English) and the spoken voice. Stored as `tts.language`.

Ask-mode answers in Swedish are spoken aloud now (they used to stay text-only
in the bubble). Try it:

```bash
python -m app.tts "Hej! Jag är Marvin. Försök att inte bli besviken."
```

## Hands-free: "Hey Marvin"

Say **"Hey Marvin"** and he opens an ask episode — no keys. Off by default;
switch it on in **Settings → Hands-free**.

```json
"wakeword": {
  "enabled": false,          // opt-in
  "phrase": "Hey Marvin",
  "model": null,             // path to a custom .onnx; null = the bundled one
  "threshold": 0.5,          // raise it if he wakes up on his own
  "debounce": 2.0,           // seconds deaf after a detection
  "silence_seconds": 1.2,    // this much quiet ends the question
  "silence_rms": 0.01,       // loudness below this counts as quiet
  "lead_in_seconds": 1.0,    // grace to start speaking before that applies
  "max_seconds": 15.0        // hard cap on one hands-free question
}
```

**It does not open a second microphone.** WhisperFlow already holds one
`InputStream` for the app's lifetime (see the `app/audio.py` docstring — that
design exists because tearing PortAudio streams up and down deadlocks
CoreAudio's HAL). The wake word is a *tap* on that same stream, so turning it on
adds no new device handling and no new permission. What it costs is one small
ONNX inference per 80 ms: a 2-second sliding window is re-scored on a worker
thread, never on the realtime audio thread.

A hands-free question has no key to release, so it ends on **silence**
(`silence_seconds` of quiet after a `lead_in_seconds` grace, or `max_seconds`).
Marvin goes deliberately deaf while recording and while speaking — otherwise
his own voice saying his own name wakes him up mid-sentence.

Measured on this model's own validation set (20.7 hours of audio):
**AUT 0.0012 · 0.10 false positives/hour · 88.6% recall** at threshold 0.5.

`threshold` ships at **0.7**, not the calibrated 0.5. On an out-of-distribution
voice, near-misses `"hey martin"` and `"hey marvel"` scored 0.73–0.79 while real
wake phrases scored 0.84–0.95 — a thinner margin than the validation set
suggests. A missed wake word costs you a repeat; a false one starts recording in
the middle of a meeting. Raise it toward 0.85 if it still trips, lower it toward
0.5 if it ignores you.

> **onnxruntime must be ≥ 1.28.** 1.27.x executes the bundled speech-embedding
> model incorrectly — the mel frontend and the classifier are both fine, so
> nothing appears broken, but every score collapses to ~0.002 and the wake word
> silently never fires. `app/wakeword.py` refuses to start on older versions
> rather than pretend to listen.

### Stopping him mid-answer

When he mishears you, a wrong answer should cost a second, not a monologue.
Three ways to cut him off, all equivalent:

- **Click Marvin** — while he's talking, a click means "shut up" (it only
  cycles the STT model when he's quiet).
- **Say "Hey Marvin"** — stops him *and* opens a fresh question. While he
  speaks, detection runs at the lower `wakeword.barge_threshold` (0.55) —
  his own voice on the speakers is drowning yours out at exactly that
  moment, and a false positive then merely cuts his own answer short.
- **Start any recording** — the dictate or ask hotkey silences him first,
  so his voice never ends up inside your recording.

### Training the model

The detector is [livekit-wakeword](https://github.com/livekit/livekit-wakeword)
(Apache 2.0), trained entirely locally from synthetic Piper TTS speech — no
recordings of you, no account, no cloud:

```bash
brew install espeak-ng ffmpeg
./scripts/train_wakeword.sh          # hours; needs ~20GB free
./scripts/sync_runtime.sh            # then restart and switch it on
```

The config is `scripts/wakeword/hey_marvin.yaml`. Two things in it are
deliberate. The wake phrase is *"hey marvin"* and the bare name **"marvin"** is
trained as a **negative** — a two-syllable bare name fires constantly in normal
speech. And `"martin"` / `"hey martin"` are negatives too, because Martin is a
common Swedish name that will be said near this microphone.

Training installs its own throwaway venv: it needs torch, torchaudio and ~17GB
of datasets, none of which belong in the app's environment. The app itself only
needs `numpy` + `onnxruntime` to listen, and the result is one ~1MB `.onnx`.

## Transcribe audio files

Drop one or more audio files **onto Marvin** (or right-click him →
"Transkribera ljudfil…"). He transcribes in the background — Swedish through
KB-Whisper as usual — and writes a `.txt` next to the source, with the full
text on the clipboard and a bubble when done.

**The gesture decides the grouping:** files dropped *together* become ONE
combined `.txt` ("`del1 +2 filer.txt`", a `## name (m:ss)` header per
recording, in drop order); files dropped one at a time each get their own.
Existing files are never overwritten (`namn 2.txt`).

Transcripts come out as **paragraphs, not a wall of text**: a speech gap
longer than `files.paragraph_gap_seconds` (1.2s) starts a new paragraph, and
recordings over two minutes get a `[m:ss]` stamp per paragraph for navigation
(`files.timestamps`).

Formats: wav, m4a (iPhone voice memos), mp3, aac, aiff, caf, flac — decoded
with macOS's built-in `afconvert`, no extra installs. Long recordings split
into ~60s chunks cut at the quietest point, so a dictation started mid-file
waits seconds, not minutes. Deliberately **no LLM cleanup** on files — the
cleanup prompt is tuned for short dictations, and recordings deserve the
faithful transcript (only the hallucination guard + your vocabulary fixes).

## Verify without a microphone

```bash
python scripts/dry_run.py                 # bundled assets/sample.wav
python scripts/dry_run.py path/to/your.wav
```

Runs the real VAD → faster-whisper → cleanup pipeline on a WAV file and prints
the final cleaned string. If Ollama isn't running it says so and falls back to
rule-based cleanup.

## Tests

```bash
python -m pytest tests/ -v
```

Keystrokes, clipboard, and Ollama calls are mocked, so tests pass anywhere.

## Privacy / offline guarantee

The core path makes **zero** cloud calls: STT is faster-whisper running
in-process (models download once from Hugging Face, then cache), and cleanup
talks only to your local Ollama server. There is no telemetry.
