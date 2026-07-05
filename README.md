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
ollama pull llama3.1:8b        # default
# lighter alternatives (set cleanup.ollama_model in config.json):
ollama pull phi3:mini
ollama pull mistral:7b
```

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
    "vad_filter": true,        // faster-whisper only
    "device": "cpu",           // faster-whisper only
    "compute_type": "int8"     // faster-whisper only
  },
  "cleanup": {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
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
