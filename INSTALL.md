# Installing WhisperFlow on a Mac

Local, offline voice dictation for macOS with an animated Marvin. Hold a hotkey,
speak, and your words are transcribed and cleaned up entirely on your own
machine — then typed into whatever app you're in. Swedish and English,
auto-detected. Marvin shakes/nods his head and his eyes glow while he listens.

## Requirements

- **Apple Silicon Mac** (M1 or newer) recommended — the fast GPU transcription
  (mlx) needs it. It falls back to a slower CPU engine otherwise.
- macOS 13 or newer.

## Install (one click)

1. Copy this whole **WhisperFlow** folder to the new Mac (AirDrop, USB, or
   iCloud — anywhere is fine).
2. Open the folder and **double-click `install.command`**.
   - If macOS says *"cannot be opened because it is from an unidentified
     developer,"* **right-click `install.command` → Open → Open** (once).
3. A Terminal window runs the whole setup automatically: Homebrew, Python,
   Ollama, the speech + cleanup models, and builds the app. This takes roughly
   **10–20 minutes**, mostly downloading the models (~6 GB total). You can use
   your Mac meanwhile.
4. When it finishes it launches the app (Marvin appears on screen) and prints
   the final step below.

## The one manual step: permissions

macOS won't let any app watch your keyboard or type for you without permission,
and it identifies this app as **"python3.x"** (because it runs on Python), not
"WhisperFlow". After the installer finishes:

1. **System Settings → Privacy & Security → Accessibility** → turn **ON** the
   `python3.x` entry.
2. **System Settings → Privacy & Security → Input Monitoring** → turn **ON** the
   same `python3.x` entry.
3. Right-click Marvin → **Quit**, then reopen **WhisperFlow** from your
   **Applications** folder.

(The installer prints the exact `python3.x` version and its path.)

## Using it

- Click into any text field, **hold Control + Shift + Space, speak, release**.
- Allow the **Microphone** prompt the first time, then dictate again.
- Speak Swedish or English — it detects the language automatically.
- **Right-click Marvin** for settings: **Model** (accuracy), **Language**,
  **AI cleanup** on/off (verbatim), **Sound cues**, **Appearance**.
- **Drag** Marvin anywhere; he remembers where you put him.

It **starts automatically at every login** — no terminal needed.

## Uninstall

```bash
rm -rf ~/Library/WhisperFlow ~/Applications/WhisperFlow.app
```

Then remove the `python3.x` entries from Accessibility / Input Monitoring, and
the Login Item in System Settings → General → Login Items.

## Alternative: clone from GitHub

If you have access to the repo, on the new Mac:

```bash
git clone <repo-url> WhisperFlow && cd WhisperFlow && bash install.command
```
