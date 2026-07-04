# Installing WhisperFlow on a Mac

Local, offline voice dictation. Hold a hotkey, speak, and your words are typed
into whatever app you're in — transcribed and cleaned up entirely on your own
machine. Swedish and English, auto-detected.

## Install (one click)

1. Copy this whole **WhisperFlow** folder to the new Mac (AirDrop, USB, or
   iCloud — anywhere is fine).
2. Open the folder and **double-click `install.command`**.
   - If macOS says *"cannot be opened because it is from an unidentified
     developer,"* **right-click `install.command` → Open → Open**. You only
     need to do this once.
3. A Terminal window opens and does everything automatically: installs
   Homebrew, Python, Ollama, downloads the models, and builds the app. This
   takes roughly **5–15 minutes**, mostly downloading. You can use your Mac
   meanwhile.
4. When it finishes it launches the app (🎤 appears in the menu bar) and prints
   the final step below.

## The one manual step: permissions

macOS won't let any app watch your keyboard or type for you without permission,
and it identifies this app as **"python3.x"** (because it runs on Python), not
"WhisperFlow". After the installer finishes:

1. **System Settings → Privacy & Security → Accessibility** → turn **ON** the
   `python3.x` entry.
2. **System Settings → Privacy & Security → Input Monitoring** → turn **ON** the
   same `python3.x` entry.
3. Click the menu-bar 🎤 → **Quit**, then reopen **WhisperFlow** from your
   **Applications** folder.

## Using it

- Click into any text field, **hold Control + Shift + Space, speak, release**.
- The first time, allow the **Microphone** prompt, then dictate again.
- Speak Swedish or English — it detects the language automatically.

## Adjusting

- **Speed vs. accuracy:** menu-bar 🎤 → **Model** (base = fastest, large = most
  accurate).
- It **starts automatically at every login** — no need to open a terminal ever.

## Uninstall

```bash
bash ~/Library/WhisperFlow/scripts/uninstall_login_item.sh   # (if present)
rm -rf ~/Library/WhisperFlow ~/Applications/WhisperFlow.app
```

Then remove the `python3.x` entries from Accessibility / Input Monitoring, and
the Login Item in System Settings → General → Login Items.
