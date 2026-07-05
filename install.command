#!/bin/bash
# WhisperFlow — one-click installer for macOS (Apple Silicon).
#
# Double-click this file in Finder. It sets up everything needed and leaves you
# with a menu-bar dictation app that starts automatically at login:
#   • Homebrew (if missing)      • Python + all libraries
#   • Ollama + the cleanup model • the speech model (pre-downloaded)
#   • WhisperFlow.app in your Applications, added to Login Items
#
# The only thing it can't do for you is grant macOS permissions — it prints
# clear instructions for that at the end (a one-time click).

set -uo pipefail

# --- pretty output --------------------------------------------------------
BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
step() { echo; echo "${BOLD}▶ $*${RESET}"; }
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }
die()  { echo "  ${RED}✗ $*${RESET}"; echo; echo "Installation stopped. Fix the above and run this again."; read -r -p "Press Return to close."; exit 1; }

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Library/WhisperFlow"          # TCC-safe runtime location
APPDIR="$HOME/Applications"
LLM_MODEL="llama3.1:8b"

clear
echo "${BOLD}WhisperFlow installer${RESET}"
echo "Local, offline voice dictation for macOS. This takes ~5–15 min mostly"
echo "downloading models. You can keep using your Mac while it runs."
echo

# --- 0. sanity ------------------------------------------------------------
step "Checking your Mac"
[[ "$(uname)" == "Darwin" ]] || die "This installer is for macOS only."
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "Not an Apple Silicon Mac — it may still work but is untested."
fi
ok "macOS $(sw_vers -productVersion) on $(uname -m)"

# --- 1. Homebrew ----------------------------------------------------------
step "Homebrew (package manager)"
if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew not found — installing it (you'll be asked for your password)."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
        || die "Homebrew install failed."
fi
# Make brew available in this shell (Apple Silicon default prefix).
if [[ -x /opt/homebrew/bin/brew ]]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
command -v brew >/dev/null 2>&1 || die "Homebrew still not on PATH."
ok "brew $(brew --version | head -1 | awk '{print $2}')"

# --- 2. Python ------------------------------------------------------------
step "Python"
PY=""
for cand in python3.12 python3.13 python3.11; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v $cand)"; break; fi
done
if [[ -z "$PY" ]]; then
    warn "Installing python@3.12 via Homebrew…"
    brew install python@3.12 || die "Python install failed."
    PY="$(brew --prefix)/bin/python3.12"
fi
ok "using $($PY --version) at $PY"
PYVER="$("$PY" -c 'import sys;print("python%d.%d"%sys.version_info[:2])')"

# --- 3. Runtime files -----------------------------------------------------
step "Installing app files to $DEST"
mkdir -p "$DEST"
for item in app scripts config.json requirements.txt assets README.md; do
    [[ -e "$SRC/$item" ]] && cp -R "$SRC/$item" "$DEST"/
done
ok "copied source"

# --- 4. Python environment ------------------------------------------------
step "Creating Python environment + installing libraries (this can take a few min)"
"$PY" -m venv "$DEST/.venv" || die "Could not create virtualenv."
"$DEST/.venv/bin/pip" install --quiet --upgrade pip
"$DEST/.venv/bin/pip" install --quiet -r "$DEST/requirements.txt" rumps \
    || die "Library install failed."
ok "libraries installed"

# --- 5. Ollama + models ---------------------------------------------------
step "Ollama (local LLM for cleanup)"
command -v ollama >/dev/null 2>&1 || brew install ollama || die "Ollama install failed."
brew services start ollama >/dev/null 2>&1 || ollama serve >/dev/null 2>&1 &
# wait for the server to answer
for _ in $(seq 1 30); do
    curl -fsS http://localhost:11434/api/version >/dev/null 2>&1 && break; sleep 1
done
ok "Ollama running"
step "Downloading cleanup model ($LLM_MODEL, ~5 GB) — grab a coffee"
ollama pull "$LLM_MODEL" || warn "Model pull failed; cleanup will fall back to rules until you run: ollama pull $LLM_MODEL"

step "Downloading speech model (large-v3-turbo, GPU) so the first dictation is instant"
"$DEST/.venv/bin/python" <<'PY' || warn "Speech-model prefetch skipped (downloads on first use instead)."
import numpy as np
silence = np.zeros(16000, dtype="float32")
try:
    import mlx_whisper
    mlx_whisper.transcribe(silence,
                           path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
    print("  ok (mlx / GPU)")
except Exception:
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    print("  ok (faster-whisper / CPU fallback)")
PY

# --- 6. Build the app + autostart ----------------------------------------
step "Building WhisperFlow.app"
bash "$DEST/scripts/build_app.sh" >/dev/null 2>&1 || die "App build failed."
mkdir -p "$APPDIR"
rm -rf "$APPDIR/WhisperFlow.app"
cp -R "$DEST/WhisperFlow.app" "$APPDIR/WhisperFlow.app"
ok "installed to $APPDIR/WhisperFlow.app"

step "Adding to Login Items (starts automatically each login)"
osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APPDIR/WhisperFlow.app\", hidden:false}" >/dev/null 2>&1 \
    && ok "added to Login Items" || warn "couldn't add Login Item automatically (you can add it in System Settings → General → Login Items)"

# --- 7. Launch + permission guidance -------------------------------------
step "Starting WhisperFlow"
open "$APPDIR/WhisperFlow.app"
sleep 3
ok "launched — look for the little Marvin face on your screen"

cat <<EOF

${BOLD}${GREEN}Almost done — one manual step (macOS security).${RESET}

WhisperFlow needs two permissions. macOS shows it as ${BOLD}"$PYVER"${RESET}
(not "WhisperFlow"), because the app runs on Python.

  1. System Settings → Privacy & Security → ${BOLD}Accessibility${RESET}
       → turn ON the ${BOLD}$PYVER${RESET} entry (add it with + if missing:
         $PY )
  2. System Settings → Privacy & Security → ${BOLD}Input Monitoring${RESET}
       → turn ON ${BOLD}$PYVER${RESET} (same path if you need to add it)
  3. Click the menu-bar Flow icon → Quit, then reopen WhisperFlow from Applications.

Then: click into any text field, ${BOLD}hold Control+Shift+Space, speak, release${RESET}.
Allow the Microphone prompt the first time. Works in Swedish and English.
Marvin shakes/nods his head and his eyes glow while he listens.

${BOLD}Settings${RESET} (model, language, sound): ${BOLD}right-click Marvin${RESET}.

EOF
read -r -p "Press Return to close this window."
