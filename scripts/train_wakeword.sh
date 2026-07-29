#!/bin/bash
# Train the "Hey Marvin" wake word and install it into assets/wakeword/.
#
# Why this is a separate environment: training pulls torch, torchaudio, librosa
# and ~7GB of datasets. The APP only needs numpy + onnxruntime to listen, so
# none of that belongs in the app venv (or the runtime one). Everything here
# lives in a throwaway workdir you can delete afterwards.
#
#   ./scripts/train_wakeword.sh              # train into the default workdir
#   WW_DIR=~/ww ./scripts/train_wakeword.sh  # keep the workdir somewhere else
#
# Takes hours and needs ~10GB free. The result is one ~1MB .onnx file.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
WW_DIR="${WW_DIR:-${TMPDIR:-/tmp}/whisperflow-wakeword}"
CONFIG="$SRC/scripts/wakeword/hey_marvin.yaml"
DEST="$SRC/assets/wakeword"

command -v espeak-ng >/dev/null || {
  echo "error: espeak-ng missing — brew install espeak-ng ffmpeg" >&2; exit 1; }
command -v ffmpeg >/dev/null || {
  echo "error: ffmpeg missing — brew install espeak-ng ffmpeg" >&2; exit 1; }

mkdir -p "$WW_DIR"
cd "$WW_DIR"

# Always drive the venv through `python -m`, never the bin/livekit-wakeword
# console script: a venv that has been MOVED keeps the absolute shebang it was
# built with, so the console script dies with "bad interpreter" while bin/python
# still works fine (it resolves sys.prefix from its own location). Same trap as
# the runtime venv's bin/pip — see scripts/sync_runtime.sh.
PY="$WW_DIR/.venv/bin/python"
if ! "$PY" -c "import livekit.wakeword" >/dev/null 2>&1; then
  echo "==> creating the training venv in $WW_DIR/.venv"
  rm -rf "$WW_DIR/.venv"
  python3 -m venv .venv
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install "livekit-wakeword[train,eval,export]"
fi
WW=("$PY" -m livekit.wakeword)

cp "$CONFIG" "$WW_DIR/hey_marvin.yaml"

echo "==> downloading Piper voices, backgrounds and impulse responses (~7GB, once)"
"${WW[@]}" setup --config hey_marvin.yaml

echo "==> generate -> augment -> train -> export (this is the long part)"
"${WW[@]}" run hey_marvin.yaml

echo "==> evaluating"
"${WW[@]}" eval hey_marvin.yaml || echo "(eval failed; the model is still usable)"

ONNX="$(find "$WW_DIR/output" -name 'hey_marvin*.onnx' -not -name '*quant*' \
        -print0 | xargs -0 ls -t 2>/dev/null | head -1)"
[[ -n "$ONNX" ]] || { echo "error: no exported .onnx found under $WW_DIR/output" >&2; exit 1; }

mkdir -p "$DEST"
cp "$ONNX" "$DEST/hey_marvin.onnx"
echo "installed $ONNX -> $DEST/hey_marvin.onnx"
echo
echo "next: ./scripts/sync_runtime.sh, restart the app, then switch on"
echo "Settings -> Hands-free (\"Hey Marvin\")."
echo "workdir left at $WW_DIR — delete it to reclaim the datasets."
