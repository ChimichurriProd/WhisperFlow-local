#!/bin/bash
# Sync source (this repo) -> the runtime install the app actually runs from.
#
# The login item / MarvinBar launches ~/Library/WhisperFlow/WhisperFlow.app,
# which runs ~/Library/WhisperFlow's OWN app code, venv and config.json.
# Editing code here in the repo does NOTHING for the running app until it is
# synced over — that's what this script is for. Run it after every change,
# then restart the app (MarvinBar menu or relaunch WhisperFlow.app).
#
# Deliberately NOT synced:
#   - config.json  (runtime keeps its own user state: pill position, skin,
#     vocabulary, tts voice choice — patch it by hand when a new section
#     appears; load_config() fills missing keys with defaults regardless)
#   - .venv        (separate install; new pip deps must be installed there
#     too: ~/Library/WhisperFlow/.venv/bin/pip install ...)
#   - WhisperFlow.app  (runtime bundle's launcher points at runtime paths;
#     rebuild it there with scripts/build_app.sh FROM the runtime dir if the
#     launcher itself must change)
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/Library/WhisperFlow"

[[ -d "$DST" ]] || { echo "error: $DST not found" >&2; exit 1; }

rsync -a --delete --exclude __pycache__ "$SRC/app/" "$DST/app/"
rsync -a --exclude __pycache__ "$SRC/tests/" "$DST/tests/"
rsync -a "$SRC/requirements.txt" "$SRC/README.md" "$DST/"
# Assets: additive update (never delete runtime-only files).
rsync -au "$SRC/assets/" "$DST/assets/"
# Stale bytecode from the old code must not shadow the sync.
find "$DST/app" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# Build stamp: the app logs this at startup, so every debugging session
# starts with WHICH code is actually running instead of a guess (the
# source-vs-runtime split has burned whole days before).
GIT_DESC="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo no-git)"
[[ -z "$(git -C "$SRC" status --porcelain -uno 2>/dev/null)" ]] || GIT_DESC="$GIT_DESC+dirty"
echo "synced $(date '+%Y-%m-%d %H:%M:%S') from $GIT_DESC" > "$DST/build_stamp.txt"

echo "synced $SRC -> $DST"
echo "restart the app to pick it up (MarvinBar menu, or relaunch WhisperFlow.app)"
