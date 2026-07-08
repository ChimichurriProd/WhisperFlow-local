#!/usr/bin/env bash
# Trim a WhisperFlow venv for packaging — removes weight that the running app
# never uses. Verified safe: STT (mlx-whisper), TTS (kokoro-onnx) and all app
# modules import and run with everything below removed.
#
# Run against a COPY of the venv you intend to ship, e.g.:
#   cp -R ~/Library/WhisperFlow/.venv /tmp/pkg-venv
#   scripts/trim_for_package.sh /tmp/pkg-venv
#
# It does NOT touch models (~/.cache/kokoro-onnx, HF whisper) — those are a
# separate bundle-vs-download decision.
set -euo pipefail

VENV="${1:?usage: trim_for_package.sh <venv-path> [--drop-cpu-fallback]}"
DROP_FALLBACK="${2:-}"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.14/site-packages"

echo "before: $(du -sh "$VENV" | cut -f1)"

# 1) torch is only used by mlx-whisper's PyTorch REFERENCE path (checkpoint
#    conversion) — never the MLX transcribe path we use. Drop it + its
#    exclusive deps. (fsspec/filelock/jinja2/typing-extensions are shared with
#    huggingface_hub etc., so leave those.)
"$PY" -m pip uninstall -y torch sympy networkx mpmath >/dev/null 2>&1 || true

# 2) Optional: drop the faster-whisper CPU fallback (Mac-only / MLX-only build).
if [ "$DROP_FALLBACK" = "--drop-cpu-fallback" ]; then
  "$PY" -m pip uninstall -y faster-whisper ctranslate2 av >/dev/null 2>&1 || true
fi

# 3) Build tooling not needed at runtime inside a shipped app.
"$PY" -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true

# 4) Cruft: test suites, __pycache__ (regenerated at runtime), C headers.
#    Keep *.dist-info — some packages read their own version via
#    importlib.metadata (e.g. kokoro-onnx).
find "$SP" -type d \( -name tests -o -name test -o -name "*Test*" \) \
  -exec rm -rf {} + 2>/dev/null || true
find "$SP" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SP" -type d -name "include" -path "*numpy*" -exec rm -rf {} + 2>/dev/null || true

echo "after:  $(du -sh "$VENV" | cut -f1)"
