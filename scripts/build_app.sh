#!/bin/bash
# Build WhisperFlow.app — a real macOS app bundle wrapping the Python engine.
#
# Why a bundle: macOS ties Accessibility/Input-Monitoring/Microphone
# permissions to a stable code identity. A bare `python -m app` launched by
# launchd resolves to the ambiguous shared "org.python.python" identity, so
# grants never stick. A bundle gives ONE identity ("WhisperFlow") that you
# grant once, plus double-click launch and a menu-bar-only presence.
#
# Re-run this any time after changing code; it only rebuilds the wrapper, not
# the venv. Output: ./WhisperFlow.app
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
APP="$ROOT/WhisperFlow.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found — create the venv first (see README)" >&2
    exit 1
fi

rm -rf "$APP"
mkdir -p "$MACOS" "$RES"

# Launcher: exec the venv Python running the menu-bar app, from the repo root.
# All output is teed to a log so failures under LaunchServices are visible.
# -u: unbuffered stdout — otherwise Python block-buffers to the log file and
# a crash/kill loses everything still sitting in the buffer.
LOG_DIR="$HOME/Library/Logs/whisperflow-local"
mkdir -p "$LOG_DIR"
cat > "$MACOS/WhisperFlow" <<EOF
#!/bin/bash
cd "$ROOT"
exec "$PYTHON" -u -m app --menubar >> "$LOG_DIR/app.log" 2>&1
EOF
chmod +x "$MACOS/WhisperFlow"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>WhisperFlow</string>
    <key>CFBundleDisplayName</key><string>WhisperFlow</string>
    <key>CFBundleIdentifier</key><string>local.whisperflow.app</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>WhisperFlow</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>WhisperFlow transcribes your dictation locally.</string>
    <key>NSHumanReadableCopyright</key><string>Local, offline dictation.</string>
</dict>
</plist>
EOF

# Ad-hoc code signature: gives the bundle a stable identity for TCC and
# clears the "damaged/unidentified" quarantine gripes on a local build.
codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "signed (ad-hoc)" || echo "codesign skipped (non-fatal)"

echo "Built: $APP"
echo "Double-click it in Finder, or run: open '$APP'"
