#!/bin/bash
# Install (or reinstall) the launchd agent that starts WhisperFlow Local's
# menu-bar app at login. Run from anywhere; paths are resolved absolutely.
# Uninstall: bash scripts/uninstall_login_item.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
LABEL="local.whisperflow.menubar"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/whisperflow-local"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found — create the venv first (see README)" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>-m</string>
        <string>app</string>
        <string>--menubar</string>
    </array>
    <key>WorkingDirectory</key><string>$ROOT</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict><key>SuccessfulExit</key><false/></dict>
    <key>StandardOutPath</key><string>$LOG_DIR/stdout.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Installed and started: $LABEL"
echo "Logs: $LOG_DIR"
echo "Look for the microphone icon in your menu bar."
