#!/bin/bash
# Remove the WhisperFlow Local login item and stop the running instance.
set -euo pipefail

LABEL="local.whisperflow.menubar"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "Removed: $LABEL"
