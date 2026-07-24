#!/bin/bash
# Build "Force Quit WhisperFlow.app" — a double-clickable kill switch.
#
# Why: when the engine hangs (frozen pill, dead hotkey), the menu-bar Quit
# can't respond, so you need something that kills the python process from
# outside. This bundle SIGKILLs any `python -m app --menubar` process and
# confirms with a notification. Installs straight into ~/Applications.
set -uo pipefail

APP="$HOME/Applications/Force Quit WhisperFlow.app"
MACOS="$APP/Contents/MacOS"

rm -rf "$APP"
mkdir -p "$MACOS"

cat > "$MACOS/ForceQuitWhisperFlow" <<'EOF'
#!/bin/bash
# Kill the WhisperFlow engine (menu-bar app or a dev terminal run alike).
pkill -9 -f -- "-m app --menubar"
sleep 0.4
if pgrep -f -- "-m app --menubar" >/dev/null; then
    osascript -e 'display notification "Process would not die — try Activity Monitor" with title "WhisperFlow"'
else
    osascript -e 'display notification "WhisperFlow stopped." with title "WhisperFlow"'
fi
EOF
chmod +x "$MACOS/ForceQuitWhisperFlow"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Force Quit WhisperFlow</string>
    <key>CFBundleDisplayName</key><string>Force Quit WhisperFlow</string>
    <key>CFBundleIdentifier</key><string>local.whisperflow.killswitch</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>ForceQuitWhisperFlow</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
EOF

codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "signed (ad-hoc)" || echo "codesign skipped (non-fatal)"

echo "Installed: $APP"
