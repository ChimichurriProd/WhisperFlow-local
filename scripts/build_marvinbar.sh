#!/bin/bash
# Build MarvinBar.app — the native menu-bar companion (see companion/marvinbar.swift
# for why the engine can't own a status item on macOS 26). Installs into
# ~/Applications. Needs Xcode Command Line Tools (swiftc).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/MarvinBar.app"
MACOS="$APP/Contents/MacOS"

rm -rf "$APP"
mkdir -p "$MACOS"
swiftc -O "$ROOT/companion/marvinbar.swift" -o "$MACOS/MarvinBar"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>MarvinBar</string>
    <key>CFBundleDisplayName</key><string>MarvinBar</string>
    <key>CFBundleIdentifier</key><string>local.whisperflow.marvinbar</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>MarvinBar</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
</dict>
</plist>
PLIST

codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "signed (ad-hoc)" || echo "codesign skipped (non-fatal)"
echo "Installed: $APP"
