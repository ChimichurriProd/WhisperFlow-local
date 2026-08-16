// MarvinBar — native menu-bar companion for WhisperFlow.
//
// Why this exists: macOS 26 (Tahoe) refuses to host NSStatusItems created by
// Python interpreter processes (Apple bug FB21015611 — status items from
// exec-trampoline/interpreter apps never reach Control Centre), so the rumps
// menu-bar item in app/menubar.py is invisible on this OS. This tiny native
// app IS hosted, and being a separate process it keeps working when the
// engine hangs — which makes it the crash indicator and kill switch:
// bright face = engine running, dimmed face = engine dead.
import AppKit

let enginePattern = "-m app --menubar"
let appPath = NSHomeDirectory() + "/Applications/WhisperFlow.app"

func firstExisting(_ paths: [String]) -> String? {
    paths.first { FileManager.default.fileExists(atPath: $0) }
}

let iconPath = firstExisting([
    NSHomeDirectory() + "/Library/WhisperFlow/assets/marvin/_skins/G/center.png",
    NSHomeDirectory() + "/Library/WhisperFlow/assets/marvin/center.png",
])

@discardableResult
func run(_ tool: String, _ args: [String]) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: tool)
    p.arguments = args
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    do { try p.run() } catch { return "" }
    p.waitUntilExit()
    return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
}

func enginePid() -> String? {
    let out = run("/usr/bin/pgrep", ["-f", "--", enginePattern])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return out.isEmpty ? nil : String(out.split(separator: "\n")[0])
}

class Delegate: NSObject, NSApplicationDelegate {
    var item: NSStatusItem!
    var aliveIcon: NSImage?
    var deadIcon: NSImage?
    let statusLine = NSMenuItem(title: "Checking…", action: nil, keyEquivalent: "")
    // Auto-revive after a CRASH (never after a manual stop): the engine has
    // died to native bugs in ML libraries (SIGBUS in mlx, SIGSEGV in
    // onnxruntime — see ~/Library/Logs/DiagnosticReports/Python-*.ips), and a
    // dictation app that stays silently dead is worse than a 3-second hiccup.
    var lastAliveAt = Date.distantPast
    var manualStop = false
    var revives: [Date] = []

    func freshCrashReport() -> Bool {
        // Only a crash leaves a Python-*.ips newer than when we last saw the
        // engine alive — a manual quit leaves nothing, so we stay quiet then.
        let dir = NSHomeDirectory() + "/Library/Logs/DiagnosticReports"
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { return false }
        let cutoff = lastAliveAt.addingTimeInterval(-60)
        for n in names where n.hasPrefix("Python-") && n.hasSuffix(".ips") {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: dir + "/" + n),
               let m = attrs[.modificationDate] as? Date, m > cutoff, lastAliveAt > .distantPast {
                return true
            }
        }
        return false
    }

    func maybeRevive() {
        revives.removeAll { $0 < Date().addingTimeInterval(-600) }
        guard !manualStop, revives.count < 3, freshCrashReport() else { return }
        revives.append(Date())
        statusLine.title = "WhisperFlow: crashed — auto-restarting (\(revives.count)/3)"
        run("/usr/bin/open", [appPath])
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let path = iconPath, let img = NSImage(contentsOfFile: path) {
            img.size = NSSize(width: 20, height: 20)
            aliveIcon = img
            let dim = NSImage(size: img.size)
            dim.lockFocus()
            img.draw(in: NSRect(origin: .zero, size: img.size),
                     from: .zero, operation: .sourceOver, fraction: 0.3)
            // Red badge: "engine NOT running" must be legible at a glance —
            // a slightly paler face alone is easy to miss.
            NSColor.systemRed.setFill()
            NSBezierPath(ovalIn: NSRect(x: img.size.width - 8, y: 0,
                                        width: 7, height: 7)).fill()
            dim.unlockFocus()
            deadIcon = dim
        } else {
            item.button?.title = "Marvin"
        }
        let menu = NSMenu()
        menu.autoenablesItems = false
        statusLine.isEnabled = false
        menu.addItem(statusLine)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(makeItem("Show Marvin — where is he?", #selector(showMarvin)))
        menu.addItem(makeItem("Start / Restart WhisperFlow", #selector(restartEngine)))
        menu.addItem(makeItem("Force Quit WhisperFlow", #selector(forceQuitEngine)))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(makeItem("Quit this indicator", #selector(quitSelf)))
        item.menu = menu
        tick()
        Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { _ in self.tick() }
    }

    func makeItem(_ title: String, _ sel: Selector) -> NSMenuItem {
        let mi = NSMenuItem(title: title, action: sel, keyEquivalent: "")
        mi.target = self
        mi.isEnabled = true
        return mi
    }

    func tick() {
        if let pid = enginePid() {
            lastAliveAt = Date()
            manualStop = false
            statusLine.title = "WhisperFlow: running (pid \(pid))"
            item.button?.image = aliveIcon
            item.button?.toolTip = "WhisperFlow is running"
        } else {
            statusLine.title = "WhisperFlow: NOT running"
            item.button?.image = deadIcon
            item.button?.toolTip = "WhisperFlow is NOT running — click to restart"
            maybeRevive()
        }
    }

    // "Where did he go?": SIGUSR1 asks the engine to rescue the pill onto a
    // visible screen and bounce it (menubar._on_reveal_signal). If the engine
    // is dead, just start it — Marvin appearing IS the answer.
    @objc func showMarvin() {
        if let pid = enginePid() {
            run("/bin/kill", ["-USR1", pid])
        } else {
            manualStop = false
            run("/usr/bin/open", [appPath])
        }
    }

    @objc func restartEngine() {
        manualStop = false
        run("/usr/bin/pkill", ["-9", "-f", "--", enginePattern])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            run("/usr/bin/open", [appPath])
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.tick() }
        }
    }

    @objc func forceQuitEngine() {
        manualStop = true   // the user chose this — never auto-revive it
        run("/usr/bin/pkill", ["-9", "-f", "--", enginePattern])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { self.tick() }
    }

    @objc func quitSelf() { NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = Delegate()
app.delegate = delegate
app.run()
