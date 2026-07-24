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

    func applicationDidFinishLaunching(_ note: Notification) {
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let path = iconPath, let img = NSImage(contentsOfFile: path) {
            img.size = NSSize(width: 20, height: 20)
            aliveIcon = img
            let dim = NSImage(size: img.size)
            dim.lockFocus()
            img.draw(in: NSRect(origin: .zero, size: img.size),
                     from: .zero, operation: .sourceOver, fraction: 0.3)
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
            statusLine.title = "WhisperFlow: running (pid \(pid))"
            item.button?.image = aliveIcon
            item.button?.toolTip = "WhisperFlow is running"
        } else {
            statusLine.title = "WhisperFlow: NOT running"
            item.button?.image = deadIcon
            item.button?.toolTip = "WhisperFlow is NOT running — click to restart"
        }
    }

    @objc func restartEngine() {
        run("/usr/bin/pkill", ["-9", "-f", "--", enginePattern])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            run("/usr/bin/open", [appPath])
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.tick() }
        }
    }

    @objc func forceQuitEngine() {
        run("/usr/bin/pkill", ["-9", "-f", "--", enginePattern])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { self.tick() }
    }

    @objc func quitSelf() { NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = Delegate()
app.delegate = delegate
app.run()
