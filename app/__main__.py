"""CLI entrypoint: `python -m app` starts the push-to-talk loop (macOS).

Use --menubar for the menu-bar + floating-pill app; without it, a plain
terminal process that logs to stdout.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Local WhisperFlow clone: hold a global hotkey to dictate; "
            "faster-whisper transcribes, Ollama cleans, and the text is "
            "injected at your cursor. Fully offline."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="path to config.json (default: ./config.json if present, else built-ins)",
    )
    parser.add_argument(
        "--menubar",
        action="store_true",
        help="run as a menu-bar app (status icon) instead of a terminal process",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def hold_instance_lock(path=None):
    """Take the single-instance lock, or return None if another engine holds it.

    Two engines at once is a real failure mode we have hit, not a hypothetical:
    the log shows a second instance (launched from a stale duplicate bundle)
    fighting the first — both grab the mic and the hotkey, and whichever saves
    config.json last clobbers the other's settings. flock() is advisory and
    dies with the process, so a crash can never leave a stale lock behind.
    The returned file object must be kept alive for the process lifetime.
    """
    import fcntl

    if path is None:
        path = Path.home() / "Library" / "Logs" / "whisperflow-local" / "engine.lock"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    import os

    f.write(str(os.getpid()))
    f.flush()
    return f


def required_hf_repos(config):
    """The HuggingFace repos this config's engines will load."""
    from .stt import _MLX_REPOS

    repos = []
    stt = config.get("stt", {})
    if stt.get("engine", "mlx") == "mlx":
        repos.append(_MLX_REPOS.get(stt.get("model"),
                                    _MLX_REPOS["large-v3-turbo"]))
        if stt.get("swedish_model") in _MLX_REPOS:
            repos.append(_MLX_REPOS[stt["swedish_model"]])
    cb = (config.get("tts") or {}).get("chatterbox_model")
    if cb:
        repos.append(cb)
        repos.append("mlx-community/S3TokenizerV2")  # chatterbox's tokenizer
    return repos


def maybe_go_offline(config, hub_dir=None):
    """Set HF_HUB_OFFLINE=1 when every needed model is already cached.

    Without it, huggingface_hub re-validates the cache ON EVERY DICTATION —
    a network round-trip per utterance in a "fully local" app ("Fetching 4
    files" spam in the log, and latency spikes on flaky wifi). Only skipped
    when something is missing, so a first install can still download. Must
    run BEFORE huggingface_hub is imported (it reads the env at import time).
    Returns True when offline mode was enabled.
    """
    import os

    if os.environ.get("HF_HUB_OFFLINE"):
        return True
    hub = Path(hub_dir or Path.home() / ".cache" / "huggingface" / "hub")
    for repo in required_hf_repos(config):
        d = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
        if not (d.is_dir() and any(d.iterdir())):
            print(f"[app] {repo} not cached yet — staying online for the "
                  f"download", flush=True)
            return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("[app] all models cached — HF_HUB_OFFLINE=1 (no network per "
          "dictation)", flush=True)
    return True


def main(argv=None):
    args = build_parser().parse_args(argv)

    lock = hold_instance_lock()
    if lock is None:
        print("[app] another WhisperFlow engine is already running — exiting "
              "(two engines fight over the mic, the hotkeys and config.json)",
              flush=True)
        return 0
    main._lock = lock  # keep the fd alive for the process lifetime

    # Which code is this, actually? The source repo and the runtime copy
    # (~/Library/WhisperFlow) have drifted apart before and cost whole days;
    # sync_runtime.sh writes the stamp, we put it at the top of every log.
    root = Path(__file__).resolve().parent.parent
    stamp = root / "build_stamp.txt"
    try:
        build = stamp.read_text(encoding="utf-8").strip()
    except OSError:
        build = "(no stamp — running straight from a source checkout?)"
    print(f"[app] code root: {root}", flush=True)
    print(f"[app] build: {build}", flush=True)

    config_path = args.config
    if config_path is None:
        # CWD first, then the project root (matters under launchd).
        for candidate in (Path("config.json"),
                          Path(__file__).resolve().parent.parent / "config.json"):
            if candidate.exists():
                config_path = candidate
                break
    config = load_config(config_path)
    maybe_go_offline(config)

    # Imports after arg parsing so --help stays instant.
    if args.menubar:
        from .menubar import run_menubar

        run_menubar(config, config_path=str(config_path) if config_path else None)
    else:
        from .hotkey import PushToTalkApp

        PushToTalkApp(config).run()


if __name__ == "__main__":
    sys.exit(main())
