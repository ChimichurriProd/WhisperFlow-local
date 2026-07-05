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


def main(argv=None):
    args = build_parser().parse_args(argv)

    config_path = args.config
    if config_path is None:
        # CWD first, then the project root (matters under launchd).
        for candidate in (Path("config.json"),
                          Path(__file__).resolve().parent.parent / "config.json"):
            if candidate.exists():
                config_path = candidate
                break
    config = load_config(config_path)

    # Imports after arg parsing so --help stays instant.
    if args.menubar:
        from .menubar import run_menubar

        run_menubar(config, config_path=str(config_path) if config_path else None)
    else:
        from .hotkey import PushToTalkApp

        PushToTalkApp(config).run()


if __name__ == "__main__":
    sys.exit(main())
