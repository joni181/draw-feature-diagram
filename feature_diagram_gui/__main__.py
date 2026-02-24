"""Run the feature diagram GUI package directly."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .app import run
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            sys.stderr.write(
                "Error: PySide6 is not installed in this environment.\n"
                "Install dependencies with: pip install -r requirements.txt\n"
            )
            return 1
        raise

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
