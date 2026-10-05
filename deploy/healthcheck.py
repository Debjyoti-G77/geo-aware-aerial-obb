"""Dependency-free Docker liveness probe; readiness is checked separately."""

from __future__ import annotations

import os
import sys
from urllib.request import urlopen


def main() -> int:
    try:
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            return 1
        with urlopen(f"http://127.0.0.1:{port}/health/live", timeout=3) as response:
            return 0 if response.status == 200 else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
