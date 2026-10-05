"""Container startup with a validated port and no shell interpolation."""

from __future__ import annotations

import os

import uvicorn


def configured_port() -> int:
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError as exc:
        raise ValueError("PORT must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535")
    return port


def main() -> None:
    uvicorn.run("src.api_service:app", host="0.0.0.0", port=configured_port(), workers=1)


if __name__ == "__main__":
    main()
