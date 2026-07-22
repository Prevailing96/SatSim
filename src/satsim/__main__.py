"""Allow ``python -m satsim`` to invoke the CLI."""

from __future__ import annotations

from satsim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
