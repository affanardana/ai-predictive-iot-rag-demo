"""Entry point for ``python -m ml``."""

from __future__ import annotations

import sys

from ml.cli import main

if __name__ == "__main__":
    sys.exit(main())
