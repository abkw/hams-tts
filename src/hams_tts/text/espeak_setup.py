"""Points `phonemizer` at the repo-vendored espeak-ng build.

espeak-ng is not installed system-wide (no root on the dev box) — it was
extracted from the Ubuntu .deb into `.local-tools/` at the repo root
instead. This module must be imported before any `phonemizer` call.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIB_DIR = _REPO_ROOT / ".local-tools" / "usr" / "lib" / "x86_64-linux-gnu"
_LIBRARY = _LIB_DIR / "libespeak-ng.so.1"
_DATA_PATH = _LIB_DIR / "espeak-ng-data"

_configured = False


def ensure_espeak_configured() -> None:
    global _configured
    if _configured:
        return

    if not _LIBRARY.exists():
        raise RuntimeError(
            f"Vendored espeak-ng library not found at {_LIBRARY}. "
            "See PLAN.md Phase 0 / AGENTS.md for how it was extracted "
            "(apt-get download + dpkg-deb -x, no root required)."
        )

    os.environ.setdefault("ESPEAK_DATA_PATH", str(_DATA_PATH))

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(str(_LIBRARY))
    _configured = True
