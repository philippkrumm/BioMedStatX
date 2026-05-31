"""Root conftest.py — adds src/ to sys.path so every test can import from src/.

Also forces a headless Qt platform by default so the GUI-touching tests run
without a display (CI, SSH, etc.). Override by exporting QT_QPA_PLATFORM
yourself (e.g. `QT_QPA_PLATFORM=xcb`) before invoking pytest.
"""
import os
import sys
from pathlib import Path

# Headless Qt unless the caller already chose a platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

for _p in (_ROOT, _SRC):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)
