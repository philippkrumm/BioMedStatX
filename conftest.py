"""Root conftest.py — adds src/ to sys.path so every test can import from src/."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

for _p in (_ROOT, _SRC):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)
