import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_stats_functions_import_is_headless_safe():
    """Importing stats_functions must not create a QApplication in headless mode."""
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "import analysis.stats_functions as stats_functions; "
        "from PyQt5.QtWidgets import QApplication; "
        "print('QAPP_EXISTS=' + str(QApplication.instance() is not None))"
    )

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0, (
        f"Import crashed with rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "QAPP_EXISTS=False" in proc.stdout
