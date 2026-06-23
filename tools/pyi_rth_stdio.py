# PyInstaller runtime hook — fix None stdio in windowed builds.
#
# In a --windowed / --noconsole PyInstaller build, sys.stdout and sys.stderr are
# None. Several scientific packages (numpy.f2py, scipy, statsmodels, nltk, ...)
# write to stdout/stderr at *import time*. Writing to None raises
# "AttributeError: 'NoneType' object has no attribute 'write'" and crashes the
# GUI before it ever shows a window.
#
# This hook runs before the frozen entry script and replaces any None stream with
# a sink that silently swallows writes. It kills the entire class of import-time
# I/O crashes once, instead of excluding offending modules one rebuild at a time.
import sys


class _NullStream:
    """Minimal write-only stream that never fails."""
    def write(self, *args, **kwargs):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("NullStream has no file descriptor")


if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()
