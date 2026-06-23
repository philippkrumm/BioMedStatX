# -*- mode: python ; coding: utf-8 -*-
# BioMedStatX.spec — PyInstaller build configuration
# Usage: pyinstaller BioMedStatX.spec
# Output: dist/BioMedStatX/ (onefolder, zip for distribution)
#
# Before building:
#   python tools/convert_icon.py   (creates .ico and .icns from PNG)
#
# macOS post-build steps (requires Apple Developer ID):
#   1. Sign:
#      codesign --deep --force --options runtime \
#        --entitlements assets/entitlements.plist \
#        --sign "Developer ID Application: Ihr Name (TeamID)" \
#        dist/BioMedStatX.app
#
#   2. Notarize:
#      xcrun notarytool submit dist/BioMedStatX.app \
#        --apple-id "ihre@mail.de" \
#        --password "app-spezifisches-passwort" \
#        --team-id "TEAMID" --wait
#
#   3. Staple:
#      xcrun stapler staple dist/BioMedStatX.app
#
# Note: target_arch='universal2' requires all deps as universal2 wheels.
# Check: pip install --upgrade scipy numpy PyQt5 (on Apple Silicon Mac with
# universal2 Python from python.org, not Homebrew).

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

icon = (
    "assets/Institutslogo.icns" if IS_MAC
    else "assets/Institutslogo.ico" if IS_WIN
    else None
)

# collect_all gathers binaries, datas, and hiddenimports automatically
# for packages with many dynamic/lazy-loaded submodules
_pkgs = ["pingouin", "statsmodels", "scipy", "sklearn", "networkx"]
all_datas, all_binaries, all_hiddenimports = [], [], []
for pkg in _pkgs:
    d, b, h = collect_all(pkg)
    all_datas         += d
    all_binaries      += b
    all_hiddenimports += h

# Drop test-fixture data files. They are large, never used at runtime, and
# on Windows the deeply-nested statsmodels test names blow past the 260-char
# MAX_PATH limit when the project lives under a long path (e.g. OneDrive).
# Each entry is (src_path, dst_path_in_bundle) — filter on the destination.
import os as _os
# Drop test fixtures (large, unused, blow Windows MAX_PATH) and panel (web
# dashboard lib pulled in transitively — deeply nested JS assets also exceed
# MAX_PATH and panel is never used at runtime).
_EXCLUDED_ROOTS = {"tests", "test", "panel"}
def _is_excluded_data(entry):
    dst = entry[1].replace("\\", "/")
    parts = dst.split("/")
    return bool(_EXCLUDED_ROOTS.intersection(parts))

def _is_excluded_hiddenimport(module_name):
    parts = module_name.split(".")
    return (
        "tests" in parts
        or "test" in parts
        or module_name.endswith(".conftest")
        or module_name == "conftest"
    )

all_datas = [e for e in all_datas if not _is_excluded_data(e)]
all_binaries = [e for e in all_binaries if not _is_excluded_data(e)]
all_hiddenimports = [m for m in all_hiddenimports if not _is_excluded_hiddenimport(m)]

if IS_WIN:
    _conda_bin = _os.path.join(sys.base_prefix, "Library", "bin")
    _conda_runtime_dlls = [
        "ffi-8.dll",
        "libbz2.dll",
        "libcrypto-3-x64.dll",
        "libexpat.dll",
        "liblzma.dll",
        "libssl-3-x64.dll",
        "sqlite3.dll",
    ]
    for _dll in _conda_runtime_dlls:
        _src = _os.path.join(_conda_bin, _dll)
        if _os.path.exists(_src):
            all_binaries.append((_src, "."))

a = Analysis(
    ["src/analysis/statistical_analyzer.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=[("assets/", "assets"), ("src/templates/", "templates")] + all_datas,
    hiddenimports=all_hiddenimports + [
        # PyQt5 extras not always auto-detected
        "PyQt5.sip",
        "PyQt5.QtPrintSupport",
        # matplotlib Qt backend
        "matplotlib.backends.backend_qt5agg",
        # NumPy exposes f2py lazily via numpy.__getattr__; SciPy's import path
        # can touch it even though the app does not call f2py directly.
        "numpy.f2py",
    ],
    hookspath=[],
    hooksconfig={},
    # Runs before the entry script: redirects None stdout/stderr (windowed build)
    # so import-time prints in scipy/statsmodels/etc. never crash the GUI.
    runtime_hooks=["tools/pyi_rth_stdio.py"],
    # PySide6/PySide2/PyQt6 may be installed in the dev environment as deps of
    # other tools (plotly, jupyter widgets, etc.). The app uses PyQt5 only, so
    # exclude the alternative Qt bindings explicitly — PyInstaller refuses to
    # ship multiple Qt bindings in one frozen app.
    # IPython/jupyter/jedi pull deeply-nested typeshed paths that break the
    # Windows MAX_PATH limit and are never used at runtime in a frozen GUI.
    excludes=[
        "tkinter",
        "PySide6", "PySide2", "PyQt6", "shiboken6", "shiboken2",
        "IPython", "jupyter", "jupyter_client", "jupyter_core",
        "ipykernel", "ipywidgets", "notebook", "nbformat", "nbconvert",
        "jedi", "parso",
        "pytest", "_pytest",
        "sphinx", "docutils",
        "panel", "bokeh", "param",
        # nltk is pulled in transitively but never used by the app. Its
        # PyInstaller runtime hook eagerly imports nltk -> scipy.stats ->
        # numpy.f2py at startup; in a windowed build (sys.stdout is None)
        # numpy.f2py.cfuncs crashes with "'NoneType' object has no attribute
        # 'write'". Excluding nltk removes the rthook and the crash.
        "nltk",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,           # onefolder mode (not onefile)
    name="BioMedStatX",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                   # no console window
    disable_windowed_traceback=False,
    argv_emulation=IS_MAC,           # macOS: support file drag-and-drop onto app icon
    target_arch='arm64' if IS_MAC else None,  # arm64 = Apple Silicon (M1+)
    codesign_identity=None,          # set to Apple Developer ID for signed distribution
    entitlements_file="assets/entitlements.plist" if IS_MAC else None,
    icon=icon,
    version="tools/win_version_info.txt" if IS_WIN else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BioMedStatX",
)

# macOS: wrap in .app bundle
if IS_MAC:
    app = BUNDLE(
        coll,
        name="BioMedStatX.app",
        icon="assets/Institutslogo.icns",
        bundle_identifier="de.ukaachen.biomed.biomedstatx",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "2.0",
            "CFBundleVersion": "2.0.0",
            "CFBundleName": "BioMedStatX",
        },
    )
