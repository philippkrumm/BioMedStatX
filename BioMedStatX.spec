# -*- mode: python ; coding: utf-8 -*-
# BioMedStatX.spec — PyInstaller build configuration
# Usage: pyinstaller BioMedStatX.spec
# Output: dist/BioMedStatX/ (onefolder, zip for distribution)
#
# Before building:
#   python tools/convert_icon.py   (creates .ico and .icns from PNG)
#
# For a macOS Universal2 build (runs on both Intel and Apple Silicon):
#   pyinstaller --target-arch universal2 BioMedStatX.spec
#   (requires all dependencies installed as universal2 wheels)

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

a = Analysis(
    ["src/statistical_analyzer.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=[("assets/", "assets")] + all_datas,
    hiddenimports=all_hiddenimports + [
        # PyQt5 extras not always auto-detected
        "PyQt5.sip",
        "PyQt5.QtPrintSupport",
        # matplotlib Qt backend
        "matplotlib.backends.backend_qt5agg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
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
    target_arch=None,                # None = current machine arch; use 'universal2' for fat binary
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
