"""
convert_icon.py — Einmalig ausführen vor dem PyInstaller-Build.
Konvertiert assets/Institutslogo.png → .ico (Windows) und .icns (macOS).

Usage:
    python tools/convert_icon.py
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "assets" / "Institutslogo.png"

if not src.exists():
    raise FileNotFoundError(f"Logo nicht gefunden: {src}")

img = Image.open(src).convert("RGBA")

# Windows .ico — mehrere Auflösungen eingebettet
ico_path = ROOT / "assets" / "Institutslogo.ico"
img.save(
    ico_path,
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
print(f"Windows icon erstellt: {ico_path}")

# macOS .icns — pillow kann .icns auf allen Plattformen schreiben
# Für ein vollwertiges .icns-Bundle: auf macOS zusätzlich iconutil verwenden
icns_path = ROOT / "assets" / "Institutslogo.icns"
try:
    img.save(icns_path, format="ICNS")
    print(f"macOS icon erstellt:   {icns_path}")
except Exception as e:
    print(f"macOS .icns konnte nicht erstellt werden (nur auf macOS möglich): {e}")
    print("Tipp: Auf macOS mit 'iconutil' aus einem .iconset-Ordner erstellen.")
