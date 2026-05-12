# BioMedStatX — Build & Release Guide

Step-by-step instructions for producing the Windows `.exe` and macOS `.app`
distributions and publishing them on the GitHub Releases page.

---

## Prerequisites (one-time setup)

### Common
* Python 3.10+ with all `requirements`-style deps installed (the venv used for
  development is enough).
* `PyInstaller >= 6.0` (`pip install pyinstaller`).
* GitHub CLI (`gh`) for uploading releases (optional — can also use the
  web UI). Portable Windows install:
  ```powershell
  $asset = (Invoke-RestMethod 'https://api.github.com/repos/cli/cli/releases/latest' `
            -Headers @{'User-Agent'='installer'}).assets |
           Where-Object name -match 'windows_amd64\.zip$' | Select-Object -First 1
  Invoke-WebRequest $asset.browser_download_url -OutFile $env:TEMP\gh.zip
  Expand-Archive $env:TEMP\gh.zip "$env:LOCALAPPDATA\gh-cli" -Force
  ```
  Add `%LOCALAPPDATA%\gh-cli\bin` to your user PATH, then `gh auth login`.

### Windows-specific
Nothing extra. The `.spec` already references
`tools/win_version_info.txt` and uses the `.ico` icon.

### macOS-specific
* **Python from python.org**, not Homebrew — required for `universal2`
  wheels. Verify:
  ```bash
  python3 -c "import platform; print(platform.python_implementation(), platform.processor())"
  ```
  Reinstall scientific deps as `universal2` so the resulting `.app` runs on
  both Intel and Apple Silicon:
  ```bash
  pip install --upgrade --force-reinstall scipy numpy PyQt5 pingouin statsmodels
  ```
* **Apple Developer ID** (Apple Developer Program, $99/year). Without signing,
  users have to right-click → Open the first time and dismiss a Gatekeeper
  warning. Notarisation removes the warning entirely.
* `iconutil` (built-in on macOS) for high-quality `.icns` generation.

---

## Build steps

### 1. Generate the icons

The repo only ships `assets/Institutslogo.png`. Convert it once per platform:

```bash
python tools/convert_icon.py
```

Produces `assets/Institutslogo.ico` (Windows) and a basic
`assets/Institutslogo.icns` (macOS, single-resolution via Pillow).

**Better `.icns` on macOS** (Retina-quality, multi-resolution):

```bash
SRC=assets/Institutslogo.png
ICONSET=/tmp/biomed.iconset
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
  sips -z $size $size       "$SRC" --out "$ICONSET/icon_${size}x${size}.png"
  sips -z $((size*2)) $((size*2)) "$SRC" --out "$ICONSET/icon_${size}x${size}@2x.png"
done
iconutil -c icns "$ICONSET" -o assets/Institutslogo.icns
rm -rf "$ICONSET"
```

### 2. Clean any previous build

```powershell
# Windows
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
```
```bash
# macOS / Linux
rm -rf dist build
```

### 3. Run PyInstaller

```bash
pyinstaller BioMedStatX.spec --noconfirm
```

* Output (Windows): `dist/BioMedStatX/` — onefolder build, ~400 MB.
* Output (macOS):   `dist/BioMedStatX.app/` — application bundle, ~100 MB.

Build time: 5–10 minutes depending on the machine. Watch `build_log.txt` if
you tee'd the output. Hidden imports for `pingouin`, `statsmodels`, `scipy`,
`sklearn`, `networkx` are handled automatically by `collect_all` in the spec.

### 4. Smoke test

Launch the resulting binary once and verify the main window comes up:

```powershell
# Windows
.\dist\BioMedStatX\BioMedStatX.exe
```
```bash
# macOS
open dist/BioMedStatX.app
```

If anything is missing at runtime, PyInstaller usually reports
`ModuleNotFoundError`. Add the offending module to `hiddenimports` in
`BioMedStatX.spec` and rebuild.

### 5. macOS: sign and notarise (release builds only)

```bash
codesign --deep --force --options runtime \
  --entitlements assets/entitlements.plist \
  --sign "Developer ID Application: Philipp Krumm (TEAMID)" \
  dist/BioMedStatX.app

xcrun notarytool submit dist/BioMedStatX.app \
  --apple-id "philipp-krumm123@outlook.de" \
  --password "<app-specific-password>" \
  --team-id "TEAMID" --wait

xcrun stapler staple dist/BioMedStatX.app
```

Without these steps the user sees a Gatekeeper warning. With them, the app
opens cleanly on any Mac. Replace `TEAMID` with your Apple Developer Team ID.

### 6. Package for distribution

Match the naming used in `v1.0.1` so users can compare versions easily.

```powershell
# Windows
Compress-Archive -Path dist\BioMedStatX\* `
                 -DestinationPath BioMedStatX_windows.zip -Force
```
```bash
# macOS — preserves symlinks + signature
ditto -c -k --sequesterRsrc --keepParent dist/BioMedStatX.app BioMedStatX_macOS.zip
```

The Excel template is shipped separately as
`BioMedStatX_Excel_Template.xlsx` (lives under `assets/templates/` or a path
you decide; not produced by PyInstaller).

---

## Publishing on GitHub

### Option A — `gh` CLI (recommended)

```bash
gh release create v2.0 \
  BioMedStatX_windows.zip \
  --title "BioMedStatX V2.0" \
  --generate-notes \
  --target feature/advanced-stats-automation
```

Add the macOS build later:

```bash
gh release upload v2.0 BioMedStatX_macOS.zip
```

(Drop the `--target` argument once the release branch is `main`.)

### Option B — Web UI

1. Browse to `https://github.com/philippkrumm/BioMedStatX/releases/new`.
2. **Choose a tag**: type `v2.0` and pick "Create new tag: v2.0 on publish".
3. **Target**: select the branch you want to tag (currently
   `feature/advanced-stats-automation`).
4. **Generate release notes** (button) — fills the description from commit
   history.
5. **Attach files**: drag `BioMedStatX_windows.zip` (and later
   `BioMedStatX_macOS.zip` + the Excel template) into the assets area.
6. Publish.

---

## Versioning conventions

The Windows `.exe` version resource and the `.app` bundle version both live
in two files:

| File | Field |
|---|---|
| `tools/win_version_info.txt` | `filevers`, `prodvers`, `FileVersion`, `ProductVersion` |
| `BioMedStatX.spec` (`BUNDLE` block) | `CFBundleShortVersionString`, `CFBundleVersion` |

Bump **both** when cutting a new release. Keep them in sync with the Git tag
(`v2.0` → `2.0.0`).

---

## Troubleshooting

* **`ModuleNotFoundError` at runtime** → add to `hiddenimports` in the spec.
* **`Qt platform plugin "windows" could not be initialised`** → the `assets/`
  directory wasn't copied into the bundle. Check the `datas=[...]` entry in
  the spec.
* **macOS "App is damaged and can't be opened"** → bundle wasn't notarised
  *or* the user downloaded the zip via a quarantined browser. Notarising +
  stapling fixes this; manually clearing the attribute also works:
  `xattr -dr com.apple.quarantine /Applications/BioMedStatX.app`.
* **Excel export missing** → `src/export_dispatcher.py` currently has the
  Excel calls commented out (HTML-only mode). Uncomment to restore.
