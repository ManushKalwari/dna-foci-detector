#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This build requires an Apple Silicon macOS runner." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$APP_DIR/.." && pwd)"
BUILD_DIR="$APP_DIR/build/macos"
DIST_DIR="$APP_DIR/dist/macos"
VIEWER_ENV="$BUILD_DIR/viewer-env"
WORKER_ENV="$BUILD_DIR/worker-env"

rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

python3 -m venv "$VIEWER_ENV"
python3 -m venv "$WORKER_ENV"
"$VIEWER_ENV/bin/python" -m pip install --upgrade pip wheel
"$WORKER_ENV/bin/python" -m pip install --upgrade pip wheel
"$VIEWER_ENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-viewer.txt" pyinstaller==6.19.0
"$WORKER_ENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-segmentation.txt" pyinstaller==6.19.0

printf '%s\n' \
    '{"model":"cyto3","check_runtime":true}' \
    '{"model":"nuclei","check_runtime":true}' \
    | "$WORKER_ENV/bin/python" "$APP_DIR/src/cellpose_worker.py"

"$WORKER_ENV/bin/python" -m PyInstaller \
    --noconfirm --onedir --console --name CellposeWorker \
    --distpath "$BUILD_DIR/worker-dist" \
    --workpath "$BUILD_DIR/worker-work" \
    --specpath "$BUILD_DIR" \
    --collect-all cellpose --collect-all fastremap \
    --exclude-module cellpose.gui --exclude-module PyQt5 --exclude-module PySide6 \
    "$APP_DIR/src/cellpose_worker.py"

"$VIEWER_ENV/bin/python" -m PyInstaller \
    --noconfirm --onedir --windowed --name DNA_Foci_Detector \
    --osx-bundle-identifier org.manushkalwari.dnafocidetector \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR/viewer-work" \
    --specpath "$BUILD_DIR" \
    --paths "$APP_DIR" \
    --collect-all napari --collect-all vispy --collect-all napari_builtins \
    --collect-all aicspylibczi --collect-all imagecodecs --collect-all magicgui \
    --recursive-copy-metadata napari \
    --exclude-module torch --exclude-module cellpose \
    --exclude-module PySide6 --exclude-module PyQt6 \
    "$SCRIPT_DIR/launch_app.py"

APP_BUNDLE="$DIST_DIR/DNA_Foci_Detector.app"
SEGMENTATION="$APP_BUNDLE/Contents/Resources/segmentation"
mkdir -p "$SEGMENTATION"
cp -R "$BUILD_DIR/worker-dist/CellposeWorker/." "$SEGMENTATION/"
mkdir -p "$SEGMENTATION/models"
for model in cyto3 nucleitorch_0 size_cyto3.npy size_nucleitorch_0.npy; do
    cp "$HOME/.cellpose/models/$model" "$SEGMENTATION/models/$model"
done
cp "$SCRIPT_DIR/READ_ME_FIRST.txt" "$DIST_DIR/READ_ME_FIRST.txt"

codesign --force --deep --sign - "$APP_BUNDLE"
"$VIEWER_ENV/bin/python" "$SCRIPT_DIR/verify_macos.py" "$APP_BUNDLE"

ARCHIVE="$DIST_DIR/DNA_Foci_Detector_macOS_AppleSilicon.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$ARCHIVE"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
echo "Created $ARCHIVE"
