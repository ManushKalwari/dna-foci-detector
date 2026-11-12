# DNA Foci Detector

Napari application for segmenting nuclei and detecting, counting, and measuring
fluorescent foci in 3D CZI and LSM microscopy images.

## Features

- Batch processing for CZI and LSM stacks
- Cellpose nucleus segmentation with `cyto3` or `nuclei`
- Multiscale 3D Difference-of-Gaussians focus detection
- Native-resolution 3D watershed measurements
- Physical dimensions from image metadata
- Manual voxel-size prompt when metadata is unavailable
- Raw, processed, and calibrated 3D review views
- Per-nucleus counts and per-focus CSV output

Source images are never modified.

## Method

1. Cellpose segments nuclei in 3D or by linked XY slices.
2. A multiscale 3D Difference-of-Gaussians detector finds local maxima inside
   the nuclear masks.
3. Weak candidates and repeated peaks across adjacent Z planes are removed.
4. A marker-controlled local half-prominence watershed measures each retained
   focus on the native-resolution foci channel.

Reported dimensions are apparent fluorescence extents. They are not
PSF-corrected biological dimensions.

## Installation

Python 3.11 and Windows PowerShell are required.

```powershell
git clone https://github.com/ManushKalwari/dna-foci-detector.git
Set-Location dna-foci-detector
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\foci_detector\requirements.txt
.\foci_detector\setup_segmentation.ps1
```

Cellpose runs in a separate CPU environment to keep its dependency versions
isolated from Napari.

## Run

```powershell
.\.venv\Scripts\python.exe .\foci_detector\app.py
```

Select or drop CZI or LSM files, review the settings, and click **Run
detection**. The default minimum spot volume is `0.2 µm³`. Setting it to zero
disables volume filtering.

## Output

The application saves two files beside the first selected image:

- `spot_counts_all_images.csv`: one row per segmented nucleus, including zero
  counts
- `focus_measurements_all_images.csv`: one row per counted focus

The measurement file contains:

| Column | Description |
| --- | --- |
| `source_image` | Source filename |
| `focus_id` | Spot number within the nucleus |
| `nucleus_id` | Nuclear label |
| `width_x_um` | Watershed extent along X |
| `width_y_um` | Watershed extent along Y |
| `depth_z_um` | Watershed extent along Z |
| `volume_um3` | Calibrated watershed volume |
| `selection_status` | Spot selection result |
| `watershed_status` | `ok`, `review`, or `failed` |

Missing measurements remain blank. Review the nuclear masks, Batch report, and
any watershed result marked `review` or `failed` before analysis.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s .\foci_detector\tests -v
```

## macOS Apple Silicon build

The `Build macOS Apple Silicon` GitHub Actions workflow creates an ad-hoc signed,
self-contained application for Apple Silicon Macs. The artifact includes the
Napari viewer, CPU Cellpose runtime, and both supported Cellpose models. Python
is not required on the destination Mac.

The build is not notarized. Follow `READ_ME_FIRST.txt` in the artifact for the
first launch.
