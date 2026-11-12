[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$RepositoryDirectory = Split-Path -Parent $PSScriptRoot
$MainPython = Join-Path $RepositoryDirectory '.venv\Scripts\python.exe'
$SegmentationDirectory = Join-Path $RepositoryDirectory '.segmentation_env'
$SegmentationPython = Join-Path $SegmentationDirectory 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $MainPython)) {
    throw 'Create .venv and install foci_detector\requirements.txt first.'
}
if (-not (Test-Path -LiteralPath $SegmentationPython)) {
    & $MainPython -m venv $SegmentationDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the segmentation environment.' }
}
& $SegmentationPython -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU PyTorch.' }
& $SegmentationPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-segmentation-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install Cellpose dependencies.' }
Write-Host 'Cellpose 3 CPU environment is ready. Weights are downloaded and cached on first use.'
