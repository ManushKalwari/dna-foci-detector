"""Verify the frozen viewer, Cellpose models, architecture, and code signature."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def main():
    bundle = Path(sys.argv[1]).resolve()
    main_executable = bundle / 'Contents' / 'MacOS' / 'DNA_Foci_Detector'
    worker = bundle / 'Contents' / 'Resources' / 'segmentation' / 'CellposeWorker'
    if not main_executable.is_file() or not worker.is_file():
        raise RuntimeError('Packaged viewer or Cellpose worker is missing.')

    for executable in (main_executable, worker):
        architecture = run(['file', str(executable)], capture_output=True).stdout
        if 'arm64' not in architecture:
            raise RuntimeError(f'Expected Apple Silicon executable: {architecture.strip()}')

    requests = ''.join(
        json.dumps({'model': model, 'check_runtime': True}) + '\n'
        for model in ('cyto3', 'nuclei')
    )
    check = run(
        [str(worker)], input=requests, capture_output=True, timeout=300,
    )
    responses = [json.loads(line) for line in check.stdout.splitlines() if line.strip()]
    if len(responses) != 2 or not all(item.get('ok') and item.get('weights_loaded') for item in responses):
        raise RuntimeError(f'Cellpose model check failed: {check.stdout}\n{check.stderr}')

    environment = os.environ.copy()
    environment['QT_QPA_PLATFORM'] = 'minimal'
    try:
        run([str(main_executable), '--check-runtime'], env=environment, timeout=180)
    except Exception:
        log = Path.home() / 'Library' / 'Logs' / 'DNA_Foci_Detector' / 'application.log'
        if log.is_file():
            print(log.read_text(encoding='utf-8', errors='replace'), file=sys.stderr)
        raise
    run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', str(bundle)])
    print('Apple Silicon viewer startup and both Cellpose model checks passed.')


if __name__ == '__main__':
    main()
