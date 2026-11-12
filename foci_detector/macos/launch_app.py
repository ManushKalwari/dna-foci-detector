"""Frozen application entry point with persistent startup diagnostics."""
import multiprocessing
import os
from pathlib import Path
import sys


if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.environ.setdefault('QT_API', 'pyqt5')
    if sys.platform == 'darwin':
        log_dir = Path.home() / 'Library' / 'Logs' / 'DNA_Foci_Detector'
    else:
        log_dir = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'DNA_Foci_Detector'
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / 'application.log', 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    print('\nStarting DNA Foci Detector', flush=True)
    check_runtime = '--check-runtime' in sys.argv
    if check_runtime:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import app
    if check_runtime:
        print('Viewer imports and sidebar initialized successfully.', flush=True)
        app.viewer.close()
    else:
        app.main()
