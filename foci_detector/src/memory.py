"""Disk-backed work arrays and full-resolution filters with bounded Z blocks."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

_current = ContextVar('image_workspace', default=None)


class DetectionCancelled(Exception):
    """Cooperative stop at a point where mapped arrays can be cleaned safely."""


class ImageWorkspace:
    """Own mappings until results/viewer layers no longer need their pixels."""
    def __init__(self):
        self.directory = TemporaryDirectory(prefix='dna_foci_', dir=os.environ.get('DNA_FOCI_TEMP_DIR'))
        self.arrays = []
        self.counter = 0
        self.progress = None
        self.cancel_event = None

    def allocate(self, shape, dtype=np.float32):
        check_cancelled()
        self.ensure_space(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        path = Path(self.directory.name) / f'{self.counter}.bin'
        self.counter += 1
        arr = np.memmap(path, mode='w+', shape=shape, dtype=dtype)
        self.arrays.append(arr)
        return arr

    def ensure_space(self, required_bytes):
        free = shutil.disk_usage(self.directory.name).free
        if required_bytes + 64 * 1024**2 > free:
            raise RuntimeError(
                f'Not enough temporary disk space: need about '
                f'{required_bytes / 1024**3:.1f} GiB, available '
                f'{free / 1024**3:.1f} GiB in {self.directory.name}. '
                'Free disk space or set DNA_FOCI_TEMP_DIR to a folder on a larger drive.'
            )

    def close(self):
        for arr in self.arrays:
            if not arr._mmap.closed:
                arr._mmap.close()
        self.arrays.clear()
        self.directory.cleanup()

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError):
            pass


@contextmanager
def using_workspace(workspace):
    token = _current.set(workspace)
    try:
        yield workspace
    finally:
        _current.reset(token)


def allocate(shape, dtype=np.float32):
    workspace = _current.get()
    return (workspace.allocate(shape, dtype) if workspace is not None
            else np.zeros(shape, dtype=dtype))


def current_workspace():
    return _current.get()


def report(message):
    check_cancelled()
    workspace = _current.get()
    if workspace is not None and workspace.progress is not None:
        workspace.progress(message)


def check_cancelled():
    workspace = _current.get()
    if (workspace is not None and workspace.cancel_event is not None
            and workspace.cancel_event.is_set()):
        raise DetectionCancelled('Detection cancelled.')


def blocks(shape, budget_mb=32):
    plane_bytes = int(np.prod(shape[1:])) * 4
    step = max(1, min(8, budget_mb * 1024**2 // max(1, plane_bytes)))
    for start in range(0, shape[0], step):
        check_cancelled()
        yield start, min(shape[0], start + step)


def release(array):
    workspace = _current.get()
    if workspace is not None:
        for index, owned in enumerate(workspace.arrays):
            if owned is array:
                filename = owned.filename
                owned._mmap.close()
                workspace.arrays.pop(index)
                Path(filename).unlink()
                break


def filter_volume(source, kind, *, sigma=None, size=None, mode='reflect', out=None):
    """Filter overlapping Z blocks with the same boundary rules as SciPy."""
    if out is None:
        out = allocate(source.shape)
    if kind == 'gaussian':
        halo = int(4 * sigma[0] + 0.5)
        function, kwargs = gaussian_filter, {'sigma': sigma}
    else:
        halo = size[0] // 2
        function = median_filter if kind == 'median' else maximum_filter
        kwargs = {'size': size}
    for start, stop in blocks(source.shape):
        report(f'{kind.capitalize()} filter: slices {start + 1} to {stop}/{source.shape[0]}')
        low, high = max(0, start - halo), min(source.shape[0], stop + halo)
        chunk = np.asarray(source[low:high], dtype=np.float32)
        filtered = function(chunk, mode=mode, **kwargs)
        out[start:stop] = filtered[start-low:stop-low]
    return out


def selected_values(source, mask=None, positive=False):
    """Gather exact statistics onto disk instead of making a full RAM copy."""
    count = 0
    for start, stop in blocks(source.shape):
        chunk = np.asarray(source[start:stop])
        valid = np.isfinite(chunk)
        if mask is not None:
            valid &= mask[start:stop]
        if positive:
            valid &= chunk > 0
        count += int(np.count_nonzero(valid))
    if count == 0:
        return np.empty(0, dtype=np.float32)
    values = allocate((count,))
    offset = 0
    for start, stop in blocks(source.shape):
        chunk = np.asarray(source[start:stop])
        valid = np.isfinite(chunk)
        if mask is not None:
            valid &= mask[start:stop]
        if positive:
            valid &= chunk > 0
        selected = chunk[valid]
        values[offset:offset + len(selected)] = selected
        offset += len(selected)
    return values


def exact_percentile(values, q):
    # values is disposable: partition the memory map in-place, avoiding NumPy's
    # default entire-array copy. No sampling or intensity quantization.
    check_cancelled()
    result = np.percentile(values, q, overwrite_input=True)
    check_cancelled()
    return result


def response_statistics(source, mask, percentile):
    vals = selected_values(source, mask, positive=True)
    if not vals.size:
        return None
    qs = [50] if percentile is None else [50, percentile]
    stats = exact_percentile(vals, qs)
    median = float(stats[0])
    for start in range(0, len(vals), 1024**2):
        check_cancelled()
        part = vals[start:start + 1024**2]
        part[:] = np.abs(part - median)
    mad = float(exact_percentile(vals, 50))
    release(vals)
    return median, 1.4826 * mad + 1e-8, float(stats[-1])
