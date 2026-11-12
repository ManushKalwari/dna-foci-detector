"""Bounded-grid segmentation with an isolated, cancellable CPU helper."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time

import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects, distance_transform_edt
from skimage.transform import resize

from src.memory import allocate, check_cancelled, report

_session = ContextVar('segmentation_session', default=None)


@dataclass(frozen=True)
class SegmentationSettings:
    backend: str = 'cellpose'
    model: str = 'cyto3'
    diameter_um: float | None = None
    max_xy: int = 512
    mode: str = '3d'
    intensity_gamma: float = 2.
    min_volume_um3: float = 100.
    network_max_xy: int = 128
    min_signal_ratio: float = 1.3

    def __post_init__(self):
        if self.backend not in ('cellpose', 'legacy'):
            raise ValueError('Segmentation backend must be cellpose or legacy.')
        if self.model not in ('cyto3', 'nuclei'):
            raise ValueError('Cellpose model must be cyto3 or nuclei.')
        if self.mode not in ('3d', 'stitched'):
            raise ValueError('Cellpose mode must be 3d or stitched.')
        if not np.isfinite(self.intensity_gamma) or not 0 < self.intensity_gamma <= 3:
            raise ValueError('Cellpose intensity_gamma must be positive and at most 3.')
        if not np.isfinite(self.min_volume_um3) or self.min_volume_um3 < 0:
            raise ValueError('Minimum nucleus volume must be nonnegative cubic micrometres.')
        if self.diameter_um is not None and (not np.isfinite(self.diameter_um) or self.diameter_um <= 0):
            raise ValueError('Nucleus diameter must be positive micrometres, or None for automatic.')
        if not isinstance(self.max_xy, int) or not 64 <= self.max_xy <= 512:
            raise ValueError('Segmentation max_xy must be an integer between 64 and 512.')
        if not isinstance(self.network_max_xy, int) or not 64 <= self.network_max_xy <= 256:
            raise ValueError('Segmentation network_max_xy must be an integer between 64 and 256.')
        if not np.isfinite(self.min_signal_ratio) or self.min_signal_ratio < 0:
            raise ValueError('Minimum nuclear signal ratio must be nonnegative (zero disables it).')


class SegmentationSession:
    """One helper per batch. This object is owned by the detection thread."""
    def __init__(self, python=None):
        default = Path(__file__).resolve().parents[2] / '.segmentation_env' / 'Scripts' / 'python.exe'
        self.packaged = bool(getattr(sys, 'frozen', False)) and python is None and not os.environ.get('DNA_FOCI_SEGMENTATION_PYTHON')
        if self.packaged:
            default = Path(sys.executable).resolve().parent / 'segmentation' / 'CellposeWorker.exe'
        self.python = Path(python or os.environ.get('DNA_FOCI_SEGMENTATION_PYTHON', default))
        self.process = None
        self.directory = TemporaryDirectory(prefix='dna_foci_cellpose_')
        self.log = None
        self.responses = Queue()

    def _start(self):
        if not self.python.is_file():
            if self.packaged:
                raise RuntimeError('Cellpose runtime is missing. Extract the complete application folder, including segmentation.')
            raise RuntimeError('Cellpose environment is missing. Run setup_segmentation.ps1 before starting detection.')
        if not Path(self.directory.name).exists():
            self.directory = TemporaryDirectory(prefix='dna_foci_cellpose_')
            self.responses = Queue()
        self.log = open(Path(self.directory.name) / 'helper.log', 'w+', encoding='utf-8')
        env = os.environ.copy()
        env.update({'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                    'NUMBA_NUM_THREADS': '4'})
        command = ([str(self.python)] if self.packaged else
                   [str(self.python), '-u', str(Path(__file__).with_name('cellpose_worker.py'))])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        process = self.process
        def read_responses():
            try:
                for line in process.stdout:
                    self.responses.put(line)
            finally:
                self.responses.put(None)
        self.reader = Thread(target=read_responses, daemon=True)
        self.reader.start()

    def infer(self, image, model, diameter_px, anisotropy, sizing_sigma_px=2., mode='3d', intensity_gamma=1., network_max_xy=128):
        check_cancelled()
        if self.process is None:
            self._start()
        directory = Path(self.directory.name)
        input_path, output_path = directory / 'input.npy', directory / 'labels.npy'
        np.save(input_path, image, allow_pickle=False)
        request = dict(input=str(input_path), output=str(output_path), model=model,
                       diameter_px=diameter_px, anisotropy=anisotropy, sizing_sigma_px=sizing_sigma_px,
                       mode=mode, intensity_gamma=intensity_gamma, network_max_xy=network_max_xy)
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            started = last_update = time.monotonic()
            while True:
                check_cancelled()
                try:
                    line = self.responses.get(timeout=0.1)
                    break
                except Empty:
                    now = time.monotonic()
                    if now - last_update >= 5:
                        report(f'Cellpose {model}: loading model / CPU segmentation ({int(now-started)} s elapsed)')
                        last_update = now
            if line is None:
                self.log.flush()
                self.log.seek(0)
                raise RuntimeError('Cellpose helper stopped: ' + self.log.read()[-3000:])
            response = json.loads(line)
            if not response.get('ok'):
                raise RuntimeError('Cellpose segmentation failed: ' + response['error'])
            return np.load(output_path, allow_pickle=False), response
        except BaseException:
            # Never leave inference running after cancellation or a broken pipe.
            self.close()
            raise

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self.reader.join(timeout=3)
            process.stdin.close()
            process.stdout.close()
        if self.log is not None:
            self.log.close()
            self.log = None
        self.directory.cleanup()


@contextmanager
def segmentation_session():
    session = SegmentationSession()
    token = _session.set(session)
    try:
        yield session
    finally:
        session.close()
        _session.reset(token)


def prepare_grid(stack, spacing, max_xy=512):
    if stack.ndim != 3 or not all(stack.shape):
        raise ValueError('Expected a nonempty Z/Y/X nucleus stack.')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Cellpose requires positive calibrated Z/Y/X voxel sizes.')
    factor = min(1., max_xy / max(stack.shape[1:]))
    shape = (stack.shape[0], max(1, round(stack.shape[1]*factor)), max(1, round(stack.shape[2]*factor)))
    image = np.empty(shape, dtype=np.float32)
    for z in range(shape[0]):
        check_cancelled()
        plane = np.asarray(stack[z], dtype=np.float32)
        if not np.all(np.isfinite(plane)):
            raise ValueError('Nucleus stack contains non-finite intensities.')
        image[z] = resize(plane, shape[1:], order=1, preserve_range=True, anti_aliasing=True)
    low, high = np.percentile(image, (1, 99.5))
    if high <= low:
        raise RuntimeError('No nuclear intensity contrast available for segmentation.')
    image = np.clip((image-low)/(high-low), 0, 1)
    grid_spacing = spacing * np.array(stack.shape) / shape
    # Cellpose uses a single XY scale. Correct rare non-square XY sampling.
    if not np.isclose(grid_spacing[1], grid_spacing[2], rtol=.02):
        raise ValueError('Cellpose currently requires approximately equal X/Y voxel sizes.')
    return image, grid_spacing


def restore_labels(labels, shape):
    """Nearest-neighbour mapping without allocating a full float volume."""
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Cellpose returned invalid instance labels.')
    out = allocate(shape, dtype=np.int32)
    zi = np.minimum(((np.arange(shape[0])+.5)*labels.shape[0]/shape[0]).astype(int), labels.shape[0]-1)
    yi = np.minimum(((np.arange(shape[1])+.5)*labels.shape[1]/shape[1]).astype(int), labels.shape[1]-1)
    xi = np.minimum(((np.arange(shape[2])+.5)*labels.shape[2]/shape[2]).astype(int), labels.shape[2]-1)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    for z in range(shape[0]):
        check_cancelled()
        plane = labels[zi[z]][np.ix_(yi, xi)]
        out[z] = np.where(plane > 0, np.searchsorted(ids, plane)+1, 0)
        # Fill enclosed dark nucleoli without taking pixels of another instance.
        occupied = out[z] != 0
        proposals = np.zeros(shape[1:], dtype=np.int32)
        conflicts = np.zeros(shape[1:], dtype=bool)
        # Work in each object's bounding box. Full-frame filling per instance
        # becomes prohibitively slow on images containing many tiny fragments.
        for new_id, region in enumerate(find_objects(out[z]), 1):
            if region is None:
                continue
            hole = binary_fill_holes(out[z][region] == new_id) & ~occupied[region]
            conflicts[region] |= hole & (proposals[region] != 0)
            proposals[region][hole] = new_id
        fill = (~occupied) & (~conflicts)
        out[z][fill] = proposals[fill]
    # A tiny object can vanish when mapped to a coarser processing grid.
    surviving = sorted(label_counts(out))
    for z in range(shape[0]):
        out[z] = np.where(out[z] > 0, np.searchsorted(surviving, out[z])+1, 0)
    return out


def label_counts(labels):
    """Count instances one plane at a time, including disk-backed native grids."""
    counts = {}
    for plane in labels:
        check_cancelled()
        ids, sizes = np.unique(plane, return_counts=True)
        for label, size in zip(ids, sizes):
            if label > 0:
                counts[int(label)] = counts.get(int(label), 0) + int(size)
    return dict(sorted(counts.items()))


def filter_small_instances(labels, spacing, minimum_um3):
    counts = label_counts(labels)
    voxel_volume = float(np.prod(spacing))
    rejected = [{'original_id': i, 'volume_um3': n*voxel_volume}
                for i, n in counts.items() if n*voxel_volume < minimum_um3]
    rejected_ids = {item['original_id'] for item in rejected}
    kept = [i for i in counts if i not in rejected_ids]
    for plane in labels:
        check_cancelled()
        if not kept:
            plane[:] = 0
            continue
        indices = np.searchsorted(kept, plane)
        valid = (indices < len(kept)) & (np.take(kept, indices, mode='clip') == plane)
        plane[:] = np.where(valid, indices+1, 0)
    return rejected


def filter_signal_instances(labels, image, spacing, minimum_um3, minimum_ratio):
    """Reject unsupported background masks without imposing a geometric shape.

    Compare the upper intensity decile to a 1 µm exterior shell. This tolerates
    dark nucleoli and boundary staining. Other instance labels are excluded
    from the background reference so touching nuclei do not invalidate each other.
    """
    rejected, unavailable = [], []
    if minimum_ratio == 0:
        return rejected, unavailable
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Cellpose returned invalid instance labels.')
    if int(labels.max()) > 1000000:
        raise ValueError('Cellpose returned an invalid instance label range.')
    padding = np.ceil(1./np.asarray(spacing)).astype(int)
    voxel_volume = float(np.prod(spacing))
    # Keep the original segmentation intact until every reference is measured.
    for label_id, box in enumerate(find_objects(labels), 1):
        check_cancelled()
        if box is None:
            continue
        volume = np.count_nonzero(labels[box] == label_id)*voxel_volume
        if volume < minimum_um3:
            continue  # The final physical-volume filter reports these separately.
        region = tuple(slice(max(0, s.start-int(p)), min(n, s.stop+int(p)))
                       for s, p, n in zip(box, padding, labels.shape))
        local = labels[region]
        inside = local == label_id
        distance = distance_transform_edt(~inside, sampling=spacing)
        shell = (distance <= 1.) & (local == 0)
        if not np.any(shell):
            unavailable.append(label_id)
            continue
        signal = float(np.quantile(image[region][inside], .9))
        background = float(np.quantile(image[region][shell], .9))
        ratio = signal/max(background, 1e-6)
        if ratio < minimum_ratio or signal-background < .03:
            rejected.append(dict(original_id=label_id, volume_um3=volume,
                                 signal_ratio=ratio, normalized_contrast=signal-background,
                                 reason='insufficient_local_nuclear_signal'))
    rejected_ids = [item['original_id'] for item in rejected]
    if rejected_ids:
        for plane in labels:
            plane[np.isin(plane, rejected_ids)] = 0
    return rejected, unavailable


def segment_cellpose(stack, spacing, settings):
    started = time.perf_counter()
    report(f'Preparing AF488 nucleus stack for Cellpose {settings.model}')
    image, grid_spacing = prepare_grid(stack, spacing, settings.max_xy)
    diameter = None if settings.diameter_um is None else settings.diameter_um / grid_spacing[2]
    arguments = dict(model=settings.model, diameter_px=diameter,
                     anisotropy=float(grid_spacing[0]/grid_spacing[2]),
                     sizing_sigma_px=float(.3/grid_spacing[2]), mode=settings.mode,
                     intensity_gamma=settings.intensity_gamma, network_max_xy=settings.network_max_xy)
    session = _session.get()
    if session is None:
        with segmentation_session() as owned:
            labels, diagnostics = owned.infer(image, **arguments)
    else:
        labels, diagnostics = session.infer(image, **arguments)
    if tuple(labels.shape) != tuple(image.shape):
        raise RuntimeError(f'Cellpose returned shape {labels.shape}; expected {image.shape}.')
    unsupported, unavailable = filter_signal_instances(
        labels, image, grid_spacing, settings.min_volume_um3, settings.min_signal_ratio)
    labels = restore_labels(labels, stack.shape)
    rejected = filter_small_instances(labels, spacing, settings.min_volume_um3)
    sizes = label_counts(labels)
    if not sizes:
        raise RuntimeError(f'Cellpose {settings.model} detected no nuclei after signal validation '
                           f'and the {settings.min_volume_um3:g} µm³ minimum-volume filter. '
                           'Review the nucleus channel, diameter, or segmentation settings. '
                           'Try the other Cellpose model or review the AF488 nuclear channel.')
    mask = allocate(labels.shape, dtype=bool)
    # Projection is display-only; smallest Z wins in overlapping XY columns.
    projection = np.zeros(labels.shape[1:], dtype=np.int32)
    for z in range(labels.shape[0]):
        check_cancelled()
        mask[z] = labels[z] > 0
        select = (projection == 0) & mask[z]
        projection[select] = labels[z][select]
    diagnostics.update(backend='cellpose', model=settings.model, mode=settings.mode, input_shape=list(stack.shape),
                       inference_shape=list(image.shape), grid_spacing_um=grid_spacing.tolist(),
                       anisotropy=float(grid_spacing[0]/grid_spacing[2]),
                       sizing_smoothing_um=.3,
                       intensity_gamma=settings.intensity_gamma,
                       min_volume_um3=settings.min_volume_um3,
                       rejected_small_instances=rejected,
                       rejected_low_support_instances=unsupported,
                       signal_support_unavailable=unavailable,
                       min_signal_ratio=settings.min_signal_ratio,
                       diameter_um=float(diagnostics['diameter_px']*grid_spacing[2]),
                       estimated_diameter_um=float(diagnostics.get('estimated_diameter_px', diagnostics['diameter_px'])*grid_spacing[2]),
                       instance_voxel_counts=sizes, total_seconds=time.perf_counter()-started)
    diagnostics['mask_review_required'] = bool(diagnostics.get('diameter_clamped_for_memory') or unavailable)
    centers = []
    for label_id in sizes:
        # Use the largest XY section; projection overlap can hide an instance.
        areas = [np.count_nonzero(plane == label_id) for plane in labels]
        plane = labels[int(np.argmax(areas))]
        centers.append(np.argwhere(plane == label_id).mean(axis=0))
    report(f'Cellpose accepted {len(sizes)} nucleus/nuclei')
    return dict(roi_proj=np.mean(stack, axis=0), mask2d=projection > 0, labels2d=projection,
                roi_mask_3d=mask, roi_labels_3d=labels, nucleus_count=len(sizes),
                centers=np.array(centers),
                radii=[], candidate_diagnostics=[diagnostics], segmentation_diagnostics=diagnostics)
