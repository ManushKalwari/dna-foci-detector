"""Native-resolution physical measurements for detected DNA foci."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, isfinite
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, label as ndi_label
from scipy.optimize import least_squares
from skimage.segmentation import watershed

from src.memory import check_cancelled, report


FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class FocusMeasurement:
    """One detected focus and its two apparent-size measurements."""

    source_image: str
    focus_id: int
    nucleus_id: int
    center_z_px: float
    center_y_px: float
    center_x_px: float
    center_z_um: float
    center_y_um: float
    center_x_um: float
    voxel_size_z_um: float
    voxel_size_y_um: float
    voxel_size_x_um: float
    calibration_source: str
    watershed_voxel_count: int | None = None
    watershed_volume_um3: float | None = None
    watershed_extent_z_um: float | None = None
    watershed_extent_y_um: float | None = None
    watershed_extent_x_um: float | None = None
    watershed_equivalent_diameter_um: float | None = None
    gaussian_fwhm_z_um: float | None = None
    gaussian_fwhm_y_um: float | None = None
    gaussian_fwhm_x_um: float | None = None
    gaussian_halfmax_volume_um3: float | None = None
    gaussian_fit_r2: float | None = None
    gaussian_center_shift_um: float | None = None
    gaussian_fit_status: str = "not_run"
    measurement_status: str = "not_run"
    qc_flags: str = ""

    def to_record(self) -> dict:
        return asdict(self)


FOCUS_MEASUREMENT_CSV_COLUMNS = [
    'source_image', 'focus_id', 'nucleus_id',
    'width_x_um', 'width_y_um', 'depth_z_um', 'volume_um3',
    'selection_status', 'watershed_status',
]


def focus_numbers_within_nuclei(nucleus_ids):
    """Number the supplied spots from one independently within each nucleus."""
    counts = {}
    for nucleus_id in nucleus_ids:
        counts[nucleus_id] = counts.get(nucleus_id, 0) + 1
        yield counts[nucleus_id]


def watershed_export_record(record, selection_status='accepted'):
    """Report watershed dimensions only; Gaussian fit quality is independent."""
    fields = {
        'width_x_um': 'watershed_extent_x_um',
        'width_y_um': 'watershed_extent_y_um',
        'depth_z_um': 'watershed_extent_z_um',
        'volume_um3': 'watershed_volume_um3',
    }
    values = {column: record.get(source) for column, source in fields.items()}
    measured = all(v is not None and isfinite(v) and v > 0 for v in values.values())
    flags = (record.get('qc_flags') or '').split(';')
    review = any(flag.startswith('watershed_') or flag == 'patch_clipped_by_image'
                 for flag in flags)
    return dict(source_image=record.get('source_image'), focus_id=record.get('focus_id'),
                nucleus_id=record.get('nucleus_id'), **values,
                selection_status=selection_status,
                watershed_status=('review' if review else 'ok') if measured else 'failed')


@dataclass
class _PatchSpec:
    measurement: FocusMeasurement
    radius_um: float
    center: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    data: np.ndarray
    clipped: bool
    read_error: bool = False


def processed_to_native_center(
    processed_zyx: Iterable[float],
    processing_scale_zyx: Iterable[int],
    trimmed_z_planes: int = 0,
) -> np.ndarray:
    """Map a detector center to the center of its native-resolution block."""
    point = np.asarray(tuple(processed_zyx), dtype=float)
    point[0] += trimmed_z_planes
    scale = np.asarray(tuple(processing_scale_zyx), dtype=float)
    return (point + 0.5) * scale - 0.5


def _make_specs(
    image_shape_zyx,
    detections,
    voxel_size_um_zyx,
    processing_scale_zyx,
    source_image,
    calibration_source,
    trimmed_z_planes=0,
):
    spacing = np.asarray(voxel_size_um_zyx, dtype=float)
    scale = np.asarray(processing_scale_zyx, dtype=float)
    processed_xy_um = float(np.mean(spacing[1:] * scale[1:]))
    shape = np.asarray(image_shape_zyx, dtype=int)
    specs = []
    for focus_id, (nucleus_id, blob) in enumerate(detections, start=1):
        center = processed_to_native_center(blob[:3], scale, trimmed_z_planes)
        radius_um = max(float(blob[3]) * processed_xy_um, float(max(spacing)))
        half_width_um = max(1.5, 3.5 * radius_um)
        half_voxels = np.maximum(2, np.ceil(half_width_um / spacing).astype(int))
        intended_start = np.floor(center).astype(int) - half_voxels
        intended_stop = np.floor(center).astype(int) + half_voxels + 1
        start = np.maximum(0, intended_start)
        stop = np.minimum(shape, intended_stop)
        clipped = bool(np.any(start != intended_start) or np.any(stop != intended_stop))
        measurement = FocusMeasurement(
            source_image=str(source_image), focus_id=focus_id,
            nucleus_id=int(nucleus_id),
            center_z_px=float(center[0]), center_y_px=float(center[1]),
            center_x_px=float(center[2]),
            center_z_um=float(center[0] * spacing[0]),
            center_y_um=float(center[1] * spacing[1]),
            center_x_um=float(center[2] * spacing[2]),
            voxel_size_z_um=float(spacing[0]),
            voxel_size_y_um=float(spacing[1]),
            voxel_size_x_um=float(spacing[2]),
            calibration_source=calibration_source,
        )
        specs.append(_PatchSpec(
            measurement=measurement, radius_um=radius_um, center=center,
            start=start, stop=stop,
            data=np.empty(tuple(stop - start), dtype=np.float32), clipped=clipped,
        ))
    return specs


def _fill_native_patches(image, spot_channel: int, specs: list[_PatchSpec]):
    """Read every needed native Z plane once and copy all intersecting patches."""
    if not specs:
        return
    z_to_specs = {}
    for spec in specs:
        for z in range(int(spec.start[0]), int(spec.stop[0])):
            z_to_specs.setdefault(z, []).append(spec)
    for index, z in enumerate(sorted(z_to_specs), start=1):
        check_cancelled()
        report(f'Measuring foci: reading native slice {index}/{len(z_to_specs)}')
        try:
            plane = image[z, spot_channel]
            if hasattr(plane, 'compute'):
                plane = plane.compute(scheduler='synchronous')
            plane = np.asarray(plane)
        except Exception:
            for spec in z_to_specs[z]:
                spec.data[z - int(spec.start[0])] = np.nan
                spec.read_error = True
            continue
        for spec in z_to_specs[z]:
            local_z = z - int(spec.start[0])
            spec.data[local_z] = plane[
                int(spec.start[1]):int(spec.stop[1]),
                int(spec.start[2]):int(spec.stop[2]),
            ]


def _physical_grids(spec: _PatchSpec, spacing: np.ndarray):
    axes = [
        (np.arange(int(spec.start[i]), int(spec.stop[i])) - spec.center[i]) * spacing[i]
        for i in range(3)
    ]
    return np.meshgrid(*axes, indexing='ij')


def _local_peak_and_background(image, grids, radius_um):
    zz, yy, xx = grids
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    core_radius = max(0.5 * radius_um, 0.12)
    shell_inner = max(1.8 * radius_um, 0.45)
    shell_outer = max(3.0 * radius_um, 0.8)
    core = image[distance <= core_radius]
    shell = image[(distance >= shell_inner) & (distance <= shell_outer)]
    if core.size == 0:
        return None
    background = float(np.median(shell)) if shell.size >= 8 else float(np.percentile(image, 20))
    peak = float(np.percentile(core, 95))
    return peak, background, distance, shell_outer


def _component_containing(mask, seed):
    labels, _ = ndi_label(mask)
    label_id = int(labels[seed])
    return labels == label_id if label_id else np.zeros_like(mask, dtype=bool)


def _measure_watershed(spec, all_specs, smooth, grids, spacing, flags):
    local = _local_peak_and_background(smooth, grids, spec.radius_um)
    if local is None:
        flags.add('watershed_empty_core')
        return False
    peak, background, distance, support_radius = local
    prominence = peak - background
    if not np.isfinite(prominence) or prominence <= 0:
        flags.add('watershed_no_prominence')
        return False

    central_threshold = background + 0.5 * prominence
    central_eligible = (smooth >= central_threshold) & (distance <= support_radius)
    union_mask = central_eligible.copy()
    markers = np.zeros(smooth.shape, dtype=np.int32)
    central_marker = None

    nearby = []
    for other in all_specs:
        relative_um = (other.center - spec.center) * spacing
        if np.all(other.center >= spec.start) and np.all(other.center < spec.stop):
            nearby.append((other, relative_um))
    for marker_id, (other, relative_um) in enumerate(nearby, start=1):
        zz, yy, xx = grids
        marker_distance = np.sqrt(
            (zz - relative_um[0]) ** 2 + (yy - relative_um[1]) ** 2 +
            (xx - relative_um[2]) ** 2
        )
        core_radius = max(0.5 * other.radius_um, 0.12)
        core_mask = marker_distance <= core_radius
        if not np.any(core_mask):
            continue
        masked = np.where(core_mask, smooth, -np.inf)
        seed = tuple(int(v) for v in np.unravel_index(np.argmax(masked), smooth.shape))
        markers[seed] = marker_id
        other_shell_inner = max(1.8 * other.radius_um, 0.45)
        other_shell_outer = max(3.0 * other.radius_um, 0.8)
        shell = smooth[(marker_distance >= other_shell_inner) &
                       (marker_distance <= other_shell_outer)]
        other_background = (float(np.median(shell)) if shell.size >= 8
                            else float(np.percentile(smooth, 20)))
        other_peak = float(np.percentile(smooth[core_mask], 95))
        if other_peak > other_background:
            union_mask |= ((smooth >= other_background + 0.5 * (other_peak - other_background)) &
                           (marker_distance <= other_shell_outer))
        if other is spec:
            central_marker = (marker_id, seed)

    if central_marker is None:
        flags.add('watershed_missing_marker')
        return False
    marker_id, seed = central_marker
    union_mask[markers > 0] = True
    basins = watershed(-smooth, markers=markers, mask=union_mask)
    region = (basins == marker_id) & central_eligible
    region[seed] = central_eligible[seed]
    region = _component_containing(region, seed)
    count = int(np.count_nonzero(region))
    if count == 0:
        flags.add('watershed_empty_region')
        return False
    indices = np.argwhere(region)
    extents = (indices.max(axis=0) - indices.min(axis=0) + 1) * spacing
    volume = count * float(np.prod(spacing))
    m = spec.measurement
    m.watershed_voxel_count = count
    m.watershed_volume_um3 = float(volume)
    m.watershed_extent_z_um = float(extents[0])
    m.watershed_extent_y_um = float(extents[1])
    m.watershed_extent_x_um = float(extents[2])
    m.watershed_equivalent_diameter_um = float((6.0 * volume / pi) ** (1.0 / 3.0))
    if np.any(indices == 0) or np.any(indices == (np.asarray(region.shape) - 1)):
        flags.add('watershed_touches_patch_boundary')
    return True


def _fit_gaussian(spec, all_specs, grids, spacing, processing_spacing, flags):
    raw = spec.data.astype(np.float64, copy=False)
    local = _local_peak_and_background(raw, grids, spec.radius_um)
    if local is None:
        flags.add('gaussian_empty_core')
        spec.measurement.gaussian_fit_status = 'failed'
        return False
    peak, background, distance, _ = local
    amplitude = peak - background
    if not np.isfinite(amplitude) or amplitude <= 0:
        flags.add('gaussian_no_prominence')
        spec.measurement.gaussian_fit_status = 'failed'
        return False

    fit_radius = max(2.5 * spec.radius_um, 0.8)
    fit_mask = distance <= fit_radius
    # Exclude voxels closer to another detected center from this focus's fit.
    for other in all_specs:
        if other is spec:
            continue
        relative_um = (other.center - spec.center) * spacing
        if np.linalg.norm(relative_um) > fit_radius + other.radius_um:
            continue
        other_distance = np.sqrt(sum(
            (grid - relative_um[axis]) ** 2 for axis, grid in enumerate(grids)
        ))
        fit_mask &= distance <= other_distance
    if np.count_nonzero(fit_mask) < 20:
        flags.add('gaussian_insufficient_samples')
        spec.measurement.gaussian_fit_status = 'failed'
        return False

    coordinates = [grid[fit_mask] for grid in grids]
    observed = raw[fit_mask]
    sigma_xy = max(spec.radius_um / 1.8, float(np.mean(spacing[1:])))
    sigma_z = max(sigma_xy, float(spacing[0]))
    initial = np.array([background, amplitude, 0.0, 0.0, 0.0,
                        sigma_z, sigma_xy, sigma_xy], dtype=float)
    center_bounds = np.asarray(processing_spacing, dtype=float)
    sigma_lower = 0.5 * spacing
    patch_half_um = 0.5 * np.asarray(spec.data.shape) * spacing
    sigma_upper = np.maximum(sigma_lower * 1.1, patch_half_um / 1.5)
    lower = np.r_[0.0, 0.0, -center_bounds, sigma_lower]
    upper = np.r_[max(float(np.max(raw)) * 2.0, background + amplitude * 2.0),
                  max(float(np.max(raw)) * 3.0, amplitude * 3.0),
                  center_bounds, sigma_upper]
    initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)

    def residual(parameters):
        base, amp, cz, cy, cx, sz, sy, sx = parameters
        exponent = (((coordinates[0] - cz) / sz) ** 2 +
                    ((coordinates[1] - cy) / sy) ** 2 +
                    ((coordinates[2] - cx) / sx) ** 2)
        return base + amp * np.exp(-0.5 * exponent) - observed

    try:
        fit = least_squares(
            residual, initial, bounds=(lower, upper), loss='soft_l1',
            f_scale=max(1.0, 0.1 * amplitude), max_nfev=250,
        )
    except Exception:
        flags.add('gaussian_optimizer_error')
        spec.measurement.gaussian_fit_status = 'failed'
        return False
    if not fit.success or not np.all(np.isfinite(fit.x)):
        flags.add('gaussian_not_converged')
        spec.measurement.gaussian_fit_status = 'failed'
        return False

    predicted_residual = residual(fit.x)
    ss_res = float(np.sum(predicted_residual ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sigmas = fit.x[5:8]
    fwhm = FWHM_FACTOR * sigmas
    center_shift = float(np.linalg.norm(fit.x[2:5]))
    m = spec.measurement
    m.gaussian_fwhm_z_um = float(fwhm[0])
    m.gaussian_fwhm_y_um = float(fwhm[1])
    m.gaussian_fwhm_x_um = float(fwhm[2])
    m.gaussian_halfmax_volume_um3 = float(pi / 6.0 * np.prod(fwhm))
    m.gaussian_fit_r2 = float(r2) if np.isfinite(r2) else None
    m.gaussian_center_shift_um = center_shift
    m.gaussian_fit_status = 'ok'
    if np.isfinite(r2) and r2 < 0.8:
        flags.add('gaussian_low_r2')
    tolerance = 0.01 * np.maximum(1.0, np.abs(upper - lower))
    if np.any(fit.x - lower <= tolerance) or np.any(upper - fit.x <= tolerance):
        flags.add('gaussian_parameter_at_bound')
    if np.any(np.abs(fit.x[2:5]) >= 0.9 * center_bounds):
        flags.add('gaussian_center_near_bound')
    return True


def measure_foci_native(
    image,
    spot_channel: int,
    nucleus_blobs: Iterable[tuple[int, np.ndarray]],
    voxel_size_um_zyx,
    processing_scale_zyx=(1, 1, 1),
    source_image="",
    calibration_source="metadata",
    trimmed_z_planes=0,
) -> list[FocusMeasurement]:
    """Measure detections in native data while preserving detection ordering."""
    spacing = np.asarray(voxel_size_um_zyx, dtype=float)
    scale = np.asarray(processing_scale_zyx, dtype=float)
    detections = [(int(label_id), np.asarray(blob, dtype=float))
                  for label_id, blob in nucleus_blobs]
    specs = _make_specs(
        (image.shape[0], image.shape[2], image.shape[3]), detections,
        spacing, scale, source_image, calibration_source, trimmed_z_planes,
    )
    _fill_native_patches(image, spot_channel, specs)
    processing_spacing = spacing * scale
    for index, spec in enumerate(specs, start=1):
        check_cancelled()
        report(f'Measuring focus {index}/{len(specs)}')
        flags = set()
        if spec.clipped:
            flags.add('patch_clipped_by_image')
        if spec.read_error:
            flags.add('native_patch_read_error')
            spec.measurement.gaussian_fit_status = 'failed'
            spec.measurement.measurement_status = 'both_failed'
            spec.measurement.qc_flags = ';'.join(sorted(flags))
            continue
        grids = _physical_grids(spec, spacing)
        # Half a processed voxel gives comparable physical smoothing for native
        # LSM and oversampled CZI data without measuring on the pooled pixels.
        sigma_voxels = np.maximum(0.0, 0.5 * processing_spacing / spacing)
        smooth = gaussian_filter(spec.data.astype(np.float32), sigma=sigma_voxels)
        try:
            watershed_ok = _measure_watershed(spec, specs, smooth, grids, spacing, flags)
        except Exception:
            watershed_ok = False
            flags.add('watershed_processing_error')
        try:
            gaussian_ok = _fit_gaussian(
                spec, specs, grids, spacing, processing_spacing, flags,
            )
        except Exception:
            gaussian_ok = False
            spec.measurement.gaussian_fit_status = 'failed'
            flags.add('gaussian_processing_error')
        if watershed_ok and gaussian_ok:
            status = 'ok_with_warnings' if flags else 'ok'
        elif watershed_ok:
            status = 'gaussian_failed'
        elif gaussian_ok:
            status = 'watershed_failed'
        else:
            status = 'both_failed'
        spec.measurement.measurement_status = status
        spec.measurement.qc_flags = ';'.join(sorted(flags))
    return [spec.measurement for spec in specs]
