from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_erosion,
    label as ndi_label,
    find_objects,
)

from src.utils import (
    open_image, extract_channels, channel_contrast_limits,
    smart_downsample_factors, resolve_voxel_size_um,
)
from src.foci_measurement import FocusMeasurement, measure_foci_native
from src.spot_selection import (
    SpotSearchSettings, SpotSizeLimits, SpotCandidate, FWHM_FACTOR, apply_spot_selection,
)
from src.nuclei_segmenter import segment_nuclei
from src.cellpose_backend import label_counts
from src.display import prepare_display_channels
from src.focus_deduplication import consolidate_z_duplicates
from src.memory import (
    ImageWorkspace, using_workspace, allocate, release, blocks, filter_volume,
    selected_values, exact_percentile, response_statistics,
    current_workspace, report,
    check_cancelled,
)
from functools import wraps


SMALL_SPOT_SIGMAS_XY = (1.6, 2.2, 2.6)
LARGE_SPOT_SIGMAS_XY = (3.4, 4.4, 5.5)


def _disk_working_arrays(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        workspace = ImageWorkspace()
        parent = current_workspace()
        workspace.progress = parent.progress if parent is not None else None
        workspace.cancel_event = parent.cancel_event if parent is not None else None
        try:
            with using_workspace(workspace):
                return function(*args, **kwargs)
        finally:
            workspace.close()
    return wrapped


def _retained_working_arrays(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        workspace = ImageWorkspace()
        workspace.progress = kwargs.get('progress_callback')
        workspace.cancel_event = kwargs.get('cancel_event')
        try:
            with using_workspace(workspace):
                result = function(*args, **kwargs)
                check_cancelled()
            if isinstance(result, dict):
                result['_workspace'] = workspace
            else:
                result._workspace = workspace
            return result
        except BaseException:
            workspace.close()
            raise
    return wrapped


# ============================================================
# OUTPUT CONTAINER
# ============================================================
@dataclass
class DetectionResult:
    file_path: str
    file_stem: str
    img: np.ndarray                  # shape: (Z, C, Y, X)
    roi_stack: np.ndarray            # nucleus channel, shape: (Z, Y, X)
    spot_stack: np.ndarray           # foci channel, shape: (Z, Y, X)

    roi_proj: np.ndarray
    mask2d: np.ndarray
    labels2d: np.ndarray

    roi_mask_3d: np.ndarray
    roi_labels_3d: np.ndarray

    blobs_raw: np.ndarray
    blobs_in_roi: np.ndarray
    blobs_filtered: np.ndarray

    nucleus1_blobs: np.ndarray
    nucleus2_blobs: np.ndarray

    nucleus1_count: int
    nucleus2_count: int

    threshold: float
    contrast_limits: list | None = None
    _workspace: ImageWorkspace | None = None
    nucleus_count: int = 2
    nucleus_blobs: dict | None = None
    nucleus_counts: dict | None = None
    segmentation_diagnostics: dict | None = None
    nucleus_diagnostics: list | None = None
    spot_thresholds: dict | None = None
    processing_scale_zyx: tuple[int, int, int] = (1, 1, 1)
    trimmed_z_planes: int = 0
    voxel_size_um_zyx: tuple[float, float, float] | None = None
    calibration_source: str | None = None
    focus_measurements: list[FocusMeasurement] | None = None
    spot_candidates: tuple[SpotCandidate, ...] | None = None
    spot_selection: object = None
    spot_search_diagnostics: dict | None = None
    uncertain_counts: dict | None = None
    processed_foci: np.ndarray | None = None
    processed_nuclei: np.ndarray | None = None
    processed_foci_limits: tuple | None = None
    roi_channel: int = 2
    spot_channel: int = 1


# ============================================================
# ROBUST HELPERS
# ============================================================
def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-8)


def _safe_percentile(x: np.ndarray, q: float, default: float = 0.0) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, q))


# ============================================================
# BACKGROUND SUBTRACTION + Z NORMALIZATION
# ============================================================
def _normalize_spot_stack(
    spot_stack: np.ndarray,
    roi_mask_3d: np.ndarray,
    bg_sigma_xy: float = 18.0,
    clip_percentiles: Tuple[float, float] = (0.1, 99.9),
) -> np.ndarray:
    """
    Convert raw foci channel to a background-subtracted, z-normalized stack.

    Why this is better than thresholding raw intensity directly:
    - removes broad nuclear/background haze
    - reduces slice-to-slice brightness drift / photobleaching effects
    - makes one threshold more comparable across z-slices
    """
    finite = selected_values(spot_stack)
    if finite.size == 0:
        return allocate(spot_stack.shape)

    lo, hi = exact_percentile(finite, clip_percentiles)
    release(finite)
    norm = allocate(spot_stack.shape)
    Z = spot_stack.shape[0]
    for z in range(Z):
        report(f'Background normalization: slice {z + 1}/{Z}')
        spot = np.clip(np.asarray(spot_stack[z], dtype=np.float32), lo, hi).astype(np.float32)
        sub = spot - gaussian_filter(spot, sigma=bg_sigma_xy)
        sub[sub < 0] = 0.0
        mask_z = roi_mask_3d[z] if z < roi_mask_3d.shape[0] else np.zeros_like(sub, dtype=bool)
        vals = sub[mask_z]
        if vals.size < 32:
            vals = sub[np.isfinite(sub)]
        med = float(np.median(vals)) if vals.size else 0.0
        sig = _robust_sigma(vals)
        plane = (sub - med) / (sig + 1e-8)
        plane[~np.isfinite(plane)] = 0.0
        plane[plane < 0] = 0.0
        norm[z] = plane
    return norm


# ============================================================
# MULTISCALE 3D DoG RESPONSE
# ============================================================
def _build_multiscale_3d_dog_response(
    spot_norm: np.ndarray,
    sigmas_xy: Tuple[float, ...] = (1.1, 1.4, 1.8, 2.2),
    dog_ratio: float = 2.4,
    sigma_z: float = 0.7,
    xy_axis_factors=(1., 1.),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 3D-ish DoG response. The z smoothing is intentionally mild because
    microscope z-spacing is usually coarser than xy spacing.
    """
    if not sigmas_xy:
        raise ValueError('At least one detection scale is required.')
    best_response = allocate(spot_norm.shape)
    best_radius = allocate(spot_norm.shape)
    small = allocate(spot_norm.shape)
    large = allocate(spot_norm.shape)

    for scale_index, s in enumerate(sigmas_xy):
        report(f'Detection scale {scale_index + 1}/{len(sigmas_xy)} (sigma={s})')
        sy, sx = s * xy_axis_factors[0], s * xy_axis_factors[1]
        filter_volume(spot_norm, 'gaussian', sigma=(sigma_z, sy, sx), out=small)
        filter_volume(
            spot_norm,
            'gaussian',
            sigma=(max(1.0, 2.0 * sigma_z), dog_ratio * sy, dog_ratio * sx),
            out=large,
        )
        for start, stop in blocks(spot_norm.shape):
            resp = small[start:stop] - large[start:stop]
            resp[resp < 0] = 0.0
            resp *= float(s ** 1.2)
            better = resp > best_response[start:stop]
            if scale_index == 0:
                best_response[start:stop] = resp
                best_radius[start:stop] = 1.8 * s
            else:
                best_response[start:stop][better] = resp[better]
                best_radius[start:stop][better] = 1.8 * s
    release(small)
    release(large)

    # Used for local prominence measurements, not thresholding.
    intensity_smooth = filter_volume(spot_norm, 'gaussian', sigma=(0.4, 0.8, 0.8))
    return best_response, best_radius, intensity_smooth


# ============================================================
# LOCAL 3D PROMINENCE FEATURES
# ============================================================
def _local_prominence_features_3d(
    intensity_3d: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius_hint: float,
    xy_axis_factors=(1., 1.),
) -> Dict[str, float]:
    Z, H, W = intensity_3d.shape
    r_xy = max(1.5, float(radius_hint))
    patch_xy = max(5, int(np.ceil(3.5 * r_xy * max(xy_axis_factors))))
    patch_z = 2

    z0 = max(0, z - patch_z)
    z1 = min(Z, z + patch_z + 1)
    y0 = max(0, y - patch_xy)
    y1 = min(H, y + patch_xy + 1)
    x0 = max(0, x - patch_xy)
    x1 = min(W, x + patch_xy + 1)

    patch = intensity_3d[z0:z1, y0:y1, x0:x1]
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]

    dz = (zz - z) / 1.5
    rr2 = ((yy - y) / xy_axis_factors[0]) ** 2 + ((xx - x) / xy_axis_factors[1]) ** 2

    center_r = max(1.2, 0.85 * r_xy)
    ring_in = max(2.2, 1.6 * r_xy)
    ring_out = max(ring_in + 1.0, 3.2 * r_xy)

    center_mask = (np.abs(dz) <= 1.0) & (rr2 <= center_r ** 2)
    ring_mask = (np.abs(dz) <= 1.5) & (rr2 >= ring_in ** 2) & (rr2 <= ring_out ** 2)

    center_vals = patch[center_mask]
    ring_vals = patch[ring_mask]

    if center_vals.size == 0 or ring_vals.size < 8:
        return {
            "center_signal": 0.0,
            "ring_median": 0.0,
            "prominence": 0.0,
            "ring_noise": 0.0,
            "snr": 0.0,
        }

    # Percentile is less noisy than max and less diluted than mean.
    center_signal = float(np.percentile(center_vals, 85))
    ring_median = float(np.median(ring_vals))
    ring_noise = _robust_sigma(ring_vals)
    prominence = center_signal - ring_median
    snr = prominence / (ring_noise + 1e-8)

    return {
        "center_signal": center_signal,
        "ring_median": ring_median,
        "prominence": float(prominence),
        "ring_noise": float(ring_noise),
        "snr": float(snr),
    }


# ============================================================
# MASK HANDLING / ASSIGNMENT
# ============================================================
def _erode_labels_3d(roi_labels_3d: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode each nucleus label independently so boundary noise does not create
    confident but wrong assignments. If erosion removes a tiny label, fall back
    to the original label for that nucleus.
    """
    if iterations <= 0:
        return roi_labels_3d

    out = allocate(roi_labels_3d.shape, dtype=roi_labels_3d.dtype)
    structure = np.ones((1, 3, 3), dtype=bool)  # xy-only erosion

    for label_id in label_counts(roi_labels_3d):
        original_count = eroded_count = 0
        for z in range(roi_labels_3d.shape[0]):
            mask = roi_labels_3d[z] == label_id
            eroded = binary_erosion(mask, structure=structure[0], iterations=iterations)
            original_count += np.count_nonzero(mask)
            eroded_count += np.count_nonzero(eroded)
            out[z][eroded] = label_id
        if eroded_count < 0.25 * original_count:
            for z in range(roi_labels_3d.shape[0]):
                out[z][out[z] == label_id] = 0
                out[z][roi_labels_3d[z] == label_id] = label_id
    return out


def _assign_label_by_local_majority(
    roi_labels_3d: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius_hint: float,
) -> int:
    """
    Assign a spot to the nucleus occupying most pixels around the spot center,
    rather than trusting one boundary voxel.
    """
    Z, H, W = roi_labels_3d.shape
    rz = 1
    rxy = max(2, int(np.ceil(0.8 * radius_hint)))

    z0 = max(0, z - rz)
    z1 = min(Z, z + rz + 1)
    y0 = max(0, y - rxy)
    y1 = min(H, y + rxy + 1)
    x0 = max(0, x - rxy)
    x1 = min(W, x + rxy + 1)

    patch = roi_labels_3d[z0:z1, y0:y1, x0:x1]
    labels, counts = np.unique(patch[patch > 0], return_counts=True)
    if labels.size == 0:
        return 0
    return int(labels[np.argmax(counts)])


# ============================================================
# 3D CANDIDATE EXTRACTION + MERGE
# ============================================================
def _merge_close_candidates(
    candidates: List[dict],
    merge_radius_xy: float = 4.5,
    merge_radius_z: int = 1,
) -> List[dict]:
    """
    Greedy final merge. This catches split maxima from one physical focus.
    It does not merge across nucleus labels.
    """
    kept: List[dict] = []
    for cand in sorted(candidates, key=lambda d: d["score"], reverse=True):
        check_cancelled()
        duplicate = False
        for prev in kept:
            if cand["label_id"] != prev["label_id"]:
                continue
            if abs(cand["z"] - prev["z"]) > merge_radius_z:
                continue
            if np.hypot(cand["y"] - prev["y"], cand["x"] - prev["x"]) <= merge_radius_xy:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return sorted(kept, key=lambda d: (d["label_id"], d["z"], d["y"], d["x"]))


def _extract_3d_candidates(
    response_3d: np.ndarray,
    intensity_3d: np.ndarray,
    radius_map: np.ndarray,
    roi_labels_3d: np.ndarray,
    peak_mad_k: float = 5.0,
    peak_percentile: float | None = 97.5,
    min_distance_xy: int = 6,
    min_distance_z: int = 1,
    min_radius_px: float = 2.0,
    max_radius_px: float = 6.5,
    prominence_noise_k: float = 3.0,
    min_prominence_abs: float = 1.2,
    boundary_erosion_px: int = 1,
    xy_axis_factors=(1., 1.),
) -> Tuple[List[dict], float]:
    eroded = _erode_labels_3d(roi_labels_3d, iterations=boundary_erosion_px)
    roi_core = allocate(roi_labels_3d.shape, dtype=bool)
    for start, stop in blocks(roi_core.shape):
        roi_core[start:stop] = eroded[start:stop] > 0
    if eroded is not roi_labels_3d:
        release(eroded)
    stats = response_statistics(response_3d, roi_core, peak_percentile)
    if stats is None:
        release(roi_core)
        return [], float("nan")

    med, sig, percentile_floor = stats
    threshold = med + peak_mad_k * sig

    # A mild percentile floor prevents very noisy stacks from exploding, but this
    # is deliberately much lower than 99.5 so dense/high-count nuclei are not crushed.
    if peak_percentile is not None:
        threshold = max(threshold, percentile_floor)

    maxima = filter_volume(
        response_3d,
        'maximum',
        size=(2 * min_distance_z + 1, 2 * min_distance_xy + 1, 2 * min_distance_xy + 1),
        mode="nearest",
    )
    cand_mask = allocate(response_3d.shape, dtype=bool)
    # Erosion protects threshold estimation from bright boundaries. It must not
    # remove real candidate centers lying inside the original nuclear outline.
    for start, stop in blocks(response_3d.shape):
        resp = response_3d[start:stop]
        cand_mask[start:stop] = ((resp == maxima[start:stop]) &
                                (resp >= threshold) & (roi_labels_3d[start:stop] > 0))
    release(maxima)
    release(roi_core)

    report('Grouping candidate peaks across all slices')
    labeled = allocate(cand_mask.shape, dtype=np.int32)
    ncomp = ndi_label(cand_mask, output=labeled)
    release(cand_mask)
    candidates: List[dict] = []

    for comp_id, region in enumerate(find_objects(labeled, max_label=ncomp), start=1):
        check_cancelled()
        if region is None:
            continue
        zs, ys, xs = np.where(labeled[region] == comp_id)
        zs += region[0].start
        ys += region[1].start
        xs += region[2].start
        if zs.size == 0:
            continue

        scores = response_3d[zs, ys, xs]
        best_i = int(np.argmax(scores))
        z = int(zs[best_i])
        y = int(ys[best_i])
        x = int(xs[best_i])
        r = float(radius_map[z, y, x])

        if r < min_radius_px or r > max_radius_px:
            continue

        feats = _local_prominence_features_3d(intensity_3d, z=z, y=y, x=x, radius_hint=r,
                                               xy_axis_factors=xy_axis_factors)
        if feats["prominence"] < min_prominence_abs:
            continue
        if feats["snr"] < prominence_noise_k:
            continue

        label_id = _assign_label_by_local_majority(roi_labels_3d, z=z, y=y, x=x, radius_hint=r)
        if label_id <= 0:
            continue

        candidates.append({
            "z": z,
            "y": y,
            "x": x,
            "radius": r,
            "score": float(response_3d[z, y, x]),
            "prominence": feats["prominence"],
            "snr": feats["snr"],
            "label_id": label_id,
        })

    release(labeled)
    candidates = _merge_close_candidates(
        candidates,
        merge_radius_xy=max(3.0, 0.75 * min_distance_xy),
        merge_radius_z=min_distance_z,
    )
    return candidates, float(threshold)


# ============================================================
# MAIN SPOT DETECTOR
# ============================================================
@_disk_working_arrays
def detect_spots(
    spot_stack: np.ndarray,
    roi_mask_3d: np.ndarray,
    roi_labels_3d: np.ndarray,
    # spot size
    sigmas_xy: Tuple[float, ...] = SMALL_SPOT_SIGMAS_XY,
    dog_ratio: float = 2.0, #how much local background to subtract
    sigma_z: float = 0.7,
    min_radius_px: float = 3.0,
    max_radius_px: float = 10.0,
    # intensity / response threshold
    peak_percentile: float | None = 99.5,
    peak_mad_k: float = 11.0,
    # local prominence filter
    prominence_noise_k: float = 8.0,
    min_prominence_abs: float = 6.0,
    # peak separation
    min_distance_xy: int = 10,
    min_distance_z: int = 3,
    # kept for backward compatibility with old tuning scripts
    link_radius_xy: float | None = None,
    max_gap_z: int | None = None,
    min_track_length: int | None = None,
    # preprocessing / mask handling
    bg_sigma_xy: float = 18.0,
    boundary_erosion_px: int = 1,
    large_spot_sigmas_xy: Tuple[float, ...] | None = None,
    search_settings=None,
    voxel_size_um_zyx=None,
    processed_output=None,
    merge_z_duplicates: bool = True,
    z_duplicate_xy_um: float = .35,
    z_duplicate_min_bridge: float = .8,
) -> Dict[str, Any]:
    """
    Robust foci detector.

    Main change from the old code:
    - detect 3D local maxima directly instead of slice peaks + greedy z-linking
    - normalize background and z-slice intensity before thresholding
    - use mild percentile floor + MAD threshold, not a hard 99.5 percentile gate
    - assign each focus by local majority label around the spot center
    """
    if not np.any(roi_mask_3d):
        raise RuntimeError("ROI mask is empty.")

    search = (SpotSearchSettings(**search_settings) if isinstance(search_settings, dict)
              else search_settings or SpotSearchSettings())
    xy_axis_factors = (1., 1.)
    if search.manual:
        spacing = np.asarray(voxel_size_um_zyx, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError('Physical spot search requires calibrated Z/Y/X processing spacing.')
        xy_um = float(np.sqrt(spacing[1] * spacing[2]))
        xy_axis_factors = tuple(xy_um / spacing[1:])
        sigmas_xy = tuple(w / FWHM_FACTOR / xy_um for w in search.widths())
        # Broad objects need a background neighborhood wider than the object.
        bg_sigma_xy = tuple(max(18., 2 * dog_ratio * max(sigmas_xy)) * f
                            for f in xy_axis_factors)
        min_radius_px, max_radius_px = 0., float('inf')

    spot_norm = _normalize_spot_stack(
    spot_stack=spot_stack,
    roi_mask_3d=roi_mask_3d,
    bg_sigma_xy=bg_sigma_xy,
    )

    # remove fine grain / salt-pepper noise before DoG amplifies it
    filtered = filter_volume(spot_norm, 'median', size=(1, 5, 5))
    release(spot_norm)
    spot_norm = filtered
    if processed_output is not None:
        if processed_output.shape != spot_norm.shape:
            raise ValueError('Processed display grid must match the spot-processing grid.')
        for start, stop in blocks(spot_norm.shape):
            processed_output[start:stop] = spot_norm[start:stop]

    # Keep the original small-scale threshold independent. Combining every
    # scale into one response can raise the global percentile floor enough
    # that introducing a bright, large focus makes smaller foci disappear.
    if large_spot_sigmas_xy is None:
        large_spot_sigmas_xy = (LARGE_SPOT_SIGMAS_XY
                               if tuple(sigmas_xy) == SMALL_SPOT_SIGMAS_XY else ())
    large_scales = tuple(s for s in large_spot_sigmas_xy
                         if min_radius_px <= 1.8*s <= max_radius_px
                         and s not in sigmas_xy)
    passes = [('small', tuple(sigmas_xy))]
    if large_scales:
        passes.append(('large', large_scales))
    if search.manual:
        passes = [(f'physical_{i + 1}', (s,)) for i, s in enumerate(sigmas_xy)]
    candidates = []
    thresholds = {}
    for pass_order, (pass_name, scales) in enumerate(passes):
        report(f'Detecting {pass_name} foci ({len(scales)} scales)')
        response_img, radius_map, intensity_img = _build_multiscale_3d_dog_response(
            spot_norm=spot_norm, sigmas_xy=scales,
            dog_ratio=dog_ratio, sigma_z=sigma_z,
            **({'xy_axis_factors': xy_axis_factors} if search.manual else {}),
        )
        for start, stop in blocks(response_img.shape):
            response_img[start:stop][~roi_mask_3d[start:stop]] = 0.0
        found, threshold = _extract_3d_candidates(
            response_3d=response_img, intensity_3d=intensity_img,
            radius_map=radius_map, roi_labels_3d=roi_labels_3d,
            peak_mad_k=peak_mad_k, peak_percentile=peak_percentile,
            min_distance_xy=min_distance_xy, min_distance_z=min_distance_z,
            min_radius_px=min_radius_px, max_radius_px=max_radius_px,
            prominence_noise_k=prominence_noise_k,
            min_prominence_abs=min_prominence_abs,
            boundary_erosion_px=boundary_erosion_px,
            **({'xy_axis_factors': xy_axis_factors} if search.manual else {}),
        )
        thresholds[pass_name] = float(threshold) if np.isfinite(threshold) else 0.0
        for candidate in found:
            check_cancelled()
            # Retain original detections and add only new large-focus centers.
            # Scores from independently thresholded passes are not ranked
            # against one another. Never merge across different nuclei.
            duplicate = any(
                candidate['label_id'] == previous['label_id']
                and abs(candidate['z']-previous['z']) <= min_distance_z
                and np.hypot(candidate['y']-previous['y'], candidate['x']-previous['x'])
                <= max(3.0, 0.75*min_distance_xy)
                for previous in candidates
            )
            if not duplicate:
                candidate['pass_order'] = pass_order
                candidates.append(candidate)
        release(response_img)
        release(radius_map)
        release(intensity_img)
    release(spot_norm)
    if merge_z_duplicates and voxel_size_um_zyx is not None:
        report('Consolidating aligned peaks across Z using signal continuity')
        candidates, z_consolidation = consolidate_z_duplicates(
            candidates, spot_stack, roi_labels_3d, voxel_size_um_zyx,
            xy_tolerance_um=z_duplicate_xy_um, min_bridge_ratio=z_duplicate_min_bridge)
    else:
        z_consolidation = dict(enabled=False, reason='disabled' if not merge_z_duplicates else 'calibration_not_supplied',
                               input_candidates=len(candidates), retained_candidates=len(candidates), removed_count=0)
    candidates.sort(key=lambda c: (c['label_id'], c['z'], c['y'], c['x']))
    threshold = next(iter(thresholds.values()))

    def to_array(label_id: int) -> np.ndarray:
        pts = [c for c in candidates if c["label_id"] == label_id]
        if not pts:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(
            [[c["z"], c["y"], c["x"], c["radius"]] for c in pts],
            dtype=np.float32,
        )

    nucleus_blobs = {i: to_array(i) for i in label_counts(roi_labels_3d)}
    nucleus_counts = {i: len(blobs) for i, blobs in nucleus_blobs.items()}
    nucleus1_blobs = nucleus_blobs.get(1, to_array(1))
    nucleus2_blobs = nucleus_blobs.get(2, to_array(2))
    blobs_filtered = (
        np.vstack([a for a in nucleus_blobs.values() if len(a) > 0]).astype(np.float32)
        if any(nucleus_counts.values())
        else np.empty((0, 4), dtype=np.float32)
    )

    # Here raw == filtered because extraction already performs biologically meaningful filters.
    # Kept as separate keys for UI/backward compatibility.
    return {
        "threshold": float(threshold) if np.isfinite(threshold) else 0.0,
        "scale_thresholds": thresholds,
        "z_consolidation": z_consolidation,
        "searched_sigmas_xy": tuple(s for _, group in passes for s in group
                                   if min_radius_px <= 1.8 * s <= max_radius_px),
        "blobs_raw": blobs_filtered,
        "blobs_in_roi": blobs_filtered,
        "blobs_filtered": blobs_filtered,
        "nucleus1_blobs": nucleus1_blobs,
        "nucleus2_blobs": nucleus2_blobs,
        "nucleus1_count": int(len(nucleus1_blobs)),
        "nucleus2_count": int(len(nucleus2_blobs)),
        "nucleus_blobs": nucleus_blobs,
        "nucleus_counts": nucleus_counts,
    }


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================
@_retained_working_arrays
def run_detection(
    file_path: str,
    roi_channel: int = 2,
    spot_channel: int = 1,
    spot_params: dict | None = None,
    progress_callback=None,
    nucleus_params: dict | None = None,
    cancel_event=None,
    smart_downsample: bool = True,
    voxel_size_override_um_zyx: tuple[float, float, float] | None = None,
    segmentation_settings=None,
    spot_search_settings=None,
    z_trim_planes: int = 0,
    spot_size_limits=None,
) -> DetectionResult:
    if spot_params is None:
        spot_params = {}
    selection_limits = (SpotSizeLimits(**spot_size_limits) if isinstance(spot_size_limits, dict)
                        else spot_size_limits if spot_size_limits is not None
                        else SpotSizeLimits(min_volume_um3=.2))

    search = (SpotSearchSettings(**spot_search_settings) if isinstance(spot_search_settings, dict)
              else spot_search_settings or SpotSearchSettings())
    if search.manual and any(k in spot_params for k in
                             ('sigmas_xy', 'large_spot_sigmas_xy', 'min_radius_px', 'max_radius_px')):
        raise ValueError('Choose physical search settings or pixel-based spot sizes, not both.')
    voxel_size, calibration_source = resolve_voxel_size_um(
        file_path, voxel_size_override_um_zyx,
    )
    report(f'Opening {Path(file_path).name}')
    img = open_image(file_path)
    scale = (smart_downsample_factors(file_path, img.shape)
             if smart_downsample else (1, 1, 1))
    processed_shape = tuple(int(np.ceil(length / factor))
                            for length, factor in zip(
                                (img.shape[0], img.shape[2], img.shape[3]), scale))
    if not isinstance(z_trim_planes, int) or z_trim_planes < 0:
        raise ValueError('Z trimming must be a nonnegative integer number of processing planes.')
    if processed_shape[0] - 2 * z_trim_planes < 3:
        raise ValueError('Detection requires at least three processing Z slices after any explicit trimming.')
    # Conservative peak: retained channels/masks plus filter/statistics scratch.
    current_workspace().ensure_space(int(np.prod(processed_shape)) * 48)
    if scale != (1, 1, 1):
        report(f'Smart CZI reduction: {scale[0]}x Z, {scale[1]}x Y, {scale[2]}x X; '
               f'processing {processed_shape[0]}x{processed_shape[1]}x{processed_shape[2]}')
    report('Reading nucleus and foci channels, one slice at a time')
    roi_stack, spot_stack = extract_channels(
        img,
        roi_channel=roi_channel,
        spot_channel=spot_channel,
        downsample_factors=scale,
    )

    # Keep the full acquisition by default: end slices can contain genuine foci.
    if z_trim_planes:
        roi_stack = roi_stack[z_trim_planes:-z_trim_planes]
        spot_stack = spot_stack[z_trim_planes:-z_trim_planes]

    report('Segmenting nuclei')
    seg = segment_nuclei(roi_stack, z_trim=0, validation_params=nucleus_params,
                         voxel_size_um_zyx=tuple(np.asarray(voxel_size) * scale),
                         settings=segmentation_settings)

    report('Detecting foci')
    processing_spacing = np.asarray(voxel_size) * scale
    det = detect_spots(
        spot_stack=spot_stack,
        roi_mask_3d=seg["roi_mask_3d"],
        roi_labels_3d=seg["roi_labels_3d"],
        search_settings=search,
        voxel_size_um_zyx=processing_spacing,
        **spot_params,
    )

    ordered_detections = [(i, blob) for i, blobs in det['nucleus_blobs'].items() for blob in blobs]
    focus_measurements = measure_foci_native(
        image=img,
        spot_channel=spot_channel,
        nucleus_blobs=ordered_detections,
        voxel_size_um_zyx=voxel_size,
        processing_scale_zyx=scale,
        source_image=Path(file_path).name,
        calibration_source=calibration_source,
        trimmed_z_planes=z_trim_planes,
    )

    processed_nuclei, processed_foci = prepare_display_channels(
        roi_stack, spot_stack, processing_spacing)
    foci_limits = (0., 1.)
    searched = np.asarray(det['searched_sigmas_xy']) * FWHM_FACTOR * np.sqrt(np.prod(processing_spacing[1:]))
    warnings = []
    resolution = float(2 * max(processing_spacing[1:]))
    if len(searched) and searched.min() < resolution:
        warnings.append('Search includes widths below two processing pixels; these sizes may be unresolved. '
                        'Use a finer processing grid before interpreting them.')
    diagnostics = dict(mode='physical' if search.manual else 'automatic',
                       searched_width_um=(float(searched.min()), float(searched.max())) if len(searched) else (None, None),
                       requested_width_um=(search.min_width_um, search.max_width_um),
                       resolution_xy_um=resolution, warnings=warnings,
                       z_trim_planes=z_trim_planes, boundary_candidates='original_mask',
                       z_consolidation=det['z_consolidation'])
    if len(ordered_detections) != len(focus_measurements):
        raise RuntimeError('Candidate/measurement alignment failed.')
    candidates = tuple(SpotCandidate.create(i, blob, measurement)
                       for (i, blob), measurement in zip(ordered_detections, focus_measurements))
    limits = []
    for channel in range(img.shape[1]):
        report(f'Preparing viewer contrast: channel {channel + 1}/{img.shape[1]}')
        display_sample = img[::scale[0], channel, ::scale[1], ::scale[2]]
        limits.append(channel_contrast_limits(display_sample))
    result = DetectionResult(
        file_path=str(file_path),
        file_stem=Path(file_path).stem,
        img=img,
        roi_stack=roi_stack,
        spot_stack=spot_stack,
        roi_proj=seg["roi_proj"],
        mask2d=seg["mask2d"],
        labels2d=seg["labels2d"],
        roi_mask_3d=seg["roi_mask_3d"],
        roi_labels_3d=seg["roi_labels_3d"],
        blobs_raw=det["blobs_raw"],
        blobs_in_roi=det["blobs_in_roi"],
        blobs_filtered=det["blobs_filtered"],
        nucleus1_blobs=det["nucleus1_blobs"],
        nucleus2_blobs=det["nucleus2_blobs"],
        nucleus1_count=det["nucleus1_count"],
        nucleus2_count=det["nucleus2_count"],
        threshold=det["threshold"],
        contrast_limits=limits,
        nucleus_count=seg['nucleus_count'],
        nucleus_blobs=det['nucleus_blobs'],
        nucleus_counts=det['nucleus_counts'],
        segmentation_diagnostics=seg.get('segmentation_diagnostics'),
        nucleus_diagnostics=seg['candidate_diagnostics'],
        spot_thresholds=det['scale_thresholds'],
        processing_scale_zyx=scale,
        trimmed_z_planes=z_trim_planes,
        voxel_size_um_zyx=voxel_size,
        calibration_source=calibration_source,
        focus_measurements=focus_measurements,
        spot_candidates=candidates,
        spot_search_diagnostics=diagnostics,
        processed_foci=processed_foci,
        processed_nuclei=processed_nuclei,
        processed_foci_limits=tuple(foci_limits),
        roi_channel=roi_channel, spot_channel=spot_channel,
    )
    apply_spot_selection(result, selection_limits)
    return result
