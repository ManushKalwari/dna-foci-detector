from typing import Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from src.memory import allocate, report, check_cancelled


@dataclass(frozen=True)
class NucleusValidationParams:
    """Initial defaults for boundary-stained nuclei; calibrate on real images."""
    max_candidates: int = 8
    min_peak_snr: float = 3.0
    min_ring_contrast: float = 0.03
    min_ring_snr: float = 1.5
    min_ring_interior_fraction: float = 0.12
    min_boundary_coverage: float = 0.65
    max_radius_cv: float = 0.18
    sectors: int = 16
    min_radius_fraction: float = 0.10
    max_radius_fraction: float = 0.30

    def __post_init__(self):
        for name in ('max_candidates', 'sectors'):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f'{name} must be an integer.')
        if self.max_candidates < 2 or self.sectors < 8:
            raise ValueError('Use at least two candidates and eight boundary sectors.')
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid nucleus validation parameter: {name}')
        for name in ('min_boundary_coverage', 'min_ring_interior_fraction'):
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be between zero and one.')
        if not 0 < self.min_radius_fraction <= self.max_radius_fraction < 0.5:
            raise ValueError('Radius fractions must satisfy 0 < min <= max < 0.5.')


class NucleusValidationError(RuntimeError):
    """Keep the rejection evidence available when no masks can be produced."""
    def __init__(self, diagnostics):
        self.candidate_diagnostics = diagnostics
        best = max(diagnostics, key=lambda item: item['peak_response'], default=None)
        detail = (f" Strongest candidate failed: {', '.join(best['rejection_reasons'])}."
                  if best is not None else '')
        super().__init__(
            'No valid nucleus detected within the configured radius range.'
            + detail + ' Check the nucleus channel or nucleus validation parameters.'
        )


def _background_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 0.0
    median = float(np.median(values))
    noise = float(1.4826 * np.median(np.abs(values - median)))
    return median, noise


def _validate_candidate(projection, response, x, y, radius, params, excluded=None):
    """Assess a native-resolution circular boundary in a local 2D crop.

    Radial bins within each angular sector tolerate uneven staining while
    measuring coverage and variation of the apparent boundary radius. A filled
    bright disk lacks boundary-to-interior contrast even if it is perfectly round.
    """
    extent = int(np.ceil(1.8 * radius))
    y0, y1 = max(0, y-extent), min(projection.shape[0], y+extent+1)
    x0, x1 = max(0, x-extent), min(projection.shape[1], x+extent+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(yy-y, xx-x) / radius
    image = projection[y0:y1, x0:x1]
    local_response = response[y0:y1, x0:x1]
    eligible = (np.ones(image.shape, dtype=bool) if excluded is None
                else ~excluded[y0:y1, x0:x1])
    background = (rr >= 1.15) & (rr <= 1.5) & eligible
    response_background = (rr >= 1.25) & (rr <= 1.8) & eligible
    bg, noise = _background_stats(image[background])
    response_bg, response_noise = _background_stats(local_response[response_background])
    peak = float(response[y, x])
    peak_snr = (peak-response_bg) / max(response_noise, 1e-6)
    interior_values = image[(rr <= 0.5) & eligible]
    interior = float(np.median(interior_values)) if interior_values.size else float('inf')
    angles = (np.arctan2(yy-y, xx-x) + np.pi) % (2*np.pi)
    sectors = np.minimum((angles * params.sectors / (2*np.pi)).astype(int), params.sectors-1)
    # Probe beyond the nominal ring as well: a narrow annulus alone can hide
    # the long axis of an ellipse and incorrectly make it appear circular.
    radial_edges = np.linspace(0.55, 1.50, 17)
    sector_signal = np.full(params.sectors, np.nan)
    sector_radius = np.full(params.sectors, np.nan)
    for sector in range(params.sectors):
        samples = []
        for low, high in zip(radial_edges[:-1], radial_edges[1:]):
            pixels = image[(sectors == sector) & (rr >= low) & (rr < high) & eligible]
            samples.append(float(pixels.mean()) if pixels.size else -np.inf)
        best = int(np.argmax(samples))
        if np.isfinite(samples[best]):
            sector_signal[sector] = samples[best]
            sector_radius[sector] = (radial_edges[best] + radial_edges[best+1]) / 2
    signal_floor = bg + max(params.min_ring_contrast, params.min_ring_snr * noise)
    supported = np.isfinite(sector_signal) & (sector_signal >= signal_floor)
    coverage = float(np.count_nonzero(supported) / params.sectors)
    finite_signal = sector_signal[np.isfinite(sector_signal)]
    ring_signal = float(np.median(finite_signal)) if finite_signal.size else 0.0
    ring_contrast = ring_signal-bg
    ring_interior_fraction = (ring_signal-interior) / max(ring_contrast, 1e-6)
    radii = sector_radius[supported]
    radius_cv = float(radii.std() / radii.mean()) if radii.size else float('inf')
    median_radius = float(np.median(radii)) if radii.size else float('inf')
    reasons = []
    if peak <= 0 or peak_snr < params.min_peak_snr:
        reasons.append('weak_peak')
    if ring_contrast < max(params.min_ring_contrast, params.min_ring_snr * noise):
        reasons.append('weak_ring')
    if coverage < params.min_boundary_coverage:
        reasons.append('incomplete_boundary')
    if radius_cv > params.max_radius_cv:
        reasons.append('irregular_radius')
    if not 0.60 <= median_radius <= 1.08:
        reasons.append('radius_mismatch')
    if ring_interior_fraction < params.min_ring_interior_fraction:
        reasons.append('filled_blob')
    return {
        'center': (int(y), int(x)), 'radius': int(radius), 'peak_response': peak,
        'peak_snr': float(peak_snr), 'boundary_coverage': coverage,
        'radius_cv': radius_cv, 'ring_contrast': float(ring_contrast),
        'measured_radius': median_radius * radius,
        'ring_interior_fraction': float(ring_interior_fraction),
        'accepted': not reasons, 'rejection_reasons': reasons,
    }


def _select_valid_nuclei(response, outer_map, inner_map, projection, params):
    # Consider more than two peaks so a rejected artifact cannot prevent a
    # lower-ranked, genuine nucleus from being evaluated.
    candidates = _pick_top_k(response, outer_map, inner_map, k=params.max_candidates)
    accepted, diagnostics = [], []
    excluded = np.zeros(projection.shape, dtype=bool)
    yy, xx = np.ogrid[:projection.shape[0], :projection.shape[1]]
    for x, y, inner, outer in candidates:
        check_cancelled()
        metrics = _validate_candidate(projection, response, x, y, outer, params, excluded)
        diagnostics.append(metrics)
        if metrics['accepted']:
            accepted.append((x, y, inner, outer))
            # Do not mistake the brighter neighboring nucleus for a distant
            # portion of this candidate's circumference/background.
            excluded |= (yy-y)**2 + (xx-x)**2 <= (1.08*outer)**2
            if len(accepted) == 2:
                break
    return accepted, diagnostics


# ============================================================
# ANNULUS KERNEL
# ============================================================
def _annulus_kernel(r_inner, r_outer):
    pad = int(np.ceil(r_outer * 1.5))
    y, x = np.mgrid[-pad:pad+1, -pad:pad+1]
    rr = np.sqrt(x**2 + y**2)

    band = ((rr >= r_inner) & (rr <= r_outer)).astype(np.float32)
    band = gaussian_filter(band, sigma=1.0)

    surround = np.exp(-0.5 * (rr / (1.2 * r_outer + 1e-8))**2)

    k = band - 0.15 * surround
    k -= k.mean()
    k /= np.sqrt((k * k).sum()) + 1e-8

    return k


# ============================================================
# NMS (TOP-K PEAKS)
# ============================================================
def _pick_top_k(resp, r_outer_map, r_inner_map, k=2):
    resp_copy = resp.copy()
    H, W = resp.shape
    yy, xx = np.mgrid[:H, :W]

    detections = []

    for _ in range(k):
        y, x = np.unravel_index(np.argmax(resp_copy), resp.shape)

        if not np.isfinite(resp_copy[y, x]):
            break

        r_out = int(r_outer_map[y, x])
        r_in = int(r_inner_map[y, x])

        detections.append((int(x), int(y), r_in, r_out))

        # suppress neighborhood
        suppress = (xx - x)**2 + (yy - y)**2 < (1.3 * r_out)**2
        resp_copy[suppress] = -np.inf

    return detections


# ============================================================
# MAIN SEGMENTATION
# ============================================================
def segment_nuclei_legacy(
    roi_stack: np.ndarray,
    channel_nucleus: int = 2,
    z_trim: int = 5,
    validation_params: dict | NucleusValidationParams | None = None,
) -> Dict[str, Any]:

    params = (validation_params if isinstance(validation_params, NucleusValidationParams)
              else NucleusValidationParams(**(validation_params or {})))

    # ---------------------------------------------------
    # projection (simple + stable)
    # ---------------------------------------------------
    if roi_stack.ndim != 3 or z_trim < 0 or roi_stack.shape[0] <= 2*z_trim:
        raise ValueError('Expected a nonempty Z/Y/X stack after nucleus Z trimming.')
    roi = roi_stack[z_trim:-z_trim] if z_trim else roi_stack
    proj = roi.mean(axis=0).astype(np.float32)
    if not np.all(np.isfinite(proj)):
        raise ValueError('Nucleus projection contains non-finite intensities.')

    H, W = proj.shape
    yy, xx = np.mgrid[:H, :W]

    # ---------------------------------------------------
    # normalize
    # ---------------------------------------------------
    p1, p99 = np.percentile(proj, [1, 99])
    proj_n = np.clip((proj - p1) / (p99 - p1 + 1e-8), 0, 1)

    # ---------------------------------------------------
    # denoise
    # ---------------------------------------------------
    proj_s = gaussian_filter(proj_n, 0.8)

    # ---------------------------------------------------
    # bandpass (membrane enhancement)
    # ---------------------------------------------------
    high = gaussian_filter(proj_s, 1.5)
    low = gaussian_filter(proj_s, 7.0)
    band = high - low

    bg = np.percentile(band, 5)
    band = np.clip(band - bg, 0, None)
    band = band / (band.max() + 1e-8)

    # ---------------------------------------------------
    # radius search
    # ---------------------------------------------------
    # Preserve the usual search on images that already work. A lower-radius
    # search is attempted only if every standard candidate fails validation.
    initial_fraction = max(params.min_radius_fraction, min(0.18, params.max_radius_fraction))
    r_outer_min = max(3, int(initial_fraction * min(H, W)))
    r_outer_max = max(r_outer_min, int(params.max_radius_fraction * min(H, W)))
    r_outer_vals = np.arange(r_outer_min, r_outer_max + 1, 2)

    best_response = np.full((H, W), -np.inf, dtype=np.float32)
    best_outer = np.zeros((H, W), dtype=np.int32)

    def search_radii(values, stage):
        for r_out in values:
            report(f'Nucleus {stage} radius search: radius {r_out}/{r_outer_max}')
            r_in = int(0.70 * r_out)
            k = _annulus_kernel(r_in, r_out)
            resp = cv2.filter2D(band, -1, k, borderType=cv2.BORDER_REFLECT)
            better = resp > best_response
            best_response[better] = resp[better]
            best_outer[better] = r_out

    search_radii(r_outer_vals, 'standard')
    best_inner = (0.65 * best_outer).astype(int)

    # ---------------------------------------------------
    # suppress borders (very important)
    # ---------------------------------------------------
    margin = int(0.9 * r_outer_max)
    best_response[:margin, :] = -np.inf
    best_response[-margin:, :] = -np.inf
    best_response[:, :margin] = -np.inf
    best_response[:, -margin:] = -np.inf

    # ---------------------------------------------------
    # accept up to two validated nuclei
    # ---------------------------------------------------
    detections, diagnostics = _select_valid_nuclei(
        best_response, best_outer, best_inner, proj_s, params,
    )

    for candidate in diagnostics:
        candidate['search_stage'] = 'standard'
    search_extended = False
    lower_radius = max(3, int(params.min_radius_fraction * min(H, W)))
    if not detections and lower_radius < r_outer_min:
        search_extended = True
        search_radii(np.arange(lower_radius, r_outer_min, 2), 'smaller')
        best_inner = (0.65 * best_outer).astype(int)
        best_response[:margin, :] = -np.inf
        best_response[-margin:, :] = -np.inf
        best_response[:, :margin] = -np.inf
        best_response[:, -margin:] = -np.inf
        detections, smaller_diagnostics = _select_valid_nuclei(
            best_response, best_outer, best_inner, proj_s, params,
        )
        for candidate in smaller_diagnostics:
            candidate['search_stage'] = 'smaller'
        diagnostics.extend(smaller_diagnostics)

    if len(detections) == 0:
        raise NucleusValidationError(diagnostics)
    report(f'Accepted {len(detections)} nucleus/nuclei')

    # ---------------------------------------------------
    # build label map
    # ---------------------------------------------------
    roi_labels_2d = np.zeros((H, W), dtype=np.int32)

    centers = []
    radii = []

    for i, (x, y, r_in, r_out) in enumerate(detections, start=1):
        mask = (xx - x)**2 + (yy - y)**2 <= r_out**2

        roi_labels_2d[mask] = i

        centers.append([y, x])  # (y, x)
        radii.append(r_out)

    # ---------------------------------------------------
    # expand to 3D
    # ---------------------------------------------------
    
    
# ---------------------------------------------------
# build softer tapered 3D masks instead of harsh ellipsoid masks
# ---------------------------------------------------
    Z = roi_stack.shape[0]
    roi_labels_3d = allocate((Z, H, W), dtype=np.uint8)

    z_thresh_frac = 0.25      # was 0.35; lower = include more top/bottom slices
    z_pad = 1                 # add extra z slices above/below valid range
    min_taper_frac = 0.65     # radius never shrinks below 65% inside valid z range
    mask_radius_scale = 1.08

    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        
        r_mask = r * mask_radius_scale
        base_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_mask**2

        z_profile = np.array([
            roi_stack[z][base_mask].mean() if np.any(base_mask) else 0.0
            for z in range(Z)
        ], dtype=np.float32)

        z_profile = gaussian_filter(z_profile, sigma=1.0)

        z_peak = int(np.argmax(z_profile))
        z_thresh = z_thresh_frac * z_profile[z_peak]

        valid_z = np.where(z_profile >= z_thresh)[0]

        if len(valid_z) == 0:
            z0, z1 = max(0, z_peak - 2), min(Z - 1, z_peak + 2)
        else:
            z0, z1 = int(valid_z.min()), int(valid_z.max())

        # expand z range slightly so top/bottom foci are not lost
        z0 = max(0, z0 - z_pad)
        z1 = min(Z - 1, z1 + z_pad)

        zc = 0.5 * (z0 + z1)
        rz = max((z1 - z0) / 2.0, 1.0)

        for z in range(Z):
            check_cancelled()
            dz = (z - zc) / rz

            if abs(dz) > 1.0:
                continue

            # softer taper: do not let radius collapse at top/bottom
            taper = np.sqrt(max(0.0, 1.0 - dz**2))
            taper = max(min_taper_frac, taper)

            r_xy = r_mask * taper

            mask_z = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_xy**2
            roi_labels_3d[z, mask_z] = i

    roi_mask_3d = allocate((Z, H, W), dtype=bool)
    for z in range(Z):
        roi_mask_3d[z] = roi_labels_3d[z] > 0


    return {
        "roi_proj": proj,
        "mask2d": roi_labels_2d > 0,
        "labels2d": roi_labels_2d,
        "roi_mask_3d": roi_mask_3d,
        "roi_labels_3d": roi_labels_3d,
        "centers": np.array(centers),
        "radii": radii,
        "nucleus_count": len(detections),
        "candidate_diagnostics": diagnostics,
        "validation_params": asdict(params),
        "radius_search_extended": search_extended,
    }


def segment_nuclei(roi_stack, channel_nucleus=2, z_trim=5, validation_params=None,
                   voxel_size_um_zyx=None, settings=None):
    """Segment whole nuclei; Cellpose adds no Z trimming to the supplied stack."""
    from src.cellpose_backend import SegmentationSettings, segment_cellpose
    config = settings if isinstance(settings, SegmentationSettings) else SegmentationSettings(**(settings or {}))
    if config.backend == 'legacy':
        return segment_nuclei_legacy(roi_stack, channel_nucleus, z_trim, validation_params)
    if voxel_size_um_zyx is None:
        raise ValueError('Physical voxel sizes are required for Cellpose segmentation.')
    return segment_cellpose(roi_stack, voxel_size_um_zyx, config)
