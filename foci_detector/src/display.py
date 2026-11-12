"""Display-only denoising and soft highlight compression, separate from analysis."""
import numpy as np
from scipy.ndimage import gaussian_filter

from src.memory import (
    allocate, check_cancelled, exact_percentile, filter_volume,
    release, report, selected_values,
)


def soft_intensity_curve(plane, black, reference, gamma=1.15):
    """Map reference intensity to a midtone without clipping brighter values.

    Unlike percentile clipping, x / (1 + x) has a smooth highlight shoulder.
    Gamma above one suppresses faint haze rather than amplifying it.
    """
    signal = np.nan_to_num(np.asarray(plane, dtype=np.float32), copy=True,
                           nan=0., posinf=0., neginf=0.)
    signal -= black
    np.maximum(signal, 0., out=signal)
    signal /= max(float(reference - black), 1e-8)
    signal /= 1. + signal
    np.power(signal, gamma, out=signal)
    return signal


def _map_display_in_place(volume):
    # One mapping for the entire stack prevents slice-to-slice contrast flicker.
    values = selected_values(volume)
    if values.size:
        black, reference = exact_percentile(values, (1., 99.5))
        maximum = float(np.max(values))
    else:
        black, reference, maximum = 0., 0., 0.
    release(values)
    if reference <= black:
        reference = maximum
    for z in range(volume.shape[0]):
        check_cancelled()
        if reference <= black:
            volume[z] = 0.
        else:
            volume[z] = soft_intensity_curve(volume[z], black, reference)
    return volume


def prepare_display_channels(nuclear_stack, foci_stack, spacing_um_zyx):
    """Prepare views from original processing-grid channels, never detector output.

    No upper intensity clipping is applied before the soft mapping. In
    particular, do not reuse the detector's percentile-clipped foci response.
    """
    spacing = np.asarray(spacing_um_zyx, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError('Display preparation requires positive calibrated voxel sizes.')
    report('Denoising nuclear display')
    nuclei = filter_volume(nuclear_stack, 'gaussian', sigma=tuple(.15 / spacing))
    report('Preparing foci display without highlight clipping')
    foci = allocate(foci_stack.shape)
    for z in range(foci_stack.shape[0]):
        check_cancelled()
        plane = np.nan_to_num(np.asarray(foci_stack[z], dtype=np.float32), copy=True,
                              nan=0., posinf=0., neginf=0.)
        smooth = gaussian_filter(plane, sigma=tuple(.08 / spacing[1:]))
        background = gaussian_filter(smooth, sigma=tuple(2. / spacing[1:]))
        np.subtract(smooth, background, out=smooth)
        np.maximum(smooth, 0., out=smooth)
        foci[z] = smooth
    report('Applying display intensity curves')
    return _map_display_in_place(nuclei), _map_display_in_place(foci)
