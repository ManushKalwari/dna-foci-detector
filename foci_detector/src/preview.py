"""Bounded, display-only 3D volumes built from the existing processing grid."""
from dataclasses import dataclass
import numpy as np

from src.memory import DetectionCancelled


@dataclass(frozen=True)
class Preview3D:
    nuclei: np.ndarray
    foci: np.ndarray
    labels: np.ndarray
    spacing_um: tuple
    origin_um: tuple


def prepare_3d_preview(result, cancel_event=None, max_xy=256, max_z=128):
    """Peak-pool foci, average nuclear context, and sample discrete labels.

    Output arrays together occupy at most 96 MiB at the default limits. Read
    processing planes individually; never materialize the original CZI volume.
    """
    source_nuclei = result.processed_nuclei
    source_foci = result.processed_foci
    source_labels = result.roi_labels_3d
    if source_nuclei is None or source_foci is None:
        raise ValueError('Run detection to prepare the processed volumes before opening 3D.')
    shape = np.asarray(source_foci.shape, dtype=int)
    if (shape.shape != (3,) or np.any(shape <= 0)
            or tuple(source_nuclei.shape) != tuple(shape)
            or tuple(source_labels.shape) != tuple(shape)):
        raise ValueError('Processed images and nuclear labels must share one 3D grid.')
    spacing = np.asarray(result.voxel_size_um_zyx, dtype=float)
    scale = np.asarray(result.processing_scale_zyx, dtype=float)
    if (spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError('Calibrated voxel spacing is required for the 3D preview.')
    limits = np.array([max_z, max_xy, max_xy])
    if np.any(limits <= 0):
        raise ValueError('3D preview grid limits must be positive.')
    factors = np.maximum(1, np.ceil(shape / limits).astype(int))
    output_shape = tuple((shape + factors - 1) // factors)
    nuclei = np.empty(output_shape, dtype=np.float32)
    foci = np.empty(output_shape, dtype=np.float32)
    labels = np.empty(output_shape, dtype=np.int32)
    sz, sy, sx = map(int, factors)
    pad_y, pad_x = (-int(shape[1])) % sy, (-int(shape[2])) % sx

    def cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise DetectionCancelled('3D preview cancelled.')

    def reduce_xy(plane, method):
        if pad_y or pad_x:
            plane = np.pad(plane, ((0, pad_y), (0, pad_x)), mode='edge')
        blocks = plane.reshape(output_shape[1], sy, output_shape[2], sx)
        return blocks.max(axis=(1, 3)) if method == 'max' else blocks.mean(axis=(1, 3))

    yi = np.minimum(np.arange(output_shape[1]) * sy + (sy - 1) // 2, shape[1] - 1)
    xi = np.minimum(np.arange(output_shape[2]) * sx + (sx - 1) // 2, shape[2] - 1)
    for out_z in range(output_shape[0]):
        cancelled()
        z0, z1 = out_z * sz, min((out_z + 1) * sz, shape[0])
        total = np.zeros(tuple(shape[1:]), dtype=np.float32)
        peak = np.full(tuple(shape[1:]), -np.inf, dtype=np.float32)
        for z in range(z0, z1):
            cancelled()
            total += np.asarray(source_nuclei[z], dtype=np.float32)
            np.maximum(peak, source_foci[z], out=peak)
        nuclei[out_z] = reduce_xy(total / (z1 - z0), 'mean')
        foci[out_z] = reduce_xy(peak, 'max')
        center_z = min(z0 + (sz - 1) // 2, shape[0] - 1)
        labels[out_z] = source_labels[center_z][np.ix_(yi, xi)]
    cancelled()
    processing_spacing = spacing * scale
    trim = getattr(result, 'trimmed_z_planes', 0)
    origin = ((scale - 1) / 2 + np.array([trim, 0, 0]) * scale) * spacing
    origin += (factors - 1) / 2 * processing_spacing
    return Preview3D(nuclei, foci, labels, tuple(processing_spacing * factors), tuple(origin))
