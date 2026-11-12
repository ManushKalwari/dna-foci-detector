"""Deferred checks for bounded 3D display preparation; no OpenGL is required."""
from threading import Event
from types import SimpleNamespace
import unittest

import numpy as np

from src.memory import DetectionCancelled
from src.preview import prepare_3d_preview


class PreviewTests(unittest.TestCase):
    def sample(self):
        shape = (7, 19, 23)
        foci = np.zeros(shape, dtype=np.float32)
        foci[0, 0, 0] = 17.
        foci[-1, -1, -1] = 23.
        return SimpleNamespace(
            processed_nuclei=np.ones(shape, dtype=np.float32), processed_foci=foci,
            roi_labels_3d=np.ones(shape, dtype=np.int32),
            voxel_size_um_zyx=(.13, .035, .035), processing_scale_zyx=(3, 3, 3),
            trimmed_z_planes=0,
        )

    def test_preview_is_bounded_preserves_bright_edges_and_does_not_modify_source(self):
        result = self.sample()
        original = result.processed_foci.copy()
        preview = prepare_3d_preview(result, max_xy=8, max_z=4)
        self.assertEqual(preview.foci.shape, (4, 7, 8))
        self.assertEqual(preview.foci[0, 0, 0], 17.)
        self.assertEqual(preview.foci[-1, -1, -1], 23.)
        np.testing.assert_array_equal(preview.nuclei, 1.)
        self.assertEqual(set(np.unique(preview.labels)), {1})
        np.testing.assert_array_equal(result.processed_foci, original)
        self.assertEqual(preview.foci.dtype, np.float32)
        self.assertEqual(preview.labels.dtype, np.int32)

    def test_preview_uses_physical_spacing_and_processing_block_centers(self):
        result = self.sample()
        preview = prepare_3d_preview(result, max_xy=8, max_z=4)
        np.testing.assert_allclose(preview.spacing_um, (.78, .315, .315))
        np.testing.assert_allclose(preview.origin_um, (.325, .14, .14))
        result.trimmed_z_planes = 5
        trimmed = prepare_3d_preview(result, max_xy=8, max_z=4)
        np.testing.assert_allclose(trimmed.origin_um, (2.275, .14, .14))

    def test_cancellation_and_invalid_calibration_are_explicit(self):
        result = self.sample()
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(DetectionCancelled):
            prepare_3d_preview(result, cancelled, max_xy=8, max_z=4)
        result.voxel_size_um_zyx = (0, .035, .035)
        with self.assertRaises(ValueError):
            prepare_3d_preview(result, max_xy=8, max_z=4)
