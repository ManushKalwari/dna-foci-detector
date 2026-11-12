import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.foci_measurement import (
    FWHM_FACTOR, FocusMeasurement, measure_foci_native,
    processed_to_native_center,
)
from src.utils import (
    FOCUS_MEASUREMENT_CSV_COLUMNS, resolve_voxel_size_um,
    save_focus_measurements_csv,
)


def gaussian_volume(shape, spacing, centers_and_sigmas):
    zz, yy, xx = np.indices(shape)
    image = np.full(shape, 100.0, dtype=np.float32)
    for center, sigmas, amplitude in centers_and_sigmas:
        exponent = sum(
            (((grid - center[axis]) * spacing[axis]) / sigmas[axis]) ** 2
            for axis, grid in enumerate((zz, yy, xx))
        )
        image += amplitude * np.exp(-0.5 * exponent)
    volume = np.zeros((shape[0], 2, shape[1], shape[2]), dtype=np.float32)
    volume[:, 1] = image
    return volume


class CountingImage:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return self.array[index]


class FociMeasurementTests(unittest.TestCase):
    def test_anisotropic_gaussian_fwhm_and_halfmax_volume(self):
        spacing = np.array((0.25, 0.1, 0.1))
        center = np.array((15.0, 40.0, 40.0))
        sigmas = np.array((0.55, 0.32, 0.24))
        image = gaussian_volume((31, 81, 81), spacing, [(center, sigmas, 900.0)])
        result = measure_foci_native(
            image, 1, [(1, np.array((10, 40, 40, 4), dtype=float))],
            spacing, source_image='synthetic.czi', calibration_source='manual', trimmed_z_planes=5,
        )[0]
        fitted = np.array((result.gaussian_fwhm_z_um, result.gaussian_fwhm_y_um,
                           result.gaussian_fwhm_x_um))
        np.testing.assert_allclose(fitted, FWHM_FACTOR * sigmas, atol=spacing.max())
        expected_volume = np.pi / 6.0 * np.prod(FWHM_FACTOR * sigmas)
        self.assertAlmostEqual(result.gaussian_halfmax_volume_um3, expected_volume, places=3)
        self.assertGreater(result.watershed_volume_um3, 0)
        self.assertEqual(result.measurement_status, 'ok')

    def test_touching_foci_are_measured_separately_and_planes_read_once(self):
        spacing = np.array((0.2, 0.1, 0.1))
        centers = [np.array((15.0, 35.0, 37.0)), np.array((15.0, 35.0, 43.0))]
        image = CountingImage(gaussian_volume(
            (31, 81, 81), spacing,
            [(centers[0], (0.4, 0.28, 0.28), 700.0),
             (centers[1], (0.4, 0.28, 0.28), 650.0)],
        ))
        blobs = [(1, np.array((10, 35, 37, 3.5))),
                 (1, np.array((10, 35, 43, 3.5)))]
        results = measure_foci_native(image, 1, blobs, spacing, trimmed_z_planes=5)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.watershed_voxel_count > 0 for result in results))
        z_reads = [index[0] for index in image.reads]
        self.assertEqual(len(z_reads), len(set(z_reads)))

    def test_downsample_mapping_uses_native_block_centers(self):
        mapped = processed_to_native_center((0, 1, 2), (3, 3, 3))
        np.testing.assert_array_equal(mapped, (1, 4, 7))
        mapped = processed_to_native_center((0, 1, 2), (3, 3, 3), trimmed_z_planes=5)
        np.testing.assert_array_equal(mapped, (16, 4, 7))

    def test_flat_patch_keeps_detection_and_exports_failure_qc(self):
        image = np.full((31, 2, 41, 41), 100, dtype=np.uint16)
        result = measure_foci_native(
            image, 1, [(2, np.array((10, 20, 20, 3)))], (0.3, 0.1, 0.1),
        )[0]
        self.assertEqual(result.focus_id, 1)
        self.assertEqual(result.measurement_status, 'both_failed')
        self.assertIn('no_prominence', result.qc_flags)

    def test_resolve_spacing_requires_complete_positive_override(self):
        with patch('src.utils.image_voxel_size_um', return_value=(0.3, 0.1, 0.1)):
            self.assertEqual(resolve_voxel_size_um('sample.czi'),
                             ((0.3, 0.1, 0.1), 'metadata'))
        for invalid in ((0.3, 0, 0.1), (0.3, 0.1), (0.3, np.nan, 0.1)):
            with self.assertRaisesRegex(ValueError, 'three positive finite values'):
                resolve_voxel_size_um('sample.czi', invalid)
        with patch('src.utils.image_voxel_size_um', return_value=None):
            with self.assertRaisesRegex(ValueError, 'voxel-size metadata'):
                resolve_voxel_size_um('sample.czi')

    def test_focus_csv_has_headers_when_empty_and_stable_row_values(self):
        with TemporaryDirectory() as folder:
            empty = SimpleNamespace(file_path=str(Path(folder) / 'empty.czi'),
                                    focus_measurements=[])
            path = save_focus_measurements_csv([empty])
            frame = pd.read_csv(path)
            self.assertEqual(frame.columns.tolist(), FOCUS_MEASUREMENT_CSV_COLUMNS)
            self.assertEqual(len(frame), 0)
            measurement = FocusMeasurement(
                source_image='one.czi', focus_id=1, nucleus_id=2,
                center_z_px=1, center_y_px=2, center_x_px=3,
                center_z_um=.3, center_y_um=.2, center_x_um=.3,
                voxel_size_z_um=.3, voxel_size_y_um=.1, voxel_size_x_um=.1,
                calibration_source='metadata', measurement_status='ok',
            )
            populated = SimpleNamespace(file_path=str(Path(folder) / 'one.czi'),
                                        focus_measurements=[measurement])
            frame = pd.read_csv(save_focus_measurements_csv([populated]))
            self.assertEqual(frame.loc[0, 'source_image'], 'one.czi')
            self.assertEqual(frame.loc[0, 'nucleus_id'], 2)


if __name__ == '__main__':
    unittest.main()
