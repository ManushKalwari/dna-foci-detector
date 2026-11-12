import unittest
from pathlib import Path
from unittest.mock import patch
from threading import Event
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, binary_erosion

from src.memory import ImageWorkspace, using_workspace, filter_volume, response_statistics, DetectionCancelled
from src.spot_detector import (
    _normalize_spot_stack, _build_multiscale_3d_dog_response, _robust_sigma,
    _erode_labels_3d, detect_spots,
    run_detection,
)
from src.utils import (
    open_image, channel_contrast_limits, lazy_stack, downsample_channel,
    smart_downsample_factors,
)


def reference_normalize(spot_stack, roi_mask_3d, bg_sigma_xy=18.0):
    spot = spot_stack.astype(np.float32, copy=False)
    lo, hi = np.percentile(spot[np.isfinite(spot)], (0.1, 99.9))
    spot = np.clip(spot, lo, hi).astype(np.float32)
    sub = spot - gaussian_filter(spot, sigma=(0.0, bg_sigma_xy, bg_sigma_xy))
    sub[sub < 0] = 0.0
    norm = np.zeros_like(sub)
    for z in range(len(sub)):
        vals = sub[z][roi_mask_3d[z]]
        if vals.size < 32:
            vals = sub[z][np.isfinite(sub[z])]
        med = float(np.median(vals)) if vals.size else 0.0
        norm[z] = (sub[z] - med) / (_robust_sigma(vals) + 1e-8)
    norm[~np.isfinite(norm)] = 0
    norm[norm < 0] = 0
    return norm


def reference_dog(spot_norm, sigmas_xy=(1.6, 2.2, 2.6), dog_ratio=2.0, sigma_z=0.7):
    responses = []
    for s in sigmas_xy:
        small = gaussian_filter(spot_norm, sigma=(sigma_z, s, s))
        large = gaussian_filter(spot_norm, sigma=(max(1.0, 2*sigma_z), dog_ratio*s, dog_ratio*s))
        response = small - large
        response[response < 0] = 0
        response *= float(s**1.2)
        responses.append(response)
    responses = np.stack(responses)
    radii = np.stack([np.full_like(spot_norm, 1.8*s) for s in sigmas_xy])
    best = responses.argmax(axis=0)
    return (responses.max(axis=0),
            np.take_along_axis(radii, best[None], axis=0)[0],
            gaussian_filter(spot_norm, sigma=(0.4, 0.8, 0.8)))


class MemoryPipelineTests(unittest.TestCase):
    def setUp(self):
        self.workspace = ImageWorkspace()
        self.context = using_workspace(self.workspace)
        self.context.__enter__()
        self.rng = np.random.default_rng(23)
        self.data = self.rng.uniform(0, 100, (25, 48, 53)).astype(np.float32)

    def tearDown(self):
        directory = Path(self.workspace.directory.name)
        self.context.__exit__(None, None, None)
        self.workspace.close()
        self.assertFalse(directory.exists())

    def test_filters_match_at_block_and_volume_edges(self):
        for sigma in [(0.7, 2.6, 2.6), (1.4, 5.2, 5.2)]:
            np.testing.assert_array_equal(filter_volume(self.data, 'gaussian', sigma=sigma),
                                          gaussian_filter(self.data, sigma=sigma))
        np.testing.assert_array_equal(filter_volume(self.data, 'median', size=(1, 5, 5)),
                                      median_filter(self.data, size=(1, 5, 5)))
        np.testing.assert_array_equal(
            filter_volume(self.data, 'maximum', size=(7, 21, 21), mode='nearest'),
            maximum_filter(self.data, size=(7, 21, 21), mode='nearest'))

    def test_normalization_and_running_scales_match_reference(self):
        mask = self.data > 20
        actual = _normalize_spot_stack(self.data, mask)
        expected = reference_normalize(self.data, mask)
        np.testing.assert_array_equal(actual, expected)
        actual_dog = _build_multiscale_3d_dog_response(actual, (1.6, 2.2, 2.6), 2.0, 0.7)
        for actual, expected in zip(actual_dog, reference_dog(expected)):
            np.testing.assert_array_equal(actual, expected)

    def test_statistics_and_erosion_preserve_global_rules(self):
        labels = np.zeros(self.data.shape, dtype=np.int32)
        labels[:, 5:25, 5:25] = 1
        labels[:, 35, 35] = 2  # Erosion fallback must retain tiny nuclei.
        expected = np.zeros_like(labels)
        for label_id in (1, 2):
            mask = labels == label_id
            eroded = binary_erosion(mask, structure=np.ones((1, 3, 3)), iterations=2)
            if eroded.sum() < 0.25 * mask.sum():
                eroded = mask
            expected[eroded] = label_id
        np.testing.assert_array_equal(_erode_labels_3d(labels, 2), expected)
        mask = labels > 0
        vals = self.data[mask]
        med, sig, percentile = response_statistics(self.data, mask, 99.5)
        self.assertEqual(med, float(np.median(vals)))
        self.assertAlmostEqual(sig, _robust_sigma(vals), places=5)
        self.assertAlmostEqual(percentile, float(np.percentile(vals, 99.5)), places=5)

    def test_spots_match_legacy_filtering_and_have_no_seam_duplicates(self):
        data = self.rng.uniform(0, 3, (25, 48, 53)).astype(np.float32)
        zz, yy, xx = np.indices(data.shape)
        # Peaks across the 8-slice processing boundaries, including volume ends.
        for z in (1, 7, 8, 15, 16, 23):
            data += 100 * np.exp(-((zz-z)**2/2 + (yy-24)**2/8 + (xx-26)**2/8))
        labels = np.ones(data.shape, dtype=np.uint8)
        mask = labels > 0
        params = dict(peak_mad_k=2, peak_percentile=80, prominence_noise_k=0,
                      min_prominence_abs=0, min_radius_px=0, boundary_erosion_px=0,
                      min_distance_z=1, min_distance_xy=2)
        actual = detect_spots(data, mask, labels, **params)
        with patch('src.spot_detector._normalize_spot_stack', reference_normalize), \
             patch('src.spot_detector._build_multiscale_3d_dog_response', reference_dog):
            expected = detect_spots(data, mask, labels, **params)
        self.assertGreater(len(actual['blobs_filtered']), 0)
        np.testing.assert_array_equal(actual['blobs_filtered'], expected['blobs_filtered'])
        self.assertAlmostEqual(actual['threshold'], expected['threshold'], places=5)

    def test_lazy_display_and_exact_contrast(self):
        mapped = self.workspace.allocate((4, 8, 9), np.uint16)
        mapped[:] = np.arange(mapped.size).reshape(mapped.shape)
        lazy = lazy_stack(mapped)
        np.testing.assert_array_equal(lazy[2].compute(scheduler='synchronous'), mapped[2])
        np.testing.assert_allclose(channel_contrast_limits(lazy), np.percentile(mapped, [1, 99.5]))

    def test_smart_reduction_preserves_single_voxel_focus_and_constant_nucleus(self):
        source = np.full((5, 7, 8), 90, dtype=np.uint16)
        source[4, 6, 7] = 4095
        nucleus = downsample_channel(source, (2, 3, 3), 'mean').compute(
            scheduler='synchronous')
        foci = downsample_channel(source, (2, 3, 3), 'max').compute(
            scheduler='synchronous')
        self.assertEqual(nucleus.shape, (3, 3, 3))
        self.assertEqual(foci.shape, (3, 3, 3))
        self.assertEqual(int(foci[-1, -1, -1]), 4095)
        self.assertGreaterEqual(float(nucleus.min()), 90.0)

    def test_complete_pipeline_retains_mappings_and_cleans_up_on_failure(self):
        pixels = self.rng.integers(0, 2000, (26, 3, 96, 96), dtype=np.uint16)
        yy, xx = np.mgrid[:96, :96]
        ring = 100 + 1800 * np.exp(-0.5 * ((np.hypot(yy-48, xx-48)-20)/1.5)**2)
        pixels[:, 2] = ring.astype(np.uint16)
        with patch('src.spot_detector.open_image', return_value=pixels):
            result = run_detection(
                'synthetic.czi', voxel_size_override_um_zyx=(0.35, 0.11, 0.11),
                segmentation_settings={'backend': 'legacy'},
            )
        self.assertEqual(result.roi_stack.shape, (26, 96, 96))
        self.assertEqual(result.processed_foci.shape[0], 26)
        self.assertEqual(result.trimmed_z_planes, 0)
        self.assertIsInstance(result.roi_stack, np.memmap)
        self.assertEqual(result.img.shape, pixels.shape)
        self.assertEqual(len(result.contrast_limits), 3)
        self.assertEqual(result.nucleus_count, 1)
        directory = Path(result._workspace.directory.name)
        self.assertTrue(directory.exists())
        result._workspace.close()
        self.assertFalse(directory.exists())
        workspace = ImageWorkspace()
        directory = Path(workspace.directory.name)
        with patch('src.spot_detector.ImageWorkspace', return_value=workspace), \
             patch('src.spot_detector.open_image', side_effect=ValueError('read failed')):
            with self.assertRaisesRegex(ValueError, 'read failed'):
                run_detection('bad.czi', voxel_size_override_um_zyx=(0.35, 0.11, 0.11))
        self.assertFalse(directory.exists())

    def test_real_czi_reads_are_lazy_and_match_native_subblocks(self):
        path = Path(__file__).resolve().parents[2] / 'data' / '1.czi'
        if not path.exists():
            self.skipTest('Local CZI sample unavailable')
        from aicspylibczi import CziFile
        with patch('src.utils._read_czi_plane') as read:
            lazy = open_image(path)
            read.assert_not_called()
        self.assertEqual(lazy.shape, (114, 4, 2196, 2196))
        self.assertEqual(smart_downsample_factors(path, lazy.shape), (3, 3, 3))
        with open(path, 'rb') as stream:
            czi = CziFile(stream)
            for z, channel in [(0, 1), (57, 2), (113, 3)]:
                expected = czi.read_image(T=0, Z=z, C=channel, cores=1)[0].squeeze()
                actual = open_image(path)[z, channel].compute(scheduler='synchronous')
                np.testing.assert_array_equal(actual, expected)

    def test_cancelling_nested_detection_cleans_all_temporary_mappings(self):
        pixels = np.zeros((26, 3, 96, 96), dtype=np.uint16)
        yy, xx = np.mgrid[:96, :96]
        pixels[:, 2] = (100 + 1800*np.exp(-0.5*((np.hypot(yy-48, xx-48)-20)/1.5)**2)).astype(np.uint16)
        event = Event()
        directories = []
        def workspace_factory():
            workspace = ImageWorkspace()
            directories.append(Path(workspace.directory.name))
            return workspace
        def progress(message):
            if message.startswith('Background normalization'):
                event.set()
        with patch('src.spot_detector.open_image', return_value=pixels), \
             patch('src.spot_detector.ImageWorkspace', side_effect=workspace_factory):
            with self.assertRaises(DetectionCancelled):
                run_detection(
                    'synthetic.czi', progress_callback=progress, cancel_event=event,
                    voxel_size_override_um_zyx=(0.35, 0.11, 0.11),
                    segmentation_settings={'backend': 'legacy'},
                )
        self.assertTrue(event.is_set())
        self.assertGreaterEqual(len(directories), 2)
        self.assertTrue(all(not directory.exists() for directory in directories))


if __name__ == '__main__':
    unittest.main()
