import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.cellpose_backend import (
    SegmentationSession, SegmentationSettings, prepare_grid, restore_labels, filter_small_instances,
    filter_signal_instances,
)
from src.memory import ImageWorkspace, using_workspace, DetectionCancelled
from src.nuclei_segmenter import segment_nuclei
from src.cellpose_worker import bounded_diameter
from src.spot_detector import _erode_labels_3d, _extract_3d_candidates, detect_spots
from src.utils import save_counts_csv


class CellposeBackendTests(unittest.TestCase):
    def test_packaged_helper_path_matches_platform_layout(self):
        with patch.object(sys, 'frozen', True, create=True), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, 'platform', 'darwin'), \
             patch.object(sys, 'executable', '/Applications/DNA_Foci_Detector.app/Contents/MacOS/DNA_Foci_Detector'):
            session = SegmentationSession()
            try:
                self.assertEqual(
                    session.python.parts[-4:],
                    ('Contents', 'Resources', 'segmentation', 'CellposeWorker'),
                )
            finally:
                session.close()

    def test_grid_caps_xy_and_preserves_physical_anisotropy(self):
        image = np.random.default_rng(3).uniform(size=(4, 600, 800)).astype(np.float32)
        grid, spacing = prepare_grid(image, (.4, .1, .1))
        self.assertEqual(grid.shape, (4, 384, 512))
        np.testing.assert_allclose(spacing, (.4, .15625, .15625))
        self.assertGreaterEqual(grid.min(), 0)
        self.assertLessEqual(grid.max(), 1)

    def test_restore_preserves_irregular_boundaries_fills_holes_and_never_overwrites_neighbors(self):
        labels = np.zeros((3, 32, 32), dtype=np.int32)
        labels[:, 3:24, 3:13] = 5
        labels[:, 17:24, 13:22] = 5  # non-circular L shape
        labels[:, 7:20, 13:25] = 20  # touching distinct nucleus
        labels[:, 9:12, 7:10] = 0  # enclosed dark nucleolus
        result = restore_labels(labels, (3, 64, 64))
        self.assertEqual(set(np.unique(result)), {0, 1, 2})
        self.assertEqual(result[1, 20, 16], 1)
        self.assertEqual(result[1, 24, 30], 2)
        self.assertEqual(result[1, 50, 8], 0)
        self.assertEqual(result[1, 30, 54], 0)

    def test_default_backend_returns_exact_input_shape_without_extra_z_trimming(self):
        stack = np.random.default_rng(3).uniform(size=(12, 48, 48)).astype(np.float32)
        labels = np.zeros(stack.shape, dtype=np.int32)
        for i, x in enumerate((2, 17, 32), 1):
            labels[:, 10:30, x:x+10] = i
        response = {'diameter_px': 18., 'helper_peak_memory_bytes': 1000}
        with patch.object(SegmentationSession, 'infer', return_value=(labels, response)):
            seg = segment_nuclei(stack, voxel_size_um_zyx=(.3, .1, .1),
                                 settings={'min_volume_um3': 0, 'min_signal_ratio': 0})
        self.assertEqual(seg['roi_labels_3d'].shape, stack.shape)
        self.assertEqual(seg['nucleus_count'], 3)
        self.assertTrue(seg['roi_mask_3d'][0].any())
        self.assertTrue(seg['roi_mask_3d'][-1].any())
        self.assertEqual(seg['segmentation_diagnostics']['model'], 'cyto3')

    def test_empty_model_result_fails_explicitly(self):
        stack = np.random.default_rng(3).uniform(size=(12, 32, 32)).astype(np.float32)
        with patch.object(SegmentationSession, 'infer', return_value=(np.zeros_like(stack, dtype=int), {'diameter_px': 18.})):
            with self.assertRaisesRegex(RuntimeError, 'detected no nuclei'):
                segment_nuclei(stack, voxel_size_um_zyx=(.3, .1, .1))

    def test_erosion_keeps_third_nucleus_and_large_label_ids(self):
        labels = np.zeros((3, 32, 32), dtype=np.int32)
        labels[:, 2:12, 2:12] = 1
        labels[:, 18:29, 18:29] = 300
        eroded = _erode_labels_3d(labels)
        self.assertEqual(set(np.unique(eroded)), {0, 1, 300})
        self.assertEqual(eroded[1, 22, 22], 300)

    def test_physical_volume_filter_is_adjustable_and_preserves_zero_filter(self):
        labels = np.zeros((4, 32, 32), dtype=np.int32)
        labels[:, 2:12, 2:12] = 1
        labels[:, 20:30, 20:30] = 3
        labels[:, 14:16, 14:16] = 2
        original = labels.copy()
        rejected = filter_small_instances(labels, (1., 1., 1.), 100.)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertEqual(rejected, [{'original_id': 2, 'volume_um3': 16.}])
        self.assertEqual(labels[2, 24, 24], 2)
        self.assertEqual(filter_small_instances(original, (1., 1., 1.), 0.), [])
        self.assertEqual(set(np.unique(original)), {0, 1, 2, 3})

    def test_signal_validation_rejects_background_and_keeps_irregular_nucleus_with_dark_hole(self):
        labels = np.zeros((8, 48, 64), dtype=np.int32)
        labels[:, 5:26, 5:20] = 1
        labels[:, 17:26, 20:29] = 1
        labels[:, 6:24, 40:56] = 2
        image = np.full(labels.shape, .1, dtype=np.float32)
        image[labels == 1] = .8
        image[:, 10:15, 10:15] = 0  # nucleus includes a dark nucleolus
        rejected, unavailable = filter_signal_instances(labels, image, (.3, .1, .1), 0., 1.3)
        self.assertEqual([r['original_id'] for r in rejected], [2])
        self.assertEqual(unavailable, [])
        self.assertEqual(set(np.unique(labels)), {0, 1})

    def test_candidate_in_third_nucleus_is_retained(self):
        response = np.zeros((5, 32, 32), dtype=np.float32)
        response[2, 16, 16] = 20
        labels = np.zeros_like(response, dtype=np.int32)
        labels[:, 5:27, 5:27] = 3
        radius = np.full_like(response, 3.)
        with patch('src.spot_detector._local_prominence_features_3d', return_value={'prominence': 10., 'snr': 10.}):
            candidates, _ = _extract_3d_candidates(response, response, radius, labels)
        self.assertEqual([c['label_id'] for c in candidates], [3])

    def test_spot_grouping_and_exports_include_all_nuclei_including_zero_spots(self):
        labels = np.zeros((12, 32, 48), dtype=np.int32)
        labels[:, 5:27, 1:15] = 1
        labels[:, 5:27, 16:30] = 2
        labels[:, 5:27, 31:46] = 3
        # Isolate the grouping contract from the unchanged peak extraction.
        peak = {'label_id': 3, 'z': 6, 'y': 16, 'x': 38, 'radius': 4., 'score': 10.}
        with patch('src.spot_detector._extract_3d_candidates', side_effect=[([peak], 1.), ([], 1.)]):
            det = detect_spots(np.random.default_rng(4).uniform(size=labels.shape).astype(np.float32), labels > 0, labels)
        self.assertEqual(det['nucleus_counts'], {1: 0, 2: 0, 3: 1})
        self.assertEqual(len(det['blobs_filtered']), 1)
        r = SimpleNamespace(file_path='test.czi', file_stem='test', nucleus_count=3,
                            nucleus_counts=det['nucleus_counts'])
        with TemporaryDirectory() as directory:
            csv = pd.read_csv(save_counts_csv([r], directory))
        self.assertEqual(csv['spot_count'].tolist(), [0, 0, 1])
        self.assertEqual(csv['filename_cellnumber'].tolist(), ['test_nucleus1', 'test_nucleus2', 'test_nucleus3'])

    def test_cancellation_stops_running_helper_and_removes_staged_arrays(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / 'slow_helper.py'
            script.write_text('import sys, time\nfor line in sys.stdin:\n time.sleep(60)\n')
            import sys
            session = SegmentationSession(python=sys.executable)
            stage = Path(session.directory.name)
            # Start a real subprocess with the same protocol but no torch dependency.
            import subprocess
            original = subprocess.Popen
            def launch(args, **kwargs):
                return original([sys.executable, '-u', str(script)], **kwargs)
            event = Event()
            workspace = ImageWorkspace()
            workspace.cancel_event = event
            process = None
            with patch('src.cellpose_backend.subprocess.Popen', side_effect=launch):
                timer = Thread(target=lambda: (time.sleep(.3), event.set()))
                timer.start()
                with using_workspace(workspace):
                    with self.assertRaises(DetectionCancelled):
                        session.infer(np.ones((3, 8, 8)), 'cyto3', 10., 1.)
                timer.join()
            self.assertIsNone(session.process)
            self.assertFalse(stage.exists())
            workspace.close()
            session.close()

    def test_bad_settings_are_rejected(self):
        for options in ({'diameter_um': -1}, {'model': 'sam'}, {'backend': 'automatic'},
                        {'max_xy': 1024}, {'mode': 'auto'}, {'intensity_gamma': 0}, {'min_volume_um3': -1}):
            with self.assertRaises(ValueError):
                SegmentationSettings(**options)

    def test_internal_network_grid_is_bounded_for_tiny_automatic_diameters(self):
        for shape, anisotropy in [((28, 512, 512), 2.57), ((100, 512, 512), 5.)]:
            diameter, clamped = bounded_diameter(shape, anisotropy, 7., 30.)
            self.assertTrue(clamped)
            rescale = 30./diameter
            self.assertLessEqual(shape[1]*rescale, 128.)
            self.assertLessEqual(shape[0]*anisotropy*rescale, 64.)
        self.assertEqual(bounded_diameter((22,512,512), 2.57, 150., 30.), (150., False))
        with self.assertRaisesRegex(ValueError, 'CPU memory budget'):
            bounded_diameter((28,512,512), 2.57, 7., 30., manual=True)

    def test_helper_is_reused_and_can_restart_after_a_failed_image(self):
        import subprocess
        import sys
        with TemporaryDirectory() as directory:
            script = Path(directory) / 'helper.py'
            script.write_text(
                'import json, sys, os, numpy as np\n'
                'for line in sys.stdin:\n'
                ' r=json.loads(line)\n'
                ' if r["model"] == "nuclei":\n'
                '  print(json.dumps({"ok":False,"error":"test image failure"}),flush=True)\n'
                ' else:\n'
                '  np.save(r["output"],np.ones(np.load(r["input"]).shape,dtype=np.int32))\n'
                '  print(json.dumps({"ok":True,"pid":os.getpid()}),flush=True)\n')
            original = subprocess.Popen
            def launch(args, **kwargs):
                return original([sys.executable, '-u', str(script)], **kwargs)
            session = SegmentationSession(python=sys.executable)
            try:
                with patch('src.cellpose_backend.subprocess.Popen', side_effect=launch) as launches:
                    labels, first = session.infer(np.ones((3, 8, 8)), 'cyto3', 10., 1.)
                    _, second = session.infer(np.ones((3, 8, 8)), 'cyto3', 10., 1.)
                    self.assertEqual(first['pid'], second['pid'])
                    self.assertEqual(launches.call_count, 1)
                    with self.assertRaisesRegex(RuntimeError, 'test image failure'):
                        session.infer(np.ones((3, 8, 8)), 'nuclei', 10., 1.)
                    _, third = session.infer(np.ones((3, 8, 8)), 'cyto3', 10., 1.)
                    self.assertNotEqual(first['pid'], third['pid'])
                    self.assertEqual(launches.call_count, 2)
                    self.assertTrue(np.all(labels == 1))
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()

