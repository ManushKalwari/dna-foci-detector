"""Check worker signals without creating an OpenGL viewer."""
import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from unittest.mock import MagicMock, patch

class WorkerUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from qtpy.QtWidgets import QApplication
        cls.qt_app = QApplication.instance() or QApplication([])
        filename = Path(__file__).resolve().parents[1] / 'app.py'
        spec = importlib.util.spec_from_file_location('worker_test_app', filename)
        cls.app = importlib.util.module_from_spec(spec)
        with patch('napari.Viewer', return_value=MagicMock()):
            spec.loader.exec_module(cls.app)

    def setUp(self):
        self.app.state.clear()
        self.app.controls.size_mode.setCurrentText('Automatic')
        self.app.controls.minimum_size.setValue(.2)
        self.app.controls.maximum_size.setValue(10.)
        self.app.controls.image_view.setCurrentText('Processed')
        self.app.controls.minimum_spot_volume.setValue(.2)
        self.app.refresh_controls()
        def points_layer(data, **kwargs):
            layer = MagicMock()
            layer.data = data
            layer.features = kwargs.get('features', {})
            return layer
        self.app.viewer.add_points.side_effect = points_layer

    def test_runtime_failure_finishes_and_reenables_controls(self):
        from qtpy.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        with patch.object(self.app, 'run_detection', side_effect=RuntimeError('test failure')), \
             patch.object(self.app, 'collect_voxel_size_overrides', return_value={}), \
             patch.object(self.app, 'show_error') as errors, \
             patch.object(self.app.traceback, 'print_exception'):
            self.app.start_detection(['sample.czi'])
            self.app.state['worker'].finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec_()
            self.assertFalse(self.app.state['busy'])
            self.assertTrue(self.app.detect_spots.enabled)
            self.assertEqual(errors.call_count, 1)
            self.assertIn('0/1 images processed; 1 skipped', errors.call_args[0][0])
            report = self.app.batch_report.toPlainText()
            self.assertIn(str(Path('sample.czi').resolve()), report)
            self.assertIn('RuntimeError: test failure', report)
            self.assertNotIn('results', self.app.state)

    def test_batch_keeps_successes_and_skips_first_middle_and_last_failures(self):
        from qtpy.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        paths = ['bad-first.czi', 'good-one.czi', 'bad-middle.lsm',
                 'good-two.lsm', 'bad-last.czi']
        first, second = object(), object()
        with patch.object(self.app, 'run_detection', side_effect=[
                ValueError('No valid nucleus'), first, OSError('Cannot read image'),
                second, RuntimeError('Detection failed')]) as detect, \
             patch.object(self.app, 'collect_voxel_size_overrides', return_value={}), \
             patch.object(self.app, 'show_results') as display, \
             patch.object(self.app, 'show_warning') as warnings, \
             patch.object(self.app, 'show_error') as errors, \
             patch.object(self.app.traceback, 'print_exception'):
            self.app.start_detection(paths)
            self.app.state['worker'].finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec_()
        self.assertEqual([call.args[0] for call in detect.call_args_list], paths)
        self.assertTrue(all(call.kwargs['smart_downsample'] for call in detect.call_args_list))
        self.assertTrue(all(call.kwargs['segmentation_settings']['backend'] == 'cellpose'
                            for call in detect.call_args_list))
        display.assert_called_once_with([first, second])
        self.assertEqual(warnings.call_count, 1)
        errors.assert_not_called()
        report = self.app.batch_report.toPlainText()
        self.assertIn('2/5 images processed; 3 skipped', report)
        for name, reason in zip(paths[::2], [
                'No valid nucleus', 'Cannot read image', 'Detection failed']):
            self.assertIn(str(Path(name).resolve()), report)
            self.assertIn(reason, report)
        self.assertFalse(self.app.state['busy'])
        self.assertTrue(self.app.detect_spots.enabled)

    def test_all_successful_batch_reports_completion(self):
        from qtpy.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        results = [object(), object()]
        with patch.object(self.app, 'run_detection', side_effect=results), \
             patch.object(self.app, 'collect_voxel_size_overrides', return_value={}), \
             patch.object(self.app, 'show_results') as display, \
             patch.object(self.app, 'show_info') as info:
            self.app.start_detection(['one.czi', 'two.czi'])
            self.app.state['worker'].finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec_()
        display.assert_called_once_with(results)
        self.assertEqual(self.app.state['failures'], [])
        self.assertIn('2/2 images processed; 0 skipped', info.call_args[0][0])

    def test_memory_limited_diameter_is_reported_as_needing_mask_review(self):
        from qtpy.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        result = SimpleNamespace(file_stem='uncertain', nucleus_count=1,
                                 segmentation_diagnostics={
                                     'model': 'cyto3', 'diameter_um': 18., 'total_seconds': 1.,
                                     'helper_peak_memory_bytes': 1000,
                                     'diameter_clamped_for_memory': True})
        with patch.object(self.app, 'run_detection', return_value=result), \
             patch.object(self.app, 'collect_voxel_size_overrides', return_value={}), \
             patch.object(self.app, 'show_results'), \
             patch.object(self.app, 'show_warning') as warnings:
            self.app.start_detection(['uncertain.czi'])
            self.app.state['worker'].finished.connect(loop.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec_()
        self.assertEqual(warnings.call_count, 1)
        self.assertIn('1 mask(s) need review', warnings.call_args[0][0])
        self.assertIn('Review mask:', self.app.batch_report.toPlainText())

    def test_accepted_mask_and_spots_align_with_original_z_coordinates(self):
        result = SimpleNamespace(
            img=np.zeros((26, 3, 32, 32), dtype=np.uint16),
            roi_labels_3d=np.ones((16, 32, 32), dtype=np.uint8),
            _workspace=None, trimmed_z_planes=5, nucleus_count=1, file_stem='one',
            contrast_limits=[(0, 1)]*3,
            blobs_filtered=np.array([[0, 4, 5, 3]], dtype=np.float32),
        )
        self.app.viewer.reset_mock()
        with patch.object(self.app, 'show_info'):
            self.app.show_results([result])
        args, kwargs = self.app.viewer.add_labels.call_args
        labels = args[0]
        self.assertEqual(labels.shape, (16, 32, 32))
        self.assertEqual(kwargs['name'], 'Accepted nuclei (1)')
        np.testing.assert_array_equal(kwargs['scale'], (1, 1, 1))
        np.testing.assert_array_equal(kwargs['translate'], (5, 0, 0))
        self.assertEqual(int(labels[0, 0, 0].compute()), 1)
        points = self.app.state['accepted_layer'].data
        np.testing.assert_array_equal(points, [[5, 4, 5]])
        self.app.viewer.dims.set_point.assert_called_with(0, 5)
        settings = self.app.viewer.add_points.call_args_list[-1].kwargs
        self.assertEqual(settings['face_color'], 'red')
        self.assertEqual(settings['canvas_size_limits'][0], 10)
        self.assertFalse(settings['out_of_slice_display'])

    def test_full_volume_display_preserves_early_slices_and_native_alignment(self):
        result = SimpleNamespace(
            img=np.zeros((77, 3, 24, 24), dtype=np.uint16),
            roi_labels_3d=np.ones((26, 8, 8), dtype=np.int32),
            processed_nuclei=np.ones((26, 8, 8), dtype=np.float32),
            processed_foci=np.ones((26, 8, 8), dtype=np.float32),
            processed_foci_limits=(0., 1.),
            _workspace=None, nucleus_count=1, file_stem='full_z',
            contrast_limits=[(0, 1)] * 3, processing_scale_zyx=(3, 3, 3),
            trimmed_z_planes=0, blobs_filtered=np.array([[4, 1, 2, 3]], np.float32),
        )
        self.app.viewer.reset_mock()
        self.app.show_results([result])
        image = self.app.viewer.add_image.call_args_list[-1]
        self.assertEqual(image.args[0].shape[0], 26)
        np.testing.assert_array_equal(image.kwargs['translate'], (1, 1, 1))
        np.testing.assert_array_equal(self.app.state['accepted_layer'].data, [[13, 4, 7]])

    def test_results_open_on_slice_with_most_spots_and_keep_all_coordinates(self):
        blobs = np.array([[1, 4, 5, 3], [9, 8, 9, 3], [9, 16, 18, 4]], dtype=np.float32)
        measurements = []
        for focus_id in range(1, 4):
            measurement = MagicMock()
            measurement.to_record.return_value = {
                'focus_id': focus_id, 'nucleus_id': 1,
                'watershed_volume_um3': focus_id / 10,
                'watershed_equivalent_diameter_um': focus_id / 5,
                'gaussian_fwhm_z_um': 1.0, 'gaussian_fwhm_y_um': 0.5,
                'gaussian_fwhm_x_um': 0.5, 'gaussian_halfmax_volume_um3': 0.2,
                'gaussian_fit_r2': 0.9, 'measurement_status': 'ok', 'qc_flags': '',
            }
            measurements.append(measurement)
        result = SimpleNamespace(
            img=np.zeros((26, 3, 32, 32), dtype=np.uint16),
            roi_labels_3d=np.ones((16, 32, 32), dtype=np.uint8),
            _workspace=None, trimmed_z_planes=5, nucleus_count=1, file_stem='example',
            contrast_limits=[(0, 1)]*3, blobs_filtered=blobs,
            focus_measurements=measurements,
        )
        self.app.viewer.reset_mock()
        with patch.object(self.app, 'show_info'):
            self.app.show_results([result])
        self.app.viewer.dims.set_point.assert_called_once_with(0, 14)
        displayed = self.app.state['accepted_layer'].data
        np.testing.assert_array_equal(displayed[:, 0], [6, 14, 14])
        features = self.app.state['accepted_layer'].features
        self.assertEqual(features['focus_id'], [1, 2, 3])
        self.assertEqual(features['watershed_volume_um3'], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(blobs[:, 0], [1, 9, 9])
        self.assertIn('3 spots across all Z slices', self.app.viewer.text_overlay.text)

    def test_empty_results_open_middle_slice(self):
        result = SimpleNamespace(
            img=np.zeros((26, 3, 32, 32), dtype=np.uint16),
            roi_labels_3d=np.ones((16, 32, 32), dtype=np.uint8),
            _workspace=None, trimmed_z_planes=5, nucleus_count=1, file_stem='empty',
            contrast_limits=[(0, 1)]*3, blobs_filtered=np.empty((0, 4), dtype=np.float32),
        )
        self.app.viewer.reset_mock()
        with patch.object(self.app, 'show_info'):
            self.app.show_results([result])
        self.app.viewer.dims.set_point.assert_called_once_with(0, 13)
        self.assertEqual(self.app.state['accepted_layer'].data.shape, (0, 3))

    def test_napari_worker_quit_cancels_detection_and_finishes(self):
        from qtpy.QtCore import QEventLoop, QTimer
        from src.memory import DetectionCancelled
        loop = QEventLoop()
        def slow_detection(path, progress_callback, cancel_event, **kwargs):
            while not cancel_event.wait(0.005):
                progress_callback('working')
            raise DetectionCancelled('Detection cancelled.')
        with patch.object(self.app, 'run_detection', slow_detection), \
             patch.object(self.app, 'collect_voxel_size_overrides', return_value={}), \
             patch.object(self.app, 'show_error') as errors, \
             patch.object(self.app, 'show_results') as results:
            self.app.start_detection(['sample.czi'])
            worker = self.app.state['worker']
            worker.finished.connect(loop.quit)
            QTimer.singleShot(20, worker.quit)
            QTimer.singleShot(5000, loop.quit)
            loop.exec_()
            self.assertTrue(worker.cancel_event.is_set())
            self.assertFalse(self.app.state['busy'])
            self.assertTrue(self.app.detect_spots.enabled)
            errors.assert_not_called()
            results.assert_not_called()

    def test_voxel_size_dialog_is_only_requested_for_missing_metadata(self):
        paths = ['with-meta.czi', 'missing-meta.lsm']
        with patch.object(self.app, 'image_voxel_size_um',
                          side_effect=[(0.3, 0.1, 0.1), None]), \
             patch.object(self.app, 'request_voxel_size',
                          return_value=(0.4, 0.2, 0.2)) as request:
            overrides = self.app.collect_voxel_size_overrides(paths)
        request.assert_called_once_with('missing-meta.lsm')
        expected_key = str(Path('missing-meta.lsm').resolve())
        self.assertEqual(overrides, {expected_key: (0.4, 0.2, 0.2)})

    def test_cancelling_missing_metadata_dialog_cancels_preflight(self):
        with patch.object(self.app, 'image_voxel_size_um', return_value=None), \
             patch.object(self.app, 'request_voxel_size', return_value=None):
            self.assertIsNone(self.app.collect_voxel_size_overrides(['missing.czi']))

    def test_unreadable_metadata_does_not_block_other_files_with_a_dialog(self):
        with patch.object(self.app, 'image_voxel_size_um',
                          side_effect=[OSError('Unreadable file'), (0.3, 0.1, 0.1)]), \
             patch.object(self.app, 'request_voxel_size') as request:
            overrides = self.app.collect_voxel_size_overrides(['bad.czi', 'good.czi'])
        self.assertEqual(overrides, {})
        request.assert_not_called()

    def test_downsampled_results_map_back_to_native_czi_coordinates(self):
        result = SimpleNamespace(
            img=np.zeros((78, 3, 24, 24), dtype=np.uint16),
            roi_labels_3d=np.ones((16, 8, 8), dtype=np.uint8),
            _workspace=None, trimmed_z_planes=5, nucleus_count=1, file_stem='scaled',
            contrast_limits=[(0, 1)]*3,
            blobs_filtered=np.array([[0, 1, 2, 3]], dtype=np.float32),
            processing_scale_zyx=(3, 3, 3),
        )
        self.app.viewer.reset_mock()
        with patch.object(self.app, 'show_info'):
            self.app.show_results([result])
        labels = self.app.viewer.add_labels.call_args[0][0]
        self.assertEqual(labels.shape, (16, 8, 8))
        settings = self.app.viewer.add_labels.call_args.kwargs
        np.testing.assert_array_equal(settings['scale'], (3, 3, 3))
        np.testing.assert_array_equal(settings['translate'], (16, 1, 1))
        self.assertEqual(int(labels[0, 0, 0].compute()), 1)
        points = self.app.state['accepted_layer'].data
        np.testing.assert_array_equal(points, [[16, 4, 7]])
        self.app.viewer.dims.set_point.assert_called_with(0, 16)

    def test_ctrl_c_handler_requests_stop_and_closes_viewer(self):
        import signal
        original_handler = signal.getsignal(signal.SIGINT)
        def fake_event_loop():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        with patch.object(self.app.napari, 'run', fake_event_loop), \
             patch.object(self.app, 'request_stop') as stop:
            self.app.viewer.reset_mock()
            self.app.main()
            self.app.viewer.close.assert_called_once()
            self.assertGreaterEqual(stop.call_count, 1)
        self.assertIs(signal.getsignal(signal.SIGINT), original_handler)

    def _cached_result(self, name='sample'):
        from src.spot_selection import SpotCandidate, apply_spot_selection
        candidate = SpotCandidate(1, 1, (1., 2., 3., 4.), dict(
            focus_id=1, nucleus_id=1, gaussian_fwhm_x_um=1., gaussian_fwhm_y_um=1.,
            gaussian_fit_status='ok', gaussian_fit_r2=.95, qc_flags='', watershed_volume_um3=1.))
        result = SimpleNamespace(file_stem=name, nucleus_blobs={1: np.empty((0, 4))},
                                 spot_candidates=(candidate,), spot_selection=None,
                                 spot_search_diagnostics={'searched_width_um': (.2, 10.)})
        apply_spot_selection(result)
        return result

    def test_viewer_numbers_restart_at_one_for_second_nucleus(self):
        from src.spot_selection import SpotCandidate
        candidates = [SpotCandidate(i, 1 if i <= 46 else 2, (1., 2., 3., 4.),
                                    dict(focus_id=i, nucleus_id=1 if i <= 46 else 2))
                      for i in range(1, 49)]
        features = self.app._candidate_features(candidates)
        self.assertEqual(features['focus_id'], list(range(1, 47)) + [1, 2])
        self.assertEqual(features['candidate_id'], list(range(1, 49)))

    def test_search_range_changes_require_rerun_and_keep_all_candidates(self):
        from src.spot_selection import SpotSearchSettings
        first, second = self._cached_result('one'), self._cached_result('two')
        self.app.state.update(results=[first, second], completed_search=SpotSearchSettings())
        controls = self.app.controls
        controls.size_mode.setCurrentText('Custom range')
        self.assertFalse(controls.custom_range.isHidden())
        self.assertFalse(hasattr(controls, 'apply_filter'))
        controls.minimum_size.setValue(2.)
        controls.maximum_size.setValue(4.)
        self.assertEqual(self.app.current_search_settings(), SpotSearchSettings(2., 4.))
        self.assertFalse(self.app.current_size_limits().width_active)
        self.assertEqual(self.app.current_size_limits().min_volume_um3, .2)
        self.assertEqual([r.nucleus_counts[1] for r in (first, second)], [1, 1])
        self.assertIn('Search range changed.', controls.pending.text())
        self.app.state['completed_search'] = SpotSearchSettings(2., 4.)
        self.app.refresh_controls()
        self.assertTrue(controls.pending.isHidden())
        controls.size_mode.setCurrentText('Automatic')
        self.assertTrue(controls.custom_range.isHidden())
        self.assertEqual(first.nucleus_counts[1], 1)
        self.assertIn('Search range changed.', controls.pending.text())

    def test_invalid_shared_range_preserves_selection_and_disables_run_export(self):
        from src.spot_selection import SpotSearchSettings
        result = self._cached_result()
        self.app.state.update(results=[result], completed_search=SpotSearchSettings())
        controls = self.app.controls
        controls.size_mode.setCurrentText('Custom range')
        previous = result.spot_selection
        controls.minimum_size.setValue(11.)
        self.assertIs(result.spot_selection, previous)
        self.assertFalse(controls.run_button.isEnabled())
        self.assertFalse(controls.export_widget.isEnabled())
        self.assertFalse(controls.error.isHidden())
        controls.maximum_size.setValue(12.)
        self.assertTrue(controls.run_button.isEnabled())
        self.assertTrue(controls.export_widget.isEnabled())

    def test_volume_changes_filter_cached_batch_without_changing_search(self):
        from src.spot_selection import SpotSearchSettings
        first, second = self._cached_result('one'), self._cached_result('two')
        self.app.state.update(results=[first, second], completed_search=SpotSearchSettings())
        self.app.controls.minimum_spot_volume.setValue(1.)
        self.assertEqual([r.nucleus_counts[1] for r in (first, second)], [0, 0])
        self.app.refresh_controls()
        self.assertTrue(self.app.controls.pending.isHidden())
        self.app.controls.minimum_spot_volume.setValue(.9)
        self.assertEqual([r.nucleus_counts[1] for r in (first, second)], [1, 1])
        self.app.controls.minimum_spot_volume.setValue(0.)
        self.assertIsNone(self.app.current_size_limits().min_volume_um3)

    def test_file_picker_and_drop_use_the_same_start_path(self):
        paths = [Path('sample.czi')]
        with patch.object(self.app, 'start_detection') as start:
            self.app.detect_spots(paths)
            start.assert_called_once_with(paths)
            start.reset_mock()
            self.app.handle_file_drop(self.app.viewer, paths + [Path('ignored.txt')])
            start.assert_called_once_with(paths)


if __name__ == '__main__':
    unittest.main()
