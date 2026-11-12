"""Deferred coverage for physical spot selection. Run only when requested."""
from dataclasses import FrozenInstanceError
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.spot_selection import (
    SpotCandidate, SpotSearchSettings, SpotSizeLimits, apply_spot_selection,
    candidate_export_records, selection_warnings,
)
from src.utils import save_counts_csv, save_focus_measurements_csv
from src.spot_detector import detect_spots, filter_volume, _extract_3d_candidates
from src.foci_measurement import watershed_export_record


def candidate(focus_id, nucleus=1, width=1., **overrides):
    record = dict(focus_id=focus_id, nucleus_id=nucleus, source_image='sample.czi',
                  gaussian_fwhm_x_um=width, gaussian_fwhm_y_um=width,
                  gaussian_fit_status='ok', gaussian_fit_r2=.95, qc_flags='')
    record.update(overrides)
    return SpotCandidate(focus_id, nucleus, (5., float(focus_id), 10., 3.), record)


def result(candidates):
    return SimpleNamespace(
        file_path='sample.czi', file_stem='sample', nucleus_count=3,
        nucleus_blobs={i: np.empty((0, 4), np.float32) for i in (1, 2, 3)},
        spot_candidates=tuple(candidates), spot_selection=None,
        spot_search_diagnostics=dict(mode='physical', searched_width_um=(.2, 10.),
                                     resolution_xy_um=.2, warnings=[]),
    )


class SpotSelectionTests(unittest.TestCase):
    def test_volume_cutoff_is_strict_and_independent_of_gaussian_fit(self):
        r = result([candidate(1, watershed_volume_um3=.49),
                    candidate(2, watershed_volume_um3=.5),
                    candidate(3, watershed_volume_um3=.51, gaussian_fit_status='failed'),
                    candidate(4, watershed_volume_um3=None),
                    candidate(5, watershed_volume_um3=float('nan'))])
        original = r.spot_candidates
        apply_spot_selection(r, SpotSizeLimits(min_volume_um3=.5))
        self.assertEqual(r.spot_selection.statuses,
                         ('below_volume_limit', 'below_volume_limit', 'accepted',
                          'volume_unavailable', 'volume_unavailable'))
        self.assertEqual(r.nucleus_counts, {1: 1, 2: 0, 3: 0})
        self.assertEqual(r.uncertain_counts[1], 2)
        self.assertEqual(r.spot_candidates[2].focus_id, 3)
        rows = list(candidate_export_records(r, include_excluded=True))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]['selection_status'], 'below_volume_limit')
        apply_spot_selection(r, SpotSizeLimits())
        self.assertEqual(r.nucleus_counts[1], 5)
        self.assertIs(r.spot_candidates, original)

    def test_boundary_peak_is_not_removed_by_threshold_mask_erosion(self):
        shape = (7, 24, 24)
        labels = np.zeros(shape, dtype=np.int32)
        labels[:, 5:19, 5:19] = 1
        response = np.zeros(shape, dtype=np.float32)
        response[3, 5, 12] = 100.  # Inside the true mask, outside its eroded interior.
        radii = np.full(shape, 4., dtype=np.float32)
        features = dict(prominence=20., snr=20.)
        with patch('src.spot_detector.response_statistics', return_value=(0., 1., 0.)), \
             patch('src.spot_detector._local_prominence_features_3d', return_value=features):
            candidates, _ = _extract_3d_candidates(
                response, response, radii, labels,
                peak_mad_k=1., peak_percentile=None, min_distance_xy=1, min_distance_z=1,
                boundary_erosion_px=1,
            )
        self.assertEqual([(c['z'], c['y'], c['x']) for c in candidates], [(3, 5, 12)])

    def test_inclusive_limits_restore_candidates_and_keep_zero_spot_nuclei(self):
        r = result([candidate(1, width=1.), candidate(2, nucleus=2, width=5.),
                    candidate(3, nucleus=2, width=8.)])
        original = r.spot_candidates
        apply_spot_selection(r, SpotSizeLimits(5., 8.))
        self.assertEqual(r.nucleus_counts, {1: 0, 2: 2, 3: 0})
        self.assertEqual(r.spot_selection.statuses, ('outside_range', 'accepted', 'accepted'))
        apply_spot_selection(r)
        self.assertEqual(r.nucleus_counts, {1: 1, 2: 2, 3: 0})
        self.assertIs(r.spot_candidates, original)
        self.assertEqual([c.focus_id for c in original], [1, 2, 3])

    def test_width_is_geometric_mean_and_snapshots_are_immutable(self):
        c = candidate(1, gaussian_fwhm_x_um=4., gaussian_fwhm_y_um=9.)
        self.assertAlmostEqual(c.width_um, 6.)
        with self.assertRaises(TypeError):
            c.measurement['gaussian_fwhm_x_um'] = 5.
        with self.assertRaises(FrozenInstanceError):
            c.focus_id = 2

    def test_uncertain_fits_are_separate_only_when_filtering(self):
        variants = [dict(gaussian_fit_status='failed'), dict(gaussian_fit_r2=None),
                    dict(gaussian_fit_r2=.7), dict(gaussian_fwhm_x_um=float('nan')),
                    dict(qc_flags='gaussian_parameter_at_bound'),
                    dict(gaussian_fwhm_y_um=0.)]
        r = result([candidate(i + 1, **v) for i, v in enumerate(variants)])
        apply_spot_selection(r, SpotSizeLimits(.5))
        self.assertEqual(set(r.spot_selection.statuses), {'size_uncertain'})
        self.assertEqual(r.uncertain_counts, {1: 6, 2: 0, 3: 0})
        apply_spot_selection(r)
        self.assertEqual(r.nucleus_counts[1], 6)
        self.assertEqual(sum(r.uncertain_counts.values()), 0)

    def test_search_sampling_and_validation(self):
        widths = np.asarray(SpotSearchSettings(.2, 10.).widths())
        self.assertEqual(widths[0], .2)
        self.assertEqual(widths[-1], 10.)
        self.assertTrue(np.all(widths[1:] / widths[:-1] <= np.sqrt(2) + 1e-12))
        self.assertEqual(SpotSearchSettings(5., 5.).widths(), (5.,))
        for args in ((0, 2), (5, 1), (None, 2)):
            with self.assertRaises(ValueError):
                SpotSearchSettings(*args)
        with self.assertRaises(ValueError):
            SpotSizeLimits(5, 1)

    def test_physical_search_uses_each_axis_calibration_and_retains_display(self):
        image = np.zeros((9, 24, 24), dtype=np.float32)
        labels = np.ones(image.shape, dtype=np.int32)
        processed = np.full(image.shape, np.nan, dtype=np.float32)
        with patch('src.spot_detector.filter_volume', wraps=filter_volume) as filters:
            detected = detect_spots(image, labels > 0, labels,
                                    search_settings=SpotSearchSettings(1., 2.),
                                    voxel_size_um_zyx=(.4, .2, .1),
                                    processed_output=processed)
        sigmas = [c.kwargs['sigma'] for c in filters.call_args_list
                  if len(c.args) > 1 and c.args[1] == 'gaussian']
        from src.spot_selection import FWHM_FACTOR
        np.testing.assert_allclose(sigmas[0][1:], (1 / FWHM_FACTOR / .2, 1 / FWHM_FACTOR / .1))
        self.assertEqual(len(detected['scale_thresholds']), len(SpotSearchSettings(1., 2.).widths()))
        np.testing.assert_array_equal(processed, image)

    def test_exports_keep_rejected_records_and_accepted_counts(self):
        r = result([candidate(1), candidate(2, nucleus=2, width=6.),
                    candidate(3, gaussian_fit_r2=None)])
        apply_spot_selection(r, SpotSizeLimits(5., 8.))
        records = list(candidate_export_records(r, include_excluded=True))
        self.assertEqual([v['selection_status'] for v in records],
                         ['outside_range', 'accepted', 'size_uncertain'])
        with TemporaryDirectory() as folder:
            counts = pd.read_csv(save_counts_csv([r], folder))
            self.assertEqual(counts.spot_count.tolist(), [0, 1, 0])
            measurements = pd.read_csv(save_focus_measurements_csv([r], folder))
            self.assertEqual(measurements.focus_id.tolist(), [1])
            self.assertEqual(measurements.selection_status.tolist(), ['accepted'])
            self.assertNotIn('gaussian_fit_r2', measurements.columns)

    def test_volume_filtered_measurements_match_counts_and_number_within_nucleus(self):
        r = result([candidate(1, watershed_volume_um3=.19),
                    candidate(2, watershed_volume_um3=.2),
                    candidate(3, watershed_volume_um3=.201),
                    candidate(4, watershed_volume_um3=None)])
        apply_spot_selection(r, SpotSizeLimits(min_volume_um3=.2))
        with TemporaryDirectory() as folder:
            counts = pd.read_csv(save_counts_csv([r], folder))
            measurements = pd.read_csv(save_focus_measurements_csv([r], folder))
            self.assertEqual(int(counts.spot_count.sum()), len(measurements))
            self.assertEqual(measurements.focus_id.tolist(), [1])
            self.assertEqual(r.spot_candidates[2].focus_id, 3)
            self.assertTrue((measurements.volume_um3 > .2).all())
        apply_spot_selection(r, SpotSizeLimits(min_volume_um3=1.))
        self.assertEqual(list(candidate_export_records(r)), [])

    def test_numbering_restarts_for_each_nucleus_after_filtering(self):
        r = result([candidate(1, nucleus=1, watershed_volume_um3=.1),
                    candidate(2, nucleus=1, watershed_volume_um3=1.),
                    candidate(3, nucleus=2, watershed_volume_um3=1.),
                    candidate(4, nucleus=2, watershed_volume_um3=1.),
                    candidate(5, nucleus=3, watershed_volume_um3=1.)])
        apply_spot_selection(r, SpotSizeLimits(min_volume_um3=.2))
        records = list(candidate_export_records(r))
        self.assertEqual([(row['nucleus_id'], row['focus_id']) for row in records],
                         [(1, 1), (2, 1), (2, 2), (3, 1)])
        self.assertEqual([c.focus_id for c in r.spot_candidates], [1, 2, 3, 4, 5])

    def test_compact_export_reports_watershed_independently_of_gaussian_fit(self):
        record = dict(source_image='sample.czi', focus_id=1, nucleus_id=2,
                      watershed_extent_x_um=1., watershed_extent_y_um=2.,
                      watershed_extent_z_um=3., watershed_volume_um3=4.,
                      gaussian_fit_status='failed', qc_flags='gaussian_not_converged')
        row = watershed_export_record(record, 'size_uncertain')
        self.assertEqual(len(row), 9)
        self.assertEqual((row['width_x_um'], row['width_y_um'], row['depth_z_um']), (1., 2., 3.))
        self.assertEqual(row['volume_um3'], 4.)
        self.assertEqual(row['watershed_status'], 'ok')
        record['qc_flags'] += ';watershed_touches_patch_boundary'
        self.assertEqual(watershed_export_record(record)['watershed_status'], 'review')
        record['watershed_volume_um3'] = None
        row = watershed_export_record(record)
        self.assertIsNone(row['volume_um3'])
        self.assertEqual(row['watershed_status'], 'failed')

    def test_out_of_search_range_is_reported_without_changing_selection(self):
        r = result([candidate(1, width=5.)])
        apply_spot_selection(r, SpotSizeLimits(5.))
        self.assertEqual(r.nucleus_counts[1], 1)
        self.assertIn('above the searched', ' '.join(selection_warnings(r)))
