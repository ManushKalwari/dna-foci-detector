import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.nuclei_segmenter import segment_nuclei_legacy as segment_nuclei, _validate_candidate, NucleusValidationParams
from src.utils import save_counts_csv


def boundary_image(cells, filled=()):
    yy, xx = np.mgrid[:160, :160]
    image = np.zeros((160, 160), dtype=np.float32) + 0.02
    for y, x, radius, brightness in cells:
        rr = np.hypot(yy-y, xx-x)
        image += brightness * np.exp(-0.5*((rr-radius)/2)**2)
    for y, x, radius, brightness in filled:
        image += brightness * (np.hypot(yy-y, xx-x) <= radius)
    image += np.random.default_rng(4).normal(0, 0.003, image.shape)
    return np.broadcast_to(image, (24, *image.shape)).copy()


class NucleusValidationTests(unittest.TestCase):
    def test_one_ring_does_not_create_second_nucleus(self):
        result = segment_nuclei(boundary_image([(80, 80, 32, 1)]))
        self.assertEqual(result['nucleus_count'], 1)
        self.assertEqual(set(np.unique(result['roi_labels_3d'])), {0, 1})
        self.assertFalse(result['radius_search_extended'])

    def test_small_nucleus_retries_size_search_without_relaxing_validation(self):
        yy, xx = np.mgrid[:160, :160]
        image = 0.02 + np.exp(-0.5*((np.hypot(yy-80, xx-80)-17)/1.0)**2)
        stack = np.broadcast_to(image.astype(np.float32), (24, 160, 160))
        result = segment_nuclei(stack)
        self.assertEqual(result['nucleus_count'], 1)
        self.assertTrue(result['radius_search_extended'])
        accepted = [c for c in result['candidate_diagnostics'] if c['accepted']]
        self.assertEqual(accepted[0]['search_stage'], 'smaller')
        self.assertLess(np.linalg.norm(result['centers'][0] - [80, 80]), 5)
        self.assertLess(result['radii'][0], int(0.18*160))

    def test_radius_fraction_bounds_are_validated(self):
        for minimum, maximum in ((0, .3), (.4, .3), (.1, .5)):
            with self.assertRaisesRegex(ValueError, 'Radius fractions'):
                NucleusValidationParams(min_radius_fraction=minimum, max_radius_fraction=maximum)

    def test_two_rings_allow_dimmer_second_nucleus(self):
        result = segment_nuclei(boundary_image([(48, 48, 30, 1), (112, 112, 32, 0.35)]))
        self.assertEqual(result['nucleus_count'], 2)
        self.assertEqual(set(np.unique(result['roi_labels_3d'])), {0, 1, 2})

    def test_filled_dot_is_rejected_beside_real_nucleus(self):
        result = segment_nuclei(boundary_image([(48, 48, 30, 0.6)], [(112, 112, 32, 1)]))
        self.assertEqual(result['nucleus_count'], 1)
        self.assertLess(np.linalg.norm(result['centers'][0] - [48, 48]), 8)
        self.assertTrue(any('filled_blob' in c['rejection_reasons']
                            for c in result['candidate_diagnostics']))

    def test_blank_and_only_filled_blob_have_no_valid_nucleus(self):
        for stack in (np.zeros((24, 160, 160), dtype=np.float32),
                      boundary_image([], [(80, 80, 32, 1)])):
            with self.assertRaisesRegex(RuntimeError, 'No valid nucleus'):
                segment_nuclei(stack)

    def test_partial_arc_and_elongated_boundary_are_rejected(self):
        yy, xx = np.mgrid[:160, :160]
        rr = np.hypot(yy-80, xx-80)
        angle = np.arctan2(yy-80, xx-80)
        arc = np.exp(-0.5*((rr-32)/2)**2) * (np.abs(angle) < np.pi/3)
        elliptical_r = np.sqrt(((yy-80)/1.4)**2 + ((xx-80)/0.7)**2)
        ellipse = np.exp(-0.5*((elliptical_r-32)/2)**2)
        # Give both artifacts a strong center response: geometric validation
        # must still reject them, independent of the peak prominence check.
        response = np.exp(-rr**2/32).astype(np.float32)
        for projection in (arc, ellipse):
            metrics = _validate_candidate(projection, response, 80, 80, 32,
                                          NucleusValidationParams())
            self.assertFalse(metrics['accepted'])
            self.assertTrue(any(reason in metrics['rejection_reasons']
                                for reason in ('incomplete_boundary', 'irregular_radius')))

    def test_csv_exports_existing_zero_spot_nucleus_only(self):
        with TemporaryDirectory() as folder:
            results = [SimpleNamespace(file_path=f'{folder}/one.czi', file_stem='one',
                                       nucleus_count=1, nucleus1_count=0, nucleus2_count=0),
                       SimpleNamespace(file_path=f'{folder}/two.czi', file_stem='two',
                                       nucleus_count=2, nucleus1_count=5, nucleus2_count=0)]
            frame = pd.read_csv(save_counts_csv(results))
            self.assertEqual(frame['filename_cellnumber'].tolist(),
                             ['one_nucleus1', 'two_nucleus1', 'two_nucleus2'])
            self.assertEqual(frame['spot_count'].tolist(), [0, 5, 0])


if __name__ == '__main__':
    unittest.main()
