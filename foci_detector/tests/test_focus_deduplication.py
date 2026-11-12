"""Deferred axial-duplicate coverage; these checks do not use Cellpose or files."""
import unittest
import numpy as np

from src.focus_deduplication import consolidate_z_duplicates


class FocusDeduplicationTests(unittest.TestCase):
    def candidate(self, z, x=15, nucleus=1, score=100.):
        return dict(z=z, y=15, x=x, radius=3., label_id=nucleus, score=score,
                    pass_order=0)

    def column(self):
        source = np.zeros((17, 31, 31), dtype=np.float32)
        source[2:15, 14:17, 14:19] = 100.
        return source, np.ones(source.shape, dtype=np.int32)

    def test_one_broad_focus_with_three_axial_peaks_counts_once(self):
        source, labels = self.column()
        candidates = [self.candidate(2, score=80), self.candidate(8, score=100),
                      self.candidate(14, score=90)]
        kept, diagnostics = consolidate_z_duplicates(candidates, source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['z'], 8)
        self.assertEqual(diagnostics['removed_count'], 2)
        self.assertEqual(len(diagnostics['groups'][0]['merged_centers_zyx']), 2)

    def test_distinct_axial_foci_with_a_deep_valley_remain_separate(self):
        zz, yy, xx = np.indices((17, 31, 31))
        source = sum(100 * np.exp(-.5 * (((zz-z)/1.)**2 +
                                        ((yy-15)/1.)**2 + ((xx-15)/1.)**2))
                     for z in (3, 12)).astype(np.float32)
        labels = np.ones(source.shape, dtype=np.int32)
        kept, diagnostics = consolidate_z_duplicates(
            [self.candidate(3), self.candidate(12)], source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 2)
        self.assertEqual(diagnostics['removed_count'], 0)

    def test_alignment_does_not_merge_across_nuclei_or_mask_gaps(self):
        source, labels = self.column()
        candidates = [self.candidate(2), self.candidate(14, nucleus=2)]
        kept, _ = consolidate_z_duplicates(candidates, source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 2)
        labels[7] = 0
        candidates[1]['label_id'] = 1
        kept, _ = consolidate_z_duplicates(candidates, source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 2)

    def test_complete_group_check_prevents_lateral_chain_merging(self):
        source, labels = self.column()
        candidates = [self.candidate(2, x=15, score=100),
                      self.candidate(8, x=16, score=90),
                      self.candidate(14, x=17, score=80)]
        kept, _ = consolidate_z_duplicates(candidates, source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 2)

    def test_lateral_and_same_slice_objects_are_not_axial_duplicates(self):
        source, labels = self.column()
        candidates = [self.candidate(2), self.candidate(14, x=21), self.candidate(2, x=16)]
        kept, _ = consolidate_z_duplicates(candidates, source, labels, (.2, .1, .1))
        self.assertEqual(len(kept), 3)
