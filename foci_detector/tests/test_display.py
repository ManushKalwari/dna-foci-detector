"""Deferred checks for display-only preprocessing; no GUI is created."""
import unittest
import numpy as np

from src.display import prepare_display_channels, soft_intensity_curve
from src.memory import ImageWorkspace, using_workspace


class DisplayTests(unittest.TestCase):
    def test_highlight_curve_is_monotonic_and_does_not_clip_at_reference(self):
        plane = np.array([[0., 1., 10., 100., 1000.]], dtype=np.float32)
        original = plane.copy()
        mapped = soft_intensity_curve(plane, black=0., reference=10.)
        self.assertEqual(mapped[0, 0], 0.)
        self.assertTrue(np.all(np.diff(mapped[0]) > 0))
        self.assertTrue(np.all(mapped < 1.))
        self.assertLess(mapped[0, 2], .5)
        self.assertGreater(mapped[0, 3], mapped[0, 2])
        np.testing.assert_array_equal(plane, original)

    def test_display_channels_do_not_modify_analysis_inputs(self):
        rng = np.random.default_rng(7)
        nuclei = rng.integers(0, 300, (9, 48, 48), dtype=np.uint16)
        foci = np.zeros(nuclei.shape, dtype=np.uint16)
        foci[4, 24, 12] = 1000
        foci[4, 24, 36] = 2000
        original_nuclei, original_foci = nuclei.copy(), foci.copy()
        with using_workspace(ImageWorkspace()) as workspace:
            try:
                nuclear_display, foci_display = prepare_display_channels(nuclei, foci, (.4, .1, .1))
                for volume in (nuclear_display, foci_display):
                    self.assertEqual(volume.shape, nuclei.shape)
                    self.assertTrue(np.all(np.isfinite(volume)))
                    self.assertGreaterEqual(float(volume.min()), 0.)
                    self.assertLess(float(volume.max()), 1.)
                self.assertGreater(foci_display[4, 24, 36], foci_display[4, 24, 12])
                np.testing.assert_array_equal(nuclei, original_nuclei)
                np.testing.assert_array_equal(foci, original_foci)
            finally:
                workspace.close()

    def test_empty_signal_remains_dark_and_finite(self):
        source = np.zeros((3, 12, 12), dtype=np.float32)
        with using_workspace(ImageWorkspace()) as workspace:
            try:
                channels = prepare_display_channels(source, source, (.4, .1, .1))
                for channel in channels:
                    np.testing.assert_array_equal(channel, np.zeros_like(source))
            finally:
                workspace.close()
