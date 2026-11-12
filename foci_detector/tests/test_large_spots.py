import unittest
import numpy as np
from src.spot_detector import detect_spots


class LargeSpotTests(unittest.TestCase):
    def image(self, spots):
        shape = (25, 128, 128)
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        data = np.random.default_rng(8).uniform(0, 2, shape).astype(np.float32)
        for y, x, sigma in spots:
            data += 600*np.exp(-((zz-12)**2/(2*1.2**2)
                                + ((yy-y)**2+(xx-x)**2)/(2*sigma**2)))
        labels = np.ones(shape, dtype=np.uint8)
        return data, labels > 0, labels

    def test_large_spot_is_added_without_losing_small_spot(self):
        args = self.image([(35, 35, 2.3), (85, 85, 7.0)])
        old = detect_spots(*args, max_radius_px=6, large_spot_sigmas_xy=())
        new = detect_spots(*args)
        np.testing.assert_array_equal(old['blobs_filtered'][:, :3], [[12, 35, 35]])
        np.testing.assert_array_equal(new['blobs_filtered'][:, :3], [[12, 35, 35], [12, 85, 85]])
        np.testing.assert_array_equal(new['blobs_filtered'][0], old['blobs_filtered'][0])
        self.assertEqual(new['threshold'], old['threshold'])
        self.assertGreater(new['blobs_filtered'][1, 3], 6)
        self.assertLessEqual(new['blobs_filtered'][1, 3], 10)

    def test_same_focus_is_not_counted_twice(self):
        args = self.image([(64, 64, 3.2)])
        small = detect_spots(*args, large_spot_sigmas_xy=())
        large = detect_spots(*args, sigmas_xy=(3.4, 4.4, 5.5), large_spot_sigmas_xy=())
        self.assertEqual(small['nucleus1_count'], 1)
        self.assertEqual(large['nucleus1_count'], 1)
        result = detect_spots(*args)
        self.assertEqual(result['nucleus1_count'], 1)
        self.assertEqual(result['nucleus2_count'], 0)

    def test_old_radius_limit_disables_out_of_range_large_scales(self):
        args = self.image([(35, 35, 2.3), (85, 85, 7.0)])
        capped = detect_spots(*args, max_radius_px=6)
        old = detect_spots(*args, max_radius_px=6, large_spot_sigmas_xy=())
        np.testing.assert_array_equal(capped['blobs_filtered'], old['blobs_filtered'])
        self.assertNotIn('large', capped['scale_thresholds'])


if __name__ == '__main__':
    unittest.main()
