"""Image-only fixtures for registration, occlusion and actual wall geometry."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architecture import BOOM, FOCAL, patches, translation, wall_face


class Architecture(unittest.TestCase):
    def test_motion_against_a_wrong_command_and_camera_fixed_payload(self):
        grain = ndimage.gaussian_filter(np.random.default_rng(34).normal(size=(256,256)), 1.1)
        a = np.repeat((100+50*grain)[:, :, None], 3, axis=2)
        b = ndimage.shift(a, (-14.4, 21.2, 0), order=1, mode="nearest")*.9+8
        a[90:180, 35:220] = b[90:180, 35:220] = [170, 80, 32]
        shift, count = translation(patches(a), patches(b), [28,-9], 13)
        np.testing.assert_allclose(shift, [21.2,-14.4], atol=.22)
        self.assertGreater(count, 5)

    def test_unobservable_blank(self):
        blank = np.full((256,256,3), 90, dtype=np.int16)
        self.assertIsNone(translation(patches(blank), patches(blank), [0,0], 10))
        self.assertEqual(wall_face(blank), (None,None))

    def test_range_and_offset_from_an_interrupted_wall(self):
        image = np.full((256,256,3), 95, dtype=np.int16)
        image[32:42,:66] = image[32:42,208:] = [102,66,42]
        depth, center = wall_face(image)
        scale = (1.3-.145)/FOCAL
        self.assertAlmostEqual(depth, BOOM+(128-41)*scale, places=6)
        self.assertAlmostEqual(center, ((65+208)/2-127.5)*scale, places=6)


if __name__ == "__main__":
    unittest.main()
