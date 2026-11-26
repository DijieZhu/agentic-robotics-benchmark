"""The post check against its registered rule (tests/review/PREREG.md): a post is made when the settled cable comes
within 10 mm of its surface, below its top, with the post on the stated side. Run in the task's image:
    /opt/venv/bin/python3 -I tests/review/test_post_check.py
A straight cable passes a post with its closest approach midway between two of its points (segments of 20 mm, the
cable's 40 segments at 0.80 m), where a check on the nearest point of the cable and one on its nearest vertex differ.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import harness_env as H  # noqa: E402

RADIUS = 0.004
SEG = 0.020


class Ep(object):
    def __init__(self, side):
        self.fixtures = [{"type": "post", "pos": np.array([0.0, 0.0]), "side": side}]
        self.radius = RADIUS


def straight_cable(gap, z=None, offset=SEG / 2):
    """Along +x, the post's surface `gap` from the cable's surface on the cable's left; a point every SEG, the
    closest approach `offset` from the nearest point."""
    y = -(H.POST_R + RADIUS + gap)
    x = -0.4 + offset + SEG * np.arange(41)
    zz = H.BOARD_T + RADIUS if z is None else z
    return np.stack([x, np.full_like(x, y), np.full_like(x, zz)], axis=1)


def made(side, P):
    return H.evaluate(Ep(side), P)["checks"][0]["ok"]


class PostCheck(unittest.TestCase):
    def test_within_10_mm_midway_between_points(self):
        for gap in (0.0, 0.005, 0.009, 0.0099):
            P = straight_cable(gap)
            vertex_gap = float(np.min(np.linalg.norm(P[:, :2], axis=1))) - H.POST_R - RADIUS
            self.assertTrue(made("left", P), gap)
            if gap >= 0.009:
                self.assertGreater(vertex_gap, H.POST_TOL, "the nearest vertex alone would have missed it")

    def test_beyond_10_mm(self):
        for gap in (0.0101, 0.012, 0.03):
            for offset in (0.0, SEG / 4, SEG / 2):
                self.assertFalse(made("left", straight_cable(gap, offset=offset)), (gap, offset))

    def test_wrong_side(self):
        for gap in (0.0, 0.009):
            self.assertFalse(made("right", straight_cable(gap)))

    def test_above_the_post(self):
        self.assertFalse(made("left", straight_cable(0.005, z=H.BOARD_T + H.POST_H + 0.001)))
        self.assertTrue(made("left", straight_cable(0.005, z=H.BOARD_T + H.POST_H - 0.001)))

    def test_on_a_vertex_unchanged(self):
        for gap in (0.0, 0.009, 0.0099):
            self.assertTrue(made("left", straight_cable(gap, offset=0.0)))

    def test_arc_length_at_the_nearest_point(self):
        P = straight_cable(0.005)
        r = H.evaluate(Ep("left"), P)
        self.assertEqual(r["routed"], 1)
        j, t, x, d = H._nearest_on_cable(P, np.array([0.0, 0.0]))
        self.assertAlmostEqual(float(x[0]), 0.0, places=9)
        self.assertAlmostEqual(t, 0.5, places=9)
        self.assertAlmostEqual(d, H.POST_R + RADIUS + 0.005, places=9)

    def test_wrapped_cable(self):
        """A cable wrapped half round the post, its surface 2 mm off, passes it on the post's left as it runs."""
        rr = H.POST_R + RADIUS + 0.002
        a = np.linspace(-math.pi / 2 - 1.2, -math.pi / 2 + 1.2, 41)
        P = np.stack([rr * np.cos(a), rr * np.sin(a), np.full_like(a, H.BOARD_T + RADIUS)], axis=1)
        # the cable runs from -x to +x below the post (y < 0): the post is on its left
        self.assertGreater(P[-1, 0], P[0, 0])
        self.assertTrue(made("left", P))
        self.assertFalse(made("right", P))


if __name__ == "__main__":
    unittest.main()
