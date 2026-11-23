"""Small deterministic fixtures for storey memory, backtracking and optical guides."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle_run import DIRECTIONS, FOCAL, RoomMemory, aperture_width, explore, guide_offset, reverse


class Routes(unittest.TestCase):
    def test_reverse_flies_into_same_well(self):
        a,b,d = (0,1,0),(0,1,1),(0,1)
        self.assertEqual(reverse((a,b,d)),(b,a,d))
        c = (1,1,0)
        self.assertEqual(reverse((a,c,(1,0))),(c,a,(-1,0)))

    def test_ports_are_taken_on_real_crossings(self):
        for flat in (False,True):
            mem = RoomMemory(flat)
            mem.remember((0,1,0),(0,2,0),10,18)
            self.assertEqual(mem.ports((0,2,0))[(0,-1,0)],18)
            self.assertEqual((0,-1,0) in mem.ports((0,2,1)),flat)
            self.assertNotIn((0,0,1),mem.ports((0,2,0)))

    def test_both_enter_upper_landing_only_xyz_checks_other_room(self):
        a,b,c,d = (0,0,0),(0,1,0),(0,1,1),(0,0,1)
        doors = {a:{(0,1):"open"}, b:{(0,-1):"open",(0,1):"landing"},
                 c:{(0,-1):"open",(0,1):"landing"},d:{(0,1):"open"}}
        class Pilot:
            def scan(self):
                return self.at,{v:doors[self.at].get(v,"wall") for v in DIRECTIONS}
            def look(self,*_):
                return None
        for flat in (False,True):
            p = Pilot()
            p.loaded,p.trace,p.at,p.obs = False,[],a,dict(done=False,frame_id=0)
            p.command = np.zeros(4)
            crossed = []
            def fly(_,origin,target,heading):
                self.assertEqual(origin,p.at)
                self.assertIn(target,(a,b,c,d))
                p.at = target
                p.obs["frame_id"] += 1
                crossed.append(target)
            with patch("oracle_run.fly",fly), patch("oracle_run.apron_color",return_value="wall"), \
                    patch("oracle_run.find_pod",side_effect=lambda cell,*_: np.zeros(2) if cell==d else None):
                _,_,target = explore(p,a,flat)
            self.assertIn(c,crossed)
            self.assertEqual(target is None,flat)
            self.assertEqual(crossed, [b,c,b,a] if flat else [b,c,d])

    def test_guide_span_measures_lateral_and_range(self):
        img = np.zeros((256,256,3),dtype=int)
        self.assertIsNone(guide_offset(img))
        for u in (48,194):
            img[:,u-4:u+5] = (218,165,37)
        np.testing.assert_allclose(guide_offset(img),
                                   [(48+194-255)*.70/146,.17+FOCAL*1.40/(.99277*146)-1.222])

    def test_loaded_route_prefers_known_crossings(self):
        m = RoomMemory()
        a,b,c,d = (0,2,1),(1,2,1),(1,2,0),(1,1,0)
        for i,(u,v,h) in enumerate(((a,b,(1,0)),(b,c,(0,1)),(c,d,(0,-1)))):
            m.remember(u,v,i,i+1,h)
        m.observed[a] = ((0,-1,0),)
        self.assertEqual([x[1] for x in m.onward(a,d)],[b,c,d])
        self.assertEqual(m.onward(d,d),[])
        self.assertEqual(m.onward(a,(0,1,0)),[])  # approach the observed, untried door here

    def test_reduced_bore_filters_known_edges_and_fresh_frontiers(self):
        m = RoomMemory()
        a,b,c,d = (0,3,1),(0,3,0),(1,3,1),(1,3,0)
        for x,y,h in ((a,b,(0,1)),(a,c,(1,0)),(c,d,(0,1)),(d,b,(-1,0))):
            m.remember(x,y,0,1,h)
        allowed = lambda at,s: not s[2] or at[0]!=0
        self.assertEqual([x[1] for x in m.onward(a,b)],[b])
        self.assertEqual([x[1] for x in m.onward(a,b,allowed)],[c,d,b])
        del m.ports(a)[(0,0,-1)]
        m.observed[a] = ((0,0,-1),)
        m.observed[c] = ((1,0,0),)
        self.assertEqual([x[1] for x in m.onward(a,(0,2,0),allowed)],[c])

    def test_inner_edges_not_frame_color_label(self):
        for span in (82,174):
            img = np.zeros((256,256,3),dtype=int)
            img[90:195,32:128-span//2] = img[90:195,128+span//2:224] = (182,87,46)
            self.assertAlmostEqual(aperture_width(img,1.75),span*(1.75-.145-.02)/FOCAL)
        with self.assertRaises(RuntimeError):
            aperture_width(np.zeros_like(img),1.75)


if __name__=="__main__":
    unittest.main()
