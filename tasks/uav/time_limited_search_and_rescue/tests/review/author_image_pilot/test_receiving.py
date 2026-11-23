"""The status reader should distinguish shapes, and abstain without evidence."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle_run import dispatch_arrow,station_view


class Receiving(unittest.TestCase):
    def test_arrow_follows_shape(self):
        im = Image.new("RGB",(256,256),"#202626")
        d = ImageDraw.Draw(im)
        d.line([(84,102),(166,102),(147,84)],fill="white",width=5)
        d.line([(166,102),(147,120)],fill="white",width=5)
        rgb = np.array(im,dtype=np.int16)
        self.assertEqual(dispatch_arrow(rgb),1)
        self.assertEqual(dispatch_arrow(rgb[:,::-1]),-1)

    def test_cross_and_abstention(self):
        im = Image.new("RGB",(256,256),"#494949")
        d = ImageDraw.Draw(im)
        self.assertEqual(station_view(np.array(im,dtype=np.int16)),"uncertain")
        d.line([(89,80),(162,153)],fill="white",width=5)
        self.assertEqual(station_view(np.array(im,dtype=np.int16)),"uncertain")
        d.line([(89,153),(162,80)],fill="white",width=5)
        self.assertEqual(station_view(np.array(im,dtype=np.int16)),"closed")


if __name__=="__main__":
    unittest.main()
