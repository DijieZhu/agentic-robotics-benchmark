"""Continuous architectural finishes, cut from the licensed concrete material."""
from functools import lru_cache
from io import BytesIO
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from scene_assets import asset_root


@lru_cache(maxsize=2)
def sheet(upper):
    # Laid in one bond across the whole storey, not restarted at room centers.
    # Pigment and stone grain vary by slab, not by room or simulation site.
    rng = np.random.default_rng(2306 if upper else 811)
    n = 1536
    source = np.asarray(Image.open(asset_root()/"textures/concrete_floor_02.png").convert("L")
                        .resize((768, 768), Image.Resampling.LANCZOS), dtype=float)
    source = (source-source.mean()) / max(1., source.std())
    grout = (63, 66, 64) if upper else (62, 60, 55)
    pigment = np.array((102, 113, 113) if upper else (115, 108, 96))
    tile_w, tile_h = (88, 71) if upper else (127, 64)
    out = np.empty((n, n, 3), dtype=np.uint8)
    out[:] = grout
    for row, y in enumerate(range(-31, n, tile_h)):
        start = -48 - (row%2)*tile_w//2 if not upper else -27
        for x in range(start, n, tile_w):
            w, h = tile_w-3, tile_h-3
            i, j = rng.integers(0, 768-max(w, h), 2)
            grain = source[j:j+h, i:i+w]
            grain = (grain+np.fliplr(grain)*.22) * rng.uniform(13, 23)
            tile = np.clip(pigment + rng.uniform(-13, 13) + grain[:, :, None], 30, 158)
            a, b, c, d = max(0, x), min(n, x+w), max(0, y), min(n, y+h)
            if a < b and c < d:
                out[c:d, a:b] = tile[c-y:d-y, a-x:b-x]
    buf = BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def install(asset, decals):
    for name, upper in (("quarry", False), ("terrazzo", True)):
        path = name+".png"
        decals[path] = sheet(upper)
        ET.SubElement(asset, "texture", name=name, type="2d", file=path)
        ET.SubElement(asset, "material", name=name, texture=name, texuniform="true",
                      texrepeat=".073 .073", specular=".025", reflectance="0")
