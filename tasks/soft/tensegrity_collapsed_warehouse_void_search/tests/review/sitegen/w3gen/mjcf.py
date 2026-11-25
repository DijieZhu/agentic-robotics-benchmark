"""Site dict -> MuJoCo XML for W3: W2's writer (w2gen.mjcf) unchanged, only the model's name says w3.

The world is the same kind of thing in both families (floor plane, static boxes, ground patches, colour chips, the
site's key light, the search camera), so the writer is reused as it is; nothing below changes a geometry number.
"""
from w2gen import mjcf as _w2

apply_site_lighting = _w2.apply_site_lighting
relief_heights = _w2.relief_heights
world_fragment = _w2.world_fragment
asset_fragment = _w2.asset_fragment


def site_xml(site, qc_markers=None, offscreen=(640, 480)):
    """Standalone MJCF of the world; W2's text with the model named w3_seed<seed>."""
    xml = _w2.site_xml(site, qc_markers=qc_markers, offscreen=offscreen)
    old = '<mujoco model="w2_seed%d">' % site["seed"]
    if not xml.startswith(old):
        raise RuntimeError("unexpected head of W2's site XML")
    return '<mujoco model="w3_seed%d">' % site["seed"] + xml[len(old):]
