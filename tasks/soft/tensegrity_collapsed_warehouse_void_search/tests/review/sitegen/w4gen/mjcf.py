"""Site dict + the heaps' collision grids -> MuJoCo XML for W4.

The world of W3 (w3gen.mjcf, unchanged: floor, static boxes, ground patches, colour chips, the site's key light, the
search camera) without the heaps' box records, plus one inline height field per heap:
  * asset  <hfield name="field_<band><k>" nrow ncol size="rx ry size_z 0.02" elevation="..."/>, the library piece's
           5 mm grid turned, mirrored and scaled (cells 5 mm x the heap's scale); the elevation rows run from +y to -y
           (the grid's rows run +y, so they are written flipped); the values are the grid divided by its max (a
           ring of zero cells and max == size_z, so MuJoCo's min..max rescaling is the identity), written with six
           decimals (the field's size and position too: 1e-6 m), so no number of the file is a long digit run;
  * geom   <geom name="heap__<band><k>" type="hfield" pos="centre of the grid, z = -0.010"/>, group 0, the boxes'
           friction; the heap's box record in site.json carries the same name.
"""
import numpy as np

from w3gen import mjcf as _w3

HEAP_RGB = {"R": (0.50, 0.49, 0.47), "M": (0.55, 0.48, 0.38)}
_f = _w3._w2._f
_v = _w3._w2._v


def _el(v):
    t = "%.6f" % v
    t = t.rstrip("0").rstrip(".")
    return t if t not in ("", "-0") else "0"


def heap_fragments(site, grids):
    """(asset lines, geom lines) of the heaps, in the order of site['heaps']['heaps']."""
    assets, geoms = [], []
    bmu = site["physics"]["box_mu"]
    for h in site["heaps"]["heaps"]:
        g = grids[h["name"]]
        H = np.asarray(g["H"], np.float64)
        ny, nx = H.shape
        sz = float(g["size_z"])
        E = np.flipud(H / sz)
        rx, ry = (nx - 1) * g["grid"] / 2.0, (ny - 1) * g["grid"] / 2.0
        cx, cy = g["origin"][0] + rx, g["origin"][1] + ry
        fname = "field_" + h["name"].split("__", 1)[1]
        assets.append('<hfield name="%s" nrow="%d" ncol="%d" size="%s %s %s 0.02" elevation="%s"/>'
                      % (fname, ny, nx, _f(rx), _f(ry), _f(sz), " ".join(_el(q) for q in E.ravel())))
        geoms.append('<geom name="%s" type="hfield" hfield="%s" pos="%s %s %s" rgba="%s 1" friction="%s 0.005 0.0001" '
                     'group="0"/>' % (h["name"], fname, _f(cx), _f(cy), _f(float(g["base_z"])),
                                      _v(HEAP_RGB[h["cls"]]), _f(bmu)))
    return assets, geoms


def site_xml(site, grids, qc_markers=None, offscreen=(640, 480)):
    """Standalone MJCF of the world with its heaps; W3's text with the model named w4_seed<seed>."""
    s = dict(site)
    s["boxes"] = [b for b in site["boxes"] if b["kind"] != "heap"]
    xml = _w3.site_xml(s, qc_markers=qc_markers, offscreen=offscreen)
    old = '<mujoco model="w3_seed%d">' % site["seed"]
    if not xml.startswith(old):
        raise RuntimeError("unexpected head of W3's site XML")
    xml = '<mujoco model="w4_seed%d">' % site["seed"] + xml[len(old):]
    assets, geoms = heap_fragments(site, grids)
    if not assets:
        return xml
    if "  <asset>" in xml:
        raise RuntimeError("W3's site XML already has an asset block (gravel relief): not supported with heaps")
    xml = xml.replace("  <worldbody>\n", "  <asset>\n" + "".join("    %s\n" % a for a in assets) + "  </asset>\n"
                      "  <worldbody>\n", 1)
    tail = "  </worldbody>\n</mujoco>\n"
    if not xml.endswith(tail):
        raise RuntimeError("unexpected tail of W3's site XML")
    return xml[:-len(tail)] + "".join("    %s\n" % g for g in geoms) + tail
