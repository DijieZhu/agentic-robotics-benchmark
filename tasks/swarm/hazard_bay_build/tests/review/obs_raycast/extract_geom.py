# Moved in from the round-4 observability study (arb-eval/pr31r4-sensors/obs_map/extract_geom.py, map-observability.md section 7) for
# check_observability.py --raycast: the pxr / numpy cross-check of the renderer's measure. Unchanged but this header.
"""Extract the real bay's visible geometry (Cube and Cylinder gprims with their world transforms and material colours)
from tests/hidden/bay_true.usda into a JSON file, for the observability study (map-observability.md).

    python -I -B extract_geom.py <bay_true.usda> <out.json>

pxr only; reads the layer as data. The YCB parts are references into /opt/assets (absent on this host): they are
skipped here and added from the days' poses and the parts' own boxes by visibility.py.
"""
import json
import sys

from pxr import Usd, UsdGeom, UsdShade

st = Usd.Stage.Open(sys.argv[1])
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
out = []


def colour(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return None, None
    for c in Usd.PrimRange(mat.GetPrim()):
        if c.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(c)
            i = sh.GetInput("diffuseColor")
            if i and i.Get() is not None:
                return mat.GetPrim().GetName(), [float(v) for v in i.Get()]
    return mat.GetPrim().GetName(), None


for p in st.Traverse():
    path = p.GetPath().pathString
    if "/Items/" in path:
        continue
    vis = UsdGeom.Imageable(p).ComputeVisibility() if p.IsA(UsdGeom.Imageable) else "inherited"
    M = xc.GetLocalToWorldTransform(p)
    # Gf matrices are row-vector: p_world = p_local * M
    R = [[M[i][j] for j in range(3)] for i in range(3)]  # rows = images of the local axes (scaled)
    T = [M[3][j] for j in range(3)]
    mat, rgb = colour(p)
    if p.IsA(UsdGeom.Cube):
        s = float(UsdGeom.Cube(p).GetSizeAttr().Get())
        out.append(dict(kind="box", path=path, center=T, axes=R, half=[s / 2] * 3, material=mat, rgb=rgb, visible=str(vis)))
    elif p.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(p)
        out.append(dict(kind="cyl", path=path, center=T, axes=R, radius=float(c.GetRadiusAttr().Get()),
                        height=float(c.GetHeightAttr().Get()), axis=str(c.GetAxisAttr().Get()), material=mat, rgb=rgb,
                        visible=str(vis)))
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(len(out), "gprims")
