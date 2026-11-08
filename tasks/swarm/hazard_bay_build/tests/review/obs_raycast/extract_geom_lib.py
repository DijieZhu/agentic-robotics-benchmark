"""The real bay's visible Cube and Cylinder gprims as the ray caster's shapes (extract_geom.py's extraction as a
function; check_observability.py --raycast). pxr only."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visibility as V  # noqa: E402


def records(usda):
    from pxr import Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(usda)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []

    def colour(prim):
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not mat:
            return None, None
        for c in Usd.PrimRange(mat.GetPrim()):
            if c.IsA(UsdShade.Shader):
                i = UsdShade.Shader(c).GetInput("diffuseColor")
                if i and i.Get() is not None:
                    return mat.GetPrim().GetName(), [float(v) for v in i.Get()]
        return mat.GetPrim().GetName(), None
    for p in st.Traverse():
        path = p.GetPath().pathString
        if "/Items/" in path:
            continue
        vis = UsdGeom.Imageable(p).ComputeVisibility() if p.IsA(UsdGeom.Imageable) else "inherited"
        M = xc.GetLocalToWorldTransform(p)
        R = [[M[i][j] for j in range(3)] for i in range(3)]
        T = [M[3][j] for j in range(3)]
        mat, rgb = colour(p)
        if p.IsA(UsdGeom.Cube):
            s = float(UsdGeom.Cube(p).GetSizeAttr().Get())
            out.append(dict(kind="box", path=path, center=T, axes=R, half=[s / 2] * 3, material=mat, rgb=rgb, visible=str(vis)))
        elif p.IsA(UsdGeom.Cylinder):
            c = UsdGeom.Cylinder(p)
            out.append(dict(kind="cyl", path=path, center=T, axes=R, radius=float(c.GetRadiusAttr().Get()), height=float(c.GetHeightAttr().Get()),
                            axis=str(c.GetAxisAttr().Get()), material=mat, rgb=rgb, visible=str(vis)))
    return out


def bay_shapes(usda):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(records(usda), f)
    try:
        return V.load_bay(f.name)
    finally:
        os.remove(f.name)
