"""Maintainers' perception fleet for the localisation check (round 4, spec section 5.1): the reference build's scene with
two more cameras on the vehicle, so that a fleet of the reference's design can see what the mission needs.

    /opt/venv/bin/python3 -I -s tests/review/perception_fleet.py <reference scene.usda> <out scene.usda>

Added to `/World/Fleet/UGV` (the reference's mast camera kept):
- an eye-in-hand `CAM-RGB-S` on the Gen3's last link (`end_effector_link`), looking along the tool axis (the link's +Z);
- a `CAM-RGB-W` on the mast, looking forward (the Jackal's +X) and 10 degrees down (the spec's 30 degrees would leave the
  dial out of its frame at the valve stand: the dial is 22 degrees above the mast there, the frame's top edge 11 degrees
  above at 30 degrees down; 10 degrees down keeps the dial, the lever's grip at its open stop (30 degrees up), Table B's
  parts from its stands (down to 39 degrees down) and the crate in frame: 51 degrees down to 31 up).
Each with the catalog's intrinsics and its catalog mass added to the body it rides on (the chassis' collision body,
as the reference build adds the mast camera's; the tool link's own mass). The vehicle then streams three cameras;
the flying robots keep their four cameras and LIDAR-3D. The variant must still pass scan.py and the fleet probe's
design checks (R4-R9). pxr only; the out scene keeps the reference's absolute library references.
"""

import json
import math
import os
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = json.load(open(os.path.join(os.path.dirname(HERE), "grader_config.json")))["catalog"]["sensors"]
UGV = "/World/Fleet/UGV"


def camera(st, path, model, pos, rot_cols):
    """A catalog camera at `pos` in its parent's frame; `rot_cols` the camera's X, Y, Z axes in the parent's frame."""
    c = CAT[model]
    cam = UsdGeom.Camera.Define(st, path)
    cam.CreateFocalLengthAttr(float(c["focal"]))
    cam.CreateHorizontalApertureAttr(float(c["hap"]))
    cam.CreateVerticalApertureAttr(float(c["vap"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    m = Gf.Matrix4d(1.0)
    r = np.array(rot_cols, dtype=float)  # (Gf is row-vector: row i is the camera's axis i in the parent's frame)
    m.SetRow3(0, Gf.Vec3d(*r[0]))
    m.SetRow3(1, Gf.Vec3d(*r[1]))
    m.SetRow3(2, Gf.Vec3d(*r[2]))
    m.SetTranslateOnly(Gf.Vec3d(*pos))
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    cam.GetPrim().CreateAttribute("arb:sensor_model", Sdf.ValueTypeNames.String).Set(model)
    return cam


def add_mass(prim, kg):
    ma = UsdPhysics.MassAPI.Apply(prim)
    old = ma.GetMassAttr().Get() or 0.0
    ma.CreateMassAttr(float(old) + float(kg))
    return float(old) + float(kg)


def main():
    src, out = sys.argv[1], sys.argv[2]
    st = Usd.Stage.Open(src)
    ee = next(p for p in Usd.PrimRange(st.GetPrimAtPath(UGV)) if p.GetName() == "end_effector_link")
    # eye-in-hand: the camera's -Z along the link's +Z (the tool axis), its +Y along the link's -Y; 4 cm off the axis
    camera(st, f"{ee.GetPath()}/EyeInHand", "CAM-RGB-S", (0.04, 0.0, -0.02), [(1, 0, 0), (0, -1, 0), (0, 0, -1)])
    m_ee = add_mass(ee, CAT["CAM-RGB-S"]["mass"])
    # the mast: forward (+X of the Jackal's base_link) and 10 degrees down; just below the reference's mast camera
    base = st.GetPrimAtPath(f"{UGV}/Base/base_link")
    ref_cam = st.GetPrimAtPath(f"{UGV}/Base/base_link/Camera")
    z = UsdGeom.Xformable(ref_cam).GetLocalTransformation().ExtractTranslation()[2] - 0.05
    a = math.radians(10.0)
    look = np.array([math.cos(a), 0.0, -math.sin(a)])
    up = np.array([math.sin(a), 0.0, math.cos(a)])
    right = np.cross(look, up)
    camera(st, f"{base.GetPath()}/MastWide", "CAM-RGB-W", (-0.13, 0.14, float(z)), [right, up, -look])
    body = st.GetPrimAtPath(f"{UGV}/Base/base_link/collisions/body")
    m_body = add_mass(body, CAT["CAM-RGB-W"]["mass"])
    st.GetRootLayer().Export(out)
    print(json.dumps(dict(out=out, eye_in_hand=f"{ee.GetPath()}/EyeInHand", mast_wide=f"{base.GetPath()}/MastWide", ee_mass=m_ee,
                          chassis_body_mass=m_body)))


if __name__ == "__main__":
    main()
