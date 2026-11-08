"""Writes the four multirotor airframes of the task's own asset folder (pxr only).

    python3 make_airframes.py <out_dir>

Each airframe is one USD file: a rigid body `Body` (hull, arms, motors, propellers, landing gear) with its empty mass
authored, a payload mount frame `Body/Mount` under the hull and a cargo hook frame `Body/Hook` below it. The take-off
limits are in the catalog the agent gets (`catalog/airframes.md`), not in the files.

The committed files in `airframes/` are this script's output plus one variant set on each root prim, `look`, added by
the maintainers for renders (2026-11-07). `proxy`, the default, is exactly what this script writes. `real` sets the
purpose of the generated shapes to `guide`, so that renders skip them (they still collide; every mass and collider
stays), and loads `look/<airframe>_look.usda` under `Body/Look`: propellers, motor bells, arms, landing gear and a body
from the Unity Asset Store pack "Realistic Drones", fitted to these layouts, without physics. The `look/` folder comes
from the task's asset dataset (`environment/assets.sha256`). Rerunning this script drops the variant set.
"""
import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

AIRFRAMES = {  # wheelbase = motor-to-motor diagonal [m]; prop radius [m]; hull (x, y, z) [m]; landing gear: height below the
    # hull's underside, half-width between the skids, skid length [m] (tall and wide: these frames carry under-slung payloads)
    "arb_quad_450": dict(rotors=4, coax=False, wheelbase=0.45, prop_r=0.127, hull=(0.16, 0.16, 0.07), gear=(0.20, 0.16, 0.36),
                         mass=1.15, color=(0.20, 0.20, 0.22)),
    "arb_quad_650": dict(rotors=4, coax=False, wheelbase=0.65, prop_r=0.190, hull=(0.22, 0.22, 0.09), gear=(0.30, 0.22, 0.46),
                         mass=2.40, color=(0.16, 0.18, 0.24)),
    "arb_hex_800": dict(rotors=6, coax=False, wheelbase=0.80, prop_r=0.190, hull=(0.26, 0.26, 0.10), gear=(0.42, 0.30, 0.56),
                        mass=4.10, color=(0.24, 0.14, 0.12)),
    "arb_x8_1000": dict(rotors=4, coax=True, wheelbase=1.00, prop_r=0.230, hull=(0.32, 0.32, 0.12), gear=(0.46, 0.36, 0.66),
                        mass=6.90, color=(0.12, 0.20, 0.16)),
}


def material(st, path, rgb, rough=0.5, metal=0.2):
    m = UsdShade.Material.Define(st, path)
    s = UsdShade.Shader.Define(st, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


def box(st, path, center, size, mat, yaw=0.0, col=True):
    g = UsdGeom.Cube.Define(st, path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    g.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    g.AddScaleOp().Set(Gf.Vec3f(*size))
    if col:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def bar(st, path, p0, p1, thick, mat, col=True):
    """A square-section strut from p0 to p1."""
    d = Gf.Vec3d(*p1) - Gf.Vec3d(*p0)
    L = d.GetLength()
    g = UsdGeom.Cube.Define(st, path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set((Gf.Vec3d(*p0) + Gf.Vec3d(*p1)) / 2)
    g.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), d / L).GetQuat()))
    g.AddScaleOp().Set(Gf.Vec3f(thick, thick, L))
    if col:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def cyl(st, path, center, r, h, mat, col=True):
    g = UsdGeom.Cylinder.Define(st, path)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(h)
    g.CreateAxisAttr("Z")
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    if col:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def build(name, spec, out_dir):
    st = Usd.Stage.CreateNew(os.path.join(out_dir, f"{name}.usda"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, f"/{name}")
    st.SetDefaultPrim(root.GetPrim())
    looks = f"/{name}/Looks"
    m_hull = material(st, f"{looks}/hull", spec["color"], 0.45, 0.3)
    m_arm = material(st, f"{looks}/arm", (0.10, 0.10, 0.11), 0.4, 0.5)
    m_motor = material(st, f"{looks}/motor", (0.70, 0.70, 0.72), 0.3, 0.8)
    m_prop = material(st, f"{looks}/prop", (0.05, 0.05, 0.05), 0.4, 0.0)
    m_gear = material(st, f"{looks}/gear", (0.85, 0.45, 0.10), 0.5, 0.0)
    body = UsdGeom.Xform.Define(st, f"/{name}/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(spec["mass"])
    for k, v in (("arb:airframe_model", name), ("arb:rotors", spec["rotors"] * (2 if spec["coax"] else 1))):
        body.GetPrim().CreateAttribute(k, Sdf.ValueTypeNames.String if isinstance(v, str) else Sdf.ValueTypeNames.Int).Set(v)
    hx, hy, hz = spec["hull"]
    b = f"/{name}/Body"
    box(st, f"{b}/hull", (0, 0, 0), (hx, hy, hz), m_hull)
    box(st, f"{b}/hull_top", (0, 0, hz / 2 + 0.015), (hx * 0.7, hy * 0.5, 0.03), m_arm)
    R = spec["wheelbase"] / 2
    n = spec["rotors"]
    for k in range(n):
        ang = 360.0 * k / n + (45.0 if n == 4 else 0.0)
        ax, ay = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        box(st, f"{b}/arm_{k}", (ax * R / 2, ay * R / 2, 0.0), (R, 0.03, 0.025), m_arm, yaw=ang)
        rot = UsdGeom.Xform.Define(st, f"{b}/Rotor_{k}")
        rot.AddTranslateOp().Set(Gf.Vec3d(ax * R, ay * R, 0.0))
        rot.GetPrim().CreateAttribute("arb:spin", Sdf.ValueTypeNames.Int).Set(1 if k % 2 == 0 else -1)
        cyl(st, f"{b}/Rotor_{k}/motor", (0, 0, 0.02), 0.022 + 0.01 * (R > 0.3), 0.04, m_motor)
        cyl(st, f"{b}/Rotor_{k}/prop", (0, 0, 0.045), spec["prop_r"], 0.005, m_prop, col=False)
        if spec["coax"]:
            cyl(st, f"{b}/Rotor_{k}/motor_lower", (0, 0, -0.03), 0.032, 0.04, m_motor)
            cyl(st, f"{b}/Rotor_{k}/prop_lower", (0, 0, -0.055), spec["prop_r"], 0.005, m_prop, col=False)
    g, gw, gl = spec["gear"]
    for k, side in enumerate((-1, 1)):  # two skids on splayed struts
        box(st, f"{b}/gear_skid_{k}", (0, side * gw, -hz / 2 - g), (gl, 0.018, 0.018), m_gear)
        for e, end in enumerate((-1, 1)):
            bar(st, f"{b}/gear_strut_{k}{e}", (end * hx * 0.4, side * hy * 0.45, -hz / 2), (end * gl * 0.4, side * gw, -hz / 2 - g + 0.01),
                0.016, m_gear)
        bar(st, f"{b}/gear_brace_{k}", (-hx * 0.4, side * hy * 0.45, -hz / 2), (gl * 0.4, side * gw, -hz / 2 - g + 0.01), 0.010, m_gear, col=False)
    mount = UsdGeom.Xform.Define(st, f"{b}/Mount")  # payload mount frame: its +z points down into the payload
    mount.AddTranslateOp().Set(Gf.Vec3d(0, 0, -hz / 2))
    hook = UsdGeom.Xform.Define(st, f"{b}/Hook")  # cargo hook at the hull's rear underside: the tether attaches here (the payload mount at the centre stays free)
    hook.AddTranslateOp().Set(Gf.Vec3d(0, -0.4 * hy, -hz / 2 - 0.03))
    cyl(st, f"{b}/Hook/ring", (0, 0, 0), 0.012, 0.004, m_motor, col=False)
    st.GetRootLayer().Save()
    return dict(file=f"{name}.usda", mass=spec["mass"], rotors=spec["rotors"] * (2 if spec["coax"] else 1),
                wheelbase=spec["wheelbase"], hull=spec["hull"], gear=spec["gear"][0], hook_z=-hz / 2 - 0.03)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, spec in AIRFRAMES.items():
        print(name, build(name, spec, out))
