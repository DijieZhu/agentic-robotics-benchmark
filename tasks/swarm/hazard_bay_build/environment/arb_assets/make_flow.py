"""Writes the task's two NVIDIA Flow effects (pxr only): a fire and a steam plume, metres, Z up, each one USD file
whose default prim holds the Flow emitter, simulation, offscreen and render prims of one layer. The prim tree and the
attribute set are those of the Flow extension's own fire preset (copied as a template so every attribute has the type
the plugin expects); the values are this task's.

    python3 make_flow.py <preset Fire.usda> <out_dir>

A scene places one by referencing the file under an Xform (translation only: Flow follows the prim's position) and
may switch it with the emitter's `enabled` attribute. Rendering needs the `omni.flowusd` extension and the RTX
setting `/rtx/flow/enabled` (the plugin turns it on when it finds a FlowRender prim).
"""
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom

PRESET, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def build(name, values, layer):
    path = os.path.join(OUT, f"{name}.usda")
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    src = Sdf.Layer.FindOrOpen(PRESET)
    Sdf.CopySpec(src, Sdf.Path("/Fire"), st.GetRootLayer(), Sdf.Path(f"/{name}"))
    root = st.GetPrimAtPath(f"/{name}")
    st.SetDefaultPrim(root)
    # the preset's names -> ours
    for old, new in (("flowEmitterSphere", "emitter"), ("flowSimulate", "simulate"), ("flowOffscreen", "offscreen"), ("flowRender", "render")):
        Sdf.CopySpec(st.GetRootLayer(), Sdf.Path(f"/{name}/{old}"), st.GetRootLayer(), Sdf.Path(f"/{name}/{new}"))
        st.RemovePrim(f"/{name}/{old}")
    for p in Usd.PrimRange(root):
        a = p.GetAttribute("layer")
        if a and a.IsValid():
            a.Set(layer)
    for rel, vals in values.items():
        p = st.GetPrimAtPath(f"/{name}/{rel}")
        assert p.IsValid(), rel
        for k, v in vals.items():
            a = p.GetAttribute(k)
            assert a and a.IsValid(), (rel, k)
            t = a.GetTypeName()
            if t == Sdf.ValueTypeNames.Float3:
                v = Gf.Vec3f(*v)
            elif t == Sdf.ValueTypeNames.Float4Array:
                v = [Gf.Vec4f(*x) for x in v]
            a.Set(v)
    st.GetRootLayer().Save()
    print(name, "->", path)


# the fire: the preset in metres (it is authored in centimetres, Y up): lengths /100, gravity along -Z
FIRE = {
    "emitter": dict(position=(0.0, 0.0, 0.05), radius=0.22, radiusIsWorldSpace=True, velocity=(0.0, 0.0, 0.3), velocityIsWorldSpace=True,
                    fuel=0.8, temperature=2.0, coupleRateFuel=2.0, coupleRateTemperature=10.0, coupleRateVelocity=2.0),
    # (forceSimulate: without it Flow does not advance in a headless app, and the emitter is only painted; simulateWhenPaused
    # must stay off, or the step size is zero)
    "simulate": dict(densityCellSize=0.02, forceSimulate=True),
    "simulate/advection": dict(gravity=(0.0, 0.0, -1.0), buoyancyPerTemp=6.0, coolingRate=1.5),
    "simulate/summaryAllocate": dict(smokeThreshold=0.02, speedThreshold=0.01),
    # (fuel and heat that are not replenished die away within a couple of seconds: a fire put out goes out)
    "simulate/advection/fuel": dict(damping=0.0, fade=0.6),
    "simulate/advection/temperature": dict(damping=0.0, fade=0.4),
    "offscreen/shadow": dict(attenuation=4.5, lightDirection=(0.3, -0.5, 1.0)),
    "render/rayMarch": dict(attenuation=5.0),
}
# the steam: no combustion; a smoke emitter with a jet out of the flange, rising and drifting; a white colormap
STEAM = {
    "emitter": dict(position=(0.0, 0.0, 0.0), radius=0.14, radiusIsWorldSpace=True, velocity=(0.0, -1.2, 1.6), velocityIsWorldSpace=True,
                    fuel=0.0, temperature=0.8, smoke=1.5, coupleRateFuel=0.0, coupleRateTemperature=4.0, coupleRateVelocity=6.0,
                    coupleRateSmoke=8.0),
    "simulate": dict(densityCellSize=0.035, forceSimulate=True),
    "simulate/advection": dict(gravity=(0.0, 0.0, -1.0), combustionEnabled=False, buoyancyPerTemp=3.0, buoyancyPerSmoke=0.8, coolingRate=0.3),
    "simulate/advection/smoke": dict(damping=0.05, fade=0.3),  # (a leak that stops is gone within a few seconds)
    "simulate/advection/velocity": dict(damping=0.08, fade=0.6),
    "simulate/advection/temperature": dict(damping=0.2, fade=0.3),
    "simulate/vorticity": dict(forceScale=1.5),
    "simulate/summaryAllocate": dict(smokeThreshold=0.02, speedThreshold=0.01),
    # (the colormap's x is the temperature; the volume is drawn as absorption plus the colormap's emission, so cold
    # steam needs an emissive, nearly white colour at every x to read as steam rather than soot)
    "offscreen/colormap": dict(colorScale=1.0, rgbaPoints=[(5.0, 5.1, 5.4, 0.9), (5.2, 5.3, 5.6, 0.95), (5.4, 5.5, 5.8, 1.0),
                                                           (5.6, 5.7, 5.9, 1.0), (5.6, 5.7, 5.9, 1.0), (5.6, 5.7, 5.9, 1.0)],
                               xPoints=[0.0, 0.2, 0.5, 0.8, 0.9, 1.0]),
    # (the self-shadow pass darkens an unlit volume to its minIntensity: steam stays bright)
    "offscreen/shadow": dict(attenuation=1.0, lightDirection=(0.3, -0.5, 1.0), minIntensity=0.75),
    "render/rayMarch": dict(attenuation=7.0, colorScale=1.3),
}
build("FlowFire", FIRE, 0)
build("FlowSteam", STEAM, 1)
