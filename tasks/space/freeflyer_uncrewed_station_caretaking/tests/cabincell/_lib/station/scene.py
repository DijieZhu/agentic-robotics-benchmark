"""Scene text, look, lighting per circuit, headlight and near clip of the station scene v0.3.

* ``scene_text(scene, look_root)``: the physics scene (collision world and display boxes, no file assets) or, with a
  look root, the same scene with the modules' meshes and textures (``data/station/look_<tag>.xml``; meshdir and
  texturedir point at the look root, the unpacked look archive, see ``assets``).  The collision geoms are the same
  in both and are clustered in 1 m static cells (spacekit.scenes.cluster_static_geoms_xml).
* Lighting per circuit (one circuit = one module; states 'general' | 'night' | 'off' = failed circuit): the module's
  spot lights and the emission of its luminaire materials (the station library's v0.2 rule, neutral material names).
* The headlight (MuJoCo's camera-attached light, one per model): the night state and a failed circuit both
  change it; ``apply_headlight(model, state)`` sets it for the state of the module a camera is in (the renderer calls
  it before each frame).  Values [G]: the camera renderer owns the render recipe and may retune them.
* Near clip: robot cameras need about 1.5 cm (spacekit 0.4.0 ``cameras.set_near_clip``).
"""
from __future__ import annotations

from . import data

LIGHT_SCALE = {"general": 1.0, "night": 0.35, "off": 0.0}          # station library v0.2 [G]
LUM_EMISSION = {"luminaires": 0.9, "misc_details_lit": 0.6}          # material class -> emission at 'general'
SPOT_DIFFUSE = 0.55
HEADLIGHT = {                                                        # ambient, diffuse, specular [G]
    "general": dict(ambient=0.32, diffuse=0.18, specular=0.03),     # the station library's scene value
    "night": dict(ambient=0.32 * LIGHT_SCALE["night"], diffuse=0.0, specular=0.0),
    "off": dict(ambient=0.32 * 0.05, diffuse=0.0, specular=0.0),
}
NEAR_CLIP_M = 0.015
LIGHT_STATES = tuple(LIGHT_SCALE)


def scene_text(scene: str = "mission", look_root: str | None = None) -> str:
    """MJCF text of the scene; with ``look_root`` the look variant (meshdir / texturedir = look_root)."""
    if look_root is None:
        return data.scene_text(scene, look=False)
    root = str(look_root).rstrip("/")
    if '"' in root or "<" in root:
        raise ValueError("bad look root path")
    return data.scene_text(scene, look=True).replace('meshdir="LOOK_ROOT"', f'meshdir="{root}"').replace(
        'texturedir="LOOK_ROOT"', f'texturedir="{root}"')


def apply_lighting(spec, lighting: dict) -> dict:
    """Set the per-circuit lighting on an MjSpec (lights ``lum_<module>_<k>``, luminaire materials ``m_<class>_<module>``).
    Returns {module: state} as applied."""
    for st in lighting.values():
        if st not in LIGHT_SCALE:
            raise ValueError(f"lighting state {st!r} not in {LIGHT_STATES}")
    for lt in spec.lights:
        parts = (lt.name or "").split("_")
        if len(parts) >= 3 and parts[0] == "lum":
            st = lighting.get(parts[1], "general")
            sc = LIGHT_SCALE[st]
            lt.diffuse = [SPOT_DIFFUSE * sc, SPOT_DIFFUSE * sc, SPOT_DIFFUSE * sc * (0.97 if st != "night" else 0.80)]
            lt.active = sc > 0
    for mt in spec.materials:
        nm = mt.name or ""
        for cls, em in LUM_EMISSION.items():
            pre = f"m_{cls}_"
            if nm.startswith(pre) and nm[len(pre):] in lighting:
                mt.emission = em * LIGHT_SCALE[lighting[nm[len(pre):]]]
    return dict(lighting)


def headlight_for(state: str) -> dict:
    return dict(HEADLIGHT[state])


def apply_headlight(model, state: str) -> dict:
    """Set the model's headlight for a camera in a module whose circuit is in ``state``; returns the values set."""
    H = HEADLIGHT[state]
    model.vis.headlight.ambient[:] = [H["ambient"]] * 3
    model.vis.headlight.diffuse[:] = [H["diffuse"]] * 3
    model.vis.headlight.specular[:] = [H["specular"]] * 3
    return dict(H)


def apply_near_clip(model, near_m: float = NEAR_CLIP_M) -> float:
    """Near clipping distance of every camera of the model [m] (model-wide); returns the previous one."""
    from spacekit.cameras import set_near_clip
    return set_near_clip(model, near_m)


def module_of_point(feats: dict, p) -> str | None:
    """Neutral id of the module whose axis span contains p and whose axis is nearest (None outside every module)."""
    import numpy as np
    best, bd = None, 1e9
    p = np.asarray(p, float)
    for mid, M in feats["modules"].items():
        o, ax = np.asarray(M["axis_origin"], float), np.asarray(M["axis_dir"], float)
        e = M["end_walls_s"]
        s = float((p - o) @ ax)
        r = float(np.linalg.norm((p - o) - ax * s))
        if (e["start"] or 0) - 0.2 <= s <= (e["end"] or M["axis_s_range_used"][1]) + 0.2 and r < bd:
            best, bd = mid, r
    return best
