"""Which of the kit's adapter corrections a reference-build model already contains (reference build v1.2).

From v1.2 on the reference build carries the four corrections the adapters used to apply (``gripper.rig_spec``'s finger
contact recipe, ``dock.rig_dock_spec``'s dock-face cups, ``dock.attach_dock_v1``'s contact-free dock display parts,
``cameras.RobotCameras``'s exclusion of the robot's own rigid structure from the depth rays) and says so in a custom
text element of every model file: name ``refbuild``, data "reference build v1.2; corrections: <tokens>".  The adapters
read it -- by file content, never by the file's path -- so that nothing is applied twice; models without it (v0, v1,
v1.1, the test placeholder) keep the adapters' edits.  An attach prefix on the text's name is allowed.
"""
from __future__ import annotations

import mujoco

__all__ = ["TEXT_NAME", "CORRECTIONS", "build_info", "has_correction"]

TEXT_NAME = "refbuild"
CORRECTIONS = ("gripper_pads", "dock_face_cups", "berth_display_only", "fwd_depth_window")


def _parse(data: str) -> dict:
    head, _, tail = data.partition("corrections:")
    version = head.strip().rstrip(";").replace("reference build", "").strip() or None
    return {"text": data, "version": version, "corrections": frozenset(t for t in tail.split() if t)}


def build_info(obj) -> dict:
    """{'text', 'version', 'corrections'} of a compiled ``MjModel`` or an ``MjSpec`` (every text named ``refbuild`` or
    ``<prefix>refbuild`` is read; their corrections are united); text None when the model has none."""
    found = []
    if isinstance(obj, mujoco.MjModel):
        for i in range(obj.ntext):
            name = mujoco.mj_id2name(obj, mujoco.mjtObj.mjOBJ_TEXT, i) or ""
            if name.endswith(TEXT_NAME):
                a, n = int(obj.text_adr[i]), int(obj.text_size[i])
                found.append(bytes(obj.text_data[a:a + n]).split(b"\0")[0].decode())
    else:
        for t in obj.texts:
            if (t.name or "").endswith(TEXT_NAME):
                found.append(t.data)
    if not found:
        return {"text": None, "version": None, "corrections": frozenset()}
    infos = [_parse(x) for x in found]
    out = infos[0]
    out["corrections"] = frozenset().union(*(i["corrections"] for i in infos))
    return out


def has_correction(obj, token: str) -> bool:
    """True if the model (MjModel or MjSpec) already contains the adapter correction ``token`` (see CORRECTIONS)."""
    return token in build_info(obj)["corrections"]
