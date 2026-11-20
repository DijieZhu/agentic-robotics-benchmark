"""Site description ``station_site/2`` (station part), its builder and the station part of the validity check.

Station library v0.3 (format station_site/2; v0.2 wrote station_site/1): one stowage unit per module at a unit site
(registered or moved); the scene read from the task's neutral
data (``data``, ``scene``) with an optional look root; the dock and the robot from paths the caller passes (the task's
vendored reference build); the soft item's loop from spacekit 0.4.0 (``soft_item(loop='edge_mid')``, byte-identical
to v0.2's insertion); per-circuit lighting with neutral material names; inlet ends over a wall box.

Station part of a site description (every field explicit; nothing is drawn at build time):
  schema 'station_site/2', site_id, scene ('mission' | 'practice'), seed (recorded only)
  cabin_air_units: {cau_<m>: {build: 'element' | 'pad', r_face, fan_stop: 'allowed' | 'inhibited', inhibit_reason,
                              flow_cfm_per_inlet (optional, default 40)}}
  inter_module:  {link_<hub>_<lab>: {flow_cfm}}
  blocker:       {inlet, family, size [lx, ly, lz], mass, n_strips, hinge_k, theta_deg, offset_m [a, b],
                  loop 'edge_mid' | 'corner' | null, strap, colour}  | null
  small_items:   [{id, inlet, size, mass, theta_deg, offset_m, colour}]
  stowage_units: {<module>: <unit site id of data/station/station_v03.json>}
  clutter:       [{site: <clutter site id>}]            (strapped transfer bags, static display + collision)
  lighting:      {<module>: 'general' | 'night' | 'off'}
  robot:         {present: bool, berth: 1 | 2}
The task's site package (the site format's station_site/2 with the register, fixture records and pre-grasp records) adds its own
fields next to these; the builder ignores fields it does not know.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from . import data
from . import inlets as IN
from . import scene as SC
from . import stowage as SW
from .util import fmt as _v
from .util import obb_corners, obb_overlap, quat_from_mat

SCHEMA = "station_site/2"
ITEM_GAP = 0.0002            # initial gap of an item's face to the inlet face [m] (seal 0.98 at t = 0)
COLOURS = dict(orange="0.90 0.55 0.20 1", teal="0.15 0.55 0.55 1", grey="0.55 0.56 0.58 1", blue="0.15 0.45 0.85 1",
               green="0.30 0.62 0.30 1", yellow="0.92 0.80 0.25 1", purple="0.50 0.35 0.65 1", beige="0.82 0.76 0.62 1")
TRANSFER_BAG = dict(size=(0.50, 0.43, 0.25), colour="0.86 0.84 0.76 1")
CORRIDOR_HALF = 0.45         # clutter and stowage units leave a centred square tube of this half-width free  [C]
POUCH_END_OFFSET = 0.25      # a pouch over an opening's end: centre this far along the long axis [m]


def mat_to_quat(R):
    return quat_from_mat(R)


inlet_by_id = data.inlet_by_id


def unit_of_inlet(feats, iid):
    return inlet_by_id(feats, iid)["cab_air_unit"]


def example_site(kind: str = "S", stowage_units: dict | None = None) -> dict:
    """A complete example site of the mission modules (kind 'S': element-type unit in L1, stops allowed; 'P':
    pad-type, stops inhibited)."""
    S = kind == "S"
    reg = data.unit_sites()["registered"]
    return dict(
        schema=SCHEMA, site_id=f"example_{kind}", scene="mission", seed=0,
        cabin_air_units=dict(
            cau_l2=dict(build="pad", r_face=64.0, fan_stop="allowed", inhibit_reason=None),
            cau_h2=dict(build="pad", r_face=64.0, fan_stop="allowed", inhibit_reason=None),
            cau_l1=dict(build="element" if S else "pad", r_face=396.6 if S else 72.0, fan_stop="allowed" if S else "inhibited",
                        inhibit_reason=None if S else "smoke detector check")),
        inter_module=dict(link_h2_l1=dict(flow_cfm=130.0), link_h2_l2=dict(flow_cfm=130.0)),
        blocker=dict(inlet="ri_l1_2", family="soft_sheet", size=[0.30, 0.28, 0.004], mass=0.10, n_strips=5, hinge_k=0.002,
                     theta_deg=0.0, offset_m=[0.03, 0.0], loop="edge_mid", strap=True, colour="orange"),
        small_items=[dict(id="pouch_1", inlet="ri_l2_2", size=[0.20, 0.12, 0.025], mass=0.10, theta_deg=0.0,
                          offset_m=[0.25, 0.0], colour="blue"),
                     dict(id="pouch_2", inlet="ri_h2_1", size=[0.18, 0.12, 0.02], mass=0.08, theta_deg=0.0,
                          offset_m=[-0.24, 0.0], colour="green")],
        stowage_units=dict(stowage_units or {m: reg[m] for m in ("l2", "h2", "l1")}),
        clutter=[dict(site="tb_l2_c3"), dict(site="tb_l1_c1")],
        lighting=dict(l2="general", h2="general", l1="night"),
        robot=dict(present=False, berth=1),
    )


# ----------------------------------------------------------------------------------------------- item poses
def item_frame_on_inlet(inl, theta_deg, offset, half_z, gap=ITEM_GAP, chain_along_b=True):
    """(centre, R) of an item lying flat on an inlet face: local z = the inlet normal, local x = the inlet's b axis
    turned by theta about the normal (theta 0: chain / long side across the opening's short axis), centre offset
    (along a, along b) from the face centre."""
    n, a, b = IN.frame_axes(inl["normal"], inl["t1"])
    th = math.radians(theta_deg)
    x0 = b if chain_along_b else a
    y0 = np.cross(n, x0)
    x = math.cos(th) * x0 + math.sin(th) * y0
    y = np.cross(n, x)
    c = IN.face_center(inl) + a * offset[0] + b * offset[1] + n * (half_z + gap)
    return c, np.column_stack([x, y, n])


def soft_sheet_xml(name, inl, B):
    """Blocker: 5-strip soft item with the carry strap on the middle strip and the handle loop at the middle of the end
    strip ('edge_mid') or near its corner ('corner')."""
    from spacekit import fixtures as FX

    lx, ly, lz = B["size"]
    n_s = int(B.get("n_strips", 5))
    sw = lx / n_s
    c, R = item_frame_on_inlet(inl, B.get("theta_deg", 0.0), B.get("offset_m", [0, 0]), lz / 2)
    root = c - R[:, 0] * sw * (n_s // 2)
    q = mat_to_quat(R)
    loop = B.get("loop", "edge_mid")
    xml, excl, half = FX.soft_item(name, root, q, n=n_s, size=(lx, ly, lz), mass=B.get("mass", 0.10),
                                   hinge_k=B.get("hinge_k", 0.002), centre_handle=bool(B.get("strap", True)),
                                   loop=loop, material="mat_item_soft")
    col = COLOURS.get(B.get("colour", "orange"), COLOURS["orange"])
    xml = xml.replace('material="mat_item_soft"', f'rgba="{col}"')
    bodies = [f"{name}{k}" for k in range(n_s)]
    return xml, excl, dict(bodies=bodies, half=list(half), centre=c.tolist(), R=R.tolist(), quat=list(q), root=root.tolist(),
                           sites=dict(strap=f"{name}_grasp_centre", loop=f"{name}_grasp_corner"))


def small_item_xml(name, inl, S):
    from spacekit import fixtures as FX

    sx, sy, sz = S["size"]
    c, R = item_frame_on_inlet(inl, S.get("theta_deg", 0.0), S.get("offset_m", [0, 0]), sz / 2, chain_along_b=False)
    q = mat_to_quat(R)
    xml, half = FX.small_item(name, c, q, size=(sx, sy, sz), mass=S.get("mass", 0.1))
    col = COLOURS.get(S.get("colour", "blue"), COLOURS["blue"])
    xml = xml.replace('material="mat_item_bag"', f'rgba="{col}"')
    return xml, dict(body=name, half=list(half), centre=c.tolist(), R=R.tolist(), site=f"{name}_grasp")


def transfer_bag_xml(name, site):
    """Strapped transfer bag on a rack face (static box + two straps)."""
    sz = TRANSFER_BAG["size"]
    n, a = np.asarray(site["normal"], float), np.asarray(site["t1"], float)
    b = np.cross(n, a)
    c = np.asarray(site["center"], float)
    xy = f'xyaxes="{_v(a)} {_v(b)}"'
    g = [f'<geom name="{name}" type="box" pos="{_v(c)}" size="{sz[0] / 2} {sz[1] / 2} {sz[2] / 2 - 0.002}" {xy} '
         f'rgba="{TRANSFER_BAG["colour"]}" contype="1" conaffinity="1" friction="0.6 0.01 0.001"/>']
    for k, s in enumerate((-0.15, 0.15)):
        g.append(f'<geom name="{name}_strap{k}" type="box" pos="{_v(c + a * s + n * 0.001)}" size="0.012 {sz[1] / 2 + 0.003} {sz[2] / 2}" '
                 f'{xy} rgba="0.20 0.25 0.32 1" contype="0" conaffinity="0"/>')
    return "\n".join(g), dict(geom=name, center=c.tolist(), size=list(sz), normal=n.tolist(), t1=a.tolist())


def unit_part(module: str, site_id: str, pad_fid_side: str = SW.PAD_FID_DEFAULT) -> SW.UnitPart:
    """The stowage unit of a module at a unit site (name 'u_<module>')."""
    return SW.unit_from_site("u_" + module, data.unit_site(site_id), pad_fid_side=pad_fid_side)


# ----------------------------------------------------------------------------------------------- builder
def build_site_spec(site: dict, look_root: str | None = None, robot_xml: str | None = None, dock_xml: str | None = None,
                    pad_fid_side: str = SW.PAD_FID_DEFAULT, extra: dict | None = None):
    """Compose the scene of one site.  look_root: the unpacked look (None = physics scene only); robot_xml / dock_xml:
    the reference build's robot and dock models (the dock is left out when None; a site with a robot needs both);
    pad_fid_side: the pouch pads' plate side (``stowage.pad_fid_side_of(site_spec)``); extra: dict(xml, excl, items=[
    dict(id, bodies, half, grid)]) of extra items attached with prefix 'fr_'.  Returns (MjSpec, info)."""
    import mujoco
    from spacekit import dock as DK
    from spacekit import fixtures as FX
    from spacekit import gripper as GR

    if site.get("slow_drift"):
        raise ValueError("station_site/2 has no slow-drift item")
    scene = site.get("scene", "mission")
    feats = data.features(scene)
    spec = mujoco.MjSpec.from_string(SC.scene_text(scene, look_root))
    info = dict(site_id=site.get("site_id"), scene=scene, look_root=look_root, edits=[])
    o = spec.option
    o.density, o.viscosity = 0.0, 0.0
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.impratio = 10.0
    o.timestep = 0.002
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # ---- inlet media per cabin air unit
    units = site["cabin_air_units"]
    world_media = []
    for inl in feats["v02"]["inlets"]:
        want = units[inl["cab_air_unit"]]["build"]
        for g in list(spec.geoms):
            if g.name.startswith(inl["id"] + "_media_"):
                spec.delete(g)
        world_media += IN.media_geoms(inl, want)
    info["inlet_builds"] = {i["id"]: units[i["cab_air_unit"]]["build"] for i in feats["v02"]["inlets"]}
    # ---- dock + robot
    dk = feats["dock"]
    info["dock_pose"] = (list(dk["pos"]), list(dk["quat_wxyz"]))
    info["dock"] = None
    if dock_xml is not None:
        # the task's dock model ships its collision meshes and its targets only (data/robot): visual=False
        dinfo = DK.attach_dock_v1(spec, dk["pos"], dk["quat_wxyz"], xml=dock_xml, visual=False)
        info["dock"] = {k: v for k, v in dinfo.items() if k in ("silenced_geoms",)}
    R = site.get("robot", {})
    info["robot"] = None
    if R.get("present", False):
        if robot_xml is None or dock_xml is None:
            raise ValueError("a site with the robot needs robot_xml and dock_xml")
        berth = DK.berth_contact_pose(dk["pos"], dk["quat_wxyz"], int(R.get("berth", 1)))
        ff = mujoco.MjSpec.from_file(robot_xml)
        rd = DK.rig_dock_spec(ff)
        frame = spec.worldbody.add_frame(pos=list(berth[0]), quat=list(berth[1]))
        spec.attach(ff, frame=frame, prefix="")
        rig = GR.rig_spec(spec)
        info["robot"] = dict(berth=int(R.get("berth", 1)), berth_pose=(np.asarray(berth[0]).tolist(), np.asarray(berth[1]).tolist()),
                             rig_removed=rig.get("removed"), dock_face_boxes=rd.get("added"))
    # ---- stowage units (prefix 'su_')
    info["units"] = []
    uparts = []
    for mod, sid in sorted((site.get("stowage_units") or {}).items()):
        U = unit_part(mod, sid, pad_fid_side)
        rec = data.unit_site(sid)
        uparts.append(U.xml())
        info["units"].append(dict(module=mod, site_id=sid, kind=rec["kind"], name=U.name, part=U, pad_fid_side=pad_fid_side))
    if uparts:
        fr = mujoco.MjSpec.from_string(FX.fragment_model("\n".join(uparts), "", "stowage_units"))
        spec.attach(fr, frame=spec.worldbody.add_frame(), prefix="su_")
    # ---- fixtures (prefix 'fx_')
    parts, excl = list(world_media), []
    B = site.get("blocker")
    info["blocker"] = None
    if B:
        x, e, bi = soft_sheet_xml("blk", inlet_by_id(feats, B["inlet"]), B)
        parts.append(x)
        excl.append(e)
        info["blocker"] = dict(bi, inlet=B["inlet"], bodies=["fx_" + b for b in bi["bodies"]],
                               sites={k: "fx_" + v for k, v in bi["sites"].items()})
    info["small_items"] = []
    for S in site.get("small_items", []):
        x, si = small_item_xml(S["id"], inlet_by_id(feats, S["inlet"]), S)
        parts.append(x)
        info["small_items"].append(dict(si, id=S["id"], inlet=S["inlet"], body="fx_" + si["body"], site="fx_" + si["site"]))
    cs = {c["id"]: c for c in feats["v02"]["clutter_sites"]}
    info["clutter"] = []
    for C in site.get("clutter", []):
        x, ci = transfer_bag_xml(C["site"], cs[C["site"]])
        parts.append(x)
        info["clutter"].append(dict(ci, site=C["site"], geom="fx_" + ci["geom"]))
    fr = mujoco.MjSpec.from_string(FX.fragment_model("\n".join(parts), "\n".join(excl), "site_fixtures"))
    spec.attach(fr, frame=spec.worldbody.add_frame(), prefix="fx_")
    info["free_items"] = []
    if extra:
        fr2 = mujoco.MjSpec.from_string(FX.fragment_model(extra["xml"], extra.get("excl", ""), "free_items"))
        spec.attach(fr2, frame=spec.worldbody.add_frame(), prefix="fr_")
        for it in extra.get("items", []):
            info["free_items"].append(dict(it, bodies=["fr_" + b for b in it["bodies"]]))
    for gg in spec.geoms:
        par = gg.parent.name if gg.parent is not None else ""
        if gg.name.startswith(("fx_", "su_", "fr_")) or par.startswith(("fx_", "su_", "fr_")):
            gg.group = 2
    # ---- lighting per circuit (= module)
    mods = list(feats["modules"])
    light = {m: (site.get("lighting") or {}).get(m, "general") for m in mods}
    info["lighting"] = SC.apply_lighting(spec, light)
    info["cabin_air_units"] = copy.deepcopy(units)
    info["inter_module"] = copy.deepcopy(site.get("inter_module", {}))
    info["site"] = copy.deepcopy(site)
    return spec, info


def build_site(site: dict, look_root: str | None = None, robot_xml: str | None = None, dock_xml: str | None = None,
               pad_fid_side: str = SW.PAD_FID_DEFAULT, extra: dict | None = None, near_clip: bool = True):
    """Compile the site; returns (MjModel, info).  The renderer's near clip is set to 1.5 cm."""
    import warnings
    spec, info = build_site_spec(site, look_root, robot_xml, dock_xml, pad_fid_side, extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec.compile()
    if near_clip:
        SC.apply_near_clip(model)
    return model, info


def initial_data(model, info):
    """MjData at t = 0: arm stowed and fingers at the touch opening (robot present)."""
    import mujoco
    from spacekit import gripper as GR

    d = mujoco.MjData(model)
    if info.get("robot"):
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        GR.set_fingers(model, d, [jid(n) for n in GR.ARM_NAMES["fingers"]], open_=False)
        GR.set_arm(model, d, [jid(n) for n in GR.ARM_NAMES["joints"]], 180.0)
    mujoco.mj_forward(model, d)
    return d


# ----------------------------------------------------------------------------------------------- checks
def _root(model, b):
    return int(model.body_rootid[b])


def module_at(feats, p):
    return SC.module_of_point(feats, p)


def pouch_end_of(inl: dict, S: dict) -> str | None:
    """'+a' / '-a' when a small item lies over an end of the opening (|offset along a| >= 0.15 m), else None."""
    off = float(S.get("offset_m", [0.0, 0.0])[0])
    if abs(off) < 0.15:
        return None
    return "+a" if off > 0 else "-a"


def check_site(site: dict, model, data_, info, tol_pen: float = 0.0005) -> dict:
    """Station part of the validity check of a built site.  Returns {ok, failures [..], details {..}}."""
    import mujoco
    from spacekit.zones import ZoneMap  # noqa: F401  (zone boxes are read directly)

    scene = site.get("scene", "mission")
    feats = data.features(scene)
    fails, det = [], {}
    # 1. initial interpenetration (between different root bodies)
    mujoco.mj_forward(model, data_)
    worst = []
    for i in range(data_.ncon):
        c = data_.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if _root(model, b1) == _root(model, b2):
            continue
        if c.dist < -tol_pen:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            worst.append((float(c.dist), n1, n2))
    det["deepest_penetration_m"] = float(-min([0.0] + [w[0] for w in worst]))
    det["penetrating_pairs"] = sorted(worst)[:10]
    if worst:
        fails.append("initial_interpenetration")
    # 2. items flat on their inlets
    flat, items = [], []
    if info.get("blocker"):
        items += [(b, info["blocker"]["inlet"], info["blocker"]["half"][2]) for b in info["blocker"]["bodies"]]
    items += [(s["body"], s["inlet"], s["half"][2]) for s in info["small_items"]]
    for body, iid, hz in items:
        inl = inlet_by_id(feats, iid)
        n = np.asarray(inl["normal"], float)
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        Rb = data_.xmat[b].reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(Rb[:, 2] @ n)))))
        gap = float((data_.xpos[b] - Rb[:, 2] * hz - IN.face_center(inl)) @ n)
        flat.append(dict(body=body, inlet=iid, tilt_deg=tilt, gap_m=gap))
        if tilt > 3.0 or not (-0.0005 <= gap <= 0.002):
            fails.append(f"item_not_flat:{body}")
    det["items_flat"] = flat
    # 3. rules on units and items
    units = site["cabin_air_units"]
    if sum(1 for u in units.values() if u["build"] == "element") > 1:
        fails.append("more_than_one_element_unit")
    for u in units.values():
        if u["build"] == "element" and u.get("fan_stop") == "inhibited":
            fails.append("element_unit_inhibited (no way to remove the blocker)")
    per_inlet = {}
    if site.get("blocker"):
        per_inlet[site["blocker"]["inlet"]] = 1
    for s in site.get("small_items", []):
        per_inlet[s["inlet"]] = per_inlet.get(s["inlet"], 0) + 1
    for iid, k in per_inlet.items():
        if k > 1:
            fails.append(f"two_items_on_inlet:{iid}")
    ends = data.v03().get("pouch_ends", {})
    for s in site.get("small_items", []):
        if units[unit_of_inlet(feats, s["inlet"])]["build"] == "element":
            fails.append(f"small_item_on_element_unit:{s['id']}")
        if site.get("blocker") and s["inlet"] == site["blocker"]["inlet"]:
            fails.append(f"small_item_on_blocked_inlet:{s['id']}")
        e = pouch_end_of(inlet_by_id(feats, s["inlet"]), s)
        if e is not None and ends.get(s["inlet"], {}).get(e) is False:
            fails.append(f"small_item_over_a_wall_box_end:{s['id']}:{s['inlet']}{e}")
    if site.get("slow_drift"):
        fails.append("slow_drift_removed_in_v0.3")
    # 4. stowage units and clutter
    zl = [z for z in feats.get("zones", []) if z.get("module")]
    keepouts = [z for z in zl if z["type"] == "keepout"]
    cs = {c["id"]: c for c in feats["v02"]["clutter_sites"]}
    vols = []
    mods_with_unit = set()
    for U in info.get("units", []):
        rec = data.unit_site(U["site_id"])
        if rec["module"] != U["module"]:
            fails.append(f"unit_site_of_another_module:{U['module']}:{U['site_id']}")
        mods_with_unit.add(U["module"])
        c, R, h = U["part"].obb(protrusion=SW.FRAME_DEPTH)
        vols.append((U["name"], c, R, h + np.array([0.01, 0.01, 0.0])))
    for m in feats["modules"]:
        if m not in mods_with_unit:
            fails.append(f"no_stowage_unit:{m}")
    for C in site.get("clutter", []):
        if C["site"] not in cs:
            fails.append(f"clutter_not_a_candidate:{C['site']}")
            continue
        c = cs[C["site"]]
        vols.append((C["site"], np.asarray(c["center"], float), np.column_stack([c["t1"], c["b"], c["normal"]]),
                     np.asarray(TRANSFER_BAG["size"], float) / 2))
    for (n1, c1, R1, h1), (n2, c2, R2, h2) in itertools.combinations(vols, 2):
        if obb_overlap(c1, R1, h1 + 0.02, c2, R2, h2 + 0.02):
            fails.append(f"sites_overlap:{n1}:{n2}")
    for name, c, R, h in vols:
        P = obb_corners(c, R, h)
        lo, hi = P.min(0), P.max(0)
        for z in keepouts:
            if np.all(hi > np.asarray(z["min"])) and np.all(lo < np.asarray(z["max"])):
                fails.append(f"in_keep_out:{name}:{z['id']}")
        mid = module_at(feats, c)
        if mid is not None:
            M = feats["modules"][mid]
            o = np.asarray(M["axis_origin"], float)
            u, w = np.asarray(M["lat_u"], float), np.asarray(M["lat_w"], float)
            rel = P - o
            if ((np.abs(rel @ u) < CORRIDOR_HALF) & (np.abs(rel @ w) < CORRIDOR_HALF)).any():
                fails.append(f"in_corridor:{name}")
    det["n_site_volumes"] = len(vols)
    return dict(ok=not fails, failures=fails, details=det)
