"""Site dict -> MuJoCo XML (world only, no robot).

* Every solid is a static box under worldbody; its kind (wall, rubble, support, roof, crate, plug) is the name
  prefix "<kind>__" (MJCF has no free attribute for it; site.json carries the same kind per box).
* Visual-only things (colour chips, QC markers) have contype = conaffinity = 0 and their own geom group
  (chips 2, QC markers 5); collision boxes and ground patches are group 0.
* Ground patches: 2 mm boxes with priority 2 and their own friction; the floor plane has priority 1 and the
  site's floor friction, so a contact with the robot takes the patch's / floor's value.
* Gravel relief: an optional height field (bumps only, >= 0) replacing the gravel patch box; off by default.
* The search camera uses resolution + sensorsize + focalpixel + principalpixel (the given K).
* Orientations are quaternions only, so the fragment means the same under compiler angle="radian" (ysim) and
  angle="degree".
"""
import math

import numpy as np

from . import family as fam

QC_GROUP = 5
CHIP_GROUP = 2


def _f(x):
    return ("%.6f" % x).rstrip("0").rstrip(".") if abs(x) >= 1e-9 else "0"


def _v(seq):
    return " ".join(_f(float(q)) for q in seq)


def relief_heights(site):
    """Bumps-only gravel relief: band-limited noise, wavelength 0.03-0.08 m, rms 8 mm, peak <= 25 mm,
    0.05 m taper to zero at the edges; own random stream default_rng([seed, 1])."""
    p = [q for q in site["patches"] if q["name"] == "gravel"][0]
    gr = site["gravel_relief"]
    step = gr["grid_m"]
    nx = max(4, int(round((p["x1"] - p["x0"]) / step)) + 1)
    ny = max(4, int(round((p["y1"] - p["y0"]) / step)) + 1)
    rng = np.random.default_rng([int(site["seed"]), 1])
    w = rng.normal(size=(ny, nx))
    fy = np.fft.fftfreq(ny, d=step)[:, None]
    fx = np.fft.fftfreq(nx, d=step)[None, :]
    f = np.hypot(fx, fy)
    band = (f >= 1.0 / gr["wavelength_m"][1]) & (f <= 1.0 / gr["wavelength_m"][0])
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * band))
    h = np.clip(n, 0.0, None)
    yy = np.arange(ny) * step
    xx = np.arange(nx) * step
    tx = np.clip(np.minimum(xx, xx[-1] - xx) / gr["edge_taper_m"], 0, 1)
    ty = np.clip(np.minimum(yy, yy[-1] - yy) / gr["edge_taper_m"], 0, 1)
    taper = ty[:, None] * tx[None, :]
    for _ in range(30):
        hh = np.minimum(h * taper, gr["peak_max_m"])
        rms = float(np.sqrt(np.mean(hh ** 2)))
        if rms <= 0:
            break
        h = h * (gr["rms_m"] / rms)
    hh = np.minimum(h * taper, gr["peak_max_m"])
    return p, hh


def world_fragment(site, include_floor=True, include_light=True, qc_markers=None, extra=""):
    """Worldbody children only (usable as ysim ModelConfig.world_xml)."""
    lx, ly = site["size_m"]
    L = []
    lt = site["lighting"]
    if include_light:
        k = lt["key_diffuse"]
        L.append('<light name="site_key" pos="%s" dir="%s" directional="true" castshadow="true" diffuse="%s" '
                 'specular="0.05 0.05 0.05"/>' % (_v([lx / 2, ly / 2, 6.0]), _v(lt["key_dir"]), _v([k, k * 0.98, k * 0.93])))
    mu = site["physics"]["floor_mu"]
    if include_floor:
        L.append('<geom name="floor" type="plane" size="%s" pos="%s" rgba="%s 1" friction="%s 0.005 0.0001" priority="1"/>'
                 % (_v([lx / 2 + 1.0, ly / 2 + 1.0, 0.05]), _v([lx / 2, ly / 2, 0]), _v(fam.FLOOR_RGB), _f(mu)))
    bmu = site["physics"]["box_mu"]
    for b in site["boxes"]:
        q = ' quat="%s"' % _v(b["quat"]) if b.get("quat") else ""
        L.append('<geom name="%s" type="box" pos="%s" size="%s"%s rgba="%s 1" friction="%s 0.005 0.0001" group="0"/>'
                 % (b["name"], _v(b["pos"]), _v(b["size"]), q, _v(b["rgb"]), _f(bmu)))
    relief = site["gravel_relief"]["enabled"]
    for p in site["patches"]:
        if relief and p["name"] == "gravel":
            L.append('<geom name="patch__gravel" type="hfield" hfield="gravel_relief" pos="%s" rgba="%s 1" '
                     'friction="%s 0.005 0.0001" priority="2" group="0"/>'
                     % (_v([(p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, 0.0]), _v(p["rgb"]), _f(p["mu"])))
            continue
        t = p["thickness"]
        L.append('<geom name="patch__%s" type="box" pos="%s" size="%s" rgba="%s 1" friction="%s 0.005 0.0001" '
                 'priority="2" group="0"/>'
                 % (p["name"], _v([(p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, t / 2]),
                    _v([(p["x1"] - p["x0"]) / 2, (p["y1"] - p["y0"]) / 2, t / 2]), _v(p["rgb"]), _f(p["mu"])))
    for c in site.get("chips", []):
        z = c["z_m"] + c["thickness_m"] / 2
        yaw = math.radians(c["yaw_deg"])
        quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if c["shape"] == "disc":
            geo = 'type="cylinder" size="%s"' % _v([c["size_m"] / 2, c["thickness_m"] / 2])
        else:
            geo = 'type="box" size="%s"' % _v([c["size_m"] / 2, c["size_m"] / 2, c["thickness_m"] / 2])
        L.append('<geom name="%s" %s pos="%s" quat="%s" rgba="%s 1" contype="0" conaffinity="0" group="%d"/>'
                 % (c["name"], geo, _v([c["xy"][0], c["xy"][1], z]), _v(quat), _v(c["rgb"]), CHIP_GROUP))
    cam = site["camera"]
    T = np.asarray(cam["true"]["T_world_camera"], float)
    R = T[:3, :3]
    xa, ya = R[:, 0], -R[:, 1]                      # MuJoCo camera: x right, y up, looks along -z
    cm = cam["model"]
    L.append('<camera name="search" pos="%s" xyaxes="%s" resolution="%d %d" sensorsize="%s" focalpixel="%s" '
             'principalpixel="%s"/>'
             % (_v(T[:3, 3]), " ".join("%.9f" % q for q in list(xa) + list(ya)), cm["width"], cm["height"],
                _v(cm["sensorsize_m"]), _v([cm["fx"], cm["fy"]]), _v(cm["principalpixel_mujoco"])))
    for k, (x, y, z) in enumerate(qc_markers or []):
        L.append('<geom name="qcmark__%d" type="cylinder" size="0.05 0.001" pos="%s" rgba="1 0 1 1" contype="0" '
                 'conaffinity="0" group="%d"/>' % (k, _v([x, y, z]), QC_GROUP))
    if extra:
        L.append(extra)
    return "\n".join("    " + s for s in L)


def asset_fragment(site):
    if not site["gravel_relief"]["enabled"]:
        return ""
    p, hh = relief_heights(site)
    top = float(max(hh.max(), 1e-4))
    rx, ry = (p["x1"] - p["x0"]) / 2, (p["y1"] - p["y0"]) / 2
    elev = " ".join("%.4f" % q for q in (hh / top).ravel())
    return ('    <hfield name="gravel_relief" nrow="%d" ncol="%d" size="%s" elevation="%s"/>'
            % (hh.shape[0], hh.shape[1], _v([rx, ry, top, 0.01]), elev))


def site_xml(site, qc_markers=None, offscreen=(640, 480)):
    """Standalone MJCF of the world (floor, boxes, patches, chips, light, search camera)."""
    lt = site["lighting"]
    a = lt["ambient"]
    head = ['<mujoco model="w2_seed%d">' % site["seed"],
            '  <visual>',
            '    <global offwidth="%d" offheight="%d"/>' % offscreen,
            '    <quality shadowsize="4096"/>',
            '    <headlight ambient="%s" diffuse="0 0 0" specular="0 0 0"/>' % _v([a, a, a]),
            '    <map znear="0.01" zfar="50"/>',
            '  </visual>']
    af = asset_fragment(site)
    if af:
        head += ['  <asset>', af, '  </asset>']
    body = ['  <worldbody>', world_fragment(site, qc_markers=qc_markers), '  </worldbody>', '</mujoco>']
    return "\n".join(head + body) + "\n"


def apply_site_lighting(m, site):
    """For a model merged into ysim (which brings its own key / fill lights and headlight): switch every light that
    is not the site's off and set the headlight to the site's ambient. Runtime model edit, no file is changed."""
    import mujoco
    for i in range(m.nlight):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, i) or ""
        if not nm.startswith("site_"):
            if hasattr(m, "light_active"):
                m.light_active[i] = 0
            m.light_diffuse[i] = 0
            m.light_specular[i] = 0
            m.light_ambient[i] = 0
    a = site["lighting"]["ambient"]
    m.vis.headlight.ambient[:] = [a, a, a]
    m.vis.headlight.diffuse[:] = [0, 0, 0]
    m.vis.headlight.specular[:] = [0, 0, 0]
