"""The search camera's plate backend `plate_filament`: the agent's image is the robot rendered per frame by MuJoCo's
physically based renderer into light computed offline for the site.

    register_backend("plate_filament", lambda **kw: PlateFilamentBackend(**kw))     (camera.py)
    be = PlateFilamentBackend(site_sha256=..., site_json_sha256=..., site_xml_sha256=..., plate_root=...)
    s = be.open(render_model, spec); rgb, depth = s.render(data); s.close()

The camera interface of camera.py: `open(model, spec)` with the cell's render model (the site's geometry and the
robot) and the search camera's CameraSpec; `render(data)` -> (rgb uint8 H x W x 3, depth float32 H x W metres along
the optical axis, 0 = nothing hit); the depth sensor model (camera.sensor_depth) stays the camera's. The episode's
site package is named by the backend's keyword arguments (camera.py's backend_kw): its sha256 picks the plate bundle
<plate_root>/<sha256>/ (plate_bundle.py); a missing, foreign or damaged bundle raises
PlateBundleError when the camera opens, so the episode cannot be built (a task error). There is no fall-back to the
classic image. `msaa_rgb` is accepted and ignored (the server passes it to every backend).

Per frame (steps R1-R9):
  R1  the render model's qpos copied by joint name to the robot-only models; forward kinematics;
  R2  the robot's depth: the classic renderer (MuJoCo's OpenGL, OSMesa in the task image), robot alone, 1x, single
      sample, the whole frame, the cell's own frustum code (ysim ClassicSession);
  R3  the window: 128 x 128 px around the robot's whole projection and its shadow on the floor plane; a robot that
      does not project into the frame -> the plate;
  R4  the robot's 4x mask in the window (classic, the same frustum cut to the window) tested against the 4x depth
      plate; no visible 4x sample (the robot entirely hidden, e.g. under a slab) -> the plate's bytes;
  R5  the robot layer (plate_robot.py) in the window at 4x: R_amb (nearest probe) and / or R_lit (the probe dimmed by
      G_amb / G_sun plus the sun with the robot's self-shadow); one pass when V_env is 0 or 1 on every visible sample;
  R6  robot radiance = (1 - V_env) G_amb R_amb + V_env G_sun R_lit, V_env from the bundle's 3D sun-visibility grid at
      each visible sample's surface point; then the fog curtain: C T_hi + alpha A_hi;
  R7  the shadow on the plate: darkening = phi (1 - V_r) + (1 - phi) sky_gain O (plate_shadow.py), E = P - (P - A) dark;
  R8  composite at 1x (coverage and premultiplied colour from the 4x samples): final = E (1 - alpha) + C, through the
      look's display transform; outside the window, and on window pixels with neither robot nor shadow, the plate's
      bytes as they are;
  R9  depth = the nearer of the depth plate and the robot's depth (0 = no hit); the camera applies sensor_depth.
With the look's probes.select = "light_class_then_nearest_xy" the image light
is chosen first by the robot's light class (sun when the mean V_env over its visible samples is >= 0.5, else shade)
among the probes of the same class (a probe's class = V_env at its own position >= 0.5), then by xy distance; no probe
of that class -> the nearest of all. "nearest_xy" (or no key) keeps the nearest probe.
One session per episode; at most one per process (Filament); the robot layer renders both passes once when it opens
(warm-up: measured, a layer that never rendered segfaults in its teardown next to the classic sessions); close()
releases the context, the bundle and the models, then trims the heap.
"""
import gc
import time
from dataclasses import replace

import numpy as np

from . import plate_bundle, plate_look, plate_robot, plate_shadow

BACKEND_VERSION = "0.2.0"          # sun colour from the look, probe by light class (both switched by the look)
PlateBundleError = plate_bundle.PlateBundleError


class PlateBackendError(RuntimeError):
    pass


def _mesh_dir():
    from .. import MESH_DIR
    return MESH_DIR


def spec_window(spec, ss, win):
    """CameraSpec of a window (u0, v0, w, h in ss-x pixels) of the camera on the ss-x film: fx, fy x ss, the
    principal point at ss cx + (ss - 1) / 2 - u0 (pixel centres at integer coordinates on both films)."""
    u0, v0, w, h = win
    return replace(spec, width=int(w), height=int(h), fx=spec.fx * ss, fy=spec.fy * ss,
                   cx=spec.cx * ss + (ss - 1) / 2.0 - u0, cy=spec.cy * ss + (ss - 1) / 2.0 - v0)


def compose_depth(depth_plate, depth_robot):
    """R9: the nearer of the two (0 = no hit in either)."""
    return np.where(depth_robot > 0, np.where(depth_plate > 0, np.minimum(depth_plate, depth_robot), depth_robot),
                    depth_plate).astype(np.float32)


class PlateFilamentBackend:
    name = "plate_filament"

    def __init__(self, msaa_rgb=None, site_sha256=None, site_json_sha256=None, site_xml_sha256=None, plate_root=None,
                 look_file=None):
        del msaa_rgb                                   # every backend gets it; the plate backend has no MSAA switch
        self.site_sha256 = site_sha256
        self.site_json_sha256 = site_json_sha256
        self.site_xml_sha256 = site_xml_sha256
        self.plate_root = plate_root
        self.look, self.look_sha256 = plate_look.load_look(look_file)
        self.bundle_meta_sha256 = None

    def version(self):
        import mujoco
        return "plate-filament/%s mujoco-%s look-%s bundle-%s" % (BACKEND_VERSION, mujoco.__version__,
                                                                   self.look_sha256[:12],
                                                                   (self.bundle_meta_sha256 or "none")[:12])

    def open(self, model, spec):
        return PlateFilamentSession(self, model, spec)


class PlateFilamentSession:
    def __init__(self, backend, model, spec):
        import mujoco
        from ysim import camera as yc
        t0 = time.perf_counter()
        self.mj = mujoco
        self.look = look = backend.look
        self.spec = spec
        self.B = plate_bundle.PlateBundle(backend.plate_root, backend.site_sha256, spec, look,
                                          backend.site_json_sha256, backend.site_xml_sha256)
        backend.bundle_meta_sha256 = self.B.meta_sha256
        self.t_bundle_s = time.perf_counter() - t0
        win = look["window"]
        self.ss = int(win["supersample"])
        self.win = int(win["size_1x"])
        self.margin = float(win["visible_margin_m"])
        if self.win > min(spec.width, spec.height):
            raise PlateBackendError("window %d px does not fit the %d x %d frame" % (self.win, spec.width, spec.height))
        sh = look["shadow"]
        if sh["vr"] != "exact_rays":
            raise PlateBackendError("shadow mode %r is not implemented" % sh["vr"])
        self.sh = sh
        self.hemi = plate_shadow.hemi_dirs(int(sh["sky_dirs"]))
        self.s1 = self.s4 = self.lay = None
        try:
            # R1 / R2 / R4: the robot alone, classic renderer, single sample (the cell's frustum code)
            mesh_dir = _mesh_dir()
            self.rm = mujoco.MjModel.from_xml_string(plate_robot.robot_xml(mesh_dir))
            self.rd = mujoco.MjData(self.rm)
            self.qmap = self._joint_map(model, self.rm)
            self.s1 = yc.ClassicSession(self.rm, spec.width, spec.height, (0, 1, 2), False, False)
            n4 = self.win * self.ss
            self.s4 = yc.ClassicSession(self.rm, n4, n4, (0, 1, 2), False, False)
            self.robot_geomgroup = np.array([0, 1, 0, 0, 0, 0], np.uint8)   # the robot's visual meshes are group 1
            self.body_ids = [mujoco.mj_name2id(self.rm, mujoco.mjtObj.mjOBJ_BODY, b) for b in plate_shadow.STRUT_BODIES]
            if min(self.body_ids) < 0:
                raise PlateBackendError("the robot model has no strut bodies %s" % (plate_shadow.STRUT_BODIES,))
            self.to_sun = -self.B.sun_dir
            self.probe_xy = np.asarray([p["xyz"][:2] for p in self.B.probes], float)
            # each probe's light class from the bundle's own sun-visibility grid at its position
            self.probe_select = str((look.get("probes") or {}).get("select") or "nearest_xy")
            if self.probe_select not in ("nearest_xy", "light_class_then_nearest_xy"):
                raise PlateBackendError("the look's probes.select %r is not known" % self.probe_select)
            self.probe_lit = plate_shadow.venv_lookup(self.B.venv, np.asarray([p["xyz"] for p in self.B.probes], float)) >= 0.5
            # R5: the robot layer on the 4x film
            cam4 = {"fx": spec.fx * self.ss, "fy": spec.fy * self.ss, "cx": spec.cx * self.ss + (self.ss - 1) / 2.0,
                    "cy": spec.cy * self.ss + (self.ss - 1) / 2.0, "R": np.asarray(spec.R, float),
                    "t": np.asarray(spec.position, float)}
            self.lay = plate_robot.RobotLayer(look, mesh_dir, self.B.sun_dir, self.B.sun_lux, self.B.probes, cam4,
                                              (n4, n4))
            self.fmap = self._joint_map(self.rm, self.lay.model)   # robot-only classic model -> the robot layer's
            self.qfil = np.zeros(self.lay.model.nq)
        except BaseException:
            self.close()
            raise
        self.rl = look["robot_layer"]
        self.g_amb, self.g_sun = float(self.rl["G_amb"]), float(self.rl["G_sun"])
        self.t_open_s = time.perf_counter() - t0
        self.last = None
        self.debug = False
        self.counts = {"frames": 0, "outside": 0, "hidden": 0, "passes": {"amb": 0, "lit": 0, "both": 0},
                       "shadow_clipped": 0}

    # ------------------------------------------------------------------ set-up helpers
    def _joint_map(self, src, dst):
        """[(src qpos adr, dst qpos adr, width)] for every joint of dst, by name; the types must agree."""
        mj = self.mj
        width = {0: 7, 1: 4, 2: 1, 3: 1}
        out = []
        for j in range(dst.njnt):
            nm = mj.mj_id2name(dst, mj.mjtObj.mjOBJ_JOINT, j)
            k = mj.mj_name2id(src, mj.mjtObj.mjOBJ_JOINT, nm) if nm else -1
            if k < 0 or int(src.jnt_type[k]) != int(dst.jnt_type[j]):
                raise PlateBackendError("the render model has no joint %r of the robot's type" % nm)
            out.append((int(src.jnt_qposadr[k]), int(dst.jnt_qposadr[j]), width[int(dst.jnt_type[j])]))
        return out

    @staticmethod
    def _copy(qmap, q_src, q_dst):
        for a, b, n in qmap:
            q_dst[b:b + n] = q_src[a:a + n]

    # ------------------------------------------------------------------ pieces of a frame
    def _depth1(self):
        """R2: the robot's depth over the whole frame (the depth arithmetic of the cell's ClassicSession)."""
        return self.s1._pass(self.s1.rd, self.rd, self.spec, False, True)[1]

    def _window(self, d1r):
        """R3: (u0, v0) of the 1x window around the robot's projection and its floor shadow; clipped flag."""
        ys, xs = np.nonzero(d1r > 0)
        u_lo, u_hi, v_lo, v_hi = xs.min(), xs.max(), ys.min(), ys.max()
        # the shadow on the floor plane z = 0 of the strut capsules' end points (lower surfaces are nearer)
        caps = plate_shadow.capsules(self.mj, self.rm, self.rd, self.body_ids, float(self.sh["capsule_half_length_m"]))
        pts = np.array([p for ab in caps for p in ab])
        k = self.B.sun_dir
        sh = pts + (np.maximum(pts[:, 2], 0.0) / max(-k[2], 1e-6))[:, None] * k
        u, v, z = self.spec.project(sh)
        ok = z > 0
        clipped = False
        if ok.any():
            su_lo, su_hi = min(u_lo, np.floor(u[ok].min())), max(u_hi, np.ceil(u[ok].max()))
            sv_lo, sv_hi = min(v_lo, np.floor(v[ok].min())), max(v_hi, np.ceil(v[ok].max()))
            if su_hi - su_lo + 1 <= self.win and sv_hi - sv_lo + 1 <= self.win:
                u_lo, u_hi, v_lo, v_hi = su_lo, su_hi, sv_lo, sv_hi
            else:
                clipped = True
        cu, cv = 0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi)
        u0 = int(max(0, min(self.spec.width - self.win, round(cu - self.win / 2))))
        v0 = int(max(0, min(self.spec.height - self.win, round(cv - self.win / 2))))
        return u0, v0, clipped

    def _mask4(self, u0, v0):
        """R4: the robot's depth on the 4x film in the window (single sample, the same frustum cut to the window)."""
        n4 = self.win * self.ss
        sp = spec_window(self.spec, self.ss, (u0 * self.ss, v0 * self.ss, n4, n4))
        return self.s4._pass(self.s4.rd, self.rd, sp, False, True)[1], sp

    @staticmethod
    def _unproject(d, sp):
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        pc = np.stack([(u - sp.cx) / sp.fx * d, (v - sp.cy) / sp.fy * d, d], -1)
        return pc @ np.asarray(sp.R, float).T + np.asarray(sp.position, float)

    def _centre(self):
        return np.mean([self.rd.xpos[i] for i in self.body_ids], axis=0)

    # ------------------------------------------------------------------ the frame
    def render(self, data):
        mj = self.mj
        tm = {}
        t0 = time.perf_counter()
        self._copy(self.qmap, data.qpos, self.rd.qpos)
        mj.mj_kinematics(self.rm, self.rd)
        t1 = time.perf_counter()
        d1r = self._depth1()
        tm["depth1"] = time.perf_counter() - t1
        depth = compose_depth(self.B.depth1, d1r)
        rgb = self.B.plate_ldr.copy()
        info = {"window": None, "hidden": False, "passes": (), "probe": None, "rays": 0, "sky_points": 0,
                "changed_px": 0}
        self.counts["frames"] += 1
        if not (d1r > 0).any():
            self.counts["outside"] += 1
            info["outside"] = True
        else:
            self._frame_window(d1r, rgb, info, tm)
        tm["total"] = time.perf_counter() - t0
        info["t"] = tm
        self.last = info
        return rgb, depth

    def _frame_window(self, d1r, rgb, info, tm):
        ss, n = self.ss, self.win
        t = time.perf_counter()
        u0, v0, clipped = self._window(d1r)
        self.counts["shadow_clipped"] += int(clipped)
        info["window"], info["shadow_clipped"] = (u0, v0), clipped
        d4r, sp4 = self._mask4(u0, v0)
        tm["mask4"] = time.perf_counter() - t
        dp4 = self.B.depth4[v0 * ss:(v0 + n) * ss, u0 * ss:(u0 + n) * ss]
        vis4 = (d4r > 0) & ((dp4 <= 0) | (d4r < dp4 - self.margin))
        if not vis4.any():                              # R4: the robot is entirely hidden: the plate as it is
            self.counts["hidden"] += 1
            info["hidden"] = True
            return
        sl = (slice(v0, v0 + n), slice(u0, u0 + n))
        # R5 / R6: the robot's radiance on its visible 4x samples
        t = time.perf_counter()
        P4 = self._unproject(d4r, sp4)
        ve = plate_shadow.venv_lookup(self.B.venv, P4[vis4])
        lo, hi = float(ve.min()), float(ve.max())
        passes = ("amb",) if hi <= 1e-3 else (("lit",) if lo >= 1.0 - 1e-3 else ("amb", "lit"))
        self.counts["passes"]["both" if len(passes) == 2 else passes[0]] += 1
        cen = self._centre()
        dxy = np.linalg.norm(self.probe_xy - cen[:2], axis=1)
        if self.probe_select == "light_class_then_nearest_xy":
            lit = bool(float(ve.mean()) >= 0.5)
            same = np.nonzero(self.probe_lit == lit)[0]
            pi = int(same[np.argmin(dxy[same])]) if len(same) else int(np.argmin(dxy))
            info["robot_lit"] = lit
        else:
            pi = int(np.argmin(dxy))
        self._copy(self.fmap, self.rd.qpos, self.qfil)
        self.lay.set_qpos(self.qfil)
        self.lay.use_probe(pi)
        tm["venv"] = time.perf_counter() - t
        t = time.perf_counter()
        out = self.lay.render((u0 * ss, v0 * ss, n * ss, n * ss), passes)
        tm["robot_layer"] = time.perf_counter() - t
        t = time.perf_counter()
        Lr = np.zeros(d4r.shape + (3,), np.float32)
        v = ve.astype(np.float32)[:, None]
        if passes == ("amb",):
            Lr[vis4] = self.g_amb * out["amb"][vis4]
        elif passes == ("lit",):
            Lr[vis4] = self.g_sun * out["lit"][vis4]
        else:
            Lr[vis4] = (1.0 - v) * self.g_amb * out["amb"][vis4] + v * self.g_sun * out["lit"][vis4]
        C = plate_look.box_down(Lr, ss)
        alpha = plate_look.box_down(vis4, ss)
        C = C * self.B.T_hi[sl][..., None] + alpha[..., None] * self.B.A_hi[sl]
        tm["robot_comp"] = time.perf_counter() - t
        # R7: the shadow on the plate
        t = time.perf_counter()
        dark = self._darkening(u0, v0, dp4, cen, info)
        tm["shadow"] = time.perf_counter() - t
        # R8: composite, display transform, the plate's bytes where nothing changed
        t = time.perf_counter()
        P = self.B.plate_lin[sl]
        E = P - (P - self.B.A[sl]) * dark[..., None]
        final = E * (1.0 - alpha[..., None]) + C
        change = (alpha > 0) | (dark > 0)
        win_rgb = rgb[sl]
        win_rgb[change] = plate_look.display(final[change], self.look)
        tm["composite"] = time.perf_counter() - t
        info.update(passes=passes, probe=pi, changed_px=int(change.sum()), robot_px_1x=float(alpha.sum()),
                    venv=(round(lo, 4), round(hi, 4)))
        if self.debug:                                  # diagnostics only: the window's layers
            cover = np.zeros(d4r.shape, bool)
            lim = float(plate_look.srgb_decode_u8(np.uint8(3)))      # Filament dithers its black background by 1-2
            for kind in passes:                                     # levels: coverage = above 3 levels
                cover |= out[kind].max(axis=-1) > lim
            info["debug"] = {"alpha": alpha, "dark": dark, "vis4": vis4, "mask4": d4r > 0, "cover4": cover,
                             "change": change}

    def _darkening(self, u0, v0, dp4, cen, info):
        """R7: phi (1 - V_r) + (1 - phi) sky_gain O on the window's 1x pixels."""
        ss, n, sh = self.ss, self.win, self.sh
        sub = int(sh["subpoints_per_px"])
        step = ss // sub
        off = step // 2
        dsub = dp4[off::step, off::step]                 # 2 x 2 sub-points per pixel of the 4x depth plate
        spw = spec_window(self.spec, ss, (u0 * ss, v0 * ss, n * ss, n * ss))
        ys_, xs_ = np.mgrid[off:n * ss:step, off:n * ss:step]
        pc = np.stack([(xs_ - spw.cx) / spw.fx * dsub, (ys_ - spw.cy) / spw.fy * dsub, dsub], -1)
        Psub = (pc @ np.asarray(spw.R, float).T + np.asarray(spw.position, float)).reshape(-1, 3)
        caps = plate_shadow.capsules(self.mj, self.rm, self.rd, self.body_ids, float(sh["capsule_half_length_m"]))
        Vr, nray = plate_shadow.vr_exact(self.mj, self.rm, self.rd, Psub, (dsub > 0).ravel(), self.to_sun, caps,
                                         float(sh["candidate_radius_m"]), self.robot_geomgroup)
        Vr1 = plate_look.box_down(Vr.reshape(dsub.shape), sub)
        sl = (slice(v0, v0 + n), slice(u0, u0 + n))
        phi = self.B.phi[sl]
        dark = phi * (1.0 - Vr1)
        # O on every stride-th 1x pixel of the depth plate near the robot, bilinear in between
        st = int(sh["sky_stride_px"])
        d1 = self.B.depth1[sl]
        gy, gx = np.arange(0, n, st), np.arange(0, n, st)
        if gy[-1] != n - 1:
            gy = np.append(gy, n - 1)
        if gx[-1] != n - 1:
            gx = np.append(gx, n - 1)
        dg = d1[np.ix_(gy, gx)]
        uu, vv = np.meshgrid(gx + u0, gy + v0)
        pcg = np.stack([(uu - self.spec.cx) / self.spec.fx * dg, (vv - self.spec.cy) / self.spec.fy * dg, dg], -1)
        Pg = pcg @ np.asarray(self.spec.R, float).T + np.asarray(self.spec.position, float)
        near = (dg > 0) & (np.linalg.norm(Pg[..., :2] - cen[:2], axis=-1) < float(sh["sky_near_radius_m"])) & \
               (Pg[..., 2] < float(sh["sky_near_max_z_m"]))
        Og = np.zeros(dg.shape, np.float32)
        if near.any():
            Og[near] = plate_shadow.sky_occlusion(Pg[near], caps, self.hemi, float(sh["sky_capsule_radius_m"]))
        O1 = _bilinear(Og, gy, gx, n) if Og.any() else np.zeros((n, n), np.float32)
        dark = dark + (1.0 - phi) * float(sh["sky_gain"]) * O1
        info.update(rays=nray, sky_points=int(near.sum()))
        return np.clip(dark, 0.0, 1.0).astype(np.float32)

    def facts(self):
        return {"t_open_s": round(self.t_open_s, 3), "t_bundle_s": round(self.t_bundle_s, 3),
                "t_layer_open_s": round(self.lay.t_open_s, 3), "t_layer_warm_s": round(self.lay.t_warm_s, 3),
                "counts": self.counts, "bundle": self.B.facts()}

    def close(self):
        """The robot layer (its objects, then its context), the classic sessions, then the bundle's arrays and the
        robot models; the freed heap goes back to the system (malloc_trim) once everything is released."""
        lay, self.lay = getattr(self, "lay", None), None
        if lay is not None:
            lay.close()
        del lay
        for s in (getattr(self, "s1", None), getattr(self, "s4", None)):
            if s is not None:
                s.close()
        self.s1 = self.s4 = None
        self.B = self.rm = self.rd = self.last = None
        gc.collect()
        plate_robot.trim_heap()


def _bilinear(G, gy, gx, n):
    """Bilinear interpolation of G (values at rows gy, columns gx of an n x n grid, both increasing, covering 0..n-1)."""
    ys, xs = np.arange(n), np.arange(n)
    iy = np.clip(np.searchsorted(gy, ys, side="right") - 1, 0, len(gy) - 2)
    ix = np.clip(np.searchsorted(gx, xs, side="right") - 1, 0, len(gx) - 2)
    fy = ((ys - gy[iy]) / (gy[iy + 1] - gy[iy]))[:, None]
    fx = ((xs - gx[ix]) / (gx[ix + 1] - gx[ix]))[None, :]
    a = G[np.ix_(iy, ix)]
    b = G[np.ix_(iy, ix + 1)]
    c = G[np.ix_(iy + 1, ix)]
    d = G[np.ix_(iy + 1, ix + 1)]
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)
