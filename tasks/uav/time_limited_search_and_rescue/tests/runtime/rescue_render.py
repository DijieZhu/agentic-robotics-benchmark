"""The aircraft's two cameras: MuJoCo 3.16's Filament renderer (physically based) on Vulkan; in the task's image
Vulkan is Mesa's lavapipe on the CPU (no GPU). rescue.World draws every image through this module, in the
commissioning service and in the graded flights alike.

What is drawn
  * every geom of the site's own model, where the simulation has it, in the model's own shape and size: walls,
    slabs, roofs, well frames, door and throat panels, shutters, floor markings, signs (the task's own pictures),
    tool stands and collets, the litter with its fitting and its mannequin, the aircraft. Nothing of the model is
    left out or moved. Their surfaces are photographed or painted materials instead of plain colours (the maps of
    the render bundle in RESCUE_RENDER_DIR, default /opt/task-assets/payload/render): plastered walls, tiled and
    concrete floors, ceilings, timber skirting; every coloured part keeps the model's colour (parts painted in a
    wayfinding colour show the model's colour under a wear pattern; floor markings are worn paint over the floor);
  * drawn only (nothing of it is in the simulated model): a glazed window in every wall segment of the building's
    outer hull; a timber frame round every ordinary doorway (boards on the walls beside the opening, a head over
    it at 2.10 m and wall above the head up to the slab: the opening is drawn at its structural width, and closed
    above 2.16 m where the structure is open to the slab). The rule is the same at every site and reads the model
    alone (_extras).
Light: an image light (ibl.ktx of the bundle: a brighter tone from above, a darker one from below, the same in
every compass direction, so no room is lit from a side) and ambient occlusion; no lamp, no shadow map (an image has
a price in time, and shadow maps are most of an image's time on a CPU). The site's tone (the model's light scale)
scales the image light.
Cameras: exactly the simulated cameras "down" and "forward": position, axes and frustum are the ones MuJoCo
computes for the site's model (mjv_updateCamera). An image is drawn at twice its size and averaged over 2 x 2
blocks in linear light (integer arithmetic).

An image is a function of the site and the state alone: the same state gives the same bytes however often it is
drawn, whatever was drawn before, in whichever process and on whichever host of the same image. Filament's
temporal dithering is off (_undither); every Filament object is made once per process, in one order; one process
draws one site: rescue.World starts a worker process of this module per site (Cameras) and ends it with the site.
Everything is drawn with one material (one atlas of all the maps), and everything that stands still is one mesh and
one draw call: the order in which Filament draws two draw calls changes from process to process, and with it which
of two surfaces a sample on their common edge shows. A body that moves is one mesh each, moved as a whole; no mesh
is uploaded after the site is built (an upload leaks memory in this binding).

    cams = Cameras(seed)                 # starts the worker: python3 -I rescue_render.py --worker
    down, forward = cams.images(qpos, mocap_pos)       # uint8 (256, 256, 3) each, rows top-down
    cams.close()

Every failure (a missing file of the bundle, a worker that dies or does not answer within RESCUE_RENDER_TIMEOUT
seconds) is raised as RenderError: an error of the environment, never of a flight program.
"""
import ctypes
import gc
import json
import math
import os
import select
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.environ.get("RESCUE_RENDER_DIR", "/opt/task-assets/payload/render")
LOOK_FILE = os.path.join(HERE, "rescue_look.json")
CALL_TIMEOUT = float(os.environ.get("RESCUE_RENDER_TIMEOUT", "300"))
SIZE = 256
MAGIC = b"RSCU"


class RenderError(RuntimeError):
    """The renderer failed: an error of the environment, not of a flight program."""


# ----------------------------------------------------------------------------------------------------------------
# The client: one worker process per site
# ----------------------------------------------------------------------------------------------------------------
class Cameras(object):
    """The two cameras of the site `seed` (rescue.World's seed), drawn by a worker process of their own."""

    def __init__(self, seed):
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        env["MESA_SHADER_CACHE_DISABLE"] = "true"      # no shader cache on disk: every worker compiles its own
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            self.proc = subprocess.Popen([sys.executable, "-I", "-B", os.path.abspath(__file__), "--worker"],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, close_fds=True)
            self.proc.stdin.write((json.dumps({"seed": int(seed)}) + "\n").encode())
            self.proc.stdin.flush()
            head = self._read(len(MAGIC) + 4)
            if head[:4] != MAGIC:
                raise RenderError("renderer: the worker did not start")
            self.n = struct.unpack("<I", head[4:])[0]  # how many numbers a state is (qpos, then mocap positions)
        except (OSError, RenderError) as exc:
            self.close()
            raise RenderError("renderer: %s" % exc)

    def _read(self, n):
        out = bytearray()
        fd = self.proc.stdout.fileno()
        while len(out) < n:
            if not select.select([fd], [], [], CALL_TIMEOUT)[0]:
                raise RenderError("the worker did not answer within %.0f s (RESCUE_RENDER_TIMEOUT)" % CALL_TIMEOUT)
            chunk = os.read(fd, n - len(out))
            if not chunk:
                raise RenderError("the worker ended (exit %s)" % self.proc.poll())
            out += chunk
        return bytes(out)

    def images(self, qpos, mocap_pos):
        state = np.concatenate([np.asarray(qpos, np.float64).ravel(), np.asarray(mocap_pos, np.float64).ravel()])
        if len(state) != self.n:
            raise RenderError("renderer: a state of %d numbers, the site has %d" % (len(state), self.n))
        try:
            self.proc.stdin.write(state.tobytes())
            self.proc.stdin.flush()
            raw = self._read(2 * SIZE * SIZE * 3)
        except (OSError, RenderError) as exc:
            raise RenderError("renderer: %s" % exc)
        a = np.frombuffer(raw, np.uint8).reshape(2, SIZE, SIZE, 3)
        return [a[0].copy(), a[1].copy()]

    def close(self):
        proc, self.proc = getattr(self, "proc", None), None
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except (OSError, AttributeError):
                pass
        try:
            proc.wait(10)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            try:
                proc.wait(10)
            except (subprocess.TimeoutExpired, OSError):
                pass

    def __del__(self):
        self.close()


# ----------------------------------------------------------------------------------------------------------------
# Shapes (numpy only): vertices, normals, texture coordinates, triangles
# ----------------------------------------------------------------------------------------------------------------
_FACES = (  # (normal, four corners counter-clockwise seen from outside), in units of the half sizes
    ((-1, 0, 0), ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1))),
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 1, 0), ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1))),
    ((0, 0, -1), ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
)
_UNIT = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
_UPRIGHT = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], float)     # a picture upright on a standing rectangle


def box_faces(h):
    """The six faces of a box of half sizes h: [(normal, corners 4 x 3)], in the order -x +x -y +y -z +z."""
    h = np.asarray(h, float)
    return [(np.array(n, float), np.array(c, float) * h) for n, c in _FACES]


def cylinder(r, hl, seg=24):
    """A cylinder of radius r and half length hl along z: (V, N, UV, I)."""
    V, N, UV, I = [], [], [], []
    for k in range(seg):
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        b = len(V)
        V += [(r * c0, r * s0, -hl), (r * c1, r * s1, -hl), (r * c1, r * s1, hl), (r * c0, r * s0, hl)]
        N += [(c0, s0, 0), (c1, s1, 0), (c1, s1, 0), (c0, s0, 0)]
        UV += [(k / seg, 0), ((k + 1) / seg, 0), ((k + 1) / seg, 1), (k / seg, 1)]
        I += [b, b + 1, b + 2, b, b + 2, b + 3]
        for z, nz in ((hl, 1.0), (-hl, -1.0)):
            b = len(V)
            V += [(0, 0, z), (r * c0, r * s0, z), (r * c1, r * s1, z)]
            N += [(0, 0, nz)] * 3
            UV += [(0.5, 0.5), (0.5 + 0.5 * c0, 0.5 + 0.5 * s0), (0.5 + 0.5 * c1, 0.5 + 0.5 * s1)]
            I += [b, b + 1, b + 2] if nz > 0 else [b, b + 2, b + 1]
    return np.array(V, float), np.array(N, float), np.array(UV, float), np.array(I, np.int64)


def ellipsoid(size, seg=20, rings=10):
    """An ellipsoid of semi-axes `size`: (V, N, UV, I)."""
    s = np.asarray(size, float)
    V, N, UV, I = [], [], [], []
    for i in range(rings + 1):
        t = math.pi * i / rings
        for k in range(seg + 1):
            p = 2 * math.pi * k / seg
            u = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])
            V.append(u * s)
            n = u / s
            N.append(n / np.linalg.norm(n))
            UV.append((k / seg, i / rings))
    for i in range(rings):
        for k in range(seg):
            a, b = i * (seg + 1) + k, (i + 1) * (seg + 1) + k
            I += [a, b, b + 1, a, b + 1, a + 1]
    return np.array(V, float), np.array(N, float), np.array(UV, float), np.array(I, np.int64)


def quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Batches(object):
    """What is drawn, per material key, in the order it is added: rectangles (four corners, a normal, texture
    coordinates) and triangle soups."""

    def __init__(self):
        self.keys = []
        self.quads = {}
        self.soups = {}

    def _key(self, key):
        if key not in self.quads:
            self.keys.append(key)
            self.quads[key], self.soups[key] = [], []

    def add(self, key, V, N, UV, I):
        self._key(key)
        I = np.asarray(I, np.int64)
        self.soups[key].append((np.asarray(V, float)[I], np.asarray(N, float)[I], np.asarray(UV, float)[I]))

    def quad(self, key, P4, n, uv):
        self._key(key)
        self.quads[key].append((np.asarray(P4, float), np.asarray(n, float), np.asarray(uv, float)))

    def triangles(self, key):
        return sum(len(t[0]) for t in self.soups[key]) // 3 + 2 * len(self.quads[key])


def _hex(rgb):
    return "%02x%02x%02x" % tuple(int(round(255 * min(1.0, max(0.0, float(x))))) for x in rgb[:3])


def _near(a, b, tol=0.02):
    return all(abs(float(x) - float(y)) < tol for x, y in zip(a[:3], b[:3]))


# ----------------------------------------------------------------------------------------------------------------
# What a site's model becomes: batches of the standing parts (world frame) and of each moving body (its frame)
# ----------------------------------------------------------------------------------------------------------------
DECK, SLAB = 2.60, 0.22
WALL_RGB = (.65, .70, .70)
SKIRT_RGB = (.39, .26, .17)
FLOOR_MARKS = ("apron", "pad", "receiving_mat")


def _uv_world(P, n, tile, storey=False):
    """Texture coordinates in metres over `tile` from the points' own coordinates: across a face by the two axes it
    spans; a storey map's v is the height above its storey's floor."""
    k = int(np.argmax(np.abs(n)))
    if k == 2:
        uv = P[:, :2].copy()
    else:
        uv = np.stack([P[:, 1 - k], -P[:, 2]], 1)      # the sampler's v runs down a picture: its top is the wall's top
        if storey:
            zc = float(P[:, 2].mean())
            uv[:, 1] += DECK if zc > DECK else 0.0
    return uv / float(tile)


TIE_STEP = 0.00037      # a face that would lie in another's plane and over it stands this much proud, per level
MOVING_SHRINK = 0.0005  # a moving body's boxes are drawn this much smaller: at rest against a standing part they never tie


def describe(model, look, _boxes=None, layout=None, seed=0):
    """(standing parts: Batches in the world frame; {body id: Batches in the body's frame}; material rules by key).
    Reads the model alone."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    G = mujoco.mjtGeom
    C = look["classes"]
    rules, moving = {}, {}
    outdoor = float(look.get("sun", {}).get("outdoor", 0.0)) or None
    furniture = []                                      # look_block_*: the standing things' boxes (they shade the sun)
    windows = []
    standing = Batches()
    boxes = []                                          # the standing axis-aligned boxes, in the model's order
    floor = None

    def rule(key, base, **over):
        if key not in rules:
            r = dict(C[base])
            r.update(over)
            rules[key] = r
        return key

    def plain(rgba, base="plain"):
        return rule("%s_%s" % (base, _hex(rgba)), base, rgb=[float(x) for x in rgba[:3]])

    PA = look.get("paints")                             # the look's paints: the model's signal colours toned, greyed, dusted

    def toned(rgba, kind):
        import rescue_marks as RM
        t = RM.tone([float(x) for x in rgba[:3]], *RM.TONES[kind])
        if kind == "apron":
            return list(t)
        l_ = [_linear(c) for c in t]
        lum = 0.2126 * l_[0] + 0.7152 * l_[1] + 0.0722 * l_[2]
        ds, du = float(PA["desat"]), float(PA["dust"][kind])
        dl = [_linear(c) for c in PA["dust_rgb"]]
        l_ = [(c * (1 - ds) + lum * ds) * (1 - du) + d_ * du for c, d_ in zip(l_, dl)]
        return [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in l_]

    def paint(rgba, kind):
        """(key) a painted part's rule: the look's paint of its kind (metal, wall, floor, mat), or the first design's"""
        if not PA:
            return None
        t = toned(rgba, kind)
        return rule("p%s_%s" % (kind, _hex(t)), "paint", rgb=t, map="norm_" + {"metal": "metal", "wall": "wall", "floor": "floor", "mat": "rubber"}[kind],
                    tint=True, norm2=True, tile={"metal": 1.0, "wall": 2.71, "floor": 3.0, "mat": 0.6}[kind])

    def flat(rgba):
        """(key) a small or turned painted part in one colour"""
        return plain(toned(rgba, "metal")) if PA else plain(rgba)

    def loose_box(dst, key, R, t, size, still):
        for n, c in box_faces(size):
            dst.quad(key, c @ R.T + t, R @ n, _UNIT)

    for g in range(model.ngeom):
        if (model.geom(g).name or "").startswith("prop_"):   # a standing thing's box: it shades the sun
            hh = np.abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_size[g]
            furniture.append((data.geom_xpos[g] - hh, data.geom_xpos[g] + hh))
        if float(model.geom_rgba[g, 3]) <= 0.0 or int(model.geom_group[g]) > 2:
            continue
        b = int(model.geom_bodyid[g])
        name = model.geom(g).name or ""
        rgba = model.geom_rgba[g].copy()
        mat = int(model.geom_matid[g])
        still = int(model.body_weldid[b]) == 0 and int(model.body_mocapid[b]) < 0
        if still:
            dst, R, t = standing, data.geom_xmat[g].reshape(3, 3).copy(), data.geom_xpos[g].copy()
        else:
            dst = moving.setdefault(b, Batches())
            R, t = quat_mat(model.geom_quat[g]), model.geom_pos[g].copy()
        kind = int(model.geom_type[g])
        size = model.geom_size[g].copy()
        dark = _near(rgba, (.075, .112, .126))

        if kind == int(G.mjGEOM_PLANE):
            if name == "floor":
                floor = (R, t)                          # drawn below, under the site's footprint
            else:                                       # a sign: the model's own picture over the plane's rectangle
                sx, sy = float(size[0]), float(size[1])
                P4 = np.array([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], float) @ R.T + t
                P4 = P4 + 0.00061 * R[:, 2]             # clear of a mat's or a housing's face in its plane
                tex = int(model.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]) if mat >= 0 else -1
                key = rule("sign_%d" % tex, "sign", model_tex=tex) if tex >= 0 else plain(rgba)
                dst.quad(key, P4, R[:, 2], _UNIT)
            continue

        if kind == int(G.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            V0 = model.mesh_vert[va:va + vn].astype(float)
            F = model.mesh_face[fa:fa + fn].astype(np.int64)
            P = V0[F.reshape(-1)] @ R.T + t
            na = int(model.mesh_normaladr[mid])
            if na >= 0:
                Nn = model.mesh_normal[na:na + int(model.mesh_normalnum[mid])].astype(float)
                Nv = Nn[model.mesh_facenormal[fa:fa + fn].astype(np.int64).reshape(-1)] @ R.T
            else:
                fnrm = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
                Nv = np.repeat(fnrm / (np.linalg.norm(fnrm, axis=1, keepdims=True) + 1e-20), 3, 0)
            ta = int(model.mesh_texcoordadr[mid])
            tex = int(model.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]) if mat >= 0 else -1
            if ta >= 0 and tex >= 0:
                FT = model.mesh_facetexcoord[fa:fa + fn].astype(np.int64)
                T = model.mesh_texcoord[ta:ta + int(model.mesh_texcoordnum[mid])].astype(float)
                uv = T[FT.reshape(-1)].copy()
                uv[:, 1] = 1.0 - uv[:, 1]               # MuJoCo keeps a mesh's v upwards; the atlas runs down a picture
                key = rule("scan_%d" % tex, "scan", model_tex=tex)
                if name.startswith("look_yard"):
                    key = rule("scan_%d_out" % tex, "scan", model_tex=tex, sun=outdoor)
            else:
                uv = np.zeros((len(P), 2))
                key = plain(model.mat_rgba[mat] if mat >= 0 else rgba, "plain" if still else "airframe")
                if name.startswith("look_yard"):
                    c_ = model.mat_rgba[mat] if mat >= 0 else rgba
                    key = rule("out_%s" % _hex(c_), "plain", rgb=[float(x) for x in c_[:3]], sun=outdoor)
            dst.add(key, P, Nv, uv, np.arange(len(P)))
            continue

        if kind == int(G.mjGEOM_CYLINDER):
            V, N, UV, I = cylinder(float(size[0]), float(size[1]))
            dst.add(plain(rgba, "metal" if dark else "plain"), V @ R.T + t, N @ R.T, UV, I)
            continue

        if kind == int(G.mjGEOM_ELLIPSOID):
            V, N, UV, I = ellipsoid(size)
            dst.add(plain(rgba, "soft"), V @ R.T + t, N @ R.T, UV, I)
            continue

        if kind != int(G.mjGEOM_BOX):
            raise RenderError("renderer: a geom of a type the cameras do not draw (%s)" % name)

        if not still:                                   # a moving body's box, in the body's frame
            loose_box(dst, flat(rgba) if PA else plain(rgba, "metal" if dark else "plain"), R, t, np.maximum(size - MOVING_SHRINK, 1e-5), False)
            continue
        if not np.allclose(np.abs(R), np.eye(3), atol=1e-9):
            # a standing box that is turned (a stowed deck's mat and pad, the arrow's vanes, a shut door's cross)
            if name.startswith(FLOOR_MARKS):
                base = "mat" if name.startswith("receiving_mat") else "floor_paint"
                key = rule("%s_%s" % (base, _hex(rgba)), base, rgb=[float(x) for x in rgba[:3]],
                           map=("mat_" if base == "mat" else "paint_") + _hex(rgba))
                for n, c in box_faces(size):
                    if n[2] > 0.5:
                        dst.quad(key, c @ R.T + t, R @ n, _UNIT)
                    else:
                        dst.quad(plain(rgba), c @ R.T + t, R @ n, np.zeros((4, 2)))
            else:
                if name.startswith("look_"):
                    loose_box(dst, plain(rgba), R, t, size, True)
                elif PA and name.startswith("dispatch_") and name != "dispatch_housing":
                    loose_box(dst, rule("led_white", "window_pane", map=None, rgb=[1.0, 0.97, 0.92]), R, t, size, True)
                else:
                    loose_box(dst, flat(rgba) if PA else plain(rgba, "metal" if dark else "plain"), R, t, size, True)
            continue

        # ---- a standing box along the axes: classified here, drawn after the ties are settled
        h = np.abs(R) @ size
        if name.startswith("look_"):
            for n, c in box_faces(size):
                standing.quad(plain(rgba), c @ R.T + t, R @ n, np.zeros((4, 2)))
            continue
        if name.startswith("slab"):
            what = "slab"
        elif name.startswith("roof"):
            what = "roof"
        elif PA and name.startswith(("pad", "bay", "throat_")) and float(np.min(h)) < 0.01:
            what = "pfloor"
        elif PA and name.startswith("solid") and _near(rgba, (.32, .57, .59)):
            what = "pwall"
        elif name.startswith(FLOOR_MARKS) or ((name.startswith("bay") or "_line" in name) and h[2] < 0.01):
            what = "mark"
        elif name.startswith("solid") and _near(rgba, WALL_RGB):
            what = "wall"
        elif name.startswith("solid") and _near(rgba, SKIRT_RGB):
            what = "skirt"
        else:
            what = "paint"
        boxes.append({"what": what, "name": name, "lo": t - h, "hi": t + h, "rgba": rgba, "dark": dark})

    if floor is not None and boxes:                     # the ground plane under the site and 3 m round it
        R, t = floor
        gm = float(look.get("ground_margin", 3.0))
        x0, y0 = np.min([bx["lo"][:2] for bx in boxes], axis=0) - gm
        x1, y1 = np.max([bx["hi"][:2] for bx in boxes], axis=0) + gm
        P4 = np.array([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], float) @ R.T + np.array([0.0, 0.0, t[2]])
        standing.quad(rule("floor_lower", "floor_lower"), P4, R[:, 2], _uv_world(P4, R[:, 2], C["floor_lower"]["tile"]))
    walls = [(0.5 * (bx["lo"] + bx["hi"]), 0.5 * (bx["hi"] - bx["lo"])) for bx in boxes if bx["what"] == "wall"]
    if not look.get("without_extras"):                  # (a review switch: the model's own geoms alone)
        n_own = len(boxes)
        windows = _extras(model, standing, boxes, walls, rule, C, panes=not look.get("window_holes"),
                          own_frames=not look.get("look_frames"), layout=layout)
        if look.get("sun"):
            _sun(standing, boxes[:n_own], windows, rule, rules, C, look["sun"], furniture)
    mark_extras = look.get("mark_extras")               # (a review switch: the drawn-only parts in one glowing colour)

    # ---- ties. Where two faces lie in one plane, look the same way and overlap, which of them a pixel shows is
    # not a function of the scene. A wall's top in the plane of a slab's top goes 2 mm under it. Of any other such
    # pair one face is drawn a fraction of a millimetre proud: the face alone, so that no other face of its box changes its extent.
    for bx in boxes:
        if bx["what"] in ("wall", "skirt", "frame", "door", "lintel") and any(abs(bx["hi"][2] - z) < 1e-6 for z in (DECK, 2 * DECK)):
            bx["hi"] = bx["hi"] - np.array([0.0, 0.0, 0.002])
        bx["proud"] = np.zeros((3, 2))

    lo = np.array([bx["lo"] for bx in boxes])
    hi = np.array([bx["hi"] for bx in boxes])
    for k in range(3):
        o = [j for j in range(3) if j != k]
        for side, plane in ((0, lo[:, k]), (1, hi[:, k])):
            level = np.zeros(len(boxes), np.int64)
            for i in range(1, len(boxes)):
                m = (np.abs(plane[:i] - plane[i]) < 1e-7)
                if not m.any():
                    continue
                for j in o:
                    m &= (np.minimum(hi[:i, j], hi[i, j]) - np.maximum(lo[:i, j], lo[i, j])) > 1e-6
                used = {int(level[j]) for j in np.nonzero(m)[0]}
                lv = 0
                while lv in used:
                    lv += 1
                level[i] = lv
                boxes[i]["proud"][k, side] = TIE_STEP * lv

    if _boxes is not None:                              # (for a review tool: the boxes as they are drawn)
        _boxes.extend(boxes)
    wells_x = _well_columns(model) if look.get("well_dim") else []
    wall_y = [float(bx["hi"][1]) for bx in boxes if bx["what"] == "wall"]
    y_north = max(wall_y) if wall_y else 0.0

    def dim_of(bx):
        """a hoist well is darker than a room: the factor of a part that stands in one"""
        c_ = 0.5 * (bx["lo"] + bx["hi"])
        if wells_x and c_[1] > y_north - 2.75 and any(abs(c_[0] - x) < 1.36 for x in wells_x):
            return float(look["well_dim"])
        return 0.0

    for bx in boxes:
        c, h = 0.5 * (bx["lo"] + bx["hi"]), 0.5 * (bx["hi"] - bx["lo"])
        z0 = float(bx["lo"][2])
        what, rgba = bx["what"], bx["rgba"]
        faces = [(n, P + c + n * bx["proud"][int(np.argmax(np.abs(n))), int(n.sum() > 0)]) for n, P in box_faces(h)]
        on_floor = z0 < 0.02 or abs(z0 - DECK) < 1e-6    # it stands on the ground or on the slab: no underside
        if mark_extras and what in ("frame", "door", "lintel"):
            for n, P4 in faces:
                standing.quad(rule("marked", "window_pane", flat=mark_extras, map=None), P4, n, np.zeros((4, 2)))
            continue
        if what in ("slab", "roof"):
            for n, P4 in faces:
                if n[2] > 0.5:
                    base = "floor_upper" if what == "slab" else "concrete"
                elif n[2] < -0.5:
                    base = "ceiling_upper" if float(P4[0, 2]) > DECK else "ceiling_lower"
                else:
                    base = "concrete"
                standing.quad(rule(base, base), P4, n, _uv_world(P4, n, C[base]["tile"]))
        elif what == "skirt" and look.get("look_frames"):
            continue                                    # the look's own skirting stands there
        elif what in ("pfloor", "pwall") or (PA and what == "mark" and bx["name"].startswith("receiving_mat")):
            kind = {"pfloor": "floor", "pwall": "wall", "mark": "mat"}[what]
            key = paint(rgba, kind)
            if dim_of(bx):
                key = rule(key + "_dim", "paint", **{kk: vv for kk, vv in rules[key].items() if kk != "roughness"}, dim=dim_of(bx))
            tile = float(rules[key]["tile"])
            for n, P4 in faces:
                if n[2] < -0.5 and on_floor:
                    continue
                standing.quad(key, P4, n, _uv_world(P4, n, tile))
        elif what == "mark" and PA and bx["name"].startswith("apron"):
            import hashlib
            t = toned(rgba, "apron")
            key = rule("apron_%s" % _hex(t), "floor_paint", apron=t, seed=int(seed))
            fl = bool(int(hashlib.md5(bx["name"].encode()).hexdigest()[:2], 16) % 2)
            for n, P4 in faces:
                if n[2] > 0.5:
                    standing.quad(key, P4, n, (1.0 - _UPRIGHT) if fl else _UPRIGHT)
                elif n[2] > -0.5:
                    standing.quad(plain(t), P4, n, np.zeros((4, 2)))
        elif what == "mark":
            base = "mat" if bx["name"].startswith("receiving_mat") else "floor_paint"
            key = rule("%s_%s" % (base, _hex(rgba)), base, rgb=[float(x) for x in rgba[:3]],
                       map=("mat_" if base == "mat" else "paint_") + _hex(rgba))
            for n, P4 in faces:
                if n[2] > 0.5:
                    standing.quad(key, P4, n, _UNIT)
                elif n[2] > -0.5:
                    standing.quad(plain(rgba), P4, n, np.zeros((4, 2)))
        elif what in ("wall", "lintel"):
            for n, P4 in faces:
                if n[2] < -0.5 and on_floor:
                    continue
                if abs(n[2]) > 0.5:
                    standing.quad(rule("concrete", "concrete"), P4, n, _uv_world(P4, n, C["concrete"]["tile"]))
                    continue
                base = "wall_upper" if float(P4[:, 2].mean()) > DECK else "wall_lower"
                key = rule(base, base)
                tile = C[base]["tile"]
                win = [w_ for w_ in windows if look.get("window_holes") and what == "wall"
                       and np.all(np.abs(w_["lo"] - bx["lo"]) < 0.01) and np.all(np.abs(w_["hi"] - bx["hi"]) < 0.01)]
                ka = int(np.argmax(np.abs(n)))
                if not win or ka != win[0]["k"]:
                    standing.quad(key, P4, n, _uv_world(P4, n, tile, storey=True))
                    continue
                w_ = win[0]
                ta = 1 - ka
                t0, t1 = float(P4[:, ta].min()), float(P4[:, ta].max())
                zl, zh = float(P4[:, 2].min()), float(P4[:, 2].max())
                a0, a1 = w_["at"] - w_["hw"], w_["at"] + w_["hw"]
                for u0, u1, v0, v1 in ((t0, a0, zl, zh), (a1, t1, zl, zh), (a0, a1, zl, w_["z0"]), (a0, a1, w_["z1"], zh)):
                    Q = _rect(ka, float(P4[0, ka]), n, u0, u1, v0, v1)
                    standing.quad(key, Q, n, _uv_world(Q, n, tile, storey=True))
                if n[ka] > 0:                           # once per wall: the opening's four sides
                    k0, k1 = float(bx["lo"][ka]), float(bx["hi"][ka])
                    for tc_, sg_ in ((a0, 1.0), (a1, -1.0)):
                        nn = np.zeros(3)
                        nn[ta] = sg_
                        Q = _rect(ta, tc_, nn, k0, k1, w_["z0"], w_["z1"])
                        standing.quad(key, Q, nn, _uv_world(Q, nn, tile, storey=True))
                    for zc_, sg_ in ((w_["z0"], 1.0), (w_["z1"], -1.0)):
                        nn = np.array([0.0, 0.0, sg_])
                        c4 = [(a0, k0), (a1, k0), (a1, k1), (a0, k1)]
                        Q = np.zeros((4, 3))
                        for i_, (tt, kk) in enumerate(c4):
                            Q[i_, ta], Q[i_, ka], Q[i_, 2] = tt, kk, zc_
                        if np.cross(Q[1] - Q[0], Q[3] - Q[0]) @ nn < 0:
                            Q = Q[::-1].copy()
                        standing.quad(rule("concrete", "concrete"), Q, nn, _uv_world(Q, nn, C["concrete"]["tile"]))
        elif what in ("skirt", "door", "frame"):
            base = {"skirt": "skirting", "door": "door_frame", "frame": "window_frame"}[what]
            key = rule(base, base)
            for n, P4 in faces:
                if n[2] < -0.5 and on_floor:
                    continue
                standing.quad(key, P4, n, _uv_world(P4, n, C[base].get("tile", 1.0)))
        else:
            key = paint(rgba, "metal") or rule("paint_%s" % _hex(rgba), "metal" if bx["dark"] else "paint", rgb=[float(x) for x in rgba[:3]])
            if PA and bx["name"] == "dispatch_housing":
                key = plain((0.035, 0.037, 0.04))
            if dim_of(bx):
                key = rule(key + "_dim", "paint", **{kk: vv for kk, vv in rules[key].items() if kk != "roughness"}, dim=dim_of(bx))
            tile = float(rules[key].get("tile", 1.0))
            for n, P4 in faces:
                if n[2] < -0.5 and on_floor:
                    continue
                standing.quad(key, P4, n, _uv_world(P4, n, tile))
    return standing, moving, rules


def _rect(axis, coord, n, u0, u1, v0, v1):
    """A rectangle in the plane `axis` = coord, facing n: its corners. On a standing plane u runs along the other
    level axis and v is the height; on a level plane u is x and v is y."""
    Q = np.zeros((4, 3))
    ua, va = ((1 - axis), 2) if axis < 2 else (0, 1)
    for i, (u, v) in enumerate(((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
        Q[i, axis], Q[i, ua], Q[i, va] = coord, u, v
    if np.cross(Q[1] - Q[0], Q[3] - Q[0]) @ np.asarray(n, float) < 0:
        Q = Q[::-1].copy()
    return Q


def _sun(dst, boxes, windows, rule, rules, C, sun, furniture):
    """Daylight: where a ray from a point of a floor, of a floor marking or of a wall towards the sun leaves through
    a window of the hull without meeting a wall, a slab, a panel or a standing piece of furniture, and reaches the
    window from outside, the surface is drawn sunlit (its own picture and a glowing copy of it). Points on a grid of
    sun["grid"] metres; rows of sunlit points become rectangles 0.6 mm over the surface."""
    d = np.array(sun["dir"], float)
    d = d / math.sqrt(float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
    up = -d
    res = float(sun.get("grid", 0.02))
    gain = float(sun.get("gain", 1.0))
    solid = [bx for bx in boxes if bx["what"] != "mark"]
    lo = np.array([bx["lo"] for bx in solid] + [f[0] for f in furniture])
    hi = np.array([bx["hi"] for bx in solid] + [f[1] for f in furniture])
    marks = [bx for bx in boxes if bx["what"] == "mark"]
    walls = [bx for bx in boxes if bx["what"] in ("wall", "lintel")]
    OFF = 0.0006

    def emit(key, axis, coord, n, i0, j0, lit, uv_of):
        open_ = {}
        for r in range(lit.shape[0] + 1):
            runs = set()
            if r < lit.shape[0]:
                row = np.concatenate([[False], lit[r], [False]])
                edges = np.nonzero(row[1:] != row[:-1])[0]
                runs = {(int(edges[2 * e]), int(edges[2 * e + 1])) for e in range(len(edges) // 2)}
            for run in sorted(set(open_) - runs):
                r0 = open_.pop(run)
                Q = _rect(axis, coord, n, (i0 + run[0]) * res, (i0 + run[1]) * res, (j0 + r0) * res, (j0 + r) * res)
                dst.quad(key, Q, n, uv_of(Q))
            for run in sorted(runs - set(open_)):
                open_[run] = r

    for w in windows:
        k, t = w["k"], 1 - w["k"]
        if abs(d[k]) < 1e-6 or d[2] > -1e-6:
            continue
        zf = w["floor"]
        beam = []                                       # the window's corners and where the sun throws them on the floor
        for a in (-1.0, 1.0):
            for z in (w["z0"], w["z1"]):
                q = np.zeros(3)
                q[k], q[t], q[2] = w["plane"], w["at"] + a * w["hw"], z
                beam += [q, q + d * ((z - zf) / (-d[2]))]
        beam = np.array(beam)
        blo, bhi = beam.min(0) - 0.05, beam.max(0) + 0.05
        own = np.all(np.abs(lo - w["lo"]) < 0.01, axis=1) & np.all(np.abs(hi - w["hi"]) < 0.01, axis=1)
        far = np.minimum(blo, blo + up * 8.0), np.maximum(bhi, bhi + up * 8.0)
        near = np.nonzero(np.all(hi > far[0], axis=1) & np.all(lo < far[1], axis=1) & ~own)[0]

        def lit_on(axis, coord, n, u0, u1, v0, v1):
            """(first grid indices, the sunlit points) of a rectangle of the plane `axis` = coord"""
            ua, va = ((1 - axis), 2) if axis < 2 else (0, 1)
            u0, u1, v0, v1 = max(u0, blo[ua]), min(u1, bhi[ua]), max(v0, blo[va]), min(v1, bhi[va])
            i0, i1, j0, j1 = int(math.ceil(u0 / res)), int(math.floor(u1 / res)), int(math.ceil(v0 / res)), int(math.floor(v1 / res))
            if i1 - i0 < 1 or j1 - j0 < 1:
                return None
            gu, gv = np.meshgrid((np.arange(i0, i1) + 0.5) * res, (np.arange(j0, j1) + 0.5) * res)
            O = np.zeros(gu.shape + (3,))
            O[..., ua], O[..., va], O[..., axis] = gu, gv, coord + 0.001 * n[axis]
            sw = (w["plane"] - O[..., k]) / up[k]
            at, z = O[..., t] + up[t] * sw, O[..., 2] + up[2] * sw
            lit = (sw > 0) & (np.abs(at - w["at"]) < w["hw"] - 0.035) & (z > w["z0"] + 0.035) & (z < w["z1"] - 0.035)
            for b in near:
                if not lit.any():
                    return None
                t0 = np.zeros(gu.shape)
                t1 = np.full(gu.shape, 60.0)
                for ax in range(3):
                    if abs(up[ax]) < 1e-12:
                        inside = (O[..., ax] > lo[b, ax]) & (O[..., ax] < hi[b, ax])
                        t1 = np.where(inside, t1, -1.0)
                        continue
                    a0, a1 = (lo[b, ax] - O[..., ax]) / up[ax], (hi[b, ax] - O[..., ax]) / up[ax]
                    t0 = np.maximum(t0, np.minimum(a0, a1))
                    t1 = np.minimum(t1, np.maximum(a0, a1))
                lit &= ~(t0 < t1)
            return (i0, j0, lit) if lit.any() else None

        up_n = np.array([0.0, 0.0, 1.0])
        base = "floor_upper" if zf > 1.0 else "floor_lower"
        got = lit_on(2, zf, up_n, blo[0], bhi[0], blo[1], bhi[1])
        if got:
            tile = C[base]["tile"]
            emit(rule(base + "_sun", base, sun=gain), 2, zf + OFF, up_n, got[0], got[1], got[2], lambda Q, tile=tile: _uv_world(Q, up_n, tile))
        for bx in marks:                                # a floor marking: its own picture, sunlit
            top = float(bx["hi"][2])
            if not (zf - 0.001 < top < zf + 0.05):
                continue
            got = lit_on(2, top, up_n, float(bx["lo"][0]), float(bx["hi"][0]), float(bx["lo"][1]), float(bx["hi"][1]))
            if not got:
                continue
            mbase = "mat" if bx["name"].startswith("receiving_mat") else "floor_paint"
            mkey = "%s_%s" % (mbase, _hex(bx["rgba"]))
            r_ = dict(rules.get(mkey, C[mbase]))
            key = rule(mkey + "_sun", mbase, **{kk: vv for kk, vv in r_.items() if kk in ("rgb", "map")}, sun=gain)
            F0 = np.array([bx["lo"][0], bx["lo"][1]])
            ext = np.array([bx["hi"][0] - bx["lo"][0], bx["hi"][1] - bx["lo"][1]])
            emit(key, 2, top + OFF, up_n, got[0], got[1], got[2], lambda Q, F0=F0, ext=ext: (Q[:, :2] - F0) / ext)
        for bx in walls:                                # the walls' faces that look towards the sun
            if np.all(np.abs(bx["lo"] - w["lo"]) < 0.01) and np.all(np.abs(bx["hi"] - w["hi"]) < 0.01):
                continue
            if np.any(bx["hi"] < blo) or np.any(bx["lo"] > bhi):
                continue
            for axis in (0, 1):
                for sg in (-1.0, 1.0):
                    if sg * d[axis] >= -1e-6:
                        continue
                    n = np.zeros(3)
                    n[axis] = sg
                    coord = float(bx["hi"][axis] if sg > 0 else bx["lo"][axis])
                    oa = 1 - axis
                    got = lit_on(axis, coord, n, float(bx["lo"][oa]), float(bx["hi"][oa]), float(bx["lo"][2]), float(bx["hi"][2]))
                    if not got:
                        continue
                    wb = "wall_upper" if 0.5 * (bx["lo"][2] + bx["hi"][2]) > DECK else "wall_lower"
                    tile = C[wb]["tile"]
                    emit(rule(wb + "_sun", wb, sun=gain), axis, coord + OFF * sg, n, got[0], got[1], got[2],
                         lambda Q, n=n, tile=tile: _uv_world(Q, n, tile, storey=True))


def _extras(model, dst, boxes, wall_boxes, rule, C, panes=True, own_frames=True, layout=None):
    """Drawn only: windows in the outer hull's wall segments and frames in the ordinary doorways. Both follow from
    the model's own plastered wall boxes. The hull: the wing's west, east and north walls on both storeys and its
    staging row's west and east walls (where the wall boxes of the wing end in x and in y), but not behind the
    hoist guides. A doorway: a gap of an ordinary door's width between two plastered wall boxes in one line that
    stand from the floor to the slab (the hoist wells' thresholds stay open). Frames are added to `boxes`; panes
    are drawn here."""
    if not wall_boxes:
        return []
    Wc = np.array([c for c, _ in wall_boxes])
    Wh = np.array([h for _, h in wall_boxes])
    k_of = np.argmin(Wh[:, :2], axis=1)
    thin = Wh[np.arange(len(Wh)), k_of] <= 0.08
    full = Wh[:, 2] >= 1.25
    wing = Wc[:, 1] > 0.5                              # the wing and its staging row (the loading rooms lie south of y = 0)
    xs = Wc[thin & full & wing & (k_of == 0), 0]
    ys = Wc[thin & full & wing & (k_of == 1), 1]
    X0, X1, Y1 = float(xs.min()), float(xs.max()), float(ys.max())
    wells = _well_columns(model)
    windows = []
    pane = rule("window_pane", "window_pane")
    if C.get("window_pane", {}).get("flat"):
        pane = rule("marked", "window_pane")
    dark = False
    hw, sill, head, fb = 0.62, 1.00, 2.10, 0.035
    zero = np.zeros(4)

    def add(what, c, h):
        boxes.append({"what": what, "name": what, "lo": c - h, "hi": c + h, "rgba": zero, "dark": dark})

    for c, h in wall_boxes:
        k = int(np.argmin(h[:2]))
        t = 1 - k
        if h[2] < 1.25 or h[t] < 1.15 or h[k] > 0.08 or c[1] < 0.5:
            continue
        if not ((k == 0 and (abs(c[0] - X0) < 0.1 or abs(c[0] - X1) < 0.1)) or (k == 1 and abs(c[1] - Y1) < 0.1)):
            continue
        if k == 1 and any(abs(c[0] - x) < 1.3 for x in wells):
            continue                                    # the hoist guides run up the back wall there
        zb = float(c[2] - h[2])
        windows.append({"k": k, "plane": float(c[k]), "at": float(c[t]), "floor": zb, "hw": hw, "z0": zb + sill, "z1": zb + head,
                        "lo": c - h, "hi": c + h})
        for side in (-1.0, 1.0):                        # on both faces of the wall
            n = np.zeros(3)
            n[k] = side
            gc = c.copy()
            gc[k] += side * (h[k] + 0.004)
            gc[2] = zb + (sill + head) / 2
            right = np.cross(np.array([0.0, 0.0, 1.0]), n)   # to the right of somebody who looks at this face
            P4 = np.array([gc + a * hw * right + np.array([0, 0, bz * (head - sill) / 2])
                           for a, bz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if panes:
                dst.quad(pane, P4, n, _UPRIGHT)
            for dt, dz, ht, hz in ((-hw + fb / 2, 0, fb / 2, (head - sill) / 2), (hw - fb / 2, 0, fb / 2, (head - sill) / 2),
                                   (0, 0, 0.018, (head - sill) / 2 - fb), (0, (head - sill) / 2 - fb / 2, hw - fb, fb / 2),
                                   (0, -(head - sill) / 2 + fb / 2, hw - fb, fb / 2), (0, 0.12, hw - fb, 0.014)):
                fc, fh = gc.copy(), np.zeros(3)
                fc[t] += dt
                fc[2] += dz
                fc[k] += side * 0.010
                fh[t], fh[k], fh[2] = ht, 0.014 - (0.002 if abs(dz) == 0.12 else 0.0), hz
                if own_frames:
                    add("frame", fc, fh)
    if layout is not None:                              # the receiving room: a glazed double door in its south wall, a window in its west wall
        pcx, pcy = (float(v) for v in layout.pocket)
        ps = pcy - 2.30
        sd = -1.0 if float(layout.receiving_pad[0]) < pcx else 1.0      # the open station's side
        for c, h in wall_boxes:
            k = int(np.argmin(h[:2]))
            zb = float(c[2] - h[2])
            w_ = None
            if k == 1 and abs(c[1] - ps) < 0.12 and h[0] > 3.0:
                w_ = {"at": pcx + sd * (5.45 - 1.25), "hw": 0.70, "z0": zb + 0.04, "z1": zb + 2.08}
            elif k == 0 and abs(c[0] - (pcx + sd * 5.45)) < 0.12 and h[1] > 2.0:
                w_ = {"at": ps + 1.25, "hw": hw, "z0": zb + sill, "z1": zb + head}
            if w_:
                w_.update({"k": k, "plane": float(c[k]), "floor": zb, "lo": c - h, "hi": c + h})
                windows.append(w_)
    seen = set()
    for i, (c, h) in enumerate(wall_boxes):
        k = int(np.argmin(h[:2]))
        t = 1 - k
        if h[k] > 0.08 or h[2] < 1.25:
            continue
        for j, (c2, h2) in enumerate(wall_boxes):
            if j <= i or int(np.argmin(h2[:2])) != k or abs(c2[k] - c[k]) > 1e-6 or abs(c2[2] - c[2]) > 1e-6 or h2[2] < 1.25:
                continue
            lo, hi = (c, h), (c2, h2)
            if c2[t] < c[t]:
                lo, hi = hi, lo
            e0, e1 = lo[0][t] + lo[1][t], hi[0][t] - hi[1][t]
            gap = e1 - e0
            if not 1.40 < gap < 1.50:
                continue
            zb = float(c[2] - h[2])
            cen = np.zeros(3)
            cen[k], cen[t] = c[k], (e0 + e1) / 2
            key = (round(float(cen[0]), 3), round(float(cen[1]), 3), round(zb, 3))
            if key in seen or any(abs(cen[0] - x) < 0.2 for x in wells) and k == 1:
                continue
            seen.add(key)
            # the frame lies on the walls beside the opening and over it, not in it: what the picture shows open is
            # what the structure leaves open, to the width
            for sgn in (-1, 1):
                jc, jh = cen.copy(), np.zeros(3)
                jc[t] += sgn * (gap / 2 + 0.03)
                jc[2] = zb + 1.05
                jh[t], jh[k], jh[2] = 0.03, 0.087, 1.05
                add("door", jc, jh)
            hc, hh = cen.copy(), np.zeros(3)
            hc[2] = zb + 2.13
            hh[t], hh[k], hh[2] = gap / 2 + 0.06, 0.087, 0.03
            if own_frames:
                add("door", hc, hh)
            wc, wh = cen.copy(), np.zeros(3)
            wc[2] = zb + (2.16 + DECK - SLAB) / 2
            wh[t], wh[k], wh[2] = gap / 2, 0.075, (DECK - SLAB - 2.16) / 2
            add("lintel", wc, wh)
    return windows


def _well_columns(model):
    """The x of the hoist wells: where the model's deck frames (aperture_<column>_...) stand."""
    xs = []
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if name.startswith("aperture_") and name.endswith("_1_1"):
            xs.append(float(model.geom_pos[g][0]))
    return xs


# ----------------------------------------------------------------------------------------------------------------
# The worker: Filament, one site, one process
# ----------------------------------------------------------------------------------------------------------------
def _linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin16():
    """sRGB 8 bit -> linear 16 bit, and back: integer tables (the supersampled image is averaged with them)."""
    f = [int(round(65535.0 * _linear(i / 255.0))) for i in range(256)]
    fwd = np.array(f, np.uint32)
    edges = (fwd[1:] + fwd[:-1] + 1) // 2             # a linear value below edges[i] is nearer to level i than to i + 1
    back = np.searchsorted(edges, np.arange(65536, dtype=np.uint32), side="right").astype(np.uint8)
    return fwd, back


def box_down(img, k, tables):
    fwd, back = tables
    h, w = img.shape[0] // k, img.shape[1] // k
    s = fwd[img].reshape(h, k, w, k, 3).sum(axis=(1, 3), dtype=np.uint32)
    return back[(s + (k * k) // 2) // (k * k)]


class Worker(object):
    VIEW_SET_DITHERING = "_ZN8filament4View12setDitheringENS_9DitheringE"
    VIEW_GET_DITHERING = "_ZNK8filament4View12getDitheringEv"

    def __init__(self, seed, look=None, render_dir=None):
        self.seed = int(seed)
        import mujoco
        self.mj = mujoco
        self.look = look or json.load(open(LOOK_FILE))
        self.dir = render_dir or RENDER_DIR
        sys.path.insert(0, HERE)
        from rescue_scene import build, layout_for
        self.layout = layout_for(int(seed) >> 1)
        xml, decals, _obstacles, _shutters = build(self.layout)
        self.model = mujoco.MjModel.from_xml_string(xml, assets=decals)
        self.data = mujoco.MjData(self.model)
        self.nq, self.nmocap = int(self.model.nq), int(self.model.nmocap)
        self.cams = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("down", "forward")]
        self.scn = mujoco.MjvScene(self.model, 1)      # for mjv_updateCamera only
        self.cam = mujoco.MjvCamera()
        self.k = max(1, int(self.look["render"].get("supersample", 1)))
        self.tables = _lin16()
        self.view = None
        self.frames = 0
        self._start()

    # -- Filament
    def _start(self):
        for key in ("DISPLAY", "WAYLAND_DISPLAY"):
            os.environ.pop(key, None)
        os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
        if self.look["render"].get("lavapipe_threads"):
            os.environ["LP_NUM_THREADS"] = str(int(self.look["render"]["lavapipe_threads"]))
        import mujoco._render_filament as F
        from mujoco.experimental.studio import window  # noqa: F401  registers Filament's resource providers
        self.F = F
        look = self.look
        self.ctx = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN))
        self.ctx.set_clear_color([float(x) for x in look["render"].get("clear_color", [0.0, 0.0, 0.0])])
        self.fscn = self.ctx.create_scene(F.SceneParams())
        spec = self.mj.MjSpec()
        for k, v in sorted(look["filament_numeric"].items()):
            spec.add_numeric(name=k, data=[float(x) for x in (v if isinstance(v, (list, tuple)) else [v])])
        for k, v in sorted(look["filament_text"].items()):
            spec.add_text(name=k, data=str(v))
        spec.worldbody.add_geom(type=self.mj.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], pos=[0, 0, -50])
        self.look_model = spec.compile()
        for _ in range(2):                             # MuJoCo 3.16 applies the colour grading one configure late
            self.fscn.configure_from_model(self.look_model)
        path = os.path.join(self.dir, look["ibl"]["file"])
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            raise RenderError("renderer: the render bundle lacks %s: %s" % (path, exc))
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = len(raw), 0, F.PixelFormat.PIXEL_FORMAT_KTX
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_CUBE, F.ColorSpace.COLORSPACE_LINEAR
        self.ibl_tex = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(raw)
        self.ibl_tex.upload(td)
        lp = F.LightParams()
        lp.type, lp.texture, lp.color = F.LightType.LIGHT_IMAGE, self.ibl_tex, [1.0, 1.0, 1.0]
        lp.intensity = float(look["ibl"]["intensity"]) * float(look.get("exposure_scale", 1.0)) * float(self.layout.tone)
        self.ibl = self.ctx.create_light(lp)
        self.fscn.add_light(self.ibl)
        self._scene(*describe(self.model, look, layout=self.layout, seed=int(getattr(self, 'seed', 0))))
        size = SIZE * self.k
        self.buf = np.zeros((size, size, 3), np.uint8)
        self.tgt = self.ctx.create_render_target(F.RenderTargetConfig(color_format=F.PixelFormat.PIXEL_FORMAT_RGB8))
        self.tgt.resize(size, size)
        self.read = F.ReadRequest()
        self.read.target = self.tgt
        self.read.set_buffer(self.buf)

    def _picture(self, key, rule):
        """(what the key's surface is painted with: an RGB picture; whether the picture repeats; whether it glows)"""
        fwd, back = self.tables
        glow = bool(rule.get("glow"))
        if rule.get("flat"):                            # (review switches: one colour as it is)
            return np.tile(np.array(rule["flat"], np.uint8), (4, 4, 1)), False, glow
        if "model_tex" in rule:                         # a sign's or a scanned prop's own picture, from the model
            m, i = self.model, int(rule["model_tex"])
            w, h, nc = int(m.tex_width[i]), int(m.tex_height[i]), int(m.tex_nchannel[i])
            a = int(m.tex_adr[i])
            img = np.asarray(m.tex_data[a:a + w * h * nc]).reshape(h, w, nc)[::-1, :, :3]   # MuJoCo keeps rows bottom-up
            step = max(1, max(w, h) // 256)             # a scanned prop's picture, thinned to the cameras' need
            return np.ascontiguousarray(img[::step, ::step]), False, glow
        rgb = rule.get("rgb")
        if rule.get("map"):
            from PIL import Image
            path = os.path.join(self.dir, str(rule["map"]) + ".png")
            try:
                img = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
            except OSError as exc:
                raise RenderError("renderer: the render bundle lacks %s: %s" % (path, exc))
            if rule.get("tint") and rgb is not None:    # the model's colour (an sRGB value) under the map (a wear pattern), in linear light
                k2 = 2.0 if rule.get("norm2") else 1.0   # norm2: the map holds half its value (it varies round 1)
                q = np.array([int(round(65536 * k2 * float(rule.get("dim") or 1.0) * _linear(min(1.0, max(0.0, float(c)))))) for c in rgb[:3]], np.uint64)
                img = back[np.minimum((fwd[img].astype(np.uint64) * q) >> 16, 65535)]
            return img, "tile" in rule, glow
        if rule.get("apron"):                           # a door apron: worn paint over the floor's concrete, from the site's seed
            import rescue_marks as RM
            from PIL import Image
            path = os.path.join(self.dir, "mark_base.png")
            try:
                base = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
            except OSError as exc:
                raise RenderError("renderer: the render bundle lacks %s: %s" % (path, exc))
            return np.ascontiguousarray(RM.apron_picture(rule["apron"], int(rule.get("seed", 0)), base, 256)), False, glow
        c = [int(round(255 * min(1.0, max(0.0, float(x))))) for x in (rgb if rgb is not None else (0.5, 0.5, 0.5))[:3]]
        return np.tile(np.array(c, np.uint8), (4, 4, 1)), False, glow

    def _scene(self, standing, moving, rules):
        """One material for everything (the surfaces' pictures in one atlas, every rectangle of a repeating picture
        cut at the repeats), one mesh and one draw call for the whole standing world, one for each simulated body.
        Where two surfaces of one draw call meet, which of them a sample on the line shows is decided by their order
        in the mesh, the same in every process. Between two draw calls it is not (measured: with a draw call per
        surface kind, a few samples on such lines changed from process to process). The simulated bodies' draw
        calls meet nothing in a picture: the aircraft and a collet on it are outside its own cameras' view, a
        collet on its stand and the litter on a floor stand clear of what they rest on (a moving body's boxes are
        drawn 0.5 mm smaller), and the litter's fitting under the aircraft is outside the down camera's view."""
        F = self.F
        G = 2                                           # texels of margin round every picture in the atlas
        pictures, order = {}, []
        for batches in [standing] + [moving[b] for b in sorted(moving)]:
            for key in batches.keys:
                if key not in pictures:
                    pictures[key] = self._picture(key, rules[key])
                    order.append(key)
        # shelves: the pictures by falling height, in rows of at most 2048 texels
        place, x, y, row_h, W = {}, 0, 0, 0, 2048
        for key in sorted(order, key=lambda k: (-pictures[k][0].shape[0], order.index(k))):
            h, w = pictures[key][0].shape[:2]
            if x + w + 2 * G > W:
                x, y, row_h = 0, y + row_h, 0
            place[key] = (x + G, y + G, w, h)
            x += w + 2 * G
            row_h = max(row_h, h + 2 * G)
        H = y + row_h
        colour = np.zeros((H, W, 3), np.uint8)
        glowing = np.zeros((H, W, 3), np.uint8)
        for key in order:
            img, tiled, glow = pictures[key]
            x0, y0, w, h = place[key]
            pad = np.pad(img, ((G, G), (G, G), (0, 0)), mode="wrap" if tiled else "edge")
            (glowing if glow else colour)[y0 - G:y0 + h + G, x0 - G:x0 + w + G] = pad
            sun = rules[key].get("sun")
            if sun:                                     # a sunlit surface: its picture, and the same picture glowing
                fwd, back = self.tables
                q = np.uint64(int(round(65536 * float(sun))))
                glowing[y0 - G:y0 + h + G, x0 - G:x0 + w + G] = back[np.minimum((fwd[pad].astype(np.uint64) * q) >> np.uint64(16), 65535)]
        self.atlas = [self._texture8(colour), self._texture8(glowing)]

        def to_atlas(key, uv):
            x0, y0, w, h = place[key]
            if w <= 4:                                  # one colour: its middle
                return np.tile(np.array([(x0 + w / 2.0) / W, (y0 + h / 2.0) / H]), (len(uv), 1))
            return np.stack([(x0 + uv[:, 0] * w) / W, (y0 + uv[:, 1] * h) / H], 1)

        def cut(P4, uv):
            """A rectangle whose picture repeats, cut at the repeats: [(corners, coordinates within one repeat)]."""
            e = [(P4[1] - P4[0], uv[1] - uv[0]), (P4[3] - P4[0], uv[3] - uv[0])]
            stops = []
            for _dp, duv in e:
                j = int(np.argmax(np.abs(duv)))
                a0, a1 = float(uv[0][j]), float(uv[0][j] + duv[j])
                if abs(a1 - a0) < 1e-12:
                    stops.append([0.0, 1.0])
                    continue
                lo_, hi_ = min(a0, a1), max(a0, a1)
                ks = [k for k in range(int(math.floor(lo_)) + 1, int(math.ceil(hi_))) if lo_ + 1e-9 < k < hi_ - 1e-9]
                stops.append(sorted({0.0, 1.0} | {(k - a0) / (a1 - a0) for k in ks}))
            out = []
            for i in range(len(stops[0]) - 1):
                for j in range(len(stops[1]) - 1):
                    s0, s1, t0, t1 = stops[0][i], stops[0][i + 1], stops[1][j], stops[1][j + 1]
                    st = ((s0, t0), (s1, t0), (s1, t1), (s0, t1))
                    Q = np.array([P4[0] + s_ * e[0][0] + t_ * e[1][0] for s_, t_ in st])
                    U = np.array([uv[0] + s_ * e[0][1] + t_ * e[1][1] for s_, t_ in st])
                    U = U - np.floor(U.mean(0))         # into the repeat this piece lies in
                    out.append((Q, np.clip(U, 0.0, 1.0)))
            return out

        m = self.look["material"]
        self.material = F.Material(color=np.ones(4, np.float32), roughness=float(m["roughness"]), metallic=float(m["metallic"]),
                                   specular=float(m["specular"]), glossiness=0.05, reflectance=0.0, emissive=float(m["glow"]),
                                   color_texture=self.atlas[0], emissive_texture=self.atlas[1])
        A, Us, Ty = F.VertexAttribute, F.VertexAttributeUsage, F.VertexAttributeType
        attrs = [A(usage=Us.VERTEX_ATTRIBUTE_USAGE_POSITION, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_NORMAL, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT3),
                 A(usage=Us.VERTEX_ATTRIBUTE_USAGE_UV, type=Ty.VERTEX_ATTRIBUTE_TYPE_FLOAT2)]
        def soup(batches, keys):
            """(positions, normals, atlas coordinates) of the keys' triangles, three vertices a triangle, in drawing order"""
            V, N, UV = [], [], []
            for key in keys:
                tiled = pictures[key][1]
                for P4, n, uv in batches.quads[key]:
                    for Q, U in (cut(P4, uv) if tiled else [(P4, uv)]):
                        idx = [0, 1, 2, 0, 2, 3]
                        V.append(Q[idx])
                        N.append(np.tile(n, (6, 1)))
                        UV.append(to_atlas(key, U[idx]))
                for Vs, Ns, Us_ in batches.soups[key]:
                    V.append(Vs)
                    N.append(Ns)
                    UV.append(to_atlas(key, np.clip(Us_, 0.0, 1.0)))
            return tuple(np.concatenate(x).astype(np.float32) for x in (V, N, UV))

        def unit(P, Nn, T):
            nv = len(P)
            cfg = F.MeshConfig(max_vertices=nv, max_indices=nv, interleaved=False, index_type=F.IndexType.INDEX_TYPE_U32,
                               primitive_type=F.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES, attributes=attrs)
            mesh = self.ctx.create_mesh(cfg)
            md = F.MeshData(compute_bounds=True)
            md.set_vertices(cfg, np.concatenate([P.ravel(), Nn.ravel(), T.ravel()]).tobytes())
            md.set_indices(cfg, np.arange(nv, dtype=np.uint32).tobytes())
            mesh.upload(md)
            r = self.ctx.create_renderable(F.RenderableParams(cast_shadow=False, receive_shadow=True))
            r.set_mesh(mesh, 0, nv)
            r.set_material(self.material)
            return mesh, r, nv // 3

        # The standing world is one mesh and one draw call. A body that is placed, not simulated (the cargo
        # shutter), belongs to it where it stands: one standing mesh per position such a body has taken (two in a
        # flight: the shutter down, the shutter up). Each simulated body (the aircraft, the litter, a collet) is a
        # mesh and a draw call of its own, moved with its body.
        mj = self.mj
        self.world = soup(standing, standing.keys)
        self.placed = [(b, soup(moving[b], moving[b].keys)) for b in sorted(moving) if int(self.model.body_mocapid[b]) >= 0]
        self.make_unit = unit
        self.standing_units, self.standing_now = {}, None
        self.units, self.triangles = [], len(self.world[0]) // 3
        for b in sorted(moving):
            if int(self.model.body_mocapid[b]) >= 0:
                self.triangles += len(self.placed[[x[0] for x in self.placed].index(b)][1][0]) // 3
                continue
            mesh, r, n = unit(*soup(moving[b], moving[b].keys))
            self.fscn.add_renderable(r)
            self.units.append((b, mesh, r))
            self.triangles += n
        del mj

    def _stand(self):
        """The standing world's mesh for where the placed bodies are in this state, made at its first use."""
        key = b"".join(np.asarray(self.data.xpos[b], np.float64).tobytes() + np.asarray(self.data.xmat[b], np.float64).tobytes()
                       for b, _s in self.placed)
        if key == self.standing_now:
            return
        if key not in self.standing_units:
            parts = [self.world]
            for b, (P, Nn, T) in self.placed:
                R = np.asarray(self.data.xmat[b], np.float64).reshape(3, 3)
                parts.append(((P @ R.T + np.asarray(self.data.xpos[b], np.float64)).astype(np.float32),
                              (Nn @ R.T).astype(np.float32), T))
            self.standing_units[key] = self.make_unit(*(np.concatenate(x) for x in zip(*parts)))
        if self.standing_now is not None:
            self.fscn.remove_renderable(self.standing_units[self.standing_now][1])
        self.fscn.add_renderable(self.standing_units[key][1])
        self.standing_now = key

    def _texture8(self, img):
        F = self.F
        h, w = img.shape[:2]
        tc = F.TextureConfig()
        tc.width, tc.height, tc.format = w, h, F.PixelFormat.PIXEL_FORMAT_RGB8
        tc.sampler_type, tc.color_space = F.SamplerType.TEXTURE_2D, F.ColorSpace.COLORSPACE_SRGB
        t = self.ctx.create_texture(tc)
        td = F.TextureData()
        td.set_image(np.ascontiguousarray(img, np.uint8).tobytes())
        t.upload(td)
        return t

    def _camera(self, camid):
        mj = self.mj
        self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = int(camid)
        mj.mjv_updateCamera(self.model, self.data, self.cam, self.scn)
        a, b = self.scn.camera[0], self.scn.camera[1]
        pos = (np.asarray(a.pos, float) + np.asarray(b.pos, float)) / 2.0
        width = float(a.frustum_width)
        if width <= 0.0:
            width = (float(a.frustum_top) - float(a.frustum_bottom)) / 2.0
        F = self.F
        return F.Camera(pos=pos.astype(np.float32), forward=np.asarray(a.forward, np.float32), up=np.asarray(a.up, np.float32),
                        frustum_center=float(a.frustum_center), frustum_width=width, frustum_bottom=float(a.frustum_bottom),
                        frustum_top=float(a.frustum_top), frustum_near=float(a.frustum_near),
                        frustum_far=float(self.look["render"].get("far", 60.0)))

    def draw(self, state):
        """The two images (down, forward) of the state (qpos, then the mocap bodies' positions)."""
        mj, F = self.mj, self.F
        self.data.qpos[:] = state[:self.nq]
        if self.nmocap:
            self.data.mocap_pos[:] = np.asarray(state[self.nq:]).reshape(self.nmocap, 3)
        mj.mj_kinematics(self.model, self.data)
        mj.mj_camlight(self.model, self.data)
        self._stand()
        for b, _mesh, r in self.units:                 # the simulated bodies where this state has them
            r.set_pose(np.asarray(self.data.xpos[b], np.float32), np.asarray(self.data.xmat[b], np.float32))
        out = []
        rv = self.look["render"]
        size = SIZE * self.k
        for camid in self.cams:
            req = F.RenderRequest(camera=self._camera(camid), draw_mode=F.DrawMode.DRAW_MODE_DEFAULT,
                                  post_processing=bool(rv["post_processing"]), shadows=False, reflections=False)
            req.scene, req.target = self.fscn, self.tgt
            req.viewport.left, req.viewport.bottom, req.viewport.width, req.viewport.height = 0, 0, size, size
            self.ctx.wait_for_frame(self.ctx.render([req], [self.read]))
            if self.view is None:                      # the process's first image: switch the dithering off, again
                self._undither(size, size, False, bool(rv["post_processing"]))
                self.ctx.wait_for_frame(self.ctx.render([req], [self.read]))
            self.frames += 1
            out.append(self.buf.copy() if self.k == 1 else box_down(self.buf, self.k, self.tables))
        return out

    def _undither(self, width, height, shadows, post):
        lib = ctypes.CDLL(self.F.__file__)
        try:
            setd, getd = getattr(lib, self.VIEW_SET_DITHERING), getattr(lib, self.VIEW_GET_DITHERING)
        except AttributeError as exc:
            raise RenderError("renderer: MuJoCo's Filament module exports no View::setDithering: %s" % exc)
        setd.restype, setd.argtypes = None, [ctypes.c_void_p, ctypes.c_uint8]
        getd.restype, getd.argtypes = ctypes.c_uint8, [ctypes.c_void_p]
        want = (0, 0, width, height, 1, 1, int(shadows), 1, int(post))
        found, seen = set(), set()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.write.restype, libc.write.argtypes = ctypes.c_ssize_t, [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        rfd, wfd = os.pipe()

        def read(addr, n):                             # write(2) from an unreadable address fails (EFAULT)
            k = libc.write(wfd, addr, n)
            return os.read(rfd, k) if k > 0 else None

        try:
            level = [(id(self.fscn), 64)]
            for _depth in range(3):                    # breadth first: the view is found the same way in every process
                nxt = []
                for addr, n in level:
                    b = read(addr, n)
                    for (p,) in struct.iter_unpack("<Q", b[:len(b) // 8 * 8] if b else b""):
                        if p in seen or p < 0x10000 or p > 0x7fffffffffff or p % 8:
                            continue
                        seen.add(p)
                        v = None if p % 16 else read(p + 0x268, 0x4e)
                        if v and len(v) == 0x4e and struct.unpack_from("<iiII5B", v[:16] + v[0x49:0x4e]) == want:
                            found.add(p)
                        nxt.append((p, 512))
                level = nxt
        finally:
            os.close(rfd)
            os.close(wfd)
        if len(found) != 1:
            raise RenderError("renderer: cannot find the Filament view to switch its dithering off (%d candidates)" % len(found))
        self.view = found.pop()
        setd(self.view, 0)                             # filament::View::Dithering::NONE
        if getd(self.view) != 0:
            raise RenderError("renderer: Filament's dithering did not switch off")

    def close(self):
        """Every Filament object before the context (destroying the context first panics)."""
        for _b, _mesh, r in getattr(self, "units", []):
            self.fscn.remove_renderable(r)
        if getattr(self, "standing_now", None) is not None:
            self.fscn.remove_renderable(self.standing_units[self.standing_now][1])
        self.units, self.standing_units, self.standing_now, self.make_unit = [], {}, None, None
        self.material = None
        self.atlas = []
        self.read = self.tgt = None
        self.ibl = self.ibl_tex = None
        self.fscn = None
        self.look_model = None
        gc.collect()
        self.ctx = None
        gc.collect()


def _worker_main():
    try:
        ctypes.CDLL("libc.so.6").mallopt(-8, 2)       # M_ARENA_MAX: before Filament's and lavapipe's threads allocate
    except (OSError, AttributeError):
        pass
    out = os.fdopen(os.dup(1), "wb", buffering=0)     # the protocol's pipe; fd 1 goes to stderr for Filament's banner
    os.dup2(2, 1)
    inp = sys.stdin.buffer
    head = json.loads(inp.readline().decode())
    worker = Worker(int(head["seed"]))
    n = worker.nq + 3 * worker.nmocap
    out.write(MAGIC + struct.pack("<I", n))
    try:
        while True:
            raw = inp.read(8 * n)
            if len(raw) < 8 * n:
                break
            frames = worker.draw(np.frombuffer(raw, np.float64))
            out.write(frames[0].tobytes() + frames[1].tobytes())
    finally:
        worker.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker_main()
    else:
        sys.exit("usage: rescue_render.py --worker   (started by rescue.World)")
