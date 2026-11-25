"""The heap library as the generator uses it: the registry, the pieces' collision grids and meshes, and the placement
transform (turn, mirror, uniform scale) from a piece's own frame into the site.

Registry: a JSON file (path: environment variable W4GEN_HEAP_REGISTRY, default REGISTRY_DEFAULT, the
maintainers' registry file, not published). Two layouts are read, told apart by the registry's own version:
  version 3 (doc.version == 3): `doc` holds `classes` (per class w, d, top and the fit line `fit`), `lib_root` and
      `version`; every piece record has `cls`, `rect` (the registered rectangle), `scale.interval`, `line.max_m` (the
      registered gap between the heap and its line), `files`, `trials` and `verdict`; the top-level list `usable` names
      the pieces the library offers;
  version 2 (top-level `classes`, `lib_root`, `version`): `class.rect_registered`, `scale_interval`,
      `mesh.line_gap_max_m`, `shape.has_vertical_faces`, `trials_ok`.
Environment variable W4GEN_HEAP_LIB overrides lib_root (the folder of the piece files). Every piece file the generator
reads (hfield.npy, hfield.json, agent.npz; nothing else: the spill sheet at a heap's foot is a picture-only layer and
is never read, so it is in neither the package nor the collision) is checked against the sha256 the registry gives for
it; a mismatch stops the generator. The sha256 of the registry file itself goes into every package, so a package says
which library made it without naming a piece.

Each piece's trial verdict as the registry states it (version 3: trials.ok, the four lines trials.lines a-d,
verdict.trials, and the piece in `usable`; version 2: trials_ok) is kept with the piece (`trials_verdict`); the class
geometry check of mounds of fill (heapchecks, G20) requires it. The scale interval the generator draws from is the
registered one, mounds of fill from 1.0 up, cut so that the scaled registered rectangle stays inside the class's w and d
ranges (`scale_cut`; the registry judges the class bounds on its own 2.5 mm lattice, the generator on the rectangle).

Piece frame (the registry's own convention): x along the line the heap leans on, centred; y in [0, D] with the line
at y = D (north); z up, floor z = 0. Registered rectangle [-W/2, W/2] x [0, D]. Collision grid: 5 mm, rows run +y,
heights above the grid's base at z = -0.010 m, a ring of zero cells, max == size_z.

Placement in the site (yaw 0: the line is north of the heap; yaw 180: the line is south of it; mirror: x -> -x in the
piece frame first; sigma: uniform scale about the floor):
    yaw 0    x = cx + s sigma xp,   y = yL - sigma (D - yp)
    yaw 180  x = cx - s sigma xp,   y = yL + sigma (D - yp)          (s = -1 when mirrored, else +1), z = sigma zp
The site's collision piece (collision_grid) is the piece's own grid under the same turn, mirror and scale, vertex
for vertex: its cells are 5 mm x sigma in the site, its heights H = max(0, sigma Hp - 0.010 (sigma - 1)) above the
base at -0.010 m (the field stays sunk 10 mm, cells outside the outline stay below the floor); the outermost ring is
set to zero and size_z = max(H). (Resampling the piece onto the site's 5 mm lattice instead was measured: bilinear
resampling loses the fit bound, 97.6-98.9 % of the points within 3 mm on 8 of 9 heaps, while the piece's own grid
keeps it, 99.2-100 %.) For the generator's rules a second, 5 mm grid on the site's lattice (lattice_grid: the piece's
grid bilinear at the inverse-mapped vertices, same height formula) gives the floor occupancy.
"""
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
REGISTRY_DEFAULT = os.path.join(DEV_DIR, "art", "heaps", "registry_v3.json")
GRID = 0.005
SINK = 0.010
MARGIN = 0.010                 # the site grid covers the placed rectangle plus this on every side
M_SCALE_MIN = 1.0              # mounds of fill are only scaled up (their core margin under a push is thin)
LINE_GAP_LIMIT = 0.15          # pieces whose registered gap to their line reaches this are not used
READS = ("hfield.npy", "hfield.json", "agent.npz")      # the only piece files the generator opens


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_layout(reg):
    """3 for the version 3 layout (doc.version == 3), 2 for the version 2 layout (top-level classes)."""
    doc = reg.get("doc")
    if isinstance(doc, dict) and doc.get("version") == 3:
        return 3
    if isinstance(reg.get("classes"), dict) and "pieces" in reg:
        return 2
    raise RuntimeError("heap library: unknown registry layout (neither doc.version 3 nor top-level classes)")


def _piece_v3(pid, rec, usable):
    tr = rec.get("trials") or {}
    lines = tr.get("lines") or {}
    lines_ok = all(lines.get(q) is True for q in "abcd")
    verdict = dict(trials_ok=tr.get("ok") is True, lines=dict((q, lines.get(q)) for q in "abcd"), lines_ok=lines_ok,
                   verdict_trials=(rec.get("verdict") or {}).get("trials") is True, usable=pid in usable)
    verdict["ok"] = bool(verdict["trials_ok"] and lines_ok and verdict["verdict_trials"] and verdict["usable"])
    return dict(cls=rec["cls"], rect=tuple(float(q) for q in rec["rect"]),
                scale=[float(q) for q in rec["scale"]["interval"]], line_gap_m=(rec.get("line") or {}).get("max_m"),
                has_vertical_faces=None, trials_verdict=verdict, files=rec["files"])


def _piece_v2(pid, rec):
    verdict = dict(trials_ok=rec.get("trials_ok") is True)
    verdict["ok"] = bool(verdict["trials_ok"])
    return dict(cls=rec["cls"], rect=tuple(float(q) for q in rec["class"]["rect_registered"]),
                scale=[float(q) for q in rec["scale_interval"]], line_gap_m=rec.get("mesh", {}).get("line_gap_max_m"),
                has_vertical_faces=bool(rec.get("shape", {}).get("has_vertical_faces")), trials_verdict=verdict,
                files=rec["files"])


class Library:
    """Pieces of the registry that the generator may use, by class, in id order."""

    def __init__(self, registry=None, lib_root=None):
        self.path = os.path.abspath(registry or os.environ.get("W4GEN_HEAP_REGISTRY") or REGISTRY_DEFAULT)
        self.sha256 = sha256_file(self.path)
        with open(self.path, encoding="utf-8") as fh:
            self.reg = json.load(fh)
        self.layout = registry_layout(self.reg)
        if self.layout == 3:
            doc = self.reg["doc"]
            self.classes, self.version, root = doc["classes"], doc["version"], doc["lib_root"]
            self.usable = list(self.reg.get("usable") or [])
        else:
            self.classes, self.version, root = self.reg["classes"], self.reg.get("version"), self.reg["lib_root"]
            self.usable = None
        self.root = lib_root or os.environ.get("W4GEN_HEAP_LIB") or root
        self.pieces = {}
        self.excluded = {}
        self.files_read = set()
        for pid in sorted(self.reg["pieces"]):
            rec = self.reg["pieces"][pid]
            p = _piece_v3(pid, rec, self.usable) if self.layout == 3 else _piece_v2(pid, rec)
            gap = p["line_gap_m"]
            if gap is not None and gap >= LINE_GAP_LIMIT:
                self.excluded[pid] = "registered line gap %.3f m >= %.2f m" % (gap, LINE_GAP_LIMIT)
                continue
            lo, hi = p["scale"]
            x0, x1, y0, y1 = p["rect"]
            W, D = x1 - x0, y1 - y0
            C = self.classes[p["cls"]]
            cut = {}
            if p["cls"] == "M" and lo < M_SCALE_MIN:
                lo, cut["lo"] = M_SCALE_MIN, "mounds of fill from 1.0 up"
            for key, ext in (("w", W), ("d", D)):
                if ext * hi > float(C[key][1]):
                    hi, cut["hi"] = float(C[key][1]) / ext, "rectangle %s x scale <= the class's %.2f m" % (key, C[key][1])
                if ext * lo < float(C[key][0]):
                    lo, cut["lo"] = float(C[key][0]) / ext, "rectangle %s x scale >= the class's %.2f m" % (key, C[key][0])
            if hi < lo:
                self.excluded[pid] = "no admissible scale"
                continue
            self.pieces[pid] = dict(id=pid, cls=p["cls"], rect=(x0, x1, y0, y1), W=W, D=D, scale=(lo, hi),
                                    scale_registered=list(p["scale"]), scale_cut=cut,
                                    has_vertical_faces=p["has_vertical_faces"], line_gap_m=gap,
                                    trials_verdict=p["trials_verdict"], files=p["files"])
        self.by_cls = {c: [p for p in sorted(self.pieces) if self.pieces[p]["cls"] == c] for c in ("R", "M")}
        self._grid = {}
        self._mesh = {}

    def _file(self, pid, name):
        if name not in READS:
            raise RuntimeError("heap library: the generator does not read %s" % name)
        p = os.path.join(self.root, pid, name)
        self.files_read.add(name)
        want = self.pieces[pid]["files"].get(name)
        got = sha256_file(p)
        if want != got:
            raise RuntimeError("heap library: %s/%s does not match the registry (%s != %s)" % (pid, name, got, want))
        return p

    def grid(self, pid):
        """(H, meta) of a piece's collision grid (checked: zero ring, max == size_z, 5 mm)."""
        if pid not in self._grid:
            H = np.load(self._file(pid, "hfield.npy")).astype(np.float64)
            with open(self._file(pid, "hfield.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            ring = np.concatenate([H[0, :], H[-1, :], H[:, 0], H[:, -1]])
            if np.any(ring != 0.0) or abs(float(H.max()) - float(meta["size_z"])) > 1e-9:
                raise RuntimeError("heap library: %s grid lacks the zero ring or max != size_z" % pid)
            if (abs(float(meta["grid"]) - GRID) > 1e-12 or abs(float(meta["base_z"]) + SINK) > 1e-12
                    or meta.get("against", "N") != "N"):
                raise RuntimeError("heap library: %s grid is not the 5 mm, -0.010 m, line-north convention" % pid)
            self._grid[pid] = (H, meta)
        return self._grid[pid]

    def mesh(self, pid):
        """(V, F) of the piece's agent-view mesh (the depth twin's mesh) in the piece frame."""
        if pid not in self._mesh:
            z = np.load(self._file(pid, "agent.npz"), allow_pickle=False)
            self._mesh[pid] = (np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64))
        return self._mesh[pid]


# ------------------------------------------------------------------------------------------------ transforms
def placed_rect(piece, sigma, yaw, cx, y_line):
    W, D = sigma * piece["W"], sigma * piece["D"]
    if yaw == 0:
        return (cx - W / 2, cx + W / 2, y_line - D, y_line)
    return (cx - W / 2, cx + W / 2, y_line, y_line + D)


def to_piece(x, y, piece, place):
    """Inverse map: site (x, y) -> piece frame (xp, yp)."""
    s = -1.0 if place["mirror"] else 1.0
    sig, cx, yl, D = place["scale"], place["cx"], place["y_line"], piece["D"]
    if place["yaw_deg"] == 0:
        return s * (x - cx) / sig, D - (yl - y) / sig
    return -s * (x - cx) / sig, D - (y - yl) / sig


def to_site(P, piece, place):
    """Forward map of piece-frame points P (n x 3) into the site."""
    s = -1.0 if place["mirror"] else 1.0
    sig, cx, yl, D = place["scale"], place["cx"], place["y_line"], piece["D"]
    P = np.asarray(P, np.float64)
    out = np.empty_like(P)
    if place["yaw_deg"] == 0:
        out[:, 0] = cx + s * sig * P[:, 0]
        out[:, 1] = yl - sig * (D - P[:, 1])
    else:
        out[:, 0] = cx - s * sig * P[:, 0]
        out[:, 1] = yl + sig * (D - P[:, 1])
    out[:, 2] = sig * P[:, 2]
    return out


def bilinear(H, meta, xp, yp):
    """The piece grid's height above its base at piece-frame points (0 outside the grid)."""
    gr = float(meta["grid"])
    ox, oy = (float(q) for q in meta["origin"])
    ny, nx = H.shape
    fj = (np.asarray(xp, np.float64) - ox) / gr
    fi = (np.asarray(yp, np.float64) - oy) / gr
    inside = (fi >= 0) & (fi <= ny - 1) & (fj >= 0) & (fj <= nx - 1)
    fi_c = np.clip(fi, 0, ny - 1)
    fj_c = np.clip(fj, 0, nx - 1)
    i0 = np.minimum(np.floor(fi_c).astype(np.int64), ny - 2)
    j0 = np.minimum(np.floor(fj_c).astype(np.int64), nx - 2)
    ti, tj = fi_c - i0, fj_c - j0
    v = (H[i0, j0] * (1 - ti) * (1 - tj) + H[i0 + 1, j0] * ti * (1 - tj) + H[i0, j0 + 1] * (1 - ti) * tj
         + H[i0 + 1, j0 + 1] * ti * tj)
    return np.where(inside, v, 0.0)


def collision_grid(lib, pid, place):
    """The collision grid of one placed heap: the piece's grid turned, mirrored and scaled vertex for vertex;
    dict(H, origin, shape, grid = cell size, size_z, base_z, top_m); rows run +y, columns +x."""
    piece = lib.pieces[pid]
    H, meta = lib.grid(pid)
    sig, gr = place["scale"], float(meta["grid"])
    ox, oy = (float(q) for q in meta["origin"])
    ny, nx = H.shape
    s = -1.0 if place["mirror"] else 1.0
    sx = s if place["yaw_deg"] == 0 else -s
    Hs = np.maximum(0.0, sig * H - SINK * (sig - 1.0))
    xs = place["cx"] + sx * sig * (ox + gr * np.arange(nx))
    if place["yaw_deg"] == 0:
        ys = place["y_line"] - sig * (piece["D"] - (oy + gr * np.arange(ny)))
    else:
        ys = place["y_line"] + sig * (piece["D"] - (oy + gr * np.arange(ny)))
    if xs[1] < xs[0]:
        Hs, xs = Hs[:, ::-1], xs[::-1]
    if ys[1] < ys[0]:
        Hs, ys = Hs[::-1, :], ys[::-1]
    Hs = np.ascontiguousarray(Hs)
    ring_max = float(max(Hs[0, :].max(), Hs[-1, :].max(), Hs[:, 0].max(), Hs[:, -1].max()))
    Hs[0, :] = Hs[-1, :] = 0.0
    Hs[:, 0] = Hs[:, -1] = 0.0
    sz = float(Hs.max())
    return dict(H=Hs, origin=[float(xs[0]), float(ys[0])], shape=[ny, nx], size_z=sz, base_z=-SINK, grid=sig * gr,
                ring_value_before_zeroing_m=ring_max, top_m=sz - SINK)


def lattice_grid(lib, pid, place):
    """The 5 mm grid on the site's lattice of one placed heap (the generator's occupancy): dict(H, origin, shape,
    size_z, ...); rows run +y."""
    piece = lib.pieces[pid]
    H, meta = lib.grid(pid)
    x0, x1, y0, y1 = place["rect"]
    gx0 = math.floor(round((x0 - MARGIN) / GRID, 6)) * GRID
    gx1 = math.ceil(round((x1 + MARGIN) / GRID, 6)) * GRID
    gy0 = math.floor(round((y0 - MARGIN) / GRID, 6)) * GRID
    gy1 = math.ceil(round((y1 + MARGIN) / GRID, 6)) * GRID
    nx = int(round((gx1 - gx0) / GRID)) + 1
    ny = int(round((gy1 - gy0) / GRID)) + 1
    xs = gx0 + GRID * np.arange(nx)
    ys = gy0 + GRID * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    xp, yp = to_piece(X, Y, piece, place)
    sig = place["scale"]
    Hs = np.maximum(0.0, sig * bilinear(H, meta, xp, yp) - SINK * (sig - 1.0))
    ring_max = float(max(Hs[0, :].max(), Hs[-1, :].max(), Hs[:, 0].max(), Hs[:, -1].max()))
    Hs[0, :] = Hs[-1, :] = 0.0
    Hs[:, 0] = Hs[:, -1] = 0.0
    sz = float(Hs.max())
    return dict(H=Hs, origin=[gx0, gy0], shape=[ny, nx], size_z=sz, base_z=-SINK, grid=GRID,
                ring_value_before_zeroing_m=ring_max, top_m=sz - SINK)
