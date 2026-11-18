"""Site packages of task 07: deterministic tar.xz archives, loading, validation and the site's frames (cell side;
DESIGN v1.2 section 2.5 "站点预先生成为确定性压缩包 tests/hidden/sites/*.tar.xz"; package P08).

A site package is an xz-compressed USTAR archive of regular files only, members sorted by name, every header
canonical (mode 0644, uid / gid 0, empty user / group names, mtime 0), so the same content always gives the same
bytes and `pack(unpack(blob)) == blob` (tests/test_world.py).  JSON members are canonical too (`dumps`: sorted keys,
compact separators, UTF-8, no NaN, trailing newline), so loading and re-serialising a site is also byte-identical.

Members:
    site.json          the site (schema subseacell-site/1; keys below)
    manifold.json      the structure description (gen/standin_manifold.py today; the measured asset's later, A01)
    panel_skid.json    the panel skid description (gen/panel_cad.py; one source with the drawings)
    textures/*.png     the tag plates (gen/tag_textures.py), 8-bit RGB, 1 mm per texel, row 0 = the plate's top edge

site.json keys (the world reads; P12's generator writes; JSON types):
    schema, label, role, seabed_depth_m, pose_o [x, y, z of the mudmat top], pose_psi            structure pose S in W
    structure {lx, ly, roof_z, mudmat_thk, ramp_m}                                                  flow envelope (P06)
    manifold, panel_skid: member file names; panel_frame_S {origin, X, Y, Z}: the panel frame P in S
    valves [{tag, branch, kind, key "c<j>r<r>", running_Nm, breakaway_Nm, theta0_deg, code_id, code_fouled}] (18)
    ports [{name, key "p<i>", kind, branch, code_id, code_fouled}] (8)
    tags [{of stem | port, key, code_id, text, fouled, texture, fouling: [prims] | null}] (one per skid tag)
    flow {U2, theta, t_slack, minor_ratio, f_inside, gm_coef, gm_tau, gm_seed}; src_W; plume {rate, w_b, seed};
    clutter {n, radius, ts_db, z_band, domain, seed}; water_c550; acoustic (null = from water_c550)            (P06)
    leak_branch, leak_p0_bar, leak_tau_s                                                                  process (P06)
    debris {kind D1..D4, X, D, psi}  stand in the panel frame (X along the face, D out of it, psi yaw to the face)
    cage {pos_w (sea-bed point under the inner box centre), yaw, code_id, texture}; stab {initial_port}
    hydro {label} | {set}: the vehicle's water-interaction draw (decision U1: one draw per task world)
    sensor_seed (int), battery_soc0; and pass-through blocks for the work order (chart, state_table, ...).

    files = SITE.unpack(blob); site = SITE.Site.from_files(files)        # or SITE.load(path)
    site.data, site.manifold, site.skid, site.texture(name), site.pose (S in W), site.panel_pose_w()
"""
from __future__ import annotations

import copy
import io
import json
import lzma
import math
import struct
import tarfile
import zlib
from typing import Dict, Optional

import numpy as np

SCHEMA = "subseacell-site/1"
SITE_JSON = "site.json"
MAX_MEMBER_BYTES = 64 << 20
MAX_TOTAL_BYTES = 256 << 20
XZ_PRESET = 6
TAR_RECORD = tarfile.RECORDSIZE


class SitePackageError(ValueError):
    pass


# ==================================================================================================== canonical JSON
def dumps(obj) -> bytes:
    """Canonical JSON bytes."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def loads(b: bytes):
    return json.loads(b.decode("utf-8"))


# ==================================================================================================== PNG (RGB, 8 bit)
def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_encode(img: np.ndarray) -> bytes:
    """Minimal deterministic PNG: 8-bit RGB, no interlace, filter 0 on every row, zlib level 9, no ancillary chunks."""
    a = np.ascontiguousarray(img, np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("png_encode wants (H, W, 3) uint8")
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")


def png_decode(b: bytes) -> np.ndarray:
    """Decoder for 8-bit RGB / RGBA PNGs without interlace (any filter type) -> (H, W, 3) uint8."""
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise SitePackageError("not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos < len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]
        tag = b[pos + 4:pos + 8]
        data = b[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat.append(data)
        elif tag == b"IEND":
            break
    if ihdr is None:
        raise SitePackageError("PNG without IHDR")
    w, h, depth, ctype, _, _, inter = ihdr
    if depth != 8 or ctype not in (2, 6) or inter != 0:
        raise SitePackageError("only 8-bit RGB / RGBA, non-interlaced PNGs")
    nc = 3 if ctype == 2 else 4
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * nc)
    out = np.zeros((h, w * nc), np.int32)
    prev = np.zeros(w * nc, np.int32)
    for r in range(h):
        f, line = int(raw[r, 0]), raw[r, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        else:
            cur = np.zeros_like(line)
            for x in range(w * nc):
                a = int(cur[x - nc]) if x >= nc else 0
                c = int(prev[x - nc]) if x >= nc else 0
                up = int(prev[x])
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + up) >> 1
                elif f == 4:
                    pa, pb, pc = abs(up - c), abs(a - c), abs(a + up - 2 * c)
                    p = a if (pa <= pb and pa <= pc) else (up if pb <= pc else c)
                else:
                    raise SitePackageError(f"PNG filter {f}")
                cur[x] = (int(line[x]) + p) & 255
        out[r] = cur
        prev = cur
    return out.reshape(h, w, nc)[..., :3].astype(np.uint8)


# ==================================================================================================== the archive
def _safe_name(name: str) -> bool:
    return (bool(name) and not name.startswith("/") and "\\" not in name and ".." not in name.split("/")
            and len(name.encode()) < 100 and all(32 <= ord(c) < 127 for c in name))


def pack(files: Dict[str, bytes]) -> bytes:
    """Deterministic tar.xz of {name: bytes} (regular files only, sorted, canonical headers)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(files):
            if not _safe_name(name):
                raise SitePackageError(f"bad member name {name!r}")
            data = bytes(files[name])
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.type = tarfile.REGTYPE
            tf.addfile(ti, io.BytesIO(data))
    return lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=XZ_PRESET)


def unpack(blob: bytes, strict: bool = True) -> Dict[str, bytes]:
    """{name: bytes} of a site package.  strict: the archive must be canonical (what pack() writes)."""
    try:
        raw = lzma.decompress(blob, format=lzma.FORMAT_XZ)
    except lzma.LZMAError as e:
        raise SitePackageError(f"not an xz stream: {e}") from None
    if len(raw) > MAX_TOTAL_BYTES:
        raise SitePackageError("package too large")
    out, names = {}, []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        for ti in tf.getmembers():
            if not ti.isreg():
                raise SitePackageError(f"member {ti.name!r} is not a regular file")
            if not _safe_name(ti.name):
                raise SitePackageError(f"bad member name {ti.name!r}")
            if ti.size > MAX_MEMBER_BYTES:
                raise SitePackageError(f"member {ti.name!r} too large")
            if strict and (ti.mode != 0o644 or ti.mtime != 0 or ti.uid or ti.gid or ti.uname or ti.gname
                           or ti.pax_headers):
                raise SitePackageError(f"member {ti.name!r}: non-canonical header")
            if ti.name in out:
                raise SitePackageError(f"duplicate member {ti.name!r}")
            out[ti.name] = tf.extractfile(ti).read()
            names.append(ti.name)
    if strict and names != sorted(names):
        raise SitePackageError("members not sorted")
    return out


def load(path_or_blob) -> "Site":
    blob = path_or_blob
    if isinstance(path_or_blob, str):
        with open(path_or_blob, "rb") as f:
            blob = f.read()
    return Site.from_files(unpack(blob))


# ==================================================================================================== frames
class SitePose:
    """Structure frame S in W: origin (mudmat top at the structure centre) and heading psi (rotation about +z)."""

    def __init__(self, origin_w, psi):
        self.o = np.asarray(origin_w, float).reshape(3)
        self.psi = float(psi)
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_w(self, p_s):
        return np.asarray(p_s, float) @ self.R.T + self.o

    def to_s(self, p_w):
        return (np.asarray(p_w, float) - self.o) @ self.R

    def vec_to_w(self, v_s):
        return np.asarray(v_s, float) @ self.R.T


def quat_of(R) -> np.ndarray:
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float)).reshape(-1))
    return q


# ==================================================================================================== the site
REQUIRED = ("schema", "label", "seabed_depth_m", "pose_o", "pose_psi", "structure", "manifold", "panel_skid",
            "panel_frame_S", "valves", "ports", "tags", "flow", "src_W", "plume", "water_c550", "leak_branch",
            "leak_p0_bar", "leak_tau_s", "debris", "cage", "stab", "hydro", "sensor_seed")


class Site:
    def __init__(self, data: dict, manifold: dict, skid: dict, files: Dict[str, bytes]):
        self.data = data
        self.manifold = manifold
        self.skid = skid
        self.files = files
        self._tex = {}
        self.pose = SitePose(data["pose_o"], data["pose_psi"])

    # -------------------------------------------------------------------------------------- construction
    @classmethod
    def from_files(cls, files: Dict[str, bytes], validate: bool = True) -> "Site":
        if SITE_JSON not in files:
            raise SitePackageError("package without site.json")
        data = loads(files[SITE_JSON])
        if data.get("schema") != SCHEMA:
            raise SitePackageError(f"site schema {data.get('schema')!r}, want {SCHEMA!r}")
        for k in REQUIRED:
            if k not in data:
                raise SitePackageError(f"site.json lacks {k!r}")
        for k in ("manifold", "panel_skid"):
            if data[k] not in files:
                raise SitePackageError(f"member {data[k]!r} ({k}) missing")
        s = cls(data, loads(files[data["manifold"]]), loads(files[data["panel_skid"]]), dict(files))
        if validate:
            probs = s.problems()
            if probs:
                raise SitePackageError("; ".join(probs))
        return s

    def to_files(self) -> Dict[str, bytes]:
        """Re-serialise (canonical JSON; textures as stored)."""
        out = dict(self.files)
        out[SITE_JSON] = dumps(self.data)
        out[self.data["manifold"]] = dumps(self.manifold)
        out[self.data["panel_skid"]] = dumps(self.skid)
        return out

    def to_package(self) -> bytes:
        return pack(self.to_files())

    def dict(self) -> dict:
        """The site dict as the cell's modules read it (P06 / P07 keys), with the structure inlined."""
        d = copy.deepcopy(self.data)
        d["structure"] = dict(self.data["structure"])
        return d

    # -------------------------------------------------------------------------------------- checks
    def problems(self) -> list:
        d, out = self.data, []
        mt = float(d["structure"]["mudmat_thk"])
        if abs(float(d["pose_o"][2]) - (-float(d["seabed_depth_m"]) + mt)) > 1e-6:
            out.append("pose_o z != -seabed_depth + mudmat_thk")
        tags = [v["tag"] for v in d["valves"]]
        keys = [v["key"] for v in d["valves"]]
        stem_keys = {s["key"] for s in self.skid["stems"]}
        if len(d["valves"]) != 18 or len(set(tags)) != 18 or set(keys) != stem_keys:
            out.append("valves: 18 distinct tags on the 18 stems of the skid")
        want = {f"{k}V{b}" for k in "BPT" for b in range(1, 7)}
        if set(tags) != want:
            out.append("valve tags must be BV1..TV6")
        for v in d["valves"]:
            if not (0.0 < float(v["running_Nm"]) <= float(v["breakaway_Nm"])):
                out.append(f"{v['tag']}: torques")
            if float(v["theta0_deg"]) not in (0.0, 90.0):
                out.append(f"{v['tag']}: as-found angle must be 0 or 90")
        port_keys = {p["key"] for p in self.skid["ports"]}
        if {p["key"] for p in d["ports"]} != port_keys or len(d["ports"]) != len(port_keys):
            out.append("ports: one entry per skid port")
        codes = [v["code_id"] for v in d["valves"]] + [p["code_id"] for p in d["ports"]] + [d["cage"]["code_id"]]
        if len(set(codes)) != len(codes) or not all(0 <= int(c) < 48 for c in codes):
            out.append("code ids must be distinct and in 0..47")
        tag_keys = {(t["of"], t["key"]) for t in self.skid["tags"]}
        site_tags = {(t["of"], t["key"]) for t in d["tags"]}
        if tag_keys != site_tags:
            out.append("tags: one entry per skid tag plate")
        for t in d["tags"] + [d["cage"]]:
            fn = t.get("texture")
            if fn not in self.files:
                out.append(f"texture {fn!r} missing")
        if sum(1 for t in d["tags"] if t.get("fouled")) > 4:
            out.append("more than 4 fouled codes")
        if d["stab"].get("initial_port") not in {p["name"] for p in d["ports"]}:
            out.append("stab initial port")
        if d["debris"]["kind"] not in ("D1", "D2", "D3", "D4"):
            out.append("debris kind")
        if not 1 <= int(d["leak_branch"]) <= 6:
            out.append("leak branch")
        return out

    # -------------------------------------------------------------------------------------- data access
    def texture(self, name: str) -> np.ndarray:
        if name not in self._tex:
            self._tex[name] = png_decode(self.files[name])
        return self._tex[name]

    def panel_pose_w(self):
        """(origin, R) of the panel frame P in W (R columns = X, Y, Z of P)."""
        pf = self.data["panel_frame_S"]
        R_sp = np.stack([np.asarray(pf["X"], float), np.asarray(pf["Y"], float), np.asarray(pf["Z"], float)], 1)
        return self.pose.to_w(np.asarray(pf["origin"], float)), self.pose.R @ R_sp

    def debris_pose_w(self):
        """Pose of the stand's body frame (x along its long side, y away from the panel, z up; origin = footprint
        centre on the mudmat plane) from the site's (X, D, psi) in the panel frame (P02's convention)."""
        o, R = self.panel_pose_w()
        db = self.data["debris"]
        X, D, psi = float(db["X"]), float(db["D"]), float(db["psi"])
        eX, eY, eZ = R[:, 0], R[:, 1], R[:, 2]
        pos = o + X * eX + D * eZ
        xb = -math.cos(psi) * eX - math.sin(psi) * eZ
        yb = -math.sin(psi) * eX + math.cos(psi) * eZ
        Rb = np.stack([xb, yb, np.cross(xb, yb)], 1)
        return pos, quat_of(Rb), Rb

    def valve_by_key(self) -> dict:
        return {v["key"]: v for v in self.data["valves"]}

    def port_by_key(self) -> dict:
        return {p["key"]: p for p in self.data["ports"]}

    def hydro_draw(self):
        """(HydroCoeffs or set name, info) of the vehicle's water-interaction draw."""
        import hashlib
        h = self.data["hydro"]
        if "set" in h:
            return h["set"], dict(set=h["set"])
        from uwkit.params import sample_hydro
        seed = int.from_bytes(hashlib.sha256(h["label"].encode()).digest()[:8], "little")
        coeffs, info = sample_hydro(np.random.default_rng(seed))
        return coeffs, dict(label=h["label"], lam_A=info["lam_A"], lam_D=info["lam_D"])
