"""The shift looks' cache: every parcel's look (mesh, 512 px maps, waybill record) made once and kept on disk, so that
a shift's cell starts in seconds.

A parcel's look is a pure function of its kit inputs: the kit's own code (``kit_digest()``: the sha256 of every file of
``parcelcell/parcelkit``), the parcel's kind, size, label face and label offset, its kit seed (from the shift's seed
and the parcel's index) and its waybill identity (from the shift's seed), and the quality. The cache's key is the
sha256 of exactly these, so a hit gives the bytes a fresh build would give (the kit's numerics pins hold for both:
the same process environment as the cell's). One file per parcel (``<key>.npz``: the mesh, the maps and the record as
JSON), written to a temporary name and renamed.

    from parcelcell import kitcache
    kitcache.prebuild(["shift_a.json", "shift_b.json"], "/some/cache", workers=4)    # before the shifts run
    # the cell's parcel visuals then read the cache (config look.kit_cache_dir, or $PARCELCELL_KIT_CACHE)

``prebuild`` runs its misses in worker processes (fresh, isolated interpreters running ``kitcache._worker``: no fork
of a process that holds a renderer), which inherit the process's numerics environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

__all__ = ["kit_digest", "entry_key", "load", "store", "parcel_jobs", "prebuild", "cache_dir"]

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_DIR = os.path.join(HERE, "parcelkit")
FORMAT = "parcelcell.kitcache/1"
_DIGEST = None


def kit_digest() -> str:
    """sha256 over the kit's files (relative path and bytes, sorted; caches excluded)."""
    global _DIGEST
    if _DIGEST is None:
        h = hashlib.sha256(FORMAT.encode())
        for dp, dns, fns in os.walk(KIT_DIR):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                h.update(os.path.relpath(p, KIT_DIR).encode() + b"\0")
                with open(p, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
        _DIGEST = h.hexdigest()
    return _DIGEST


def cache_dir(options=None):
    """The configured cache folder: options["kit_cache_dir"], else $PARCELCELL_KIT_CACHE, else None (no cache)."""
    o = options or {}
    d = o.get("kit_cache_dir") or os.environ.get("PARCELCELL_KIT_CACHE")
    return os.path.abspath(d) if d else None


def entry_key(spec_d: dict, pseed: int, identity: dict, quality: str) -> str:
    doc = {"kit": kit_digest(), "spec": {"kind": str(spec_d["kind"]), "size": [float(v) for v in spec_d["size"]],
                                         "label_face": int(spec_d["label_face"]),
                                         "label_uv": [float(v) for v in spec_d["label_uv"]]},
           "seed": int(pseed), "identity": identity, "quality": str(quality)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path(cdir, key):
    return os.path.join(cdir, key[:2], key + ".npz")


def load(cdir, key):
    """(record without arrays, arrays {V, N, UV, F, tex}) or None."""
    if not cdir:
        return None
    p = _path(cdir, key)
    try:
        with np.load(p, allow_pickle=False) as z:
            rec = json.loads(bytes(z["record"]).decode())
            tex = {k[4:]: z[k] for k in z.files if k.startswith("tex_")}
            arr = {"V": z["V"], "N": z["N"], "UV": z["UV"], "F": z["F"], "tex": tex}
    except (OSError, KeyError, ValueError):
        return None
    if rec.get("_cache_key") != key:
        return None
    rec.pop("_cache_key", None)
    return rec, arr


def store(cdir, key, rec_json: dict, arr: dict) -> bool:
    """Write one entry (temporary file, then rename); False when the folder cannot be written."""
    if not cdir:
        return False
    p = _path(cdir, key)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        doc = dict(rec_json, _cache_key=key)
        buf = io.BytesIO()
        tex = {"tex_" + k: np.ascontiguousarray(v) for k, v in arr["tex"].items()}
        np.savez_compressed(buf, record=np.frombuffer(json.dumps(doc, sort_keys=True).encode(), np.uint8),
                 V=np.asarray(arr["V"]), N=np.asarray(arr["N"]), UV=np.asarray(arr["UV"]), F=np.asarray(arr["F"]), **tex)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".tmp-", suffix=".npz")
        with os.fdopen(fd, "wb") as fh:
            fh.write(buf.getvalue())
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def parcel_jobs(shift, quality="agent"):
    """[(index, spec dict, kit seed, identity dict, key)] of a shift's parcels (the cell's own derivation)."""
    from .parcelkit.identity import identities
    from .visuals import KitVisuals
    seed = KitVisuals.shift_seed(shift)
    ids = identities(len(shift.arrivals), seed)
    out = []
    for i, (a, ident) in enumerate(zip(shift.arrivals, ids)):
        p = a.parcel
        spec_d = {"kind": p.kind, "size": tuple(p.size), "label_face": int(p.label_face),
                  "label_uv": tuple(p.label_uv)}
        pseed = (seed * 1000003 + i) & 0x7FFFFFFF
        idd = ident.to_dict()
        out.append((i, spec_d, pseed, idd, entry_key(spec_d, pseed, idd, quality)))
    return out


def build_entry(spec_d, pseed, ident_d, quality, name):
    """A fresh build (the cell's own call), as (record JSON, arrays)."""
    import cv2
    from .parcelkit.identity import WaybillIdentity
    from .parcelkit.parcel import make_parcel, strip
    opt = cv2.useOptimized()
    cv2.setUseOptimized(False)          # the plain kernels give the same bytes on every host
    try:
        rec = make_parcel(spec_d, pseed, (quality,), None, name=name, identity=WaybillIdentity.from_dict(ident_d),
                          check=False)
    finally:
        cv2.setUseOptimized(opt)
    arr = rec["_arrays"][quality]
    return json.loads(json.dumps(strip(rec))), arr


def prebuild(packages, cdir, workers=4, quality="agent", log=None):
    """Fill the cache for the parcels of the given shift packages (paths or dicts) that it lacks. Returns
    {"parcels", "hits", "built", "wall_s", "workers"}."""
    from .shiftgen import from_package
    log = log or (lambda s: None)
    t0 = time.time()
    os.makedirs(cdir, exist_ok=True)
    jobs, seen, n = [], set(), 0
    for pk in packages:
        if not isinstance(pk, dict):
            with open(pk) as fh:
                pk = json.load(fh)
        shift, _ = from_package(pk)
        for i, spec_d, pseed, idd, key in parcel_jobs(shift, quality):
            n += 1
            if key in seen or os.path.isfile(_path(cdir, key)):
                continue
            seen.add(key)
            jobs.append({"spec": {k: list(v) if isinstance(v, tuple) else v for k, v in spec_d.items()},
                         "seed": pseed, "identity": idd, "key": key, "name": "parcel%d" % i, "quality": quality})
    hits = n - len(jobs)
    procs = []
    k = max(1, min(int(workers), len(jobs)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    tmpfiles = []
    for w in range(k if jobs else 0):
        part = jobs[w::k]
        fd, jf = tempfile.mkstemp(dir=cdir, prefix=".jobs-", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(part, fh)
        tmpfiles.append(jf)
        code = ("import sys; sys.path.insert(0, %r); from parcelcell import kitcache; kitcache._worker(%r, %r)"
                % (os.path.dirname(HERE), cdir, jf))
        procs.append(subprocess.Popen([sys.executable, "-I", "-B", "-c", code], env=env, cwd=cdir,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _out, err = p.communicate()
        if p.returncode != 0:
            errs.append((err or b"")[-400:].decode(errors="replace"))
    for jf in tmpfiles:
        try:
            os.remove(jf)
        except OSError:
            pass
    missing = [j["key"] for j in jobs if not os.path.isfile(_path(cdir, j["key"]))]
    res = {"parcels": n, "hits": hits, "built": len(jobs) - len(missing), "missing": len(missing),
           "wall_s": round(time.time() - t0, 2), "workers": len(procs)}
    if errs:
        res["errors"] = errs[:2]
    log("[kitcache] %s" % json.dumps(res))
    return res


def _worker(cdir, jobfile):
    with open(jobfile) as fh:
        jobs = json.load(fh)
    for j in jobs:
        spec_d = dict(j["spec"])
        spec_d["size"] = tuple(spec_d["size"])
        spec_d["label_uv"] = tuple(spec_d["label_uv"])
        rec, arr = build_entry(spec_d, j["seed"], j["identity"], j["quality"], j["name"])
        store(cdir, j["key"], rec, arr)

