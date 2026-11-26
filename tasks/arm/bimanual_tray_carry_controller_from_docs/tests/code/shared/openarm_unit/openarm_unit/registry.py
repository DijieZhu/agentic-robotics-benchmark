import hashlib
import importlib.metadata
import json
import os

from . import law as lawmod
from . import unit as unitmod
from .state import SessionState, UnitState, canonical_json, public_params_sha256
from .unit import UnitInvalid

REGISTRY_FORMAT = "openarm_unit/Registry/2"
LOG_FORMAT = "openarm_unit/RegistryLog/1"
REGISTRY_CODE_FILES = ("law.py", "registry.py", "hs.py", "rng.py", "thermal.py")
ZEROED_CODE_FILES = ("refhost.py",)


class RegistryError(ValueError):
    ""


class UnitStale(RegistryError):

    def __init__(self, msg, differs=()):
        super().__init__(msg)
        self.differs = tuple(differs)


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


LIBRARIES = ("mujoco", "numpy")
LIBRARY_KEYS = ("version", "own_files_sha256")


def record_own_lines(text):
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        path = line.rsplit(",", 2)[0] if line.count(",") >= 2 else line
        parts = path.replace("\\", "/").split("/")
        if any(p.endswith(".dist-info") for p in parts) or path.endswith(".pyc") or "__pycache__" in parts \
                or path.startswith("../") or path.startswith("/"):
            continue
        out.append(line)
    return sorted(out)


def record_digests(text):
    own = "\n".join(record_own_lines(text))
    return {"own_files_sha256": hashlib.sha256(own.encode("utf-8")).hexdigest(),
            "record_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def _dist_digest(name):
    d = importlib.metadata.distribution(name)
    out = {"version": d.version}
    out.update(record_digests(d.read_text("RECORD") or ""))
    return out


def _files_digest(h, names):
    for name in names:
        with open(os.path.join(lawmod.PACKAGE_DIR, name), "rb") as f:
            b = f.read()
        h.update(name.encode("ascii") + b"\0" + str(len(b)).encode("ascii") + b"\0" + b)
    return h.hexdigest()


def code_digest():
    return _files_digest(hashlib.sha256(unitmod.code_digest().encode("ascii") + b"\0"), REGISTRY_CODE_FILES)


def zeroed_code_digest():
    return _files_digest(hashlib.sha256(b"openarm_unit zero routine code\0"), ZEROED_CODE_FILES)


def current_digests(profile_path):
    return {"params_public_sha256": public_params_sha256(), "params_law_sha256": lawmod.law_sha256(),
            "profile_sha256": _sha_file(profile_path), "code_sha256": code_digest(),
            "law_provider_digest": lawmod.LawPlugins(lawmod.law_params(lawmod.PARAMS_LAW_PATH)).digest(),
            **{lib: _dist_digest(lib) for lib in LIBRARIES}}


def _differs(key, recorded, now):
    if key in LIBRARIES:
        if not isinstance(recorded, dict) or not isinstance(now, dict):
            return True
        return any(recorded.get(k) != now.get(k) for k in LIBRARY_KEYS)
    return recorded != now


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        f.write(canonical_json(obj))
    os.replace(tmp, path)


def _read(path):
    with open(path, "r", encoding="ascii") as f:
        return json.loads(f.read())


class Registry:

    def __init__(self, root, header, log):
        self.root, self.header, self.log = os.fspath(root), header, log
        self.key = bytes.fromhex(header["key_hex"])
        with open(os.path.join(self.root, "profile.json"), "rb") as f:
            pbytes = f.read()
        want = header["digests"]["profile_sha256"]
        got = hashlib.sha256(pbytes).hexdigest()
        if got != want:
            raise RegistryError(f"the registry's own copy of the profile (profile.json, sha256 {got[:12]}...) is not "
                                f"the profile it recorded (sha256 {want[:12]}...): it was changed in place")
        self.profile = lawmod.load_profile(pbytes, strict=True)
        if self.profile.sha256 != want:
            raise RegistryError("the profile read is not the profile checked")
        self.law = lawmod.law_params(lawmod.PARAMS_LAW_PATH)
        self.now = None

    @classmethod
    def create(cls, root, profile_path, entropy=None):
        root = os.fspath(root)
        if os.path.exists(os.path.join(root, "registry.json")):
            raise RegistryError(f"{root} already holds a registry")
        ppath = os.path.realpath(os.fspath(profile_path))
        rroot = os.path.realpath(root)
        if ppath == os.path.join(rroot, "profile.json"):
            raise RegistryError("the task's profile is not the registry's own copy (profile.json in the folder)")
        rel = os.path.relpath(ppath, rroot)
        rel = None if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel) else \
            rel.replace(os.sep, "/")
        prof = lawmod.load_profile(profile_path, strict=True)
        if entropy is None:
            key, src = os.urandom(32), "os.urandom"
        else:
            key, src = bytes(entropy), "given (a test key, never a task's)"
            if len(key) != 32:
                raise RegistryError("the key is 32 bytes")
        os.makedirs(os.path.join(root, "units"), exist_ok=True)
        os.makedirs(os.path.join(root, "constructed"), exist_ok=True)
        with open(profile_path, "rb") as f:
            pbytes = f.read()
        with open(os.path.join(root, "profile.json"), "wb") as f:
            f.write(pbytes)
        dig = current_digests(profile_path)
        header = {"format": REGISTRY_FORMAT, "key_hex": key.hex(), "entropy_source": src,
                  "profile_file": os.path.basename(ppath), "profile_relpath": rel, "profile_name": prof.name,
                  "task": prof.task, "law_version": lawmod.law_params()["law_version"], "digests": dig,
                  "acceptance": prof.report, "profile_notes": prof.notes, "bumped_stops": dict(lawmod.BUMPED_STOPS),
                  "zero_mode": prof.zero_mode, "units_wanted": prof.units, "sessions": list(prof.sessions)}
        _write(os.path.join(root, "registry.json"), header)
        log = {"format": LOG_FORMAT, "next_unit_index": 0, "steps": [], "units": {}, "frozen": None}
        _write(os.path.join(root, "log.json"), log)
        r = cls(root, header, log)
        r.now = dig
        return r

    @classmethod
    def open(cls, root, profile_path=None):
        root = os.fspath(root)
        header = _read(os.path.join(root, "registry.json"))
        if header.get("format") != REGISTRY_FORMAT:
            raise RegistryError(f"not a registry of the format {REGISTRY_FORMAT!r}")
        log = _read(os.path.join(root, "log.json"))
        if profile_path is not None:
            pp = os.fspath(profile_path)
        elif header.get("profile_relpath"):
            pp = os.path.join(root, *header["profile_relpath"].split("/"))
        else:
            raise RegistryError(f"Registry.open needs the task's profile file ({header.get('profile_file')!r}, "
                                f"sha256 {header['digests']['profile_sha256'][:12]}...): pass profile_path")
        if not os.path.isfile(pp):
            raise RegistryError(f"the task's profile file {pp!r} does not exist")
        if os.path.realpath(pp) == os.path.realpath(os.path.join(root, "profile.json")):
            raise RegistryError("the task's profile is not the registry's own copy (profile.json in the folder)")
        r = cls(root, header, log)
        r.now = current_digests(pp)
        return r

    @property
    def registry_id(self):
        return hashlib.sha256(b"openarm_unit registry\0" + self.key).hexdigest()[:16]

    def stale(self):
        rec = self.header["digests"]
        return sorted(k for k in rec if _differs(k, rec[k], self.now.get(k)))

    def _require_fresh(self):
        d = self.stale()
        if d:
            raise UnitStale("the registry's recorded digests differ from the tree's: " + ", ".join(d), d)

    def _save(self):
        _write(os.path.join(self.root, "log.json"), self.log)

    def _step(self, kind, **kw):
        self.log["steps"].append(dict(kind=kind, **kw))
        self._save()

    def _require_open(self):
        if self.log["frozen"] is not None:
            raise RegistryError("the registry is frozen")

    def draw(self, label, sanity=True, sanity_hold_s=None, max_units=8):
        self._require_open()
        self._require_fresh()
        if not label or not all(c.isalnum() or c in "_-" for c in label) or label in self.log["units"]:
            raise RegistryError(f"label {label!r}: new, letters, digits, '_' or '-'")
        for _ in range(max_units):
            ui = int(self.log["next_unit_index"])
            self.log["next_unit_index"] = ui + 1
            try:
                st, ses, rec = lawmod.draw_unit(self.key, self.profile, ui, self.law, unit_id=label)
            except lawmod.LawError as exc:
                self._step("rejected", unit_index=ui, label=label, step="draw", reason=str(exc))
                continue
            if sanity:
                ok, rep = lawmod.sanity(st, ses[0], self.law, hold_s=sanity_hold_s)
                if not ok:
                    self._step("rejected", unit_index=ui, label=label, step="sanity", reason="; ".join(rep["reasons"]))
                    continue
                rec["sanity"] = rep
            paths = self._store(label, st, ses)
            self.log["units"][label] = {"unit_index": ui, "unit_digest": st.digest, "files": paths,
                                        "session_digests": rec["session_digests"], "state": "drawn",
                                        "digests": self.header["digests"]}
            self._step("drawn", unit_index=ui, label=label, record=rec)
            return st, ses
        raise RegistryError(f"no admissible unit in {max_units} candidates (see log.json)")

    def _store(self, label, st, ses, tag="unit"):
        d = os.path.join(self.root, "units")
        p = {"unit": f"{label}.{tag}.json" if tag != "unit" else f"{label}.unit.json", "sessions": {}}
        st.dump(os.path.join(d, p["unit"]))
        for s in ses:
            fn = f"{label}.{s.session_id}.session.json"
            s.dump(os.path.join(d, fn))
            p["sessions"][s.session_id] = fn
        return p

    def reject(self, label, step, reason):
        self._require_open()
        u = self.log["units"].get(label)
        if u is None or u["state"] == "rejected":
            raise RegistryError(f"no live unit {label!r}")
        u["state"] = "rejected"
        u["rejected"] = {"step": str(step), "reason": str(reason)}
        old = dict(u)
        self.log["units"][f"{label}~rejected~{u['unit_index']}"] = old
        del self.log["units"][label]
        self._step("rejected", unit_index=u["unit_index"], label=label, step=str(step), reason=str(reason))

    def zero(self, label, routine):
        self._require_open()
        self._require_fresh()
        if self.header["zero_mode"] != "routine":
            raise RegistryError("the profile's zero_mode is not routine")
        st, ses = self.load_unit(label, as_zeroed=False)
        try:
            z, rep = lawmod.zero_by_routine(st, ses[0], routine, self.law)
        except (lawmod.LawError, UnitInvalid) as exc:
            self.reject(label, "zero routine", str(exc) if isinstance(exc, lawmod.LawError) else
                        f"the plant cannot integrate physics step {exc.step} ({exc.kind})")
            raise
        fn = f"{label}.as_zeroed.unit.json"
        z.dump(os.path.join(self.root, "units", fn))
        u = self.log["units"][label]
        u["as_zeroed"] = {"file": fn, "digest": z.digest, "zero_routine_code_sha256": zeroed_code_digest()}
        u["state"] = "zeroed"
        self._step("zeroed", label=label, report=rep)
        return z

    def bumped_stops(self):
        return dict(self.header["bumped_stops"])

    def store_bracket(self, end):
        self._require_open()
        rec, st, ses = lawmod.bracket(self.profile, end, self.law)
        path = os.path.join(self.root, "constructed", f"{end}.json")
        _write(path, rec)
        self._step("constructed", end=end, digest=rec["digest"])
        return rec, st, ses

    def load_bracket(self, end):
        rec = _read(os.path.join(self.root, "constructed", f"{end}.json"))
        body = {k: v for k, v in rec.items() if k != "digest"}
        if hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest() != rec["digest"]:
            raise RegistryError(f"constructed state {end}: its digest does not match its content")
        return rec, UnitState.from_dict(rec["unit"]), SessionState.from_dict(rec["session"])

    def freeze(self, require_spread=True):
        self._require_open()
        self._require_fresh()
        live = [k for k, u in self.log["units"].items() if u["state"] in ("drawn", "zeroed")]
        spread = None
        if len(live) > 1 and self.profile.raw.get("spread_rule"):
            try:
                ok, det = lawmod.spread_check([self.load_unit(k)[0] for k in live], self.profile)
                spread = {"ok": ok, "detail": det}
                if not ok:
                    raise RegistryError("the spread rule does not hold over the live units")
            except lawmod.LawError as exc:
                if require_spread:
                    raise RegistryError(str(exc)) from exc
                spread = {"ok": None, "not_applied": str(exc)}
        files = {}
        for dirpath, _, names in os.walk(os.path.join(self.root, "units")):
            for n in sorted(names):
                files[n] = _sha_file(os.path.join(dirpath, n))
        self.log["frozen"] = {"units": live, "spread": spread, "files": files,
                              "registry_json_sha256": _sha_file(os.path.join(self.root, "registry.json"))}
        self._save()
        return self.log["frozen"]

    def labels(self):
        return [k for k, u in self.log["units"].items() if u["state"] in ("drawn", "zeroed")]

    def load_unit(self, label, as_zeroed=None):
        self._require_fresh()
        u = self.log["units"].get(label)
        if u is None:
            raise RegistryError(f"no unit {label!r}")
        if u.get("digests", self.header["digests"]) != self.header["digests"]:
            raise UnitStale(f"unit {label}: drawn under other digests", ("unit_record",))
        d = os.path.join(self.root, "units")
        z = u.get("as_zeroed") if as_zeroed in (None, True) else None
        if as_zeroed is True and z is None:
            raise RegistryError(f"unit {label} has no 'as zeroed' state")
        if z and z.get("zero_routine_code_sha256") != zeroed_code_digest():
            raise UnitStale(f"unit {label}: the zero routine's code ({', '.join(ZEROED_CODE_FILES)}) differs from the "
                            "code that zeroed it", ("zero_routine_code_sha256",))
        st = UnitState.load(os.path.join(d, z["file"] if z else u["files"]["unit"]))
        want = z["digest"] if z else u["unit_digest"]
        if st.digest != want:
            raise RegistryError(f"unit {label}: the stored state's digest {st.digest[:12]} is not the recorded one")
        if lawmod.is_constructed(st):
            raise RegistryError("a constructed state is never a hidden unit")
        ses = []
        for sid in self.header["sessions"]:
            s = SessionState.load(os.path.join(d, u["files"]["sessions"][sid]))
            if s.digest != u["session_digests"][sid]:
                raise RegistryError(f"unit {label}, session {sid}: digest mismatch")
            ses.append(s)
        return st, ses

    def noise_key(self, label):
        u = self.log["units"][label]
        from . import rng
        return rng.digest(self.key, f"noise_key/u{int(u['unit_index'])}", 0)

    def record(self, label):
        u = self.log["units"][label]
        return {"registry_id": self.registry_id, "label": label, "unit_index": u["unit_index"],
                "unit_digest": (u.get("as_zeroed") or {}).get("digest", u["unit_digest"]),
                "params_law_sha256": self.header["digests"]["params_law_sha256"],
                "profile_sha256": self.header["digests"]["profile_sha256"]}

    def build(self, label, session=None, world=None, noise=True, **kw):
        st, ses = self.load_unit(label)
        s = ses[0] if session is None else next(x for x in ses if x.session_id == session)
        return unitmod.OpenArmUnit.build(st, s, world, self.noise_key(label) if noise else None,
                                         law=lawmod.LawPlugins(self.law), registry=self.record(label), **kw)
