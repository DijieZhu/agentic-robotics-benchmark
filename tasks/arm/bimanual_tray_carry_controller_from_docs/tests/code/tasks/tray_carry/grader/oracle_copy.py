import hashlib
import os
import re
import shutil
import stat as statmod

TASK_REL = "tasks/tray_carry"
UNIT_REL = "shared/openarm_unit"
PATTERNS = (("task", "ref", r"ref/[^/]+\.py"),
            ("task", "contract", r"contract/[^/]+\.py"),
            ("task", "contract", r"contract/profile_stage05\.json"),
            ("unit", "openarm_unit", r"openarm_unit/(?![^/]*_hidden\.json$)[^/]+\.(py|json)"),
            ("unit", "base", r"base/.+"),
            ("unit", None, r"NOTICE"))
DENIED_SEGMENTS = re.compile(r"^(hidden|registry|candidates|constructed|units|solution|grader|"
                             r"runtime|rules|tests|__pycache__)$")
DENIED_NAMES = re.compile(r"(oracle_token|frozen_set|registry\.json|generator\.json|draw_log\.json|\.npz$|\.pyc$|"
                          r"_hidden\.json$)")
FILE_MODE, DIR_MODE = 0o644, 0o755


class OracleCopyRefused(RuntimeError):
    ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _walk_files(base):
    out = []
    for dp, dns, fns in os.walk(base, followlinks=False):
        dns[:] = sorted(d for d in dns if not os.path.islink(os.path.join(dp, d)))
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(os.path.relpath(p, base).replace(os.sep, "/"))
    return out


def _denied(rel):
    segs = rel.split("/")
    return any(DENIED_SEGMENTS.match(s) for s in segs[:-1]) or bool(DENIED_NAMES.search(segs[-1]))


def file_list(task_dir, unit_dir):
    bases = {"task": (os.path.abspath(task_dir), TASK_REL), "unit": (os.path.abspath(unit_dir), UNIT_REL)}
    out = {}
    for which, folder, pat in PATTERNS:
        base, dest = bases[which]
        rx = re.compile("^" + pat + "$")
        top = base if folder is None else os.path.join(base, folder)
        pre = "" if folder is None else folder + "/"
        if not os.path.isdir(top):
            rels = []
        elif pat.endswith(".+"):
            rels = [pre + r for r in _walk_files(top)]
        else:
            rels = [pre + n for n in sorted(os.listdir(top)) if os.path.isfile(os.path.join(top, n))
                    and not os.path.islink(os.path.join(top, n))]
        for rel in rels:
            if rx.match(rel):
                out[dest + "/" + rel] = os.path.join(base, rel)
    bad = sorted(r for r in out if _denied(r))
    if bad:
        raise OracleCopyRefused("the list names files that may not be in the copy: " + ", ".join(bad[:5]))
    for need in (TASK_REL + "/ref/reference_controller.py", TASK_REL + "/ref/shift.py",
                 TASK_REL + "/contract/constants.py", UNIT_REL + "/openarm_unit/__init__.py"):
        if need not in out:
            raise OracleCopyRefused("the copy's source lacks " + need)
    return out


def _digest(files):
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("ascii")).hexdigest()


def check(dest_root, files, owner_uid=None, reachable=None):
    root = os.path.abspath(dest_root)
    uid = (0 if os.getuid() == 0 else os.getuid()) if owner_uid is None else int(owner_uid)
    reach = (os.getuid() == 0) if reachable is None else bool(reachable)
    problems = []
    if os.path.islink(root) or not os.path.isdir(root):
        raise OracleCopyRefused(f"{root}: not a folder")
    up = os.path.dirname(root)
    while reach:
        try:
            if not os.stat(up).st_mode & statmod.S_IXOTH:
                problems.append(f"{up}: not searchable by others")
        except OSError as e:
            problems.append(f"{up}: {e}")
        if up == os.path.dirname(up):
            break
        up = os.path.dirname(up)
    seen = set()
    for dp, dns, fns in os.walk(root, followlinks=False):
        for name in list(dns) + list(fns):
            p = os.path.join(dp, name)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            st = os.lstat(p)
            if statmod.S_ISLNK(st.st_mode):
                problems.append("a symbolic link: " + rel)
                continue
            if st.st_uid != uid:
                problems.append(f"not owned by uid {uid}: {rel}")
            if st.st_mode & 0o022:
                problems.append("writable by group or others: " + rel)
            if statmod.S_ISDIR(st.st_mode):
                if (st.st_mode & 0o005) != 0o005:
                    problems.append("a folder not readable and searchable by others: " + rel)
            elif statmod.S_ISREG(st.st_mode):
                seen.add(rel)
                if not st.st_mode & statmod.S_IROTH:
                    problems.append("not readable by others: " + rel)
                if _denied(rel):
                    problems.append("may not be in the copy: " + rel)
                if rel not in files:
                    problems.append("not on the copy's list: " + rel)
                elif _sha256(p) != files[rel]:
                    problems.append("differs from its source: " + rel)
            else:
                problems.append("neither a file nor a folder: " + rel)
        dns[:] = [d for d in dns if not os.path.islink(os.path.join(dp, d))]
    for rel in sorted(set(files) - seen):
        problems.append("missing: " + rel)
    st = os.lstat(root)
    if st.st_uid != uid or st.st_mode & 0o022 or (st.st_mode & 0o005) != 0o005:
        problems.append("the copy's folder is not owned by uid %d with mode 755" % uid)
    if problems:
        raise OracleCopyRefused("the oracle's instrument copy is refused: " + "; ".join(problems[:8])
                                + ("" if len(problems) <= 8 else f" (and {len(problems) - 8} more)"))
    return {"root": root, "ref_dir": os.path.join(root, TASK_REL, "ref"), "files": dict(sorted(files.items())),
            "digest": _digest(files), "n_files": len(files)}


def make(dest_root, task_dir, unit_dir, forbid_inside=()):
    root = os.path.abspath(dest_root)
    if root in ("/", "") or root == os.path.dirname(root):
        raise OracleCopyRefused(f"{root}: not a place for the copy")
    for f in list(forbid_inside) + [task_dir, unit_dir]:
        fa = os.path.abspath(f)
        if fa == root or fa.startswith(root + os.sep) or root.startswith(fa + os.sep):
            raise OracleCopyRefused(f"{root}: overlaps {fa}")
    src = file_list(task_dir, unit_dir)
    if os.path.islink(root):
        os.unlink(root)
    elif os.path.exists(root):
        shutil.rmtree(root)
    old = os.umask(0o022)
    try:
        os.makedirs(root, mode=DIR_MODE)
        files = {}
        for rel in sorted(src):
            dst = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), mode=DIR_MODE, exist_ok=True)
            shutil.copyfile(src[rel], dst)
            os.chmod(dst, FILE_MODE)
            files[rel] = _sha256(src[rel])
        is_root = os.getuid() == 0
        for dp, dns, fns in os.walk(root):
            os.chmod(dp, DIR_MODE)
            if is_root:
                os.chown(dp, 0, 0)
                for fn in fns:
                    os.chown(os.path.join(dp, fn), 0, 0)
    finally:
        os.umask(old)
    return check(root, files)
