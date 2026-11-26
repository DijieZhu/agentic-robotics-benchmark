import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
for _p in (os.path.join(TASK_DIR, "ref"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shift as SH
from openarm_unit import rng as urng
from openarm_unit.registry import Registry

import generator as GEN

REGISTRY_DEV = os.path.join(HERE, "registry")
N_SEATS = 8


def open_registry(root=REGISTRY_DEV):
    return Registry.open(root, SH.K.PROFILE_PATH)


def load_dev(reg, label, cell=None):
    st, sess = reg.load_unit(label)
    site = SH.draw_site(reg.key, label, N_SEATS, ambient_C=sess[0].ambient_C)
    u = reg.build(label, world=SH.make_world(site, cell=cell), policy="stage05", truth_profile=SH.trace_fields(cell))
    return st, sess[0], site, u


def make_dev_builder(reg, label, site, cell=None):
    def build():
        return reg.build(label, world=SH.make_world(site, cell=cell), policy="stage05",
                         truth_profile=SH.trace_fields(cell))
    return build


def bracket(reg, end):
    p = os.path.join(reg.root, "constructed", f"{end}.json")
    if os.path.exists(p):
        _, st, ses = reg.load_bracket(end)
    else:
        _, st, ses = reg.store_bracket(end)
    return st, ses


def bracket_site(site, end, session):
    out = json.loads(json.dumps(site))
    mu = GEN.bracket_pad_friction()
    out["pad_friction"] = {"left": mu, "right": mu}
    out["ambient_C"] = float(session.ambient_C)
    out["label"] = f"{site.get('label')}@{end}"
    return out


def make_bracket_builder(reg, end, site_b, cell=None):
    st, ses = bracket(reg, end)

    def build():
        return SH.build_unit(st, ses, site_b, noise_key=urng.digest(reg.key, f"noise_key/{end}", 0), cell=cell)
    return st, ses, build


def main():
    argv = list(sys.argv[1:])
    reg_root = REGISTRY_DEV
    if "--registry" in argv:
        k = argv.index("--registry")
        reg_root = argv[k + 1]
        del argv[k:k + 2]
    if argv[0] == "shift":
        label, n_decline, out = argv[1], int(argv[2]), argv[3]
        with open("/proc/loadavg") as f:
            la0 = f.read().split()[:3]
        reg = open_registry(reg_root)
        st, ses, site, u = load_dev(reg, label)
        sh = GEN.generate_shift(reg.key, label, st, ses, u, n_decline)
        res = {"registry_id": reg.registry_id, "unit": label, "unit_digest": st.digest, "site": site, "shift": sh,
               "code_digest": GEN.code_digest()}
        if sh["ok"]:
            res["certification"] = GEN.certification_rows(sh)
        with open("/proc/loadavg") as f:
            res["loadavg"] = {"before": la0, "after": f.read().split()[:3]}
        res["date"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(out + ".tmp", "w") as f:
            json.dump(GEN.plain(res), f, indent=1, sort_keys=True)
        os.replace(out + ".tmp", out)
        print(out, sh["ok"], sh["reason"], sh["shift_digest"], round(sh["cpu_s"], 1))
    elif argv[0] in ("carry", "decline"):
        import label as LB
        trace = None
        if "--trace" in argv:
            k = argv.index("--trace")
            trace = argv[k + 1]
            del argv[k:k + 2]
        tdir = None
        if "--trace-dir" in argv:
            k = argv.index("--trace-dir")
            tdir = argv[k + 1]
            del argv[k:k + 2]
        with open(argv[1], encoding="utf-8") as f:
            gen = json.load(f)
        sh, label, out = gen["shift"], gen["unit"], argv[3]
        with open("/proc/loadavg") as f:
            la0 = f.read().split()[:3]
        reg = open_registry(reg_root)
        if reg.registry_id != gen["registry_id"]:
            raise SystemExit("the shift's registry is not this registry")
        site = gen["site"]
        if argv[0] == "carry":
            which = argv[2]
            if which == "true":
                st, sess = reg.load_unit(label)
                res = LB.carry_run(sh, "true", st, sess[0], make_dev_builder(reg, label, site), site,
                                   trace_path=trace)
            else:
                stb, sesb = bracket(reg, which)
                site_b = bracket_site(site, which, sesb)
                stb, sesb, build = make_bracket_builder(reg, which, site_b)
                res = LB.carry_run(sh, which, stb, sesb, build, site_b, trace_path=trace)
        else:
            i = int(argv[2])
            stb, sesb = bracket(reg, "U_best")
            site_b = bracket_site(site, "U_best", sesb)
            stb, sesb, build = make_bracket_builder(reg, "U_best", site_b)
            feas = LB.decline_feasibility(sh, i, stb, sesb, build())
            runs = []
            for fz in feas:
                if fz["passes"]:
                    tp = None if tdir is None else os.path.join(tdir, f"{label}_best_{i}_{int(fz['swivel_deg'])}.npz")
                    runs.append(LB.decline_run(sh, i, fz["job"], stb, sesb, build, site_b, trace_path=tp))
            res = {"slot_index": i, "feasibility": [{k: v for k, v in fz.items() if k != "job"} for fz in feas],
                   "runs": runs}
        res["shift_digest"] = sh["shift_digest"]
        res["unit"] = label
        res["registry_id"] = reg.registry_id
        with open("/proc/loadavg") as f:
            res["loadavg"] = {"before": la0, "after": f.read().split()[:3]}
        res["date"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(out + ".tmp", "w") as f:
            json.dump(GEN.plain(res), f, indent=1, sort_keys=True)
        os.replace(out + ".tmp", out)
        print(out, "done")


if __name__ == "__main__":
    main()
