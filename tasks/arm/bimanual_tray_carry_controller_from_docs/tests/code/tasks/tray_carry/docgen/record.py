import _genenv

import hashlib
import json
import os

FORMAT = "openarm05/DeliveryRecord/1"


def stream(noise_key, name):
    return hashlib.sha256(b"openarm05/record/" + name.encode() + b"/" + noise_key).digest()


def stream_names():
    return ("unit_id", "pads") + tuple(f"serial/{i}" for i in range(16))


def delivery_record(unit, noise_key):
    from openarm_unit import evidence as EV
    from openarm_unit.state import public_params
    import tables
    P = public_params()
    C = _genenv.contract()
    prof = tables.profile_table()
    eds = P["editions"]
    recs = EV.module_record(unit)
    mods = []
    for r in recs:
        i = r["index"]
        mods.append({"label": r["label"], "product": r["product"], "edition": eds[r["edition"]]["name"]["value"],
                     "firmware": r["firmware"], "serial": "DM" + stream(noise_key, f"serial/{i}").hex()[:10].upper(),
                     "Imax_A": float(r["Imax_A"]), "register_dump": EV.register_dump_text(unit, i, header=False)})
    c1 = prof["C1"]["value"][1]
    pl_t = prof["PL1_translation"]["value"][1] * 1000.0
    pl_r = prof["PL1_rotation"]["value"][1]
    home = [float(x) for x in C.HOME_POSE_RAD]
    return {
        "format": FORMAT,
        "unit_id": "OAP-" + stream(noise_key, "unit_id").hex()[:6].upper(),
        "modules": mods,
        "zeroing": f"the maker's routine, script edition {P['host_profiles']['zero_routine']['value']}; fixture check "
                   f"passed: every joint within {c1:.1f} deg",
        "mounting": f"arms mounted to the pedestal drawing: {pl_t:.1f} mm, {pl_r:.2f} deg",
        "pads": "lot PD-" + stream(noise_key, "pads").hex()[:5].upper(),
        "home_pose_rad": {"left": home, "right": home, "grippers": "open"},
        "not_identified": "joint friction, damping and delivered torque were not identified on this unit",
    }


def dumps(rec):
    return json.dumps(rec, indent=1, sort_keys=True) + "\n"


def example_record(registry_dir=None, label="example"):
    from openarm_unit.registry import Registry
    reg = Registry.open(registry_dir or _genenv.REGISTRY_DEV, _genenv.PROFILE_PATH)
    stale = reg.stale()
    if stale:
        raise SystemExit(f"the example's registry is stale: {stale} (draw it again under a new registry)")
    unit = reg.build(label, noise=False)
    rec = delivery_record(unit, reg.noise_key(label))
    rec["_registry_id"] = reg.registry_id
    return rec


def public_record(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def example_bytes(registry_dir=None):
    return dumps(public_record(example_record(registry_dir)))
