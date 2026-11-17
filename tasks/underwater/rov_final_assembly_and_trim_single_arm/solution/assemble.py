"""Reference submission of rov_final_assembly_and_trim_single_arm: submitted with the token (.oracle_token), this
program's cells are privileged and it asks the cell to run the privileged reference program
(`arm.assembly_rov.reference`, grader side: it weighs the vehicle on the load cells, solves the trim from the work order
and the printed trim sheet, fits the ballast at keel_up, turns the vehicle and fits the guards, the vertical thrusters and
the electronics enclosure from the exact unit poses) through the cell's own control ticks and safety watch, on the same
fixed hidden episodes as every other submission. It first takes one frame of each camera and reads the work order and
the load cells through the public calls, so every validation also runs the agent's channels."""


def solve(cell):
    info = cell.describe()
    wo = info["work_order"]
    print("[reference] work order %s: %s" % (wo.get("unit"), wo.get("delivery_site")), flush=True)
    st = cell.get_state()
    print("[reference] fixture %s" % (st.get("fixture"),), flush=True)
    for name in ("top", "wrist"):
        f = cell.get_camera(name)
        print("[reference] camera %s: %s" % (name, {k: v for k, v in f.items() if not k.endswith("_b64")}), flush=True)
        if not f.get("ok"):
            raise RuntimeError("camera %s: %s" % (name, f))
    r = cell._call("reference_assemble")
    print("[reference] reference program: %s" % (r,), flush=True)
    if not (isinstance(r, dict) and r.get("ok")):
        raise RuntimeError("the reference program did not run: %s" % (r,))
