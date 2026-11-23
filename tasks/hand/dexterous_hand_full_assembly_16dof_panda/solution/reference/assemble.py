"""Oracle of dexterous_hand_full_assembly_16dof_panda: submitted with the oracle token, this program's cells are
privileged (the nominal layout) and it asks the cell to run the privileged reference program
(`arm.assembly_hand.reference`: exact unit poses, the cell's own model for planning and collision checks) through the
cell's own control ticks and safety watch. It first asks each camera for one frame, so that every validation also goes
through the agent cameras' path; a camera call that fails, a reply that is not a frame, or a failed reference call
fails loudly."""


def solve(cell):
    for name in ("top", "wrist"):
        f = cell.get_camera(name)
        print("[oracle] camera %s: %s" % (name, {k: v for k, v in f.items() if not k.endswith("_b64")}), flush=True)
        if not f.get("ok"):
            raise RuntimeError("camera %s: %s" % (name, f))
    r = cell._call("reference_assemble")
    print("[oracle] reference: %s" % ({k: v for k, v in r.items() if k != "report"},), flush=True)
    if not r.get("ok"):
        raise RuntimeError("reference_assemble: %s" % (r,))
