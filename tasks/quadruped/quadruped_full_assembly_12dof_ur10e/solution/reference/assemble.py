"""Oracle of quadruped_full_assembly_12dof_ur10e: submitted with the oracle token, this program's cells are privileged (the
bench plan's nominal layout) and it asks the cell to run the privileged reference program (`arm.assembly_quad.reference`:
exact unit poses, the cell's own model for planning and collision checks) through the cell's own control ticks and
safety watch. It first takes one frame of each camera, so every validation also renders through the agent cameras'
path."""


def solve(cell):
    for name in ("top", "wrist"):
        f = cell.get_camera(name)
        print("[oracle] camera %s: %s" % (name, {k: v for k, v in f.items() if not k.endswith("_b64")}), flush=True)
        if not f.get("ok"):
            raise RuntimeError("camera %s: %s" % (name, f))
    r = cell._call("reference_assemble")
    print("[oracle] reference: %s" % (r,), flush=True)
