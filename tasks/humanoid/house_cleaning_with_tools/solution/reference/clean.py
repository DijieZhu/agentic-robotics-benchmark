"""Oracle of house_cleaning_with_tools: submitted with the oracle token, this program's cells are privileged (on the
same fixed hidden scenes the agents get) and it asks the cell to run the privileged reference cleaner (house_cell.reference: true object poses and
process states, the G1 driven through the cell's own controller, grip rule, safety watch and cleaning processes) inside
one call. It first takes one frame of each public camera, so every validation also exercises the agents' camera path
(a camera that cannot render fails the oracle, and with it the validation)."""

CAMERAS = ("ceiling_kitchen", "ceiling_dining", "ceiling_living", "head")


def solve(cell):
    info = cell.describe()
    print("[oracle] cell %s, cameras %s" % (info.get("api"), sorted(info.get("cameras", {}))), flush=True)
    for name in CAMERAS:
        f = cell.get_camera(name)
        print("[oracle] camera %s: %s" % (name, {k: v for k, v in f.items() if not k.endswith("_b64")}), flush=True)
    r = cell._call("reference_clean")
    print("[oracle] reference: %s" % (r,), flush=True)
