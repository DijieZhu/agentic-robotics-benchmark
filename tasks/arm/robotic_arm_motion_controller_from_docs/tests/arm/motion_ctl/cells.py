"""Desktop cells of the motion-controller task (`motion-cell/1`): the true geometry and the survey the controller gets.

Three kinds, scaled to the arm's reach (the TCP reaches about 0.46 m from the base axis; pointing down it works between
about 0.16 and 0.36 m):

  shelf     a shelf organiser (2 levels x 2-3 compartments, open towards the arm), a fixture block and one or two posts
            (light / camera stands)
  bins      two open bins with 4 mm walls, a fixture block and one or two posts; the parts are carried in the gripper
  dispense  a board on a wedge (tilted 18-40 deg towards the arm), a cup and a clamp block on or beside it; the needle
            is clamped in the gripper

Frames: the TRUE geometry is built in the robot base frame (origin on the mounting plane at the base-yaw axis, z up, +y
forward at yaw 0). The cell frame (a corner of the table) is a random rigid transform of it; the scene description that
the controller receives is written in the cell frame: every object and the robot base's own pose in the cell, each off
the truth by the survey error (objects +-2 mm per axis and +-1 deg about z; the base +-1 mm and +-0.3 deg).

Every object is a list of primitive geoms (box: half sizes; cylinder: radius, half height) in the object's frame; the
object frame sits on the table (z = 0 at its underside) and turns about z. Regions (`regions`) are what the job
generator samples targets in. numpy only; names no source.
"""

import math

import numpy as np

CELL_FORMAT = "motion-cell/1"
KINDS = ("shelf", "bins", "dispense")
SURVEY_OBJ_MM, SURVEY_OBJ_DEG = 2.0, 1.0
SURVEY_BASE_MM, SURVEY_BASE_DEG = 1.0, 0.3
WALL = 0.004
COLOURS = {"shelf": "0.78 0.72 0.6 1", "fixture": "0.35 0.38 0.42 1", "post": "0.5 0.52 0.55 1", "bin": "0.2 0.42 0.7 1",
           "board": "0.85 0.85 0.8 1", "wedge": "0.4 0.4 0.4 1", "cup": "0.9 0.9 0.92 1", "clamp": "0.3 0.3 0.32 1"}


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quat_from_R(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


class Pose(object):
    def __init__(self, p, R):
        self.p = np.asarray(p, float)
        self.R = np.asarray(R, float)

    def __mul__(self, o):
        return Pose(self.p + self.R @ o.p, self.R @ o.R)

    def inv(self):
        return Pose(-self.R.T @ self.p, self.R.T)

    def apply(self, x):
        return self.p + self.R @ np.asarray(x, float)

    def rot(self, v):
        return self.R @ np.asarray(v, float)

    def yaw(self):
        return math.atan2(self.R[1, 0], self.R[0, 0])

    def to_json(self):
        return {"pos": [round(float(v), 6) for v in self.p], "yaw_deg": round(math.degrees(self.yaw()), 4)}

    @staticmethod
    def from_json(d):
        return Pose(d["pos"], Rz(math.radians(d["yaw_deg"])))


def _box(name, half, pos, R=None, colour="fixture"):
    return {"name": name, "type": "box", "size": [float(v) for v in half], "pos": [float(v) for v in pos],
            "R": (np.eye(3) if R is None else np.asarray(R, float)).tolist(), "colour": colour}


def _cyl(name, radius, half_h, pos, colour="post"):
    return {"name": name, "type": "cylinder", "size": [float(radius), float(half_h)], "pos": [float(v) for v in pos],
            "R": np.eye(3).tolist(), "colour": colour}


# ------------------------------------------------------------------------------------------------------- objects
def shelf_object(r, name):
    """Shelf organiser in its frame: the open face looks along -y (towards the arm when the object yaws to face it)."""
    ncol = int(r.integers(2, 4))
    w_c = float(r.uniform(0.085, 0.105))                 # compartment width (the gripper is 52 x 65 mm across)
    h_c = float(r.uniform(0.100, 0.120))                 # compartment height
    dep = float(r.uniform(0.060, 0.085))
    z0 = float(r.uniform(0.020, 0.070))                  # lowest shelf board above the table (a plinth below)
    W = ncol * w_c + (ncol + 1) * WALL
    H = z0 + 2 * h_c + 3 * WALL
    g = [_box(name + "_plinth", [W / 2, dep / 2, z0 / 2], [0, 0, z0 / 2], colour="shelf"),
         _box(name + "_back", [W / 2, WALL / 2, (H - z0) / 2], [0, dep / 2 - WALL / 2, z0 + (H - z0) / 2], colour="shelf")]
    for k in range(3):                                   # boards
        z = z0 + k * (h_c + WALL) + WALL / 2
        g.append(_box("{}_board{}".format(name, k), [W / 2, dep / 2, WALL / 2], [0, 0, z], colour="shelf"))
    for c in range(ncol + 1):                            # side walls / dividers
        x = -W / 2 + WALL / 2 + c * (w_c + WALL)
        g.append(_box("{}_wall{}".format(name, c), [WALL / 2, dep / 2, (H - z0) / 2], [x, 0, z0 + (H - z0) / 2], colour="shelf"))
    regions = []
    for lev in range(2):
        for c in range(ncol):
            x = -W / 2 + WALL + c * (w_c + WALL) + w_c / 2
            zb = z0 + lev * (h_c + WALL) + WALL
            regions.append({"kind": "compartment", "object": name, "centre": [x, 0.0, zb + h_c / 2], "half": [w_c / 2, dep / 2, h_c / 2],
                            "floor_z": zb, "open": [0.0, -1.0, 0.0], "board": "{}_board{}".format(name, lev)})
    return {"name": name, "kind": "shelf", "geoms": g, "regions": regions, "footprint": [W / 2, dep / 2], "height": H}


def fixture_object(r, name):
    a, b, h = float(r.uniform(0.03, 0.05)), float(r.uniform(0.03, 0.05)), float(r.uniform(0.03, 0.09))
    return {"name": name, "kind": "fixture", "geoms": [_box(name, [a, b, h / 2], [0, 0, h / 2], colour="fixture")],
            "regions": [{"kind": "top", "object": name, "centre": [0, 0, h], "half": [a, b, 0.0], "floor_z": h}],
            "footprint": [a, b], "height": h}


def post_object(r, name):
    rad, h = float(r.uniform(0.010, 0.020)), float(r.uniform(0.12, 0.26))
    return {"name": name, "kind": "post", "geoms": [_cyl(name, rad, h / 2, [0, 0, h / 2])], "regions": [],
            "footprint": [rad, rad], "height": h}


def bin_object(r, name):
    a, b, h = float(r.uniform(0.05, 0.07)), float(r.uniform(0.04, 0.06)), float(r.uniform(0.045, 0.08))
    g = [_box(name + "_floor", [a, b, WALL / 2], [0, 0, WALL / 2], colour="bin"),
         _box(name + "_wxn", [WALL / 2, b, h / 2], [-a + WALL / 2, 0, h / 2], colour="bin"),
         _box(name + "_wxp", [WALL / 2, b, h / 2], [a - WALL / 2, 0, h / 2], colour="bin"),
         _box(name + "_wyn", [a, WALL / 2, h / 2], [0, -b + WALL / 2, h / 2], colour="bin"),
         _box(name + "_wyp", [a, WALL / 2, h / 2], [0, b - WALL / 2, h / 2], colour="bin")]
    reg = [{"kind": "bin", "object": name, "centre": [0, 0, WALL + (h - WALL) / 2], "half": [a - WALL, b - WALL, (h - WALL) / 2],
            "floor_z": WALL, "rim_z": h}]
    return {"name": name, "kind": "bin", "geoms": g, "regions": reg, "footprint": [a, b], "height": h}


def board_object(r, name):
    """A board on a wedge, tilted by `tilt` (18-40 deg) about the object x axis so that its surface faces -y (the arm) and up."""
    a, b = float(r.uniform(0.07, 0.10)), float(r.uniform(0.05, 0.07))
    tilt = math.radians(float(r.uniform(18.0, 40.0)))
    h0 = float(r.uniform(0.010, 0.030))                  # height of the board's front edge above the table (wedge)
    t = 0.006
    R = Rx(tilt)                                         # surface normal (0, -sin, cos): faces the arm (object -y)
    n = R @ np.array([0.0, 0.0, 1.0])
    centre = np.array([0.0, 0.0, h0 + b * math.sin(tilt)])
    g = [_box(name, [a, b, t / 2], centre - n * t / 2, R=R, colour="board")]
    # two rails carry the board (front and back), each 1 mm below the board's underside: nothing reaches its work face
    cz = float(centre[2])
    for k, yr in enumerate((-0.82 * b * math.cos(tilt), 0.82 * b * math.cos(tilt))):
        z_under = cz - t / math.cos(tilt) + yr * math.tan(tilt)
        top = max(0.004, z_under - 0.001 - 0.004 * math.tan(tilt))
        g.append(_box("{}_rail{}".format(name, k), [a * 0.9, 0.005, top / 2], [0, yr, top / 2], colour="wedge"))
    reg = [{"kind": "surface", "object": name, "centre": centre.tolist(), "half": [a - 0.008, b - 0.008, 0.0], "R": R.tolist(),
            "normal": n.tolist(), "tilt_deg": math.degrees(tilt)}]
    return {"name": name, "kind": "board", "geoms": g, "regions": reg, "footprint": [a, b * math.cos(tilt)], "height": float(centre[2] + b)}


def cup_object(r, name):
    rad, h = float(r.uniform(0.020, 0.030)), float(r.uniform(0.035, 0.060))
    return {"name": name, "kind": "cup", "geoms": [_cyl(name, rad, h / 2, [0, 0, h / 2], colour="cup")], "regions": [],
            "footprint": [rad, rad], "height": h}


# ---------------------------------------------------------------------------------------------------------- cells
class Cell(object):
    """`objects` = [{"name", "kind", "geoms", "regions", "pose" (Pose in the base frame)}], `base_in_cell` (Pose)."""

    def __init__(self, kind, objects, base_in_cell, survey, seed):
        self.kind = kind
        self.objects = objects
        self.base_in_cell = base_in_cell
        self.survey = survey                              # {"objects": {name: Pose error}, "base": Pose error}
        self.seed = seed

    # true geometry in the base frame (the grader's MJCF)
    def geoms(self):
        out = []
        for o in self.objects:
            P = o["pose"]
            for g in o["geoms"]:
                R = P.R @ np.asarray(g["R"], float)
                out.append({"name": "cell_" + g["name"], "type": g["type"], "size": g["size"], "pos": P.apply(g["pos"]).tolist(),
                            "quat": quat_from_R(R).tolist(), "rgba": COLOURS.get(g["colour"], "0.7 0.7 0.72 1"), "object": o["name"]})
        return out

    def object(self, name):
        return next(o for o in self.objects if o["name"] == name)

    def region_world(self, reg):
        """A region's centre / axes in the base frame."""
        P = self.object(reg["object"])["pose"]
        R = P.R @ np.asarray(reg.get("R", np.eye(3)), float)
        return P.apply(reg["centre"]), R

    # what the controller gets
    def description(self):
        """Scene description in the CELL frame with the survey errors applied."""
        base_err = self.survey["base"]
        base_desc = self.base_in_cell * base_err
        objs = []
        for o in self.objects:
            P_cell = self.base_in_cell * o["pose"] * self.survey["objects"][o["name"]]
            prims = []
            for g in o["geoms"]:
                prims.append({"name": g["name"], "type": g["type"], "size": [round(float(v), 5) for v in g["size"]],
                              "pos": [round(float(v), 5) for v in g["pos"]],
                              "quat": [round(float(v), 6) for v in quat_from_R(g["R"])]})
            objs.append({"name": o["name"], "kind": o["kind"], "pose": P_cell.to_json(), "primitives": prims})
        return {"format": "scene-description/1", "frame": "cell (the table's reference corner): x, y >= 0 on the table, z up; metres, degrees",
                "robot_base": base_desc.to_json(), "objects": objs,
                "table": {"z": 0.0, "note": "the robot base stands on the table; the table top is z = 0"},
                "survey_tolerance": {"objects_mm": SURVEY_OBJ_MM, "objects_deg": SURVEY_OBJ_DEG,
                                     "robot_base_mm": SURVEY_BASE_MM, "robot_base_deg": SURVEY_BASE_DEG}}

    def described(self):
        """The cell as the scene description puts it, in the base frame the description implies (base_desc^-1 x
        object_desc): what a documents-only controller plans against."""
        base_desc = self.base_in_cell * self.survey["base"]
        objs = []
        for o in self.objects:
            P_cell = self.base_in_cell * o["pose"] * self.survey["objects"][o["name"]]
            oo = dict(o)
            oo["pose"] = base_desc.inv() * P_cell
            objs.append(oo)
        zero = {"base": Pose(np.zeros(3), np.eye(3)), "objects": {o["name"]: Pose(np.zeros(3), np.eye(3)) for o in self.objects}}
        return Cell(self.kind, objs, base_desc, zero, self.seed)

    def to_cell(self, x_base):
        return self.base_in_cell.apply(x_base)

    def dir_to_cell(self, v_base):
        return self.base_in_cell.rot(v_base)

    def summary(self):
        return {"format": CELL_FORMAT, "kind": self.kind, "seed": self.seed,
                "objects": [{"name": o["name"], "kind": o["kind"], "pose_base": o["pose"].to_json()} for o in self.objects],
                "base_in_cell": self.base_in_cell.to_json()}


def _place(r, rmin, rmax, az_range, existing, footprint, gap=0.03, face=True, tries=200):
    """A pose in the base frame at radius / azimuth (azimuth 0 = +y), facing the base (object -y towards the axis),
    away from the robot's base block and from `existing` footprints (circles)."""
    rad_o = math.hypot(*footprint)
    for _ in range(tries):
        rr = float(r.uniform(rmin, rmax))
        az = math.radians(float(r.uniform(*az_range)))
        p = np.array([-rr * math.sin(az), rr * math.cos(az), 0.0])
        if any(np.linalg.norm(p[:2] - q[:2]) < rad_o + qr + gap for q, qr in existing):
            continue
        if np.linalg.norm(p[:2]) - rad_o < 0.11:          # the base block and the shoulder sweep
            continue
        yaw = az + (math.radians(float(r.uniform(-4.0, 4.0))) if face else float(r.uniform(-math.pi, math.pi)))
        existing.append((p, rad_o))
        return Pose(p, Rz(yaw))
    raise ValueError("no place found")


def draw_cell(seed, kind):
    from .unit import rng
    r = rng(seed, "cell/" + kind)
    objs, used = [], []
    if kind == "shelf":
        sh = shelf_object(r, "shelf")
        sh["pose"] = _place(r, 0.30 + sh["footprint"][1], 0.35 + sh["footprint"][1], (-45, 45), used, sh["footprint"])
        objs.append(sh)
        fx = fixture_object(r, "fixture")
        fx["pose"] = _place(r, 0.20, 0.28, (-120, 120), used, fx["footprint"], face=False)
        objs.append(fx)
        for k in range(int(r.integers(1, 3))):
            po = post_object(r, "post" if k == 0 else "post{}".format(k + 1))
            po["pose"] = _place(r, 0.16, 0.32, (-120, 120), used, po["footprint"], gap=0.05, face=False)
            objs.append(po)
    elif kind == "bins":
        for k in range(2):
            b = bin_object(r, "bin{}".format(k + 1))
            b["pose"] = _place(r, 0.20, 0.27, (-110, 110), used, b["footprint"], gap=0.04, face=False)
            objs.append(b)
        for k in range(int(r.integers(1, 3))):
            po = post_object(r, "post" if k == 0 else "post{}".format(k + 1))
            po["pose"] = _place(r, 0.16, 0.32, (-110, 110), used, po["footprint"], gap=0.05, face=False)
            objs.append(po)
        fx = fixture_object(r, "fixture")
        fx["pose"] = _place(r, 0.20, 0.28, (-120, 120), used, fx["footprint"], face=False)
        objs.append(fx)
    elif kind == "dispense":
        bd = board_object(r, "board")
        bd["pose"] = _place(r, 0.25, 0.32, (-35, 35), used, bd["footprint"])
        objs.append(bd)
        cp = cup_object(r, "cup")
        cp["pose"] = _place(r, 0.22, 0.30, (-110, 110), used, cp["footprint"], gap=0.03, face=False)
        objs.append(cp)
        cl = fixture_object(r, "clamp")
        cl["pose"] = _place(r, 0.17, 0.32, (-110, 110), used, cl["footprint"], gap=0.03, face=False)
        objs.append(cl)
    else:
        raise ValueError(kind)
    # cell frame: the table corner; the base stands at (0.25-0.40, 0.20-0.35) and turns by a quarter turn +- 10 deg, then
    # the frame moves so that every object (and the base) lies at x, y >= 0.05 m (all coordinates positive)
    yaw = math.radians(90.0 * int(r.integers(0, 4)) + float(r.uniform(-10.0, 10.0)))
    base_in_cell = Pose([float(r.uniform(0.25, 0.40)), float(r.uniform(0.20, 0.35)), 0.0], Rz(yaw))
    pts = [base_in_cell.apply([0.0, 0.0, 0.0])]
    for o in objs:
        fx, fy = o["footprint"]
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append(base_in_cell.apply(o["pose"].apply([sx * fx, sy * fy, 0.0])))
    pts = np.array(pts)
    shift = np.array([max(0.0, 0.05 - pts[:, 0].min()), max(0.0, 0.05 - pts[:, 1].min()), 0.0])
    base_in_cell = Pose(base_in_cell.p + shift, base_in_cell.R)
    survey = {"base": Pose(r.uniform(-1, 1, 3) * np.array([SURVEY_BASE_MM, SURVEY_BASE_MM, 0.0]) * 1e-3,
                           Rz(math.radians(float(r.uniform(-SURVEY_BASE_DEG, SURVEY_BASE_DEG))))),
              "objects": {o["name"]: Pose(r.uniform(-1, 1, 3) * SURVEY_OBJ_MM * 1e-3 * np.array([1, 1, 0.0]),
                                          Rz(math.radians(float(r.uniform(-SURVEY_OBJ_DEG, SURVEY_OBJ_DEG))))) for o in objs}}
    return Cell(kind, objs, base_in_cell, survey, int(seed))
