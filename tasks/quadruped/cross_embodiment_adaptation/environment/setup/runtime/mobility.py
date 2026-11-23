"""Physical warehouse fittings and a fixed inspection load on the native Go1."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from props import SADDLE_KG
from scene_assets import geom, vec


CLEARANCE, DUCT_DEPTH = .426, .068
DECK_Z, BEVEL, TOE = .065, .33, .012
GUIDE_RISE, COVER_OFFSET = .082, .155
TERRAIN_BIT = 1 << 39
SUPPORT = ("deck_table", "deck_west_ramp", "deck_east_ramp", "deck_center_guide",
           "deck_south_shoulder", "deck_north_shoulder")
DUCT = ("duct_box", "duct_left_shoe", "duct_right_shoe")
# The equipment the Go1 carries (collision envelopes of real devices):
#   lidar_head        a 3D lidar of the Ouster OS1 class: a cylinder 87 mm in diameter, 74.5 mm tall, 0.447 kg
#   lidar_mast        its mast, 24 mm in diameter
#   compute_puck      the compute and IMU unit under the mast, 100 x 80 x 30 mm
#   lidar_cage_*      a tubular guard frame round the lidar, 12 mm tube: a 0.204 x 0.170 m rectangle at the top
#                     (front, rear, left, right) on four posts (fl, fr, rl, rr); its top is the equipment's top
#   profiler_head     a line profiler of the LMI Gocator 2340 class mounted crosswise: 49 x 197 x 75 mm, 0.74 kg
#   profiler_bracket  its bracket under the nose
CAGE_TOP = ("lidar_cage_front", "lidar_cage_rear", "lidar_cage_left", "lidar_cage_right")
CAGE = (*CAGE_TOP, "lidar_cage_fl", "lidar_cage_fr", "lidar_cage_rl", "lidar_cage_rr")
PROTECTED = ("lidar_head", "profiler_head", "profiler_bracket", "lidar_mast", "compute_puck", *CAGE)


@dataclass(frozen=True)
class Fitting:
    kind: str
    x: float
    lo: float
    hi: float
    length: float
    clearance: float = CLEARANCE    # duct: underside above the floor
    rise: float = GUIDE_RISE        # deck: guide crest above the deck


def fittings(seed, racks, props=None):
    duct_first = bool(seed & TERRAIN_BIT)
    clearance = props.duct_clearance_m if props else CLEARANCE
    rise = props.guide_rise_m if props else GUIDE_RISE
    result = []
    for index, rack in enumerate(racks[::2]):
        duct = duct_first == (index == 0)
        result.append(Fitting("duct" if duct else "deck", (rack.near+rack.far)/2,
                              *rack.wide, .44 if duct else .62, clearance, rise))
    return result


def add_load(trunk, props=None):
    # A fixed child contributes mass and inertia but no joint, actuator or stabilizer.
    body = ET.SubElement(trunk, "body", name="inspection_equipment")
    if props and props.payload_kg > SADDLE_KG:
        # The instrument pack on the trunk's rail: mass and inertia only, no collision, not drawn.
        geom(body, "instrument_pack", pos=(props.ballast_x_m, 0, .060), size=(.030, .040, .030),
             rgba=(0, 0, 0, 0), mass=props.payload_kg-SADDLE_KG)
    for name, kind, xyz, size, mass in (
        ("compute_puck", "box", (0, 0, .072), (.050, .040, .015), .25),
        ("lidar_mast", "cylinder", (0, 0, .09975), (.012, .01275), .033),
        ("lidar_head", "cylinder", (0, 0, .14975), (.0435, .03725), .447),      # top 0.187 m above the origin
        # the guard frame: top 0.199 m above the origin
        ("lidar_cage_front", "box", (.096, 0, .193), (.006, .085, .006), .025),
        ("lidar_cage_rear", "box", (-.096, 0, .193), (.006, .085, .006), .025),
        ("lidar_cage_left", "box", (0, .079, .193), (.102, .006, .006), .025),
        ("lidar_cage_right", "box", (0, -.079, .193), (.102, .006, .006), .025),
        ("lidar_cage_fl", "box", (.096, .079, .122), (.006, .006, .065), .025),
        ("lidar_cage_fr", "box", (.096, -.079, .122), (.006, .006, .065), .025),
        ("lidar_cage_rl", "box", (-.096, .079, .122), (.006, .006, .065), .025),
        ("lidar_cage_rr", "box", (-.096, -.079, .122), (.006, .006, .065), .025),
        ("profiler_bracket", "box", (.300, 0, -.060), (.075, .020, .010), .10),
        ("profiler_head", "box", (.3655, 0, -.1255), (.0245, .0985, .0375), .74),  # bottom 0.163 m below the origin
    ):
        geom(body, name, kind, xyz, size, rgba=(.42, .43, .46, 1), mass=mass, contype=1, conaffinity=1)


def add_fittings(root, world, items):
    metal = (.43, .46, .48, 1)

    def prism(name, x, y, half, section, color):
        verts = [(xx, yy, zz) for yy in (-half, half) for xx, zz in section]
        ET.SubElement(root.find("asset"), "mesh", name=name, vertex=vec(v for pt in verts for v in pt))
        geom(world, name, "mesh", (x, y, 0.), (), rgba=color, mesh=name,
             contype=1, conaffinity=1, friction="1 .02 .001")

    for f in items:
        w, y, x = (f.hi-f.lo)/2, (f.hi+f.lo)/2, f.x
        cover_x = x
        if f.kind == "duct":
            top = f.clearance+DUCT_DEPTH
            geom(world, "duct_box", pos=(x, y, (top+f.clearance)/2), size=(f.length/2, w+.045, DUCT_DEPTH/2),
                 rgba=metal, contype=1, conaffinity=1)
            for name, rim in zip(DUCT[1:], (f.lo-.025, f.hi+.025)):
                geom(world, name, pos=(x, rim, (.32+f.clearance)/2), size=(.075, .024, (f.clearance-.32)/2),
                     rgba=metal, contype=1, conaffinity=1)
        else:
            top = DECK_Z
            geom(world, SUPPORT[0], pos=(x, y, top/2), size=(f.length/2, w, top/2),
                 rgba=metal, contype=1, conaffinity=1, friction="1 .02 .001")
            for sign, name in zip((-1, 1), SUPPORT[1:3]):
                prism(name, x+sign*f.length/2, y, w,
                      ((0., 0.), (sign*BEVEL, 0.), (sign*BEVEL, TOE), (0., top)), metal)
            # Two wheel tracks at +/-0.125 m. All guide surfaces are legal footholds.
            section = ((-.305, top), (.305, top), (.249, top+f.rise), (.061, top+f.rise))
            for name, (lo, hi) in zip(SUPPORT[3:], ((-.050, .050), (-w, -.200), (.200, w))):
                prism(name, x, y+(lo+hi)/2, (hi-lo)/2, section, (.55, .56, .54, 1))
            top += f.rise
            cover_x += COVER_OFFSET
        # Identical flat access covers on both fittings. Hue locates a cover, not its height.
        geom(world, f.kind+"_cover", pos=(cover_x, y, top+.0007), size=(.080, .039, .0007),
             rgba=(.06, .65, .68, 1))
