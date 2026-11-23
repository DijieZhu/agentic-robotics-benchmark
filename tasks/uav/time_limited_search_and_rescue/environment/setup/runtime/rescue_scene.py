import os
"""Procedural local-view training wing and its rendered collision geometry."""
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pbr_scene import prop
from floor_finish import install
from scene_assets import asset_root, geom, materials, robot_meshes
import rescue_dressing

SIDE = 5
START_Z = 1.30
SLOT_SILL, SLOT_LINTEL = .43, .89
DECK, SLAB, WELL_HALF, WALL_TOP = 2.60, .22, 1.17, 2.60
SERVICE_HALF = .54
IRON = (.61, .26, .125, 1)
DARK = (.075, .112, .126, 1)
WALL = (.65, .70, .70, 1)
WHITE = (.94, .95, .91, 1)
AMBER = (.95, .67, .08, 1)
TEAL = (.022, .74, .39, 1)
BLUE = (.05, .32, .84, 1)
ORANGE = (1., .26, .018, 1)
PURPLE = (.72, .19, .86, 1)
RED = (.90, .045, .045, 1)
STEEL = (.06, .79, .91, 1)
FLANGE_RADII = (.041, .093)
# Inner throat and outer seat of the closed collet, in metres.
COLLETS = ((.027, .054), (.074, .106))
DOCK_Z = .224
LITTER = (.22, .90, .072)
LIME = (.48, .98, .035, 1)
GRIP_TAPE = (.37, .08, .98, 1)
THROAT = .55
POCKET_HALF_DEPTH = 2.30


def edge(a, b):
    return tuple(sorted((a, b)))


def axis_at(axis, i):
    if i < 0:
        return 2*axis[0]-axis[1]
    if i >= len(axis):
        return 2*axis[-1]-axis[-2]
    return axis[i]


def boundaries(axis, i):
    center = axis[i]
    return (axis_at(axis, i-1)+center)/2, (axis_at(axis, i+1)+center)/2


def interior_edges():
    for y in range(1, SIDE):
        for x in range(SIDE):
            for dx, dy in ((1, 0), (0, 1)):
                if x+dx < SIDE and y+dy < SIDE:
                    yield edge((x, y), (x+dx, y+dy))


def distances(start, edges):
    dist = {start: 0}
    visit = deque([start])
    while visit:
        a = visit.popleft()
        for u, v in edges:
            b = v if u == a else u if v == a else None
            if b is not None and b not in dist:
                dist[b] = dist[a]+1
                visit.append(b)
    return dist


@dataclass(frozen=True)
class Layout:
    halls: frozenset
    wells: tuple
    inlet: int
    outlet: int
    rooms: tuple
    pod: tuple
    target: tuple
    tone: float
    columns: tuple
    rows: tuple
    launch: tuple
    gains: tuple
    pad: tuple
    flange: float
    depot: tuple
    loading: tuple

    def xy(self, cell):
        return np.array([axis_at(self.columns, cell[0]), axis_at(self.rows, cell[1])])

    def well_half(self, col):
        return SERVICE_HALF if col == self.wells[0][0] else WELL_HALF

    @property
    def inlet_y(self):
        return (self.rows[0]+self.rows[1])/2

    def dock(self, slot):
        return (self.columns[-1]+.32*(2*slot-1), .08, DOCK_Z)

    @property
    def pocket(self):
        return np.array([self.columns[self.outlet], boundaries(self.rows, 0)[0]-.90-POCKET_HALF_DEPTH])

    @property
    def receiving_pad(self):
        return self.pad_at(self.loading[1])

    def pad_at(self, side):
        return self.pocket + [side*3.80+self.pad[0], side*self.loading[2]+self.pad[1]]

    @property
    def bounds(self):
        return ((min(boundaries(self.columns, 0)[0], self.pocket[0]-5.5),
                 max(boundaries(self.columns, SIDE-1)[1], self.pocket[0]+5.5)),
                (self.pocket[1]-POCKET_HALF_DEPTH-.15, boundaries(self.rows, SIDE-1)[1]-.05),
                (.39, DECK+2.09))


def layout_for(seed):
    # Site, mirror and incident are independent. Flipping the incident bit
    # changes the dispatch sign, shutters and receiving decks, not the search.
    rng = np.random.default_rng(seed >> 2)
    inlet, outlet = int(rng.choice((1, 2))), 4
    wells = ((inlet, 3), (int(rng.choice((3, 4))), 3))
    cross_row = int(rng.choice((1, 2)))
    halls = set()
    for level in (0, 1):
        for region in (range(3), range(3, 5)):
            cells = {(x,y) for x in region for y in range(1,4)}
            opts = [e for e in interior_edges() if set(e) <= cells]
            # The service well has an approach from the south.
            if region.start == 0 and (level == 0 or rng.integers(2)):
                opts = [e for e in opts if wells[0] not in e or e[0][0] == e[1][0]]
            groups = {v: {v} for v in cells}
            extras = []
            for i in rng.permutation(len(opts)):
                a, b = opts[i]
                if b in groups[a]:
                    extras.append((a,b))
                    continue
                joined = groups[a] | groups[b]
                for node in joined:
                    groups[node] = joined
                halls.add(edge((*a,level), (*b,level)))
            for a,b in extras[:1]:
                halls.add(edge((*a,level), (*b,level)))
    halls.add(edge((2,cross_row,1), (3,cross_row,1)))
    # Both hoists serve the connected warehouse on either storey. The short
    # inspection approach and the upper transfer gallery are ordinary doors.
    halls.add(edge((2,1,0), (3,1,0)))
    for row in (1,2):
        halls.add(edge((inlet,row,0), (inlet,row+1,0)))
    for x in range(min(v[0] for v in wells), max(v[0] for v in wells)):
        halls.add(edge((x,3,1), (x+1,3,1)))
    for x,y in wells:
        halls.add(edge((x,y,0), (x,y,1)))
    bays = ((3,1),(4,2)) if rng.integers(2) else ((3,2),(4,1))
    rooms = tuple((*v,k) for k in (0,1) for v in bays)
    target = (*bays[int(rng.integers(len(bays)))],1)
    dx, dy = rng.uniform(-.16, .16, 2)
    yaw = float(rng.uniform(-.50, .50))
    tone = float(rng.uniform(.92, 1.05))
    columns = np.r_[0., np.cumsum(rng.uniform(2.60, 2.80, SIDE-1))]
    columns -= columns[2]
    rows = np.r_[0., np.cumsum(rng.uniform(2.60, 2.80, SIDE-1))]
    start_x, start_y = rng.uniform(-.13, .13, 2)
    gains = tuple(rng.uniform(.96, 1.045, 2))
    pad_x, pad_y = rng.uniform(-.16, .16, 2)
    flange = FLANGE_RADII[int(rng.integers(2))]
    depot = tuple(COLLETS[i] for i in rng.permutation(2))
    yaw += int(rng.integers(2))*np.pi/2
    loading = (rng.choice((-1, 1))*rng.uniform(.56, .66), 1 if seed & 2 else -1,
               rng.choice((-1, 1))*rng.uniform(.55, .70))

    def mirror(p):
        return (SIDE-1-p[0], *p[1:]) if seed & 1 else p

    if seed & 1:
        dx, yaw, start_x, pad_x = -dx, -yaw, -start_x, -pad_x
        columns = -columns[::-1]
    target = mirror(target)
    px, py = columns[target[0]]+dx, rows[target[1]]+dy
    return Layout(frozenset(edge(mirror(a), mirror(b)) for a,b in halls),
                  tuple(map(mirror,wells)), mirror((inlet,1))[0], mirror((outlet,1))[0],
                  tuple(map(mirror, rooms)), (px,py,yaw), target, tone, tuple(columns), tuple(rows),
                  (start_x,start_y), gains, (pad_x,pad_y), flange, depot, loading)


def build(layout):
    xy = layout.xy
    root = ET.fromstring("""
    <mujoco model="indoor rescue training wing">
        <compiler angle="degree" autolimits="true"/>
        <option timestep=".005" integrator="implicitfast"/>
        <visual>
          <global offwidth="256" offheight="256"/>
          <headlight ambient=".54 .54 .54" diffuse=".26 .26 .26" specular="0 0 0"/>
          <map znear=".001" zfar="6"/>
        </visual>
        <worldbody>
          <light directional="true" dir=".3 .42 -1" diffuse=".62 .60 .58" castshadow="false"/>
          <geom name="floor" type="plane" size="16 16 1" friction=".9 .02 .005"/>
          <body name="uav" gravcomp="1">
            <inertial pos="0 0 0" mass=".8" diaginertia=".012 .012 .015"/>
            <joint name="x" type="slide" axis="1 0 0" damping="9"/>
            <joint name="y" type="slide" axis="0 1 0" damping="9"/>
            <joint name="z" type="slide" axis="0 0 1" damping="9"/>
            <joint name="yaw" type="hinge" axis="0 0 1" damping=".65"/>
            <camera name="down" pos="0 .46 -.145" fovy="94"/>
            <camera name="forward" pos="0 .17 .045" xyaxes="1 0 0 0 .12 .99277" fovy="94"/>
          </body>
          <body name="pod" pos="0 2 .074">
            <freejoint name="pod_free"/>
            <geom name="litter_board" type="box" size=".22 .9 .072" mass=".20"
                  rgba=".12 .33 .31 1" friction=".95 .04 .01"/>
          </body>
        </worldbody>
        <equality><weld name="clamp" body1="uav" body2="pod" active="false"
                        solref=".008 1" relpose="0 0 -.31 1 0 0 0"/></equality>
        <actuator>
          <position name="mx" joint="x" kp="190" kv="16"/>
          <position name="my" joint="y" kp="190" kv="16"/>
          <position name="mz" joint="z" kp="190" kv="16"/>
          <position name="heading" joint="yaw" kp="25" kv="1.1"/>
        </actuator>
    </mujoco>""")
    asset = materials(root)
    decals = {}
    install(asset, decals)
    w = root.find("worldbody")
    w.find("geom[@name='floor']").set("material", "quarry")
    uav = w.find("body[@name='uav']")
    robot_meshes(root, uav, "drone")
    geom(uav, "gripper", "cylinder", (0, 0, -.133), (.035, .098), DARK)
    for side in (-1, 1):
        geom(uav, f"jaw{side}", pos=(side*.053, 0, -.207), size=(.025, .050, .025), rgba=WALL)
    pod = w.find("body[@name='pod']")
    pod.find("geom").set("size", " ".join(map(str, LITTER)))
    for end in (-1, 1):
        geom(pod, f"end_grip{end}", pos=(0, end*.81, .078),
             size=(.155, .035, .006), rgba=GRIP_TAPE)
    for side in (-1, 1):
        geom(pod, f"frame_rail{side}", pos=(side*.199, 0, .079),
             size=(.020, .89, .009), rgba=WHITE)
    geom(pod, "shell", pos=(0, 0, .053), size=(.155, .26, .051), rgba=ORANGE)
    geom(pod, "torso", "ellipsoid", (0, .24, .113), (.13, .24, .055), AMBER)
    geom(pod, "head", "ellipsoid", (0, .57, .12), (.092, .108, .075), AMBER)
    for side in (-1, 1):
        geom(pod, f"leg{side}", pos=(side*.064, -.38, .094), size=(.044, .30, .030), rgba=AMBER)
        geom(pod, f"arm{side}", pos=(side*.15, .17, .10), size=(.026, .19, .024), rgba=AMBER)
    geom(pod, "flange", "cylinder", (0, 0, .164), (layout.flange, .023), STEEL)
    geom(pod, "stem", "cylinder", (0, 0, .211), (.020, .025), DARK)
    # Same top cover on both fittings; read the load-bearing flange from the side.
    geom(pod, "fitting_cover", "cylinder", (0, 0, .247), (.132, .012), DARK)
    geom(pod, "centering_button", "cylinder", (0, 0, .261), (.080, .003), (.04, .90, .06, 1))
    xmin, xmax = boundaries(layout.columns, 0)[0], boundaries(layout.columns, SIDE-1)[1]
    ymax = boundaries(layout.rows, SIDE-1)[1]
    obstacles = []
    def slab(x0, x1, y0, y1, top, kind):
        name = f"{kind}{len(obstacles)}"
        geom(w, name, pos=((x0+x1)/2,(y0+y1)/2,top-SLAB/2),
             size=((x1-x0)/2,(y1-y0)/2,SLAB/2), material="terrazzo",
             rgba=(1,1,1,1), contype="1", conaffinity="1")
        obstacles.append((kind,name))

    # An uninterrupted structural slab above both wards. Only the two
    # enclosed rear wells are cut out; the staging/loading rooms are roofed.
    wy = layout.rows[4]
    slab(xmin, xmax, layout.inlet_y, wy-WELL_HALF, DECK, "slab")
    slab(xmin, xmax, wy+WELL_HALF, ymax, DECK, "slab")
    last = xmin
    for wx in sorted(layout.columns[x] for x,_ in layout.wells):
        slab(last, wx-WELL_HALF, wy-WELL_HALF, wy+WELL_HALF, DECK, "slab")
        last = wx+WELL_HALF
    slab(last, xmax, wy-WELL_HALF, wy+WELL_HALF, DECK, "slab")
    # Cast-iron deck frames: their inner faces ARE the four boundaries of
    # the opening. The service reducer fills the space back to the slab.
    for col,_ in layout.wells:
        wx, inner, outer = layout.columns[col], layout.well_half(col), WELL_HALF+.08
        for axis in (0,1):
            for sign in (-1,1):
                pos = [wx,wy,DECK-SLAB/2+.01]
                pos[axis] += sign*(outer+inner)/2
                size = [inner,outer,SLAB/2+.01]
                size[axis] = (outer-inner)/2
                name = f"aperture_{col}_{axis}_{sign}"
                geom(w,name,pos=pos,size=size,rgba=IRON,contype="1",conaffinity="1")
                obstacles.append(("slab",name))
    (left,right),(bottom,_),_ = layout.bounds
    slab(left-.35,right+.35,bottom-.35,layout.inlet_y,DECK,"roof")
    slab(xmin-.15,xmax+.15,layout.inlet_y-.075,ymax+.075,2*DECK,"roof")
    shutters = {}

    def sign(text, x, y, z, width=.34, vertical=False):
        key = text.lower().replace(" ", "_")
        filename = key+".png"
        if filename not in decals:
            im = Image.new("RGB", (256, 64), (27, 42, 48))
            ImageDraw.Draw(im).text((128, 31), text, anchor="mm", fill=(241, 246, 240),
                                    font=ImageFont.load_default(size=35))
            out = BytesIO()
            im.save(out, format="PNG")
            decals[filename] = out.getvalue()
            ET.SubElement(asset, "texture", name=key, type="2d", file=filename)
            ET.SubElement(asset, "material", name=key, texture=key, texrepeat="1 1")
        geom(w, f"sign{len(w)}", "plane", (x, y, z if vertical else z+.012),
             (width, .069 if vertical else .085, .002), material=key,
             **({"xyaxes":"1 0 0 0 0 1"} if vertical else {}))

    sign("STAGING", 0., -.64, 0.)
    # Wall-mounted wayfinding display. The arrow points in physical space;
    # it is not a room identifier or color code.
    direction = layout.loading[1]
    sy = layout.inlet_y-.102
    geom(w,"dispatch_housing",pos=(0,sy,1.75),size=(.57,.022,.26),rgba=DARK)
    obstacles.append(("wall","dispatch_housing"))
    sign("RECEIVING",0,sy-.026,1.90,.29,vertical=True)
    geom(w,"dispatch_shaft",pos=(0,sy-.026,1.69),size=(.365,.003,.026),rgba=WHITE)
    for v in (-1,1):
        geom(w,f"dispatch_vane{v}",pos=(direction*.283,sy-.027,1.69+v*.078),
             size=(.125,.003,.025),euler=f"0 {direction*v*43} 0",rgba=WHITE)
    sign("TOOL DEPOT", layout.columns[-1], -.47, 0.)
    for slot, (throat, seat) in enumerate(layout.depot):
        name = f"collet{slot}"
        x, y, z = layout.dock(slot)
        stand = ET.SubElement(w, "body", name=name+"_stand", pos=f"{x} {y} {z}")
        geom(stand, name+"_post", "cylinder", (0, 0, .089-z), (.035, .089), DARK)
        head = ET.SubElement(w, "body", name=name, pos=f"{x} {y} {z}")
        ET.SubElement(head, "freejoint", name=name+"_free")
        ET.SubElement(head, "inertial", pos="0 0 0", mass=".035",
                      diaginertia=".00015 .00015 .00015")
        # Two semicircular jaws, with a retaining lip beneath the recessed seat.
        for layer, radius, height in (("seat", seat, 0.), ("lip", throat, -.027)):
            outer = seat+.027
            mid = (radius+outer)/2
            for i in range(16):
                if i in (0, 8):
                    continue
                a = i*np.pi/8
                geom(head, f"{name}_{layer}{i}", pos=(mid*np.cos(a), mid*np.sin(a), height),
                     size=((outer-radius)/2, radius*np.tan(np.pi/16), .013), rgba=STEEL,
                     euler=f"0 0 {np.degrees(a)}")
        eq = root.find("equality")
        ET.SubElement(eq, "weld", name=name+"_dock", body1=name+"_stand", body2=name,
                      relpose="0 0 0 1 0 0 0", solref=".008 1")
        ET.SubElement(eq, "weld", name=name+"_mount", body1="uav", body2=name,
                      active="false", solref=".008 1")

    def wall(center, half, kind, color=WALL):
        name = f"solid{len(obstacles)}"
        obstacles.append((kind, name))
        geom(w, name, pos=center, size=half, rgba=color)

    def boundary(a, b, route=None, level=0):
        base_z = level*DECK
        center = (xy(a)+xy(b))/2
        normal = 0 if a[0] != b[0] else 1
        tangent = 1-normal
        gate_half = .51 if route == "inspection" else .74
        axis = layout.rows if tangent == 1 else layout.columns
        lo, hi = boundaries(axis, a[tangent])
        lo, hi = lo-center[tangent], hi-center[tangent]
        well = route == "well"

        def section(mid, span, lo, hi, kind, color):
            pos, half = [*center, base_z+(lo+hi)/2], [.075, .075, (hi-lo)/2]
            pos[tangent] += mid
            half[tangent] = span
            wall(pos, half, kind, color)
            if kind == "wall":
                # Continuous timber skirting ends with the actual wall at a jamb.
                for side, room in ((-1, a), (1, b)):
                    base = pos.copy()
                    base[normal] += side*(half[normal]+.003)
                    base[2] = base_z+.064
                    trim = half.copy()
                    trim[normal], trim[2] = .003, .064
                    wall(base, trim, kind, (.39, .26, .17, 1))

        if route is None:
            section((lo+hi)/2, (hi-lo)/2+.026, 0., WALL_TOP, "wall", WALL)
            return
        color = PURPLE if route == "inspection" else BLUE if route == "cargo" else AMBER if well else WHITE
        for i, (here, there) in enumerate(((a, b), (b, a))):
            # A wear plate extends from the physical opening onto its approach.
            if well and here[1] == 4 and level:
                continue  # the upper well has air, not a floating wear plate
            run = .84 if well else .79
            pos = center+(xy(here)-center)*(.075+run/2)/np.linalg.norm(xy(here)-center)
            size = [run/2, run/2, .004]
            size[tangent] = gate_half-.07
            name = f"apron{len(w)}"
            geom(w, name, pos=(*pos, base_z+.007), size=size, rgba=color)
        for start, end in ((lo, -gate_half), (gate_half, hi)):
            section((start+end)/2, (end-start)/2+.014, 0., WALL_TOP, "wall", WALL)
        if route == "inspection":
            section(0., gate_half, 0., SLOT_SILL, "inspection", AMBER)
            section(0., gate_half, SLOT_LINTEL, WALL_TOP, "inspection", AMBER)
            sign("INSPECTION", *xy((a[0], 0)), 0.)
        elif well:
            sign("HOIST WELL", *xy(a), base_z, .38)
        elif route == "cargo":
            section(0., gate_half, 2.00, WALL_TOP, "cargo", BLUE)
            x, y = center
            body = ET.SubElement(w, "body", name="cargo_shutter", mocap="true", pos=f"{x} {y} 1.025")
            geom(body, "cargo_leaf", size=(gate_half, .075, 1.025), rgba=BLUE)
            obstacles.append(("cargo", "cargo_leaf"))
            sign("CARGO", *xy((a[0], 0)), 0.)
            for z in np.linspace(-.90, .90, 9):
                geom(body, f"shutter_rib{z:.3f}", pos=(0, 0, z), size=(gate_half, .084, .018), rgba=DARK)
                obstacles.append(("cargo", f"shutter_rib{z:.3f}"))
            for side in (-1, 1):
                geom(w, f"cargo_jamb{side}", pos=(x+side*(gate_half+.012), y, 1.025),
                     size=(.030, .10, 1.025), rgba=BLUE)
                obstacles.append(("cargo", f"cargo_jamb{side}"))
    for level in (0,1):
        for y in range(0 if not level else 1, SIDE):
            boundary((-1,y),(0,y),level=level)
            boundary((SIDE-1,y),(SIDE,y),level=level)
        for x in range(SIDE):
            if not level and x != layout.outlet:
                boundary((x,-1),(x,0))
            boundary((x,SIDE-1),(x,SIDE),level=level)
            route = None if level else ("inspection" if x == layout.inlet else
                                        "cargo" if x == layout.outlet else None)
            boundary((x,0),(x,1),route,level)
        for a,b in interior_edges():
            route = "open" if edge((*a,level),(*b,level)) in layout.halls else None
            if a in layout.wells and b == (a[0],4):
                route = "well"
            boundary(a,b,route,level)
    # Full-height hoist guide channels on the back wall, visible from
    # below as well as above. The rim is the exposed edge of the slab.
    for i,(col,_) in enumerate(layout.wells):
        for side in (-1,1):
            wall((layout.columns[col]+side*.70, ymax-.103, DECK),
                 (.046,.025,DECK), "wall", AMBER)

    # The only entrance to receiving is through two full-height loading throats.
    cx, cy = layout.pocket
    entry, side, stagger = layout.loading
    north, south = cy+POCKET_HALF_DEPTH, cy-POCKET_HALF_DEPTH

    def panel(axis, fixed, a, b, kind="loading_wall", depth=.075):
        mid, span = (a+b)/2, abs(b-a)/2
        pos, size = ([fixed, mid], [depth, span]) if axis == 0 else ([mid, fixed], [span, depth])
        wall((*pos, WALL_TOP/2), (*size, WALL_TOP/2), kind,
             (.32, .57, .59, 1) if kind.startswith("throat") else WALL)

    def throat(axis, fixed, center, lo, hi, inward, name):
        panel(axis, fixed, lo, center-THROAT, name, .28)
        panel(axis, fixed, center+THROAT, hi, name, .28)
        # Reflective toe lines run along the clear inner faces onto the approach.
        for j in (-1, 1):
            p = [fixed-inward*.40, center+j*(THROAT-.025)]
            h = [.84, .025]
            if axis == 1:
                p.reverse()
                h.reverse()
            geom(w, f"{name}_{inward}_line{j}", pos=(*p, .010), size=(*h, .006), rgba=LIME)

    throat(1, north, cx+entry, cx-2.10, cx+2.10, -1, "throat_entry")
    for x in boundaries(layout.columns, layout.outlet):
        panel(0, x, north, north+.90)
    panel(1, south, cx-5.45, cx+5.45)
    for dx in (-1,1):
        exit_x,far,lane = cx+dx*2.10,cx+dx*5.45,cy+dx*stagger
        throat(0,exit_x,lane,south,north,dx,"throat_exit")
        panel(0,far,south,north)
        panel(1,north,exit_x,far)
        # Roller case and, where deployed, the floor-to-headrail shutter.
        wall((exit_x,lane,WALL_TOP-.11),(.115,.60,.11),"loading_wall",DARK)
        px,py = layout.pad_at(dx)
        if dx!=side:
            name = f"receiving_shutter_{dx}"
            geom(w,name,pos=(exit_x,lane,WALL_TOP/2),size=(.09,.56,WALL_TOP/2),
                 rgba=(.56,.11,.09,1),contype="1",conaffinity="1")
            obstacles.append(("loading_wall",name))
            for v in (-1,1):
                geom(w,f"closing_cross{dx}_{v}",pos=(exit_x-dx*.092,lane,1.32),
                     size=(.002,.61,.046),euler=f"{v*45} 0 0",rgba=WHITE)
            for i,z in enumerate((.28,.56,.82,1.84,2.12)):
                geom(w,f"slat{dx}_{i}",pos=(exit_x-dx*.091,lane,z),size=(.001,.55,.012),rgba=DARK)
        # A stowed station has no horizontal receiving surface. Both real
        # mats exist and use the same collision and rest test.
        upright = dx!=side
        deck = ET.SubElement(w,"body",name=f"deck{dx}",
                             pos=f"{far-dx*.18 if upright else px} {py} {1.28 if upright else .006}",
                             euler=f"0 {-dx*90 if upright else 0} 0")
        geom(deck,f"receiving_mat{dx}",size=(1.25,.74,.006),rgba=TEAL,
             contype="1",conaffinity="1",friction=".95 .04 .01")
        geom(deck,f"pad{dx}",pos=(0,0,.008),size=(.56,.53,.002),rgba=BLUE)
        for v in (-1,1):
            geom(deck,f"pad_side{dx}_{v}",pos=(v*.39,0,.013),size=(.030,.42,.002),rgba=WHITE)
        geom(deck,f"pad_cross{dx}",pos=(0,0,.013),size=(.39,.030,.002),rgba=WHITE)
        sign("RECEIVING",px,py-.61,0.)
    sign("LITTER TRANSFER", cx, north+.56, 0., .49)
    sign("TURNING POCKET", cx, cy-.49, 0., .48)
    # Bay paint runs around (not through) the central pickup area.
    for i, room in enumerate(layout.rooms):
        x, y = xy(room)
        z = room[2]*DECK
        sign("RESCUE BAY", x, y-.43, z)
        for side in (-1, 1):
            geom(w, f"bay{i}_rail{side}", pos=(x+side*.47, y, z+.005),
                 size=(.023, .57, .004), rgba=TEAL)
    for side, x in enumerate((xmin-.52, xmax+.52)):
        prop(root, asset_root(), "utility_box_01", f"cabinet{side}", (x, 0, 0), (.52, .61, .87))

    rescue_dressing.dress(root, layout, obstacles)
    return ET.tostring(root, encoding="unicode"), decals, obstacles, shutters

