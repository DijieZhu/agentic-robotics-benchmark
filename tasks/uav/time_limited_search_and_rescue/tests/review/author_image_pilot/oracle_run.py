"""RGB-only local exploration and evacuation; no layout or simulator imports."""
import base64
from collections import deque
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from architecture import patches, translation, wall_face

sys.path.insert(0, "/opt/visual_robotics/client")
PITCH, APRON, DECK = 2.70, .67, 2.60
DOWN_OFFSET = .46
PACE = np.array([.32, .32, .19])
TRAVEL, SLOT = 1.30, .71
TURN = math.pi/6
FOCAL = 128/math.tan(math.radians(47))
DIRECTIONS = ((0, 1), (1, 0), (-1, 0), (0, -1))


def basis(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def decode(observation, camera=0):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(observation["images"][camera]))),
                      dtype=np.int16)


def apron_color(rgb, local):
    scale = FOCAL/(TRAVEL-.145-.011)
    u, v = np.rint(np.array([127.5, 127.5])+(local-[0., DOWN_OFFSET])*[scale, -scale]).astype(int)
    r, g, b = np.median(rgb[v-3:v+4, u-3:u+4], axis=(0, 1))
    if r > 150 and g > 120 and b < 108:
        return "landing"
    if r > 155 and g < 100 and b < 100:
        return "closed"
    if b > 145 and r > 145 and g < min(r, b)-45:
        return "inspection"
    if b > 140 and b > r+60 and b > g+40:
        return "cargo"
    if min(r, g, b) > 175 and max(r, g, b)-min(r, g, b) < 45:
        return "open"
    return "wall"


def find_pod(rgb, command, floor):
    r, g, b = rgb.transpose(2, 0, 1)
    labels, _ = ndimage.label((g > 155) & (r < 90) & (b < 75))
    for i, count in enumerate(np.bincount(labels.ravel())):
        if not i or not 110 < count < 950:
            continue
        rows, cols = np.where(labels == i)
        if not (12 < np.ptp(cols) < 32 and 12 < np.ptp(rows) < 32):
            continue
        scale = (command[2]-.145-floor-.337)/FOCAL
        local = [(cols.mean()-127.5)*scale, DOWN_OFFSET-(rows.mean()-127.5)*scale]
        return command[:2]+basis(command[3]) @ local


def steel(rgb):
    r, g, b = rgb.transpose(2, 0, 1)
    return (r < 115) & (g > 145) & (b > 165) & (g > r+65)


def survey(p, target, cell):
    # Pick the side with room to offset from the sled. At cruise height the
    # common cover hides the flange; see under it in the forward camera.
    floor = DECK*cell[2]
    relative = target-PITCH*np.asarray(cell[:2])
    side = -1 if relative[1] > 0. else 1
    offset = np.array([0., side*.55])
    diagonal = abs(relative[1]) < .09
    if diagonal:
        # Keep both horizontal components clear of the nearest walls.
        offset = np.where(relative > 0., -1., 1.)*(.55/math.sqrt(2))
    p.face(*(-offset))
    p.go(*(target+offset), .43+floor, settle=3)
    rows, cols = np.where(steel(decode(p.obs, 1)[130:244, 75:181]))
    if rows.size < 40:
        raise RuntimeError("no flange visible under the fitting cover")
    width = int(np.ptp(cols)+1)
    wide = width >= 37
    p.trace.append({"event": "flange_survey", "cell": cell, "width_px": width,
                    "profile": "wide" if wide else "narrow", "frame": p.obs["frame_id"]})
    p.go(*p.command[:2], TRAVEL+floor)
    if diagonal:
        p.face(0, -side)
    p.cell(cell, TRAVEL+floor, settle=1)
    p.register(cell)
    return wide


def rack_heads(rgb):
    mask = steel(rgb)
    tools = []
    scale = (TRAVEL-.145-.237)/FOCAL
    for left, right in ((60, 127), (128, 196)):
        rows, cols = np.where(mask[145:206, left:right])
        if rows.size < 50:
            return []
        u = left+(cols.min()+cols.max())/2
        v = 145+(rows.min()+rows.max())/2
        tools.append((int(np.ptp(cols)), (u-127.5)*scale, DOWN_OFFSET-(v-127.5)*scale))
    return sorted(tools)


def equip(p, wide, depot):
    p.look(0,1)
    tools = rack_heads(decode(p.obs))
    if not tools:
        raise RuntimeError("depot collets not visible")
    width, dx, dy = tools[int(wide)]
    p.trace.append({"event": "depot_choice", "profile": "wide" if wide else "narrow",
                    "slot": int(dx > 0), "width_px": width, "frame": p.obs["frame_id"]})
    center = p.command[:2].copy()
    p.go(*(center+[dx, dy]), .43, settle=3)
    p.grip(1.)
    p.go(*(center+[dx, dy]), TRAVEL)
    p.trace.append({"event": "depot_departure", "frame": p.obs["frame_id"]})
    p.cell((depot, 0), TRAVEL, settle=1)
    p.register((depot, 0))


def floor_points(p, mask, height):
    rows, cols = np.where(mask)
    scale = (p.command[2]-.145-height)/FOCAL
    local = np.column_stack(((cols-127.5)*scale, DOWN_OFFSET-(rows-127.5)*scale))
    return p.command[:2]+local @ basis(p.command[3]).T


def sight_litter(p, target, floor):
    # Center the camera, not the UAV, over the board to see both end grips.
    p.face(0, 1)
    p.go(target[0], target[1]-DOWN_OFFSET, TRAVEL+floor, settle=2)
    r, g, b = decode(p.obs).transpose(2, 0, 1)
    labels, n = ndimage.label((b > 150) & (b > r+55) & (b > g+80) & (r > g+18))
    marks = [floor_points(p, labels == i, floor+.157).mean(axis=0)
             for i, size in enumerate(np.bincount(labels.ravel())) if i and size >= 35]
    pairs = [(a, b) for i, a in enumerate(marks) for b in marks[i+1:]
             if 1.48 < np.linalg.norm(a-b) < 1.76]
    if len(pairs) != 1:
        raise RuntimeError("cannot see both litter end grips")
    a, b = pairs[0]
    axis = math.atan2(a[0]-b[0], b[1]-a[1])
    heading = round((axis+math.pi/2)/(math.pi/2))*math.pi/2
    p.trace.append({"event": "litter_axis", "span_m": float(np.linalg.norm(a-b)),
                    "axis_deg": math.degrees(axis), "frame": p.obs["frame_id"]})
    p.face(-math.sin(heading), math.cos(heading))
    p.litter_relative = math.remainder(axis-p.command[3], math.pi)
    return (a+b)/2


def read_throat(p, axis, toward=None):
    # The two lime toe strips lie along the inner jamb faces. Fit their spacing
    # and use the approach ends (1.24 m before the throat center plane).
    r, g, b = decode(p.obs).transpose(2, 0, 1)
    labels, n = ndimage.label((g > 155) & (g > r+20) & (r > 90) & (b < 95))
    strips = []
    for i, count in enumerate(np.bincount(labels.ravel())):
        if not i or count < 20:
            continue
        pts = floor_points(p, labels == i, .016)
        if np.ptp(pts[:, axis]) > .23 and np.ptp(pts[:, 1-axis]) < .12:
            strips.append(pts)
    for i, a in enumerate(strips):
        for b in strips[i+1:]:
            v, w = np.median(a[:, 1-axis]), np.median(b[:, 1-axis])
            if not .94 < abs(v-w) < 1.17:
                continue
            coord = np.r_[a[:, axis], b[:, axis]]
            side = 1 if np.mean(coord) > p.command[axis] else -1
            if toward is not None and side!=toward:
                continue
            plane = side*(np.min(side*coord)+1.24)
            return float((v+w)/2), float(plane), side
    return None


def white_shapes(rgb):
    neutral = (rgb.min(axis=2)>175) & (np.ptp(rgb,axis=2)<46)
    labels,_ = ndimage.label(neutral)
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:
            continue
        h,w = [v.stop-v.start for v in sl]
        if w>=20 and h>=12 and w<190 and h<190:
            yield labels[sl]==i,h,w


def dispatch_arrow(rgb):
    # A horizontal shaft with taller vanes toward its tip. No text or hue.
    for shape,h,w in white_shapes(rgb):
        if not 1.8<w/h<4.8 or w<38:
            continue
        ys,xs = np.where(shape)
        wings = xs[abs(ys-(h-1)/2)>.27*h]
        if len(wings)>.11*len(xs) and len(wings) and abs(wings.mean()/w-.5)>.17:
            # The unadorned end of the shaft must be a thin central stroke.
            d = 1 if wings.mean()>w/2 else -1
            tail = ys[xs<w*.25] if d==1 else ys[xs>w*.75]
            if len(tail) and np.ptp(tail)<.35*h:
                return d
    return None


def station_view(rgb):
    for shape,h,w in white_shapes(rgb):
        if not .70<w/h<1.45 or min(w,h)<30:
            continue
        y,x = np.where(shape)
        u,v = x/(w-1),y/(h-1)
        if (np.mean(np.minimum(abs(u-v),abs(1-u-v))<.13)>.85 and
                all(np.mean((u<.27 if a==0 else u>.73) & (v<.27 if b==0 else v>.73))>.035
                    for a,b in ((0,0),(1,0),(0,1),(1,1)))):
            return "closed"
    # Positive evidence through the opening: blue target on the deployed
    # green deck. A missing cross by itself is not a clear passage.
    r,g,b = rgb[126:232,48:208].transpose(2,0,1)
    if ((b>130)&(b>r+65)&(b>g+40)).sum()>45 and ((g>105)&(g>r+55)&(g>b+15)).sum()>40:
        return "open"
    return "uncertain"


def choose_station(p,pivot,h):
    first = p.fixed_side or p.receiving_hint or 1
    for dx in (first,-first):
        p.look(dx,0)
        gate = read_throat(p,0,toward=dx)
        if gate is None:
            continue
        lane,plane,side = gate
        p.traverse(pivot[0],lane,h,h)
        p.look(dx,0)
        state = station_view(decode(p.obs,1))
        p.trace.append(dict(event="station_inspection",side=dx,state=state,frame=p.obs["frame_id"]))
        if state=="open" or p.fixed_side==dx:
            return gate
        p.traverse(*pivot,h,h)
    raise RuntimeError("no visibly open receiving deck from the turning pocket")


def transfer(p, hold_heading=False):
    h = 1.65
    sx, sy = p.command[:2].copy()
    p.go(sx, sy, h)
    p.face(0, -1)
    p.grip()
    entry = read_throat(p, 1)
    if entry is None or entry[2] != -1:
        raise RuntimeError("two loading entrance jambs not visible")
    lane, plane, _ = entry
    pivot = np.array([sx, plane-2.30])
    p.trace.append({"event": "entry_jambs", "offset_m": lane-sx, "frame": p.obs["frame_id"]})
    p.align_load(0, -1)
    p.traverse(lane, sy, h, h)
    p.traverse(lane, plane+1.30, h, h)
    near = read_throat(p, 1)
    if near is not None:
        lane = near[0]
        p.go(lane, p.command[1], h, settle=1)
    p.traverse(lane, pivot[1], h, h)
    p.traverse(*pivot, h, h)
    lane, plane, side = choose_station(p,pivot,h)
    p.trace.append({"event": "exit_jambs", "offset_m": lane-pivot[1], "side": side,
                    "frame": p.obs["frame_id"]})
    if hold_heading:
        p.align_load(0, -1)
    else:
        p.align_load(side, 0)
    p.traverse(pivot[0], lane, h, h)
    p.traverse(plane+side*1.70, lane, h, h)
    p.go(*p.command[:2], TRAVEL, settle=2)


def pad_offset(rgb):
    r, g, b = rgb.transpose(2, 0, 1)
    labels, count = ndimage.label((b > 145) & (b > r+65) & (b > g+45))
    if not count:
        return None
    largest = 1+np.argmax(np.bincount(labels.ravel())[1:])
    rows, cols = np.where(labels == largest)
    if rows.size < 2200:
        return None
    scale = (TRAVEL-.145-.008)/FOCAL
    u, v = (cols.min()+cols.max())/2, (rows.min()+rows.max())/2
    return (np.array([u, v])-127.5)*[scale, -scale]+[0., DOWN_OFFSET]


class Pilot:
    def __init__(self, client, feedback="rgb"):
        self.client = client
        self.command = np.array([0., 0., TRAVEL, 0.])
        self.obs = client.observe()
        self.trace = []
        self.path = []
        self.feedback, self.loaded = feedback, False
        self.keyframes = {}
        self.apertures = {}
        self.receiving_hint,self.fixed_side = None,None

    def act(self, action):
        if not self.obs["done"]:
            self.obs = self.client.act(action)
            delta = PACE*action[:3]
            delta[:2] = basis(self.command[3]) @ delta[:2]
            self.command[:3] += delta
            self.command[3] += TURN*action[3]

    def grip(self, value=0.):
        self.act([0., 0., 0., 0., value])

    def face(self, dx, dy):
        heading = math.atan2(-dx, dy)
        error = math.remainder(heading-self.command[3], 2*math.pi)
        while not self.obs["done"] and abs(error) > 1e-7:
            move = float(np.clip(error/TURN, -1, 1))
            self.act([0., 0., 0., move, 0.])
            error -= move*TURN

    def align_load(self, dx, dy):
        desired = math.atan2(-dx, dy)
        heading = self.command[3]+math.remainder(desired-self.command[3]-self.litter_relative, math.pi)
        self.face(-math.sin(heading), math.cos(heading))
        self.grip()

    def go(self, x, y, z, settle=0):
        target = np.array([x, y, z])
        while not self.obs["done"] and np.max(np.abs(target-self.command[:3])) > 1e-7:
            error = target-self.command[:3]
            error[:2] = basis(self.command[3]).T @ error[:2]
            move = error/PACE
            # Preserve a straight world-space segment at arbitrary payload yaw.
            move[:2] /= max(1., np.max(np.abs(move[:2])))
            self.act([*np.clip(move, -1., 1.).tolist(), 0., 0.])
        for _ in range(settle):
            self.grip()

    def look(self, dx, dy):
        self.face(dx, dy)
        self.grip()
        return decode(self.obs)

    @property
    def localize(self):
        return self.feedback != "odometry" and not (self.feedback == "prelift" and self.loaded)

    def traverse(self, x, y, z, height):
        start = self.command[:2].copy()
        target = np.array([x, y])
        distance = np.linalg.norm(target-start)
        if self.obs["done"] or distance < .12:
            self.go(x, y, z, settle=1)
            return
        self.grip()
        ax = basis(self.command[3])
        first = self.obs["frame_id"]
        scale = (height-.145)/FOCAL
        old = patches(decode(self.obs))
        total = np.zeros(2)
        matches = []
        example, last = None, first
        pieces = max(1, math.ceil(np.max(np.abs(ax.T @ (target-start)))/.55))
        previous = start
        for dest in np.linspace(start, target, pieces+1)[1:]:
            self.go(*dest, z)
            if self.obs["done"]:
                return
            step = ax.T @ (dest-previous)
            new = patches(decode(self.obs))
            prior = step*[-1, 1]/scale
            match = translation(old, new, prior, 18)
            if match is not None:
                shift, count = match
                total += ax @ (shift*[-scale, scale])
                matches.append(count)
                if example is None or count > example["inliers"]:
                    example = dict(before=last, after=self.obs["frame_id"], inliers=count,
                                   prior_px=prior.tolist(), shift_px=shift.tolist())
            else:
                total += dest-previous
            old, previous, last = new, dest, self.obs["frame_id"]
        self.grip()
        match = translation(old, patches(decode(self.obs)), [0, 0], 15)
        if match is not None:
            total += ax @ (match[0]*[-scale, scale])
        error = target-start-total
        self.trace.append({"event": "floor_track", "frame": self.obs["frame_id"], "loaded": self.loaded,
                           "start_frame": first, "target_delta": (target-start).tolist(),
                           "example": example,
                           "pieces": pieces, "registered": len(matches), "inliers": sum(matches),
                           "correction_m": error.tolist(), "applied": self.localize})
        if self.localize and np.linalg.norm(error) > .014 and matches:
            self.go(*(target+np.clip(error, -.48, .48)), z, settle=1)
        self.command[:2] = target

    def cell(self, cell, z, settle=0, track=True):
        x, y = PITCH*np.asarray(cell[:2])
        if (track and abs(self.command[2]-z) < .001 and z >= TRAVEL
                and np.linalg.norm(self.command[:2]-[x, y]) > 1.65):
            self.traverse(x, y, z, TRAVEL)
        else:
            self.go(x, y, z, settle)

    def register(self, cell):
        if self.obs["done"]:
            return
        north = self.look(0, 1)
        before = self.obs["frame_id"]
        current = patches(north)
        closure = (translation(self.keyframes[cell][1], current, [0, 0], 42)
                   if cell in self.keyframes else None)
        correction = np.zeros(2)
        faces = []
        if closure is not None:
            correction = closure[0]*[1., -1.]*(TRAVEL-.145)/FOCAL
        else:
            def face_distance(image, axis, sign):
                depth, jamb = wall_face(image)
                if depth is not None:
                    correction[axis] = sign*(depth-(PITCH/2-.081))
                    faces.append(axis)
                return jamb

            # A floor edge seen through the hoist doorway can be several
            # metres below the landing, so it is not a local floor fix.
            well_ahead = apron_color(north, np.array([0.,APRON])) == "landing"
            jamb = None if well_ahead else face_distance(north, 1, 1)
            if 1 not in faces and cell[1] > 0:
                face_distance(self.look(0, -1), 1, -1)
            if jamb is not None:
                correction[0] = jamb
            elif cell[1] > 0 or abs(cell[0]) == 2:
                sign = -1 if cell[1] == 0 and cell[0] == -2 else 1
                face_distance(self.look(sign, 0), 0, sign)
                if 0 not in faces and cell[1] > 0:
                    face_distance(self.look(-sign, 0), 0, -sign)
        self.face(0, 1)
        self.grip()
        anchor = PITCH*np.asarray(cell[:2])
        if self.localize and np.max(np.abs(correction)) > .012:
            self.go(*(self.command[:2]+np.clip(correction, -.45, .45)), self.command[2], settle=1)
        self.command[:2] = anchor
        if cell not in self.keyframes:
            self.keyframes[cell] = (self.obs["frame_id"], patches(decode(self.obs)))
        self.trace.append({"event": "station_fix", "cell": cell, "loaded": self.loaded,
                           "method": "patch_revisit" if closure is not None else "wall_geometry",
                           "reference_frame": self.keyframes[cell][0],
                           "matches": closure[1] if closure is not None else 0, "faces": faces,
                           "before_m": round(float(np.linalg.norm(correction)), 4),
                           "applied": self.localize, "before_frame": before, "frame": self.obs["frame_id"]})

    def scan(self):
        rgb = decode(self.obs)
        transform = basis(self.command[3]).T
        return rgb, {d: apron_color(rgb, transform @ np.asarray(d)*APRON) for d in DIRECTIONS}


def guide_offset(image):
    # At the forward camera's horizon, each channel intersects the
    # same horizontal plane as the camera. Range follows from separation.
    r,g,b = image[106:119].transpose(2,0,1)
    mask = ((r>145) & (g>105) & (b<100) & (r>g+12)).mean(axis=0) > .7
    cols, = np.where(mask)
    runs = [v.mean() for v in np.split(cols,np.flatnonzero(np.diff(cols)>1)+1) if 3<=len(v)<=24]
    if len(runs) != 2:
        return None
    left,right = runs
    span = right-left
    return np.array([(left+right-255)*.70/span, .17+FOCAL*1.40/(.99277*span)-1.222])


def aperture_width(rgb, height):
    """Distance between the two inner edges of the same cast-iron deck frame."""
    r,g,b = rgb.transpose(2,0,1)
    # Use the illuminated horizontal face, not its darker vertical reveal
    # (projecting that lower edge at deck height underestimates the bore).
    iron = (r>g*1.45) & (g>b*1.22) & (g>58) & (r>150)
    spans = []
    for row in iron[107:189:2]:
        xs, = np.where(row)
        runs = [v for v in np.split(xs,np.flatnonzero(np.diff(xs)>1)+1) if len(v)>=3]
        left = [v for v in runs if 12<v[0]<65 and v[-1]<117]
        right = [v for v in runs if 190<v[-1]<244 and v[0]>139]
        if left and right:
            spans.append(right[-1][0]-left[0][-1]-1)
    if len(spans)<6:
        raise RuntimeError("cannot resolve the two inner edges of the hoist frame")
    return float(np.median(spans)*(height-.145-.02)/FOCAL)


def inspect_frame(p, at):
    height = 1.75
    xy = p.command[:2].copy()
    p.go(*xy,DECK+height,settle=2)
    width = aperture_width(decode(p.obs),height)
    p.apertures[at[:2]] = width
    p.trace.append(dict(event="aperture_measurement",cell=at,loaded=p.loaded,
                        frame=p.obs["frame_id"],width_m=width))
    p.go(*xy,DECK+TRAVEL,settle=1)


def approach_well(p, at, d):
    h = TRAVEL+at[2]*DECK
    xy = PITCH*(np.asarray(at[:2])+d)
    (p.align_load if p.loaded else p.face)(*d)
    p.go(*xy,h,settle=2)
    for attempt in range(4):
        p.look(*d)
        offset = guide_offset(decode(p.obs,1)) if not p.obs["done"] else None
        if offset is None:
            if p.obs["done"]:
                return
            if attempt == 0:
                p.go(*(p.command[:2]-.18*np.asarray(d)),h,settle=2)
                continue
            raise RuntimeError("both physical hoist guides not visible")
        if np.linalg.norm(offset) < .04:
            break
        p.go(*(p.command[:2]+basis(p.command[3])@np.clip(offset,-.45,.45)),h,settle=2)
    else:
        raise RuntimeError("hoist guides did not converge")
    p.command[:2] = xy
    return offset


def leave_well(p, at, d):
    (p.align_load if p.loaded else p.face)(*(-np.asarray(d)))
    p.go(*(PITCH*np.asarray(at[:2])),TRAVEL+at[2]*DECK,settle=2)
    p.register(at)


def cross_well(p, at, there, d):
    offset = approach_well(p,at,d)
    if p.obs["done"]:
        return
    xy = p.command[:2].copy()
    if at[2] and at[:2] not in p.apertures:
        inspect_frame(p,at)
    before = p.obs["frame_id"]
    p.go(*xy,DECK-.11,settle=1)
    middle = p.obs["frame_id"]
    p.go(*xy,TRAVEL+there[2]*DECK,settle=2)
    if p.obs["done"]:
        return
    if there[2] and there[:2] not in p.apertures:
        inspect_frame(p,there)
    p.trace.append(dict(event="well_flight",cell=at,to=there,loaded=p.loaded,
                        before=before,middle=middle,frame=p.obs["frame_id"],guide_error_m=offset.tolist()))
    leave_well(p,there,d)


def fly(p, a, b, d):
    if a[2] != b[2]:
        cross_well(p,a,b,d)
    else:
        (p.align_load if p.loaded else p.face)(*d)
        p.cell(b, TRAVEL+b[2]*DECK, settle=1)
        p.register(b)


def reverse(link):
    a,b,d = link
    return (b,a,d if a[2]!=b[2] else tuple(-v for v in d))


class RoomMemory:
    """A port is exhausted by actually crossing it, never by a storey flag."""
    def __init__(self, planar=False):
        self.planar, self.crossings = planar, {}
        self.headings, self.observed = {}, {}

    def key(self, at):
        return at[:2] if self.planar else at

    def ports(self, at):
        return self.crossings.setdefault(self.key(at), {})

    def remember(self, a, b, frame, arrival, heading=None):
        delta = tuple(v-u for u,v in zip(a,b))
        back = tuple(-v for v in delta)
        heading = delta[:2] if heading is None else heading
        self.ports(a).setdefault(delta,frame)
        self.ports(b).setdefault(back,arrival)
        self.headings[self.key(a),delta] = heading
        self.headings[self.key(b),back] = heading if delta[2] else tuple(-v for v in heading)

    def onward(self, start, goal, allowed=lambda at,d: True):
        """A proven route to cargo, else to the best observed untried doorway."""
        queue,seen = deque([(start,[])]), {self.key(start)}
        choices = []
        while queue:
            at,path = queue.popleft()
            if at == goal:
                return path
            used = self.ports(at)
            for step in self.observed.get(self.key(at),()):
                if step not in used and allowed(at,step):
                    dest = tuple(a+b for a,b in zip(at,step))
                    distance = abs(dest[0]-goal[0])+abs(dest[1]-goal[1])+2*abs(dest[2]-goal[2])
                    choices.append((distance,len(path),path))
            for step in used:
                b = tuple(x+y for x,y in zip(at,step))
                if self.key(b) in seen or b[2] not in (0,1) or not allowed(at,step):
                    continue
                seen.add(self.key(b))
                queue.append((b,path+[(at,b,self.headings[self.key(at),step])]))
        return min(choices,key=lambda c:c[:2])[2] if choices else None


def explore(p, start, planar, cargo=None, memory=None, allowed=lambda at,d: True):
    at, trail = start, []
    memory = RoomMemory(planar) if memory is None else memory
    while not p.obs["done"]:
        rgb, view = p.scan()
        if not p.loaded:
            target = find_pod(rgb,p.command,at[2]*DECK)
            if target is not None and np.max(abs(target-p.command[:2])) < .48:
                return at,trail,target
        # A rear threshold may be outside the image, or hidden by the
        # carried board. Look into it before declaring that branch closed.
        for d in DIRECTIONS:
            if p.loaded or view[d] == "wall":
                front = p.look(*d)
                samples = [apron_color(front,np.array([0.,dist])) for dist in (.65,.84,1.01)]
                view[d] = max(samples,key=samples.count)
        if p.loaded and "cargo" in view.values():
            return at,trail,next(d for d,kind in view.items() if kind=="cargo")
        ports = {}
        for d,kind in view.items():
            if kind == "open":
                ports[(*d,0)] = d
            elif kind == "landing":  # visible hoist threshold into a physical well
                ports[(0,0,1-2*at[2])] = d
                if p.loaded and at[2] and at[:2] not in p.apertures:
                    approach_well(p,at,d)
                    if p.obs["done"]:
                        return at,trail,None
                    inspect_frame(p,at)
                    leave_well(p,at,d)
        used = memory.ports(at)
        memory.observed[memory.key(at)] = tuple(ports)
        frame = p.obs["frame_id"]
        p.trace.append(dict(event="room_scan",cell=at,loaded=p.loaded,frame=frame,
                            open=[list(port) for port in ports],
                            skipped=[(port,used[port]) for port in ports if port in used]))
        fresh = [port for port in ports if port not in used and allowed(at,port)]
        if cargo is not None:
            goal = (cargo,1,0)  # the inside of the cargo shutter observed from staging
            route = memory.onward(at,goal,allowed)
            if route:
                p.trace.append(dict(event="known_egress",cell=at,to=route[-1][1],
                                    edges=len(route),frame=frame))
                for link in route:
                    if p.obs["done"]:
                        break
                    fly(p,*link)
                    at = link[1]
                continue
        if fresh:
            def priority(port):
                if cargo is None:
                    return (-port[2],-port[1])
                b = tuple(x+y for x,y in zip(at,port))
                return (abs(b[0]-cargo)+abs(b[1]-1)+2*b[2],-abs(port[2]))
            step = min(fresh,key=priority)
            b = tuple(x+y for x,y in zip(at,step))
            link = at,b,ports[step]
            fly(p,*link)
            if not p.obs["done"]:
                memory.remember(at,b,frame,p.obs["frame_id"],ports[step])
                trail.append(link)
                at = b
        elif trail and cargo is None:
            link = reverse(trail.pop())
            fly(p,*link)
            at = link[1]
        else:
            p.trace.append(dict(event="exhausted",cell=at,loaded=p.loaded,frame=frame))
            break
    return at,trail,None


def loaded_return(p, bay, trail, surveyed, cargo, replay, hold_heading=False):
    # Prefer our actual empty approach back to the lower warehouse. The
    # ablation keeps the XYZ trail and all vision/servos, but reuses empty fit.
    def allowed(at,port):
        return (not port[2] or replay or
                p.apertures.get(at[:2],math.inf)*math.sqrt(2) >= 1.80)

    at = bay
    for link in map(reverse,reversed(trail)):
        a,b,d = link
        if not a[2]:
            break
        if b[2] != a[2]:
            accept = allowed(a,(0,0,-1))
            p.trace.append(dict(event="return_route",cell=a,frame=p.obs["frame_id"],
                                width_m=p.apertures[a[:2]],reuse=accept))
            if not accept:
                break
        fly(p,*link)
        if p.obs["done"]:
            return p
        at = b
    at,_,d = explore(p,at,surveyed.planar,cargo,memory=surveyed,allowed=allowed)
    if d is None:
        return p
    return unload(p,at,d,hold_heading)


def run(client, *, planar=False, fixed_tool=None, hold_heading=False, feedback="rgb", replay=False):
    p = Pilot(client,feedback)
    def staging_to(start,end):
        for col in range(start,end,1 if start<end else -1):
            cell = (col+(1 if start<end else -1),0)
            p.cell(cell,TRAVEL,settle=1)
            p.register(cell)

    def slot(a,b):
        p.face(0,1 if b[1]>a[1] else -1)
        p.cell(a,SLOT,settle=1)
        p.cell(b,SLOT)
        p.cell(b,TRAVEL,settle=1)
        p.register(b)

    # The aircraft starts facing the wing. This is an optional useful hint;
    # the actual doorway/deck is inspected again at a safe standoff.
    p.receiving_hint = dispatch_arrow(decode(p.obs,1))
    p.trace.append(dict(event="dispatch_view",side=p.receiving_hint,frame=p.obs["frame_id"]))
    inlet = cargo = depot = None
    for col in (0,1,2,1,0,-1,-2):
        p.cell((col,0),TRAVEL,settle=1)
        p.register((col,0))
        rgb, view = p.scan()
        if rack_heads(rgb):
            depot = col
        if view[(0,1)] == "inspection":
            inlet = (col,1,0)
        if view[(0,1)] == "cargo":
            cargo = col
        if inlet is not None and cargo is not None and depot is not None:
            break
    if inlet is None or cargo is None or depot is None:
        raise RuntimeError("staging hardware not found")
    base = (inlet[0],0)
    staging_to(col,inlet[0])
    slot(base,inlet)
    surveyed = RoomMemory(planar)
    bay, trail, target = explore(p,inlet,planar,memory=surveyed)
    if target is None:
        return p
    p.trace.append(dict(event="sled_found",cell=bay,frame=p.obs["frame_id"]))
    wide = survey(p,target,bay) if fixed_tool is None else fixed_tool=="wide"
    for link in reversed(trail):
        fly(p,*reverse(link))
    slot(inlet,base)
    staging_to(inlet[0],depot)
    equip(p,wide,depot)
    staging_to(depot,inlet[0])
    slot(base,inlet)
    for link in trail:
        fly(p,*link)
    floor = bay[2]*DECK
    target = find_pod(decode(p.obs),p.command,floor)
    if target is None:
        raise RuntimeError("litter not seen on the remembered return route")
    p.trace.append(dict(event="sled_reacquired",cell=bay,frame=p.obs["frame_id"]))
    px,py = sight_litter(p,target,floor)
    p.go(px,py,.43+floor,settle=3)
    p.grip(1.)
    p.loaded = True
    p.go(px,py,TRAVEL+floor)
    p.cell(bay,TRAVEL+floor,settle=1)
    p.register(bay)
    p.trace.append(dict(event="pickup_attempt",cell=bay,frame=p.obs["frame_id"]))
    if find_pod(decode(p.obs),p.command,floor) is not None:
        return p
    return loaded_return(p,bay,trail,surveyed,cargo,replay,hold_heading)


def unload(p,at,d,hold_heading):
    p.align_load(*d)
    staging = (at[0]+d[0],at[1]+d[1])
    p.cell(staging,TRAVEL,settle=1)
    p.register(staging)
    if not p.obs["done"]:
        transfer(p,hold_heading)
    p.look(0,1)
    if not p.obs["done"]:
        offset = pad_offset(decode(p.obs))
        if offset is None:
            raise RuntimeError("blue delivery pad not in the down view")
        p.go(*(p.command[:2]+offset),.43,settle=3)
    p.grip(-1.)
    for _ in range(3):
        p.grip()
    return p


if __name__ == "__main__":
    from client import Client

    pilot = run(Client())
    out = Path("/home/user/submission")
    for i, frame in enumerate(pilot.obs["images"]):
        (out/f"final-camera-{i}.png").write_bytes(base64.b64decode(frame))
    (out/"reference-route.json").write_text(json.dumps(pilot.trace, indent=2))
    print(json.dumps({k: v for k, v in pilot.obs.items() if k != "images"}), flush=True)
