"""The oracle's gait: an analytical trot from joint targets, with the stance height and the trunk's pitch
held by feedback on the robot's own joint angles and gravity reading (the payload, the servo stiffness and
the fittings' clearances differ per site)."""
import math

import numpy as np

LEG = .213
FOOT = .023                      # foot radius: trunk height above the tread = stance leg height + FOOT
GUARD_TOP, SCANNER_LOW = .199, .163
NOMINAL = .255                   # trunk height on the open floor


def leg_height(hip, thigh, calf):
    """Vertical reach of one leg below its thigh axis, from measured joint angles."""
    return LEG*(math.cos(thigh)+math.cos(thigh+calf))*math.cos(hip)


class Walk:
    def __init__(self, waypoints, fittings):
        self.waypoints = [np.asarray(p, dtype=float) for p in waypoints]
        self.fittings = fittings
        self.index = 0
        self.phase = 0.
        self.still = False
        self.stride = np.zeros(2)
        self.height = NOMINAL        # wanted trunk height (the corrections are added to it)
        self.lift = 0.               # integral: commanded minus measured stance height
        self.trim = 0.               # integral: front-rear height difference that levels the trunk
        self.start_ticks = 0
        self.aligned = False
        self.turn_center = None
        self.origin = None

    def wanted_height(self, xy):
        """Trunk height the place asks for: low under a duct, high on a deck, nominal elsewhere."""
        for f in self.fittings:
            if not f["lo"]-.2 < xy[1] < f["hi"]+.2:
                continue
            if f["kind"] == "duct" and abs(xy[0]-f["x"]) < .83:
                # Guard below the soffit, scanner above the floor: split the spare height, the larger
                # share to the scanner, which swings with the trunk's pitch.
                spare = f["clearance"]-GUARD_TOP-SCANNER_LOW
                return f["clearance"]-GUARD_TOP-max(.014, .40*spare), "duct"
            if f["kind"] == "deck" and abs(xy[0]-f["x"]) < 1.05:
                # The scanner (0.225-0.393 m ahead of the trunk origin) has to pass over the centre guide,
                # which climbs from the deck's west edge to its crest: ask for the height that place needs.
                ahead = xy[0]-f["x"]+.393+.03
                guide = f["rise"]*min(1., max(0., (ahead+.305)/.366)) if ahead < .64 else 0.
                # East of the crest the scanner is clear and the trunk is levelled again on the way down.
                zone = "climb" if xy[0] < f["x"]-.45 else "deck" if xy[0] < f["x"]+.30 else "descent"
                return max(NOMINAL, guide+SCANNER_LOW+.043), zone
        return NOMINAL, "floor"

    def feedback(self, obs, target, level=True, nose_up=0.):
        q = obs["joint_rad"]
        stance = [i for i in range(4) if obs["foot_contact"][i]]
        if len(stance) >= 2:
            measured = float(np.mean([leg_height(*q[3*i:3*i+3]) for i in stance]))+FOOT
            self.lift = float(np.clip(self.lift+.02*(target-measured), -.02, .09))
        pitch = -math.asin(max(-1., min(1., obs["gravity_body_unit"][0])))     # nose up positive
        # Climbing a deck the trunk follows the slope: there the trim only ever lifts the nose.
        if level or pitch < nose_up:
            self.trim = float(np.clip(self.trim+.004*(pitch-nose_up), -.03, .03))

    def stand(self, ticks):
        angles = []
        for i in range(4):
            fore = 1 if i < 2 else -1
            h = min(.40, self.height+self.lift-fore*self.trim-FOOT)
            bend = math.acos(h/(2*LEG))
            angles.extend([0., bend, -2*bend])
        return {"target_rad": angles, "ticks": ticks}

    def action(self, xy, yaw, obs):
        ticks = obs["ticks_remaining"]
        target, where = self.wanted_height(xy)
        self.height += float(np.clip(target-self.height, -.0008, .0020))      # lower slowly: no undershoot
        # Under a duct the scanner ahead of the feet is the low point: carry the nose a little high.
        self.feedback(obs, self.height, where not in ("deck", "climb"), .03 if where in ("duct", "deck", "climb") else 0.)
        if self.start_ticks < 600:
            n = min(12, ticks)
            self.start_ticks += n
            return self.stand(n)
        delta = self.waypoints[self.index]-xy
        final = self.index == len(self.waypoints)-1
        if not final and np.linalg.norm(delta) <= .06:
            self.index += 1
            delta = self.waypoints[self.index]-xy
            final = self.index == len(self.waypoints)-1
        if final and np.linalg.norm(delta) < .070 and abs(yaw) < .19:
            self.still = True
        if self.still:
            return self.stand(min(ticks, 32))

        co, si = math.cos(yaw), math.sin(yaw)
        speed = .108 if where == "floor" else .085
        # Along the leg to the waypoint at the gait's stride, plus a pull back onto the leg's line: a long
        # sideways leg in a bay must not creep toward the rack behind.
        start = self.waypoints[self.index-1] if self.index else self.origin
        if start is None:
            start = self.origin = np.array(xy, dtype=float)
        leg = self.waypoints[self.index]-start
        along = leg/max(np.linalg.norm(leg), 1e-6)
        off = (xy-start)-along*float(np.dot(xy-start, along))
        pull = -off*min(1., .05/max(np.linalg.norm(off), 1e-6)) if np.linalg.norm(off) > .01 else np.zeros(2)
        world = delta*min(.62, speed/max(np.linalg.norm(delta), 1e-6))+pull
        wanted = np.array([co*world[0]+si*world[1], -si*world[0]+co*world[1]])
        self.aligned |= abs(yaw) < .13
        if not self.aligned:
            if self.turn_center is None:
                self.turn_center = xy + [.10, 0.]
            ex, ey = self.turn_center-xy
            wanted = np.array([co*ex+si*ey, -si*ex+co*ey])
            wanted *= min(.35, .045/max(np.linalg.norm(wanted), 1e-6))
        self.stride += .28*(wanted-self.stride)
        turn = float(np.clip(-.31*yaw, -.102, .102))
        ticks = min(ticks, 12)
        self.phase += ticks*.002*1.90
        angles = []
        for i, offset in enumerate((0., .5, .5, 0.)):
            side = -1 if i in (0, 2) else 1
            fore = 1 if i < 2 else -1
            stride = self.stride + turn*np.array([-.135*side, .185*fore])
            p = (self.phase+offset) % 1.
            base = min(.40, self.height+self.lift-fore*self.trim-FOOT)
            if p < .59:
                fraction, height = .5-p/.59, base
            else:
                s = (p-.59)/.41
                fraction = -.5+s-math.sin(2*math.pi*s)/(2*math.pi)
                # A low step under a duct; a high one onto a deck's apron, which starts with a toe.
                lift = .050 if where == "duct" else .110 if where == "climb" else .082
                height = base-lift*math.sin(math.pi*s)
            x, y = stride*fraction + [0., side*.08]
            sag = math.sqrt(max(.01, y*y+height*height-.08**2))
            hip = math.atan2(y, height)-math.atan2(side*.08, sag)
            bend = math.acos(min(1., math.hypot(x, sag)/(LEG*2)))
            angles.extend([hip, math.atan2(-x, sag)+bend, -2*bend])
        return {"target_rad": angles, "ticks": ticks}
