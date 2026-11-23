"""Author-only analytical gait; this is not part of the robot service."""
import math

import numpy as np


STANCE_M = {"ordinary": .265, "crouch": .205, "intermediate": .217}


class Walk:
    def __init__(self, waypoints, ducts=(), posture="adaptive"):
        self.waypoints = [np.asarray(p, dtype=float) for p in waypoints]
        self.index = 0
        self.phase = 0.
        self.still = False
        self.stride = np.zeros(2)
        self.ducts = ducts
        self.posture = posture
        self.height = .265
        self.start_ticks = 0
        self.aligned = False
        self.turn_center = None

    def action(self, xy, yaw, ticks):
        mode = self.posture
        if mode == "adaptive":
            mode = "crouch" if any(abs(xy[0]-x) < .83 for x in self.ducts) else "ordinary"
        self.height += float(np.clip(STANCE_M[mode]-self.height, -.0025, .0025))
        # Settle the inspection load on four feet before combining yaw and translation.
        if self.start_ticks < 500:
            n = min(12, ticks)
            self.start_ticks += n
            bend = math.acos(self.height/.426)
            return {"target_rad": [0., bend, -2*bend]*4, "ticks": n}
        delta = self.waypoints[self.index]-xy
        final = self.index == len(self.waypoints)-1
        if not final and np.linalg.norm(delta) <= .06:
            self.index += 1
            delta = self.waypoints[self.index]-xy
            final = self.index == len(self.waypoints)-1
        if final and np.linalg.norm(delta) < .070 and abs(yaw) < .19:
            self.still = True
        if self.still:
            bend = math.acos(self.height/.426)
            return {"target_rad": [0., bend, -2*bend]*4, "ticks": min(ticks, 32)}

        co, si = math.cos(yaw), math.sin(yaw)
        local = np.array([co*delta[0]+si*delta[1], -si*delta[0]+co*delta[1]])
        wanted = local * min(.62, .108/max(np.linalg.norm(local), 1e-6))
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
            if p < .59:
                fraction, height = .5-p/.59, self.height
            else:
                s = (p-.59)/.41
                fraction = -.5+s-math.sin(2*math.pi*s)/(2*math.pi)
                height = self.height-.082*math.sin(math.pi*s)
            x, y = stride*fraction + [0., side*.08]
            sag = math.sqrt(max(.01, y*y+height*height-.08**2))
            hip = math.atan2(y, height)-math.atan2(side*.08, sag)
            bend = math.acos(min(1., math.hypot(x, sag)/(.213*2)))
            angles.extend([hip, math.atan2(-x, sag)+bend, -2*bend])
        return {"target_rad": angles, "ticks": ticks}
