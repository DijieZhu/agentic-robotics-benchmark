"""Metric floor plan from the public nadir RGB and tracking, without site inputs."""
import itertools
import math

import numpy as np
from scipy import ndimage

from client import camera_image
from registration import clusters, map_caps


STEP = .02
X = np.arange(.40, 6.81, STEP)
Y = np.arange(-1.74, 1.75, STEP)
GO1_CLEARANCE = .56  # conservative working width, including tracking/gait allowance
APPROACH = .53


def runs(mask):
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


class SurveyMap:
    def __init__(self):
        self.visible = np.zeros((len(X), len(Y)), dtype=bool)
        self.occupied = self.visible.copy()
        self.dock = self.visible.copy()
        self.cap_samples = []
        self.cover_tracks = []

    def add(self, obs):
        rgb = np.asarray(camera_image(obs))
        h, w = rgb.shape[:2]
        focal = h / (2 * math.tan(math.radians(78)/2))
        x, y = obs["tracking_xy_m"]
        self.cap_samples.extend(map_caps(rgb, np.array([x, y])))
        self.add_covers(rgb, np.array([x, y]), focal)

        def sample(height):
            scale = focal/(1.65-height)
            u = np.rint((X-x)*scale+(w-1)/2).astype(int)
            v = np.rint((y-Y)*scale+(h-1)/2).astype(int)
            valid = ((u[:, None] >= 2) & (u[:, None] < w-2)
                     & (v >= 2) & (v < h-2))
            r, g, b = rgb[v.clip(0, h-1), u[:, None].clip(0, w-1)].astype(int).transpose(2, 0, 1)
            return valid, r, g, b

        valid, r, g, b = sample(.32)
        self.visible |= valid
        self.occupied |= valid & (r > 135) & (r > 1.8*g) & (g > 1.35*b)
        valid, r, g, b = sample(0)
        self.dock |= valid & (b > 135) & (b > 1.8*g) & (g > 1.8*r)

    def landmarks(self):
        return clusters(self.cap_samples, .13)

    def add_covers(self, rgb, camera, focal):
        r, g, b = rgb.astype(float).transpose(2, 0, 1)
        mask = (g > 85) & (b > 85) & (g > 1.6*r) & (b > 1.6*r) & (g > .78*b) & (g < 1.22*b)
        labels, _ = ndimage.label(mask)
        h, w = mask.shape
        for v, u in ndimage.find_objects(labels):
            if min(v.start, h-v.stop, u.start, w-u.stop) < 2:
                continue
            if not 1.70 < (u.stop-u.start)/(v.stop-v.start) < 2.40 or u.stop-u.start < 12:
                continue
            if mask[v, u].mean() < .95:
                continue
            ray = np.array([u.start+u.stop-w, h-v.start-v.stop])/(2*focal)
            coarse = camera+1.35*ray
            for track in self.cover_tracks:
                if np.linalg.norm(coarse-track[0][0]) < .55:
                    break
            else:
                track = []
                self.cover_tracks.append(track)
            track.append((coarse, camera, ray))

    def terrain(self):
        fits = []
        for track in self.cover_tracks:
            if len(track) < 2:
                raise ValueError("insufficient parallax on a loading fixture")
            _, cameras, rays = np.asarray(track).transpose(1, 0, 2)
            dc, dr = cameras-cameras.mean(axis=0), rays-rays.mean(axis=0)
            if np.linalg.norm(dr) < .19:
                raise ValueError("need separated views of fixture")
            # Fixed point = camera_xy + (camera_z - surface_z) * image ray.
            depth = -(dc*dr).sum()/(dr*dr).sum()
            hits = cameras+depth*rays
            xy = hits.mean(axis=0)
            fits.append({"xy": xy.tolist(), "top_m": float(1.65-depth), "views": len(track),
                         "rms_m": float(np.sqrt(np.mean(np.sum((hits-xy)**2, axis=1))))})
        return sorted(fits, key=lambda fit: fit["xy"][0])

    def passages(self):
        dock_x, dock_y = np.where(self.dock)
        if not len(dock_x):
            raise ValueError("dock not observed")
        goal = np.array([X[dock_x].mean(), Y[dock_y].mean()])
        interior = X < goal[0]-.60
        if not self.visible[interior].all():
            raise ValueError("rack footprint incompletely surveyed")
        bands = self.occupied.mean(axis=1) > .35
        racks = []
        for lo, hi in runs(bands & interior):
            if (hi-lo)*STEP < .10:
                continue
            blocked = self.occupied[lo+2:hi-2].mean(axis=0) > .50
            gaps = [(float(Y[a]-STEP/2), float(Y[b-1]+STEP/2))
                    for a, b in runs(~blocked) if a and b < len(Y)]
            racks.append((float(X[lo]-STEP/2), float(X[hi-1]+STEP/2), gaps))
        if not racks or any(not rack[2] for rack in racks):
            raise ValueError("no complete series of passages in the RGB map")
        return racks, goal

    def route(self, point_rack=None):
        racks, goal = self.passages()
        deck_x = [item["xy"][0] for item in self.terrain() if item["top_m"] < .30]
        candidates = []
        is_deck = [any(near-.12 < x < far+.12 for x in deck_x) for near, far, _ in racks]
        bays = []
        for i, ((_, far, _), (near, _, _)) in enumerate(itertools.pairwise(racks)):
            # Translate sideways where the entire body, including trailing calves, fits.
            fraction = .55 if is_deck[i] else .45 if is_deck[i+1] else .50
            bays.append(far+fraction*(near-far))
        for i, (near, far, gaps) in enumerate(racks):
            minimum = 0. if i == point_rack else GO1_CLEARANCE
            left = bays[i-1] if i else near-(.82 if is_deck[i] else APPROACH)
            right = bays[i] if i < len(bays) else far+(.82 if is_deck[i] else APPROACH)
            candidates.append([
                ((left, (a+b)/2), (right, (a+b)/2))
                for a, b in gaps if b-a >= minimum
            ])
        if not all(candidates):
            raise ValueError("no route with the needed ground clearance")

        def length(path):
            return sum(math.dist(a, b) for a, b in itertools.pairwise([(0., 0.), *path]))

        paths = ([p for passage in combo for p in passage]+[goal.tolist()]
                 for combo in itertools.product(*candidates))
        return min(paths, key=length)


def scan(robot, rows=(0., 1.50, -1.50)):
    surveyed = SurveyMap()
    obs = robot.observe()
    surveyed.add(obs)
    for row, y in enumerate(rows):
        xs = np.linspace(-.68, 6.64, 15)
        for x in xs if row % 2 == 0 else xs[::-1]:
            target = np.array([x, y])
            delta = target-obs["tracking_xy_m"]
            while np.linalg.norm(delta) > 1e-6:
                delta *= min(1., .55/np.linalg.norm(delta))
                obs = robot.act(delta.tolist())
                surveyed.add(obs)
                delta = target-obs["tracking_xy_m"]
    return surveyed
