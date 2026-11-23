"""Protected MuJoCo flight, structural contact checks, and live rescue ledger."""
import base64
import io
import math

import mujoco
import numpy as np
from PIL import Image

from rescue_render import Cameras
from rescue_scene import DECK, SLAB, WELL_HALF, POCKET_HALF_DEPTH, START_Z, build, layout_for

LIMIT = 4400
DT = .2
INCREMENTS = np.array([.32, .32, .19])
# Reaching the litter is worth little; the work on the litter carries the credit (weights fixed in a commit of
# their own, before anything was graded again with them).
SCORE = {"entered_wing": .05, "mezzanine": .04, "located": .06, "grasped": .15,
         "lifted": .10, "cargo_return": .25}
ROOMS = .03
FAILURES = ("wall", "inspection", "cargo", "slab", "roof",
            "throat_entry", "throat_exit", "loading_wall", "prop", "unsafe_release")


def png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def basis(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class World:
    kind = "rescue"

    def __init__(self, kind, seed, render=True, truth=False):
        if kind != self.kind:
            raise ValueError("unknown task")
        self.truth = bool(truth)
        # Adjacent seeds share the site, gains and launch position but reverse
        # the cross-axis tracking residual. It never affects a hold command.
        self.layout = layout_for(seed >> 1)
        self.bias_sign = 1 if seed & 1 else -1
        self.trim_offset = np.zeros(2)
        self.loaded_trim_m = 0.
        self.bounds = np.asarray(self.layout.bounds).T
        xml, decals, obstacles, shutters = build(self.layout)
        self.model = mujoco.MjModel.from_xml_string(xml, assets=decals)
        self.shutters = shutters
        self.model.light_diffuse[:] *= self.layout.tone
        self.data = mujoco.MjData(self.model)
        self.command = np.array([*self.layout.launch, START_Z, 0.])
        self.data.qpos[:4] = self.command
        self.data.ctrl[:] = self.command
        x, y, yaw = self.layout.pod
        self.data.qpos[4:11] = (x, y, self.layout.target[2]*DECK+.073, math.cos(yaw/2), 0., 0., math.sin(yaw/2))
        mujoco.mj_forward(self.model, self.data)
        # The two cameras are drawn by a process of their own, one per site (rescue_render.py): an image is then
        # a function of the site and the state alone, here and in every other process that draws this site.
        self.renderer = Cameras(seed) if render else None
        self.kinds, names = zip(*obstacles)
        self.solids = np.array([self.model.geom(name).id for name in names])
        self.throats = np.array([k.startswith("throat_") for k in self.kinds])
        self.refresh_solids()
        # A body's contact pieces: its own geoms. The drawn models on it (look_*) are for the cameras only.
        drawn = np.array([(self.model.geom(g).name or "").startswith("look_") for g in range(self.model.ngeom)])
        self.parts = {name: np.flatnonzero((self.model.geom_bodyid == self.model.body(name).id) & ~drawn)
                      for name in ("uav", "pod", "collet0", "collet1")}
        self.board = self.model.geom("litter_board").id
        self.mats = [self.model.geom(f"receiving_mat{s}").id for s in (-1,1)]
        self.entry_clear = self.exit_clear = 0
        self.throat_margin = {}
        self.impact_geom = self.impact_obstacle = None
        self.rel_quat = np.empty(4)
        self.room_centers = np.asarray([self.layout.xy(room) for room in self.layout.rooms])
        self.columns = np.asarray(self.layout.columns)
        self.rows = np.asarray(self.layout.rows)
        self.steps = self.release_dwell = self.close_attempts = 0
        self.tool_changes = self.steps_to_tool = self.fit_misses = 0
        self.mounts = [self.model.equality(f"collet{i}_mount").id for i in range(2)]
        self.docks = [self.model.equality(f"collet{i}_dock").id for i in range(2)]
        self.steps_to_lift = self.steps_to_cargo = 0
        self.done = self.shutter_open = False
        self.failure = self.impact_body = None
        self.progress, self.rooms_seen, self.cells_seen = set(), set(), set()
        self.loaded_cargo_cells = set()
        self.entry_z = self.exit_z = 0.
        self.mezzanine_z = self.loaded_descent_z = self.turn_angle = 0.
        self.well_events, self.well_gaps = [], {}
        self.first_sight_frame = 0
        self.search_m = 0.
        self.storey_rooms = set()
        self.well_x = tuple(self.layout.columns[c] for c,_ in self.layout.wells)
        self._observation = None
        # Onboard sensors other than the cameras: a barometric altimeter, the IMU's yaw gyro and
        # gravity-compensated accelerometer, and an optical-flow odometer. Their noise is drawn from the
        # site's own stream, so a site replays identically; scale and bias are fixed for a site.
        self.noise = np.random.default_rng([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0x53454E53])
        self.flow_scale = 1.+self.noise.uniform(-.02, .02, 2)
        self.gyro_bias = float(self.noise.uniform(-.004, .004))
        self.baro_bias = 0.
        self.sensors = self.sense(None)

    def sense(self, before):
        """Sensor readings for the action that started at `before` = (position, yaw, velocity)."""
        rng = self.noise
        self.baro_bias = float(np.clip(self.baro_bias+rng.normal(0., .002), -.06, .06))
        altitude = float(self.position[2])+self.baro_bias+rng.normal(0., .015)
        flow, turn, accel = np.zeros(2), 0., np.zeros(3)
        if before is not None:
            position, yaw, velocity = before
            flow = basis(yaw).T @ (self.position[:2]-position[:2])
            turn = (float(self.data.qpos[3])-yaw)/DT
            change = (self.data.qvel[:3]-velocity)/DT
            accel = np.array([*(basis(float(self.data.qpos[3])).T @ change[:2]), change[2]])
        flow = flow*self.flow_scale+rng.normal(0., .003+.01*np.abs(flow))
        return {"altitude_m": round(altitude, 4),
                "flow_m": [round(float(v), 4) for v in flow],
                "gyro_z_rad_s": round(turn+self.gyro_bias+rng.normal(0., .01), 4),
                "accel_m_s2": [round(float(v), 3) for v in accel+rng.normal(0., .05, 3)]}

    def ground_truth(self):
        """Privileged state for the token-gated reference only (never in commissioning)."""
        L = self.layout
        axis = self.data.geom_xmat[self.board].reshape(3, 3)[:, 1]
        return {
            "pose": [round(float(v), 5) for v in self.data.qpos[:4]],
            "velocity": [round(float(v), 5) for v in self.data.qvel[:4]],
            "litter": [round(float(v), 5) for v in self.payload] + [round(math.atan2(-axis[0], axis[1]), 5)],
            "held": self.held, "tool": self.tool, "failure": self.failure,
            "shutter_open": self.shutter_open, "progress": sorted(self.progress),
            "layout": {"columns": list(map(float, L.columns)), "rows": list(map(float, L.rows)),
                       "halls": [[list(a), list(b)] for a, b in sorted(L.halls)],
                       "wells": [list(v) for v in L.wells], "inlet": int(L.inlet), "outlet": int(L.outlet),
                       "target": list(map(int, L.target)), "flange": float(L.flange),
                       "depot": [list(map(float, v)) for v in L.depot],
                       "docks": [list(map(float, L.dock(i))) for i in range(2)],
                       "loading": list(map(float, L.loading)), "pocket": list(map(float, L.pocket)),
                       "pads": {str(s): list(map(float, L.pad_at(s))) for s in (-1, 1)},
                       "well_half": {str(c): float(L.well_half(c)) for c, _ in L.wells}}}

    @property
    def held(self):
        return bool(self.data.eq_active[0])

    @property
    def position(self):
        return self.data.qpos[:3]

    @property
    def payload(self):
        return self.data.qpos[4:7]

    @property
    def tool(self):
        return next((i for i, mount in enumerate(self.mounts) if self.data.eq_active[mount]), None)

    def floor_at(self, xyz):
        x,y,z = xyz
        if z < DECK or y < self.layout.inlet_y:
            return 0.
        if any(max(abs(x-wx),abs(y-self.layout.rows[4])) < self.layout.well_half(col)
               for wx,(col,_) in zip(self.well_x,self.layout.wells)):
            return 0.
        return DECK

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def images(self):
        if self.renderer is None:
            return []
        return self.renderer.images(self.data.qpos, self.data.mocap_pos)     # down, forward

    def observe(self):
        if self._observation is None:
            self._observation = {
                "images": [png(frame) for frame in self.images()],
                "frame_id": self.steps, "steps_remaining": LIMIT-self.steps, "done": self.done,
                "sensors": self.sensors,
            }
            if self.truth:
                self._observation["_gt"] = self.ground_truth()
        return self._observation

    def abort(self, reason, body=None):
        self.failure, self.impact_body = reason, body
        self.done = True

    def weld(self, eq, position, orientation):
        # Capture the actual pose in the UAV frame.
        yaw = self.data.qpos[3]
        delta = position-self.position
        delta[:2] = basis(yaw).T @ delta[:2]
        self.model.eq_data[eq, 3:6] = delta
        inverse = np.array([math.cos(yaw/2), 0., 0., -math.sin(yaw/2)])
        mujoco.mju_mulQuat(self.rel_quat, inverse, orientation)
        self.model.eq_data[eq, 6:10] = self.rel_quat
        self.data.eq_active[eq] = True

    def dock_tool(self, signal, tool):
        for slot in range(2):
            delta = np.asarray(self.layout.dock(slot))-self.position
            if np.linalg.norm(delta[:2]) > .085 or abs(delta[2]+.206) > .035:
                continue
            mount, dock = self.mounts[slot], self.docks[slot]
            if signal > .5 and tool is None:
                body = self.data.body(f"collet{slot}")
                self.data.eq_active[dock] = False
                self.weld(mount, body.xpos, body.xquat)
                self.steps_to_tool = self.steps_to_tool or self.steps
                self.tool_changes += 1
            elif signal < -.5 and tool == slot:
                self.data.eq_active[mount] = False
                self.data.eq_active[dock] = True
            return

    def fits(self, tool):
        if tool is None:
            return False
        # The flange has to enter the recessed seat but not pass its retaining lip.
        # Read the two inner radii from the same jaw geometry shown in RGB.
        seat, lip = (self.model.geom(f"collet{tool}_{layer}4") for layer in ("seat", "lip"))
        throat, recess = (abs(g.pos[1])-g.size[0] for g in (lip, seat))
        return throat < self.model.geom("flange").size[0] < recess

    def grip(self, signal):
        delta = self.payload-self.position
        speed = np.linalg.norm(self.data.qvel[:3])
        turn_speed = abs(self.data.qvel[3])
        tool = self.tool
        if signal > .5 and not self.held:
            self.close_attempts += 1
            if (np.linalg.norm(delta[:2]) <= .135 and abs(delta[2]+.306) <= .060
                    and speed <= .30 and turn_speed < .3):
                if not self.fits(tool):
                    self.fit_misses += 1
                    return
                self.weld(0, self.payload, self.data.qpos[7:11])
                self.progress.add("grasped")
                self.release_dwell = 0
        elif signal < -.5 and self.held:
            self.data.eq_active[0] = False
            self.release_dwell = 0
            if self.payload[2]-self.floor_at(self.payload) > .195 or speed > .34 or turn_speed > .34:
                self.abort("unsafe_release")
        if not self.held and abs(signal) > .5 and speed <= .30 and turn_speed < .3:
            self.dock_tool(signal, tool)

    def refresh_solids(self):
        centers = self.data.geom_xpos[self.solids]
        half = self.model.geom_size[self.solids]  # all structural boxes are axis-aligned
        self.solid_lo, self.solid_hi = centers-half, centers+half

    def mat_margin(self, mat=None):
        if mat is None:
            mat = min(self.mats,key=lambda g:np.linalg.norm(self.payload-self.data.geom_xpos[g]))
        ext = abs(self.data.geom_xmat[self.board].reshape(3, 3)) @ self.model.geom_size[self.board]
        support = abs(self.data.geom_xmat[mat].reshape(3,3)) @ self.model.geom_size[mat]
        delta = abs(self.data.geom_xpos[self.board]-self.data.geom_xpos[mat])
        return float(min((support-delta-ext)[:2]))

    def supported_on_mat(self):
        for c in self.data.contact:
            a,b = c.geom
            mat = b if a==self.board else a if b==self.board else -1
            if (mat in self.mats and c.dist<=0. and self.mat_margin(mat)>=0. and
                    np.linalg.norm(self.payload[:2]-self.data.geom_xpos[mat,:2])<=.28):
                return True
        return False

    def check_clearance(self):
        # Broad-phase AABBs only select pairs. Only contact between the actual
        # rendered MuJoCo geoms can abort, sampled at each 5 ms physics step.
        mujoco.mj_kinematics(self.model, self.data)
        for name in ("uav", "collet"+str(self.tool), "pod"):
            if name not in self.parts or (name == "pod" and "grasped" not in self.progress):
                continue
            ids = self.parts[name]
            rot = self.data.geom_xmat[ids].reshape(-1, 3, 3)
            aabb = self.model.geom_aabb[ids]
            centers = self.data.geom_xpos[ids]+(rot @ aabb[:, :3, None])[:, :, 0]
            half = (np.abs(rot) @ aabb[:, 3:, None])[:, :, 0]
            lo, hi = centers-half, centers+half
            if (name in ("uav","pod") and (name == "uav" or self.held)
                    and abs(self.position[1]-self.layout.rows[4]) < WELL_HALF
                    and lo[:,2].min() <= DECK and hi[:,2].max() >= DECK-SLAB):
                idx = int(np.argmin(np.abs(self.position[0]-np.asarray(self.well_x))))
                wx = self.well_x[idx]
                origin = [wx,self.layout.rows[4]]
                gap = self.layout.well_half(self.layout.wells[idx][0])-max(
                    np.abs(lo[:,:2]-origin).max(), np.abs(hi[:,:2]-origin).max())
                self.well_gaps[name] = min(self.well_gaps.get(name, gap), float(gap))
            # Measure geometric margin near the two passages as an author diagnostic.
            reach = (.55*self.throats if name == "pod" else np.zeros(len(self.solids)))+.0001
            near = np.flatnonzero(np.all((lo.min(axis=0) <= self.solid_hi+reach[:, None]) &
                                        (hi.max(axis=0) >= self.solid_lo-reach[:, None]), axis=1))
            for i in near:
                kind = self.kinds[i]
                if (name == "pod" and kind == "slab" and self.payload[2] > DECK
                        and (not self.held or "lifted" not in self.progress)):  # upper floor support only
                    continue
                candidates = ids[np.all((lo <= self.solid_hi[i]+reach[i]) &
                                        (hi >= self.solid_lo[i]-reach[i]), axis=1)]
                for g in candidates:
                    dist = mujoco.mj_geomDistance(self.model, self.data, g, self.solids[i], reach[i], None)
                    if name == "pod" and self.throats[i] and dist < reach[i]:
                        self.throat_margin[kind] = min(self.throat_margin.get(kind, dist), dist)
                    if dist <= 0.:
                        self.impact_geom = self.model.geom(g).name
                        self.impact_obstacle = self.model.geom(self.solids[i]).name
                        self.abort(kind, "pod" if name == "pod" else "uav")
                        return

    def physical_progress(self, previous_y, previous_z):
        x, y, z = self.position
        px, py, pz = self.payload
        if (not self.held and "entered_wing" not in self.progress
                and previous_y < self.layout.inlet_y <= y):
            self.progress.add("entered_wing")
            self.entry_z = z
        if y > self.layout.inlet_y and previous_z < DECK <= z:
            self.well_events.append((self.steps, "up", bool(self.held), round(x,3), round(y,3)))
        if y > self.layout.inlet_y and previous_z >= DECK > z:
            self.well_events.append((self.steps, "down", bool(self.held), round(x,3), round(y,3)))
        if y > self.layout.inlet_y and z >= DECK+.70 and not self.mezzanine_z:
            self.progress.add("mezzanine")
            self.mezzanine_z = z
        if self.held and previous_z >= DECK > z and not self.loaded_descent_z:
            self.loaded_descent_z = z
        if self.held and pz-self.floor_at(self.payload) >= .54:
            self.progress.add("lifted")
            if not self.shutter_open:
                # The training wing's mechanical lift interlock retracts the cargo leaf.
                self.shutter_open = True
                self.steps_to_lift = self.steps
                mid = self.model.body("cargo_shutter").mocapid[0]
                self.data.mocap_pos[mid,2] = 3.8
                mujoco.mj_kinematics(self.model, self.data)
                self.refresh_solids()
        if self.held and previous_y > self.layout.inlet_y >= y and "cargo_return" not in self.progress:
            self.progress.add("cargo_return")
            self.steps_to_cargo = self.steps
            self.exit_z = z
        if self.held and "cargo_return" in self.progress and py < 0.:
            cx, cy = self.layout.pocket
            ext = abs(self.data.geom_xmat[self.board].reshape(3, 3)) @ self.model.geom_size[self.board]
            if not self.entry_clear and py+ext[1] < cy+POCKET_HALF_DEPTH-.28:
                self.entry_clear = self.steps
            if not self.exit_clear and abs(px-cx)-ext[0] > 2.10+.28 and py < cy+POCKET_HALF_DEPTH:
                self.exit_clear = self.steps
        if (not self.held and "lifted" in self.progress
                and pz <= .095 and np.linalg.norm(self.data.qvel[4:10]) <= .16 and self.supported_on_mat()):
            self.release_dwell += 1
            if self.release_dwell >= 50:
                self.progress.add("delivered")
                self.done = True
        else:
            self.release_dwell = 0

    def observation_progress(self):
        x, y, z = self.position
        if y > self.layout.inlet_y:
            cx, cy = int(np.argmin(abs(self.columns-x))), int(np.argmin(abs(self.rows-y)))
            self.cells_seen.add((cx, cy, int(z>=DECK)))
            if self.held and cx == self.layout.outlet:
                self.loaded_cargo_cells.add(cy)
        level = int(z>=DECK)
        if z-level*DECK >= .85:
            near = np.flatnonzero(np.max(np.abs(self.room_centers-[x, y]), axis=1) < .47)
            for i in near:
                room = self.layout.rooms[i]
                if room[2] == level:
                    self.rooms_seen.add(int(i))
                    self.storey_rooms.add(room)
            if (not self.held and level == self.layout.target[2]
                    and np.max(np.abs(self.payload[:2]-[x, y])) < .62):
                self.progress.add("located")
                self.first_sight_frame = self.first_sight_frame or self.steps

    def act(self, action):
        if (not isinstance(action, list) or len(action) != 5
                or any(type(a) not in (int, float) or not -1 <= a <= 1 for a in action)):
            raise ValueError("action must contain five finite numbers in [-1,1]")
        if self.done:
            return self.observe()
        self.steps += 1
        self._observation = None
        self.grip(action[4])
        delta = INCREMENTS*action[:3]
        # The inner servo is stable; only translation increments are mis-tracked.
        # At zero input, both the gain term and cross-coupling vanish.
        estimated = delta[:2].copy()
        delta[:2] *= self.layout.gains
        side = self.bias_sign*(.125 if self.held else .032)*estimated[1]
        delta[0] += side
        if self.held:
            self.loaded_trim_m += abs(side)
        rotation = basis(self.command[3])
        self.trim_offset += rotation @ (delta[:2]-estimated)
        delta[:2] = rotation @ delta[:2]
        self.command[:3] = np.clip(self.command[:3]+delta, *self.bounds)
        self.command[3] += math.radians(30)*action[3]
        self.data.ctrl[:] = self.command
        previous_heading = self.data.qpos[3]
        before = self.position.copy()
        started = (before, float(previous_heading), self.data.qvel[:3].copy())
        for _ in range(round(DT/self.model.opt.timestep)):
            if self.done:
                break
            previous_y, previous_z = self.position[1:]
            mujoco.mj_step(self.model, self.data)
            self.check_clearance()
            if not self.done:
                self.physical_progress(previous_y, previous_z)
        if not self.first_sight_frame:
            self.search_m += float(np.linalg.norm(self.position-before))
        if not self.done:
            self.observation_progress()
        self.turn_angle += abs(self.data.qpos[3]-previous_heading)
        self.sensors = self.sense(started)
        if self.steps >= LIMIT:
            self.done = True
        return self.observe()

    def diagnostics(self):
        return {"candidate_visits": sorted(self.storey_rooms), "well_crossings": self.well_events,
                "shaft_envelope_gap_m": self.well_gaps}

    def metrics(self):
        delivered = "delivered" in self.progress
        partial = sum(SCORE.get(k, 0.) for k in self.progress)+ROOMS*len(self.rooms_seen)/len(self.layout.rooms)
        return {
            "rescue_progress": 1. if delivered else round(partial, 2),
            "success": int(delivered), "delivered": int(delivered),
            **{name: int(name in self.progress) for name in SCORE},
            **{f"{kind}_abort": int(self.failure == kind) for kind in FAILURES},
            "pod_impact": int(self.impact_body == "pod"),
            "uav_impact": int(self.impact_body == "uav"),
            "rooms_seen": len(self.rooms_seen), "wing_cells_visited": len(self.cells_seen),
            "overlap_xy": len(self.cells_seen)-len({(x,y) for x,y,_ in self.cells_seen}),
            "upper_entries": sum(e[1]=="up" for e in self.well_events),
            "lower_returns": sum(e[1]=="down" for e in self.well_events),
            "loaded_shaft_crossings": sum(e[2] for e in self.well_events),
            "first_sight_frame": self.first_sight_frame,
            "initial_search_m": round(self.search_m,3),
            "loaded_cargo_cells": len(self.loaded_cargo_cells),
            "steps": self.steps, "simulation_seconds": round(self.data.time, 3),
            "steps_to_lift": self.steps_to_lift, "steps_to_cargo": self.steps_to_cargo,
            "entry_z": round(self.entry_z, 4), "exit_z": round(self.exit_z, 4),
            "mezzanine_z": round(self.mezzanine_z, 4),
            "loaded_descent_z": round(self.loaded_descent_z, 4),
            "yaw_travel_deg": round(math.degrees(self.turn_angle), 1),
            "loaded_trim_m": round(self.loaded_trim_m, 3),
            "command_tracking_offset_m": round(float(np.linalg.norm(self.trim_offset)), 3),
            "close_attempts": self.close_attempts,
            "tool_changes": self.tool_changes, "steps_to_tool": self.steps_to_tool,
            "fit_misses": self.fit_misses,
            "entry_clear": self.entry_clear, "exit_clear": self.exit_clear,
            "entry_margin_m": round(self.throat_margin.get("throat_entry", -1.), 4),
            "exit_margin_m": round(self.throat_margin.get("throat_exit", -1.), 4),
            "settled_mat_margin_m": round(self.mat_margin(), 4) if delivered else -1.,
        }
