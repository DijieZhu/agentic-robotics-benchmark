# DA-6 desktop arm - kinematics sheet

Document DA6-KIN-001, revision A. Valid for every DA-6 unit. All values are the nominal (CAD) design values; lengths in mm and angles in degrees unless a column says otherwise. Lengths are given to 0.0001 mm so that the tables reproduce the CAD model to better than 0.001 mm; the as-built tolerances of a unit are in section 7.

The arm has six revolute joints, each driven by one BUS-SERVO-12V-A bus servo: five arm joints (base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll) and the gripper jaw. With five arm joints the approach direction (TCP +z) always lies in the vertical plane of the arm (perpendicular to the three parallel pitch axes; it turns with the base yaw); the wrist roll then sets the jaw axis about the approach direction.

## 1. Base frame and zero pose

- **Base frame {B}** (= DH frame {0}): origin on the mounting plane (the underside of the base, which stands on the table) on the base-yaw axis; **z** up, along the base-yaw axis; **+y** forward = the direction the arm leans when the shoulder pitches negative at base yaw 0; **x** = y x z (to the right when looking forward).
- **Design angles** q1..q6 (radians on the data interfaces, degrees in this sheet): q = 0 is the **zero pose**. Positive directions follow the right-hand rule about the joint axes of section 3.
- **Zero pose** (all q = 0): the arm stands straight up. The upper arm (shoulder-pitch axis to elbow axis), the forearm (elbow axis to wrist-pitch axis) and the wrist-roll axis are vertical; the jaw is at its reference q6 = 0 (the two finger pads parallel, 15.80 mm apart; negative q6 closes the jaw further, positive q6 opens it; section 5.2). Every link CAD frame is then parallel to {B} (section 6).
- Figure 1 ([figures/zero_pose.png](figures/zero_pose.png)): front view (from +y) and side view (from +x) of the zero pose with {B} = {0}, the DH frames {1}..{5} and the TCP frame.

Frames at the zero pose (origins in {B}, mm; axis directions in {B}):

| frame | origin x | origin y | origin z | x axis | y axis | z axis | located at |
|---|---|---|---|---|---|---|---|
| {0} | 0.0000 | 0.0000 | 0.0000 | +x | +y | +z | mounting plane, base-yaw axis |
| {1} | 0.0000 | 0.0000 | 132.3190 | +y | -x | +z | base-yaw axis, height of the shoulder-pitch axis |
| {2} | 0.1778 | 30.3992 | 132.3190 | +z | -y | +x | shoulder-pitch axis, arm plane |
| {3} | 0.1778 | 30.3992 | 256.9419 | +z | -y | +x | elbow axis, arm plane |
| {4} | 0.1778 | 30.3992 | 406.5024 | -y | -z | +x | wrist-pitch axis = roll axis crossing |
| {5} | 0.1778 | 30.3992 | 475.2655 | -y | +x | +z | roll axis, roll joint (gripper flange) |
| TCP | 0.3960 | 30.4581 | 565.9068 | -y | +x | +z | gripper, between the finger pads (section 5.1) |
| needle tip | 0.3960 | 30.4591 | 610.9068 | -y | +x | +z | dispensing needle, 45.0 mm along TCP +z (section 5.3) |

## 2. Modified Denavit-Hartenberg parameters (Craig convention)

Convention (J. J. Craig, *Introduction to Robotics*, modified DH): frame {i} is fixed to link i; **z_i** lies on the axis of joint i and points in its positive direction; **x_i** lies on the common normal from z_i to z_(i+1) and points towards z_(i+1); y_i = z_i x x_i. The parameters are:

- alpha_(i-1): angle from z_(i-1) to z_i about x_(i-1);
- a_(i-1): distance from z_(i-1) to z_i along x_(i-1);
- d_i: distance from x_(i-1) to x_i along z_i;
- theta_i: angle from x_(i-1) to x_i about z_i, **theta_i = q_i + offset_i**.

Link transform (frame {i} expressed in frame {i-1}):

```
T(i-1, i) = Rot_x(alpha_(i-1)) * Trans_x(a_(i-1)) * Rot_z(theta_i) * Trans_z(d_i)

          | cos(th)             -sin(th)              0            a          |
        = | sin(th)*cos(al)      cos(th)*cos(al)     -sin(al)     -sin(al)*d  |
          | sin(th)*sin(al)      cos(th)*sin(al)      cos(al)      cos(al)*d  |
          | 0                    0                    0            1          |

with al = alpha_(i-1), a = a_(i-1), th = theta_i = q_i + offset_i, d = d_i.

T(B, TCP)(q)  = T(B, 0) * T(0, 1)(q1) * T(1, 2)(q2) * T(2, 3)(q3) * T(3, 4)(q4) * T(4, 5)(q5) * T(5, TCP)
T(B, tip)(q)  = T(B, 0) * T(0, 1)(q1) * ... * T(4, 5)(q5) * T(5, tip)
T(B, 0)       = identity (DH frame {0} coincides with the base frame {B})
```

| i | joint (servo id) | alpha_(i-1) [deg] | a_(i-1) [mm] | d_i [mm] | theta_i offset [deg] |
|---|---|---|---|---|---|
| 1 | base yaw (1) | 0 | 0.0000 | 132.3190 | +90 |
| 2 | shoulder pitch (2) | 90 | 30.3992 | 0.1778 | +90 |
| 3 | elbow pitch (3) | 0 | 124.6229 | 0.0000 | 0 |
| 4 | wrist pitch (4) | 0 | 149.5605 | 0.0000 | +90 |
| 5 | wrist roll (5) | 90 | 0.0000 | 68.7631 | 0 |

Notes on the table:

- d_1 = 132.3190 mm is the height of the shoulder-pitch axis above the mounting plane; frame {1} sits on the base-yaw axis at that height, x_1 pointing forward (+y of {B} at q1 = 0).
- a_1 = 30.3992 mm is the horizontal offset of the shoulder-pitch axis in front of the base-yaw axis.
- The pitch axes z_2, z_3, z_4 are parallel; their common normals are taken in the **arm plane**, the vertical plane x_B = 0.1778 mm that contains the wrist-roll axis, so d_3 = d_4 = 0 and d_2 = 0.1778 mm is the lateral offset of the arm plane from the base-yaw axis (towards +x at q1 = 0).
- a_2 = 124.6229 mm (shoulder-pitch axis to elbow axis) and a_3 = 149.5605 mm (elbow axis to wrist-pitch axis) are the upper-arm and forearm axis distances.
- The wrist-roll axis z_5 intersects the wrist-pitch axis z_4 at right angles (a_4 = 0, alpha_4 = 90 deg); d_5 = 68.7631 mm places frame {5} at the roll joint (the gripper flange). x_5 = x_4 at q5 = 0.

Tool transforms (constant, mm; rotation part = identity: the TCP frame is parallel to frame {5}):

T(5, TCP) (the TCP of section 5.1):

| | | | |
|---|---|---|---|
| 1 | 0 | 0 | -0.0589 |
| 0 | 1 | 0 | 0.2182 |
| 0 | 0 | 1 | 90.6413 |
| 0 | 0 | 0 | 1 |

T(5, tip) (the dispensing-needle tip of section 5.3: in the TCP frame (-0.0010, 0, 45.0) mm, i.e. 45.0 mm along the TCP's +z on the needle axis):

| | | | |
|---|---|---|---|
| 1 | 0 | 0 | -0.0599 |
| 0 | 1 | 0 | 0.2182 |
| 0 | 0 | 1 | 135.6413 |
| 0 | 0 | 0 | 1 |

The same table in machine-readable form (mm, deg):

```json
{
 "convention": "modified DH (Craig): T(i-1,i) = Rot_x(alpha) Trans_x(a) Rot_z(q + offset) Trans_z(d)",
 "rows": [
  {
   "i": 1,
   "joint": "base_yaw",
   "servo_id": 1,
   "alpha_deg": 0.0,
   "a_mm": 0.0,
   "d_mm": 132.319,
   "offset_deg": 90.0
  },
  {
   "i": 2,
   "joint": "shoulder_pitch",
   "servo_id": 2,
   "alpha_deg": 90.0,
   "a_mm": 30.3992,
   "d_mm": 0.1778,
   "offset_deg": 90.0
  },
  {
   "i": 3,
   "joint": "elbow_pitch",
   "servo_id": 3,
   "alpha_deg": 0.0,
   "a_mm": 124.6229,
   "d_mm": 0.0,
   "offset_deg": 0.0
  },
  {
   "i": 4,
   "joint": "wrist_pitch",
   "servo_id": 4,
   "alpha_deg": 0.0,
   "a_mm": 149.5605,
   "d_mm": 0.0,
   "offset_deg": 90.0
  },
  {
   "i": 5,
   "joint": "wrist_roll",
   "servo_id": 5,
   "alpha_deg": 90.0,
   "a_mm": 0.0,
   "d_mm": 68.7631,
   "offset_deg": 0.0
  }
 ],
 "T_base_0": "identity",
 "tcp_in_frame5_mm": [
  -0.0589,
  0.2182,
  90.6413
 ],
 "needle_tip_in_frame5_mm": [
  -0.0599,
  0.2182,
  135.6413
 ],
 "tool_rotation_in_frame5": "identity"
}
```

Check values (computed with the table above; positions in {B}, mm; TCP axes as unit vectors in {B}):

| q1..q5 [deg] | TCP x | TCP y | TCP z | TCP x axis | TCP z axis (approach) | needle tip (x, y, z) |
|---|---|---|---|---|---|---|
| 0, 0, 0, 0, 0 | 0.396 | 30.458 | 565.907 | (0.0000, -1.0000, 0.0000) | (0.0000, 0.0000, 1.0000) | (0.396, 30.459, 610.907) |
| 30, -45, -60, -30, 45 | -187.530 | 325.392 | 68.877 | (0.3624, 0.7866, 0.5000) | (-0.3536, 0.6124, -0.7071) | (-203.440, 352.948, 37.057) |
| -90, 30, -90, 60, -120 | 97.393 | -0.120 | 474.430 | (0.5000, 0.8660, 0.0000) | (0.0000, 0.0000, 1.0000) | (97.392, -0.121, 519.430) |

## 3. Joints

All six joints are revolute. Axis directions are given in {B} at the zero pose; each axis passes through the listed point (zero pose, {B}, mm), which is also the origin of the link CAD frame the joint carries (section 6). The drive direction is the sign between the design angle and the servo's position reading: +1 means the servo's ticks increase when q increases (the same value appears as `direction` in each unit's calibration record, [../unit/calibration_record.md](../unit/calibration_record.md)).

| servo id | joint | type | axis (zero pose) | through point (x, y, z) | positive rotation | drive direction | carries link |
|---|---|---|---|---|---|---|---|
| 1 | base yaw | revolute | +z | (0.0000, 0.0000, 70.8240) | counter-clockwise seen from above | +1 | shoulder |
| 2 | shoulder pitch | revolute | +x | (18.2778, 30.3992, 132.3190) | upper arm tilts backwards (towards -y) | +1 | upper arm |
| 3 | elbow pitch | revolute | +x | (18.2778, 30.3992, 256.9419) | forearm tilts backwards relative to the upper arm | +1 | forearm |
| 4 | wrist pitch | revolute | +x | (18.2778, 30.3992, 406.5024) | gripper tilts backwards relative to the forearm | -1 | wrist |
| 5 | wrist roll | revolute | +z | (0.1778, 30.3992, 475.2655) | counter-clockwise seen from the fingertips | +1 | gripper |
| 6 | jaw | revolute | +x | (-18.6222, 10.1992, 498.6654) | jaw opens (moves away from the fixed finger) | +1 | jaw |

The wrist-roll and jaw axes move with the links that carry them: the roll axis is z_5 of the DH chain; the jaw hinge is fixed on the gripper link at (-18.8000, -20.2000, 23.4000) mm in the gripper CAD frame with its axis along +x of that frame (gripper drawing).

## 4. Design-angle limits

| servo id | joint | lower [deg] | upper [deg] | lower [rad] | upper [rad] |
|---|---|---|---|---|---|
| 1 | base yaw | -105.29 | 107.14 | -1.8377 | 1.8699 |
| 2 | shoulder pitch | -96.71 | 84.17 | -1.6879 | 1.4691 |
| 3 | elbow pitch | -102.15 | 18.44 | -1.7828 | 0.3218 |
| 4 | wrist pitch | -82.24 | 83.25 | -1.4354 | 1.4530 |
| 5 | wrist roll | -157.25 | 131.21 | -2.7445 | 2.2901 |
| 6 | jaw | -5.12 | 94.19 | -0.0894 | 1.6440 |

The angle limits stored in each servo's EEPROM are these design limits converted to ticks through the unit's calibration record ([../unit/calibration_record.md](../unit/calibration_record.md); q = direction x (ticks - zero_ticks) x 2 pi / 4096, rounded inwards to whole ticks: `min_ticks`, `max_ticks`); the servo clamps goal positions outside them. Because the record's zero has the stated calibration uncertainty, the limits of a particular unit, expressed in design angles, are known to the same uncertainty.

## 5. TCP, pads, jaw and clamped tools

### 5.1 TCP and pads

- **TCP** (tool centre point): fixed on the gripper link, in the gap between the finger pads, at the height of the fixed pad's lower edge (10.0 mm below the fixed fingertip). In the gripper CAD frame: (0.2182, 0.0589, 90.6413) mm, i.e. on the wrist-roll axis to 0.2260 mm (0.0589 mm towards the fixed finger along the jaw axis, 0.2182 mm towards +x of the gripper frame) and 90.6413 mm beyond the roll joint along the roll axis.
- **TCP frame**: **+z = approach** (out of the fingers, along the roll axis); **+x = jaw axis** (from the fixed finger towards the moving jaw); +y = z x x. The TCP frame is parallel to DH frame {5}; at the zero pose x_TCP = -y_B, y_TCP = +x_B, z_TCP = +z_B.
- **Fixed finger pad**: the flat top 10.0 mm of the fixed finger (from the TCP's height to the fingertip). Its face is a plane perpendicular to x_TCP at **x_TCP = -7.8410 mm** (gripper CAD frame y = 7.8999 mm); below the pad the finger's face is set back 2.0 mm.
- **Jaw pad at q6 = 0**: parallel to the fixed pad, its face at x_TCP = +7.9589 mm (gripper CAD frame y = -7.9000 mm): the **pad gap at q6 = 0 is 15.7999 mm**. The TCP is the middle of the gap when the pads are 15.68 mm apart (the clamped needle holder, section 5.3); at q6 = 0 the middle of the gap lies 0.0589 mm towards the jaw from the TCP.
- Gripper drawing: [drawings/gripper.png](drawings/gripper.png) (DA6-GR-201).

### 5.2 Jaw: pad gap against the jaw angle

The pad gap is the clear distance between the two pad faces along x_TCP (the jaw axis), within 4 mm of the TCP's height (z_TCP from -4 to +4 mm) over the fingers' middle (y_TCP from -6 to +6 mm). It grows with q6 (positive q6 opens the jaw); between the rows it is linear. The table ends at q6 = 45.16 deg: at larger angles the jaw's pad has left that band. With a part or the needle holder of width w (along x_TCP) clamped between the pads, the jaw stops at the q6 at which the pad gap equals w; while a tool is clamped servo 6 holds the clamp.

| q6 [deg] | q6 [rad] | pad gap [mm] |
|---|---|---|
| -5.124 | -0.089435 | 9.366 |
| -3.867 | -0.067494 | 10.992 |
| -2.610 | -0.045552 | 12.548 |
| -1.353 | -0.023610 | 14.130 |
| -0.096 | -0.001668 | 15.682 |
| 0.000 | 0.000000 | 15.800 (zero pose) |
| 1.162 | 0.020273 | 17.185 |
| 2.419 | 0.042215 | 18.690 |
| 3.676 | 0.064157 | 20.199 |
| 4.933 | 0.086099 | 21.712 |
| 6.190 | 0.108040 | 23.227 |
| 7.447 | 0.129982 | 24.746 |
| 8.705 | 0.151924 | 26.265 |
| 9.962 | 0.173866 | 27.786 |
| 11.219 | 0.195807 | 29.307 |
| 12.476 | 0.217749 | 30.827 |
| 13.733 | 0.239691 | 32.346 |
| 14.990 | 0.261633 | 33.863 |
| 16.248 | 0.283574 | 35.377 |
| 17.505 | 0.305516 | 36.888 |
| 18.762 | 0.327458 | 38.394 |
| 20.019 | 0.349400 | 39.896 |
| 21.276 | 0.371341 | 41.392 |
| 22.533 | 0.393283 | 42.881 |
| 23.791 | 0.415225 | 44.363 |
| 25.048 | 0.437167 | 45.838 |
| 26.305 | 0.459108 | 47.303 |
| 27.562 | 0.481050 | 48.760 |
| 28.819 | 0.502992 | 50.207 |
| 30.076 | 0.524934 | 51.643 |
| 31.334 | 0.546875 | 53.068 |
| 32.591 | 0.568817 | 54.480 |
| 33.848 | 0.590759 | 55.880 |
| 35.105 | 0.612701 | 57.687 |
| 36.362 | 0.634642 | 59.431 |
| 37.620 | 0.656584 | 61.516 |
| 38.877 | 0.678526 | 63.485 |
| 40.134 | 0.700468 | 65.427 |
| 41.391 | 0.722409 | 67.607 |
| 42.648 | 0.744351 | 69.769 |
| 43.905 | 0.766293 | 71.977 |
| 45.163 | 0.788235 | 74.267 |

The same table in machine-readable form:

```json
{"pad_gap_against_q6": "rows [q6 rad, pad gap mm]; linear between the rows",
 "rows": [
  [-0.089435307, 9.366],
  [-0.067493556, 10.992],
  [-0.045551806, 12.548],
  [-0.023610055, 14.13],
  [-0.001668305, 15.682],
  [0.0, 15.8],
  [0.020273446, 17.185],
  [0.042215196, 18.69],
  [0.064156947, 20.199],
  [0.086098697, 21.712],
  [0.108040448, 23.227],
  [0.129982198, 24.746],
  [0.151923949, 26.265],
  [0.173865699, 27.786],
  [0.19580745, 29.307],
  [0.2177492, 30.827],
  [0.239690951, 32.346],
  [0.261632701, 33.863],
  [0.283574452, 35.377],
  [0.305516202, 36.888],
  [0.327457953, 38.394],
  [0.349399703, 39.896],
  [0.371341454, 41.392],
  [0.393283204, 42.881],
  [0.415224955, 44.363],
  [0.437166705, 45.838],
  [0.459108456, 47.303],
  [0.481050206, 48.76],
  [0.502991957, 50.207],
  [0.524933707, 51.643],
  [0.546875458, 53.068],
  [0.568817208, 54.48],
  [0.590758959, 55.88],
  [0.612700709, 57.687],
  [0.63464246, 59.431],
  [0.65658421, 61.516],
  [0.678525961, 63.485],
  [0.700467711, 65.427],
  [0.722409462, 67.607],
  [0.744351212, 69.769],
  [0.766292963, 71.977],
  [0.788234713, 74.267]
 ]}
```

### 5.3 Clamped tools

- **Gripped part** (a box with its faces parallel to the TCP frame, half sizes hx, hy, hz): it rests with its -x face on the fixed pad face and is centred on y_TCP = 0, so its centre lies at x_TCP = -7.8410 mm + hx; its height z_TCP (the seat height) comes with the part's data (the cell's `tool`, [../cells/cell_manual.md](../cells/cell_manual.md)). The jaw closes on its +x face at the q6 of section 5.2 for the gap 2 hx.
- **Dispensing needle in its holder**: the holder is a block of 15.68 x 12.00 x 18.00 mm (along x, y, z of the TCP frame) clamped between the pads, its -x face on the fixed pad face; its centre lies at (-0.0010, 0.0000, -3.0000) mm in the TCP frame (from 12.0 mm below to 6.0 mm above the TCP). The jaw closes on its +x face at q6 = -0.0972 deg (-0.001697 rad; pad gap 15.68 mm). The needle (diameter 2.4 mm) runs along the TCP's +z on the holder's centre line (x_TCP = -0.0010 mm, y_TCP = 0): it leaves the holder's top face at z_TCP = 6.0 mm and ends in a full radius R1.2 at z_TCP = 45.0 mm.
- **Needle tip** (the tool point of dispensing jobs): the end of the needle, (-0.0010, 0, 45.0) mm in the TCP frame: 45.0 mm along the TCP's +z on the needle axis (T(5, tip) of section 2; needle drawing [drawings/needle.png](drawings/needle.png), DA6-TL-301). Needle direction = the TCP's +z.
- **Needle and holder, mass properties**: 20.0 g; centre of mass at (-0.0010, 0.0000, 8.0000) mm in the TCP frame (on the needle axis); inertia about the centre of mass, TCP axes: Ixx = 4.0e-06, Iyy = 4.0e-06, Izz = 2.0e-06 kg m^2, products 0.
- While clamped, the part, or the holder and the needle, belong to the arm's collision envelope ([cad/collision/README.md](cad/collision/README.md)).
- Offsets of the TCP from the roll axis and along it are fixed by the gripper; the wrist-pitch axis to tool distance along the roll axis carries the per-unit tolerance of section 7, note 4.

## 6. Link CAD frames

The link CAD files (`cad/<link>.stl`), the collision envelope (`cad/collision/<link>.stl`) and the mass properties (`mass_properties.md`) are given in each link's CAD frame. **At the zero pose every CAD frame is parallel to the base frame {B}**; its origin lies on the axis of the joint that carries the link. Origins at the zero pose (in {B}, mm):

| link | carried by joint | origin x | origin y | origin z |
|---|---|---|---|---|
| base | - (fixed; = {B}) | 0.0000 | 0.0000 | 0.0000 |
| shoulder | base yaw (1) | 0.0000 | 0.0000 | 70.8240 |
| upper arm | shoulder pitch (2) | 18.2778 | 30.3992 | 132.3190 |
| forearm | elbow pitch (3) | 18.2778 | 30.3992 | 256.9419 |
| wrist | wrist pitch (4) | 18.2778 | 30.3992 | 406.5024 |
| gripper | wrist roll (5) | 0.1778 | 30.3992 | 475.2655 |
| jaw | jaw (6) | -18.6222 | 10.1992 | 498.6654 |

Placing a link at a configuration q (the CAD chain; the parent of the jaw is the gripper):

```
T(B, link)(q) = T(B, parent)(q) * Trans(p_link - p_parent) * Rot(a_j, q_j)
```

with p the zero-pose origins above, j the joint that carries the link, a_j its axis direction from section 3 (the same vector in the parent's CAD frame) and T(B, base) = identity. The same frames relative to the DH frames (constant; origin of the CAD frame in frame {i}, mm, and the CAD axes expressed in frame {i}):

| link | DH frame | CAD origin in {i} | CAD x axis | CAD y axis | CAD z axis |
|---|---|---|---|---|---|
| shoulder | {1} | (0.0000, 0.0000, -61.4950) | -y_1 | +x_1 | +z_1 |
| upper arm | {2} | (0.0000, 0.0000, 18.1000) | +z_2 | -y_2 | +x_2 |
| forearm | {3} | (0.0000, 0.0000, 18.1000) | +z_3 | -y_3 | +x_3 |
| wrist | {4} | (0.0000, 0.0000, 18.1000) | +z_4 | -x_4 | -y_4 |
| gripper | {5} | (0.0000, 0.0000, 0.0000) | +y_5 | -x_5 | +z_5 |

(The base CAD frame is {B} = {0}. The jaw CAD frame is the gripper CAD frame translated to the jaw hinge and rotated by q6 about its +x axis.)

## 7. Unit tolerances (tolerance block of the drawings)

Every unit is built to the nominal values of this sheet within:

```
UNIT TOLERANCES
1. Joint-frame position (every joint, jaw hinge included): +-0.3 mm in each of x, y, z
   of the link frame.
2. Pitch-axis direction (shoulder, elbow, wrist pitch): within 0.15 deg about each of the two
   axes perpendicular to the nominal axis (y and z of the link frame).
3. Axis distances: upper arm 124.62 mm and forearm 149.56 mm: +-0.3 % (+-0.37 mm, +-0.45 mm).
4. Wrist to tip: wrist-pitch axis to TCP (and needle tip) along the roll axis +-0.5 mm.
5. Link masses and inertias: +-5 % (mass properties).
Calibration: the zero of each servo is set per unit with the assembly jig and delivered in the unit's
calibration record (uncertainty +-0.5 deg arm joints, +-1.0 deg jaw).
```

Notes 1-4 act on the DH table: 1 on the joint positions of section 3 (all three coordinates), 2 on the directions of z_2, z_3, z_4, 3 on a_2 and a_3, 4 on d_5 (the TCP and the needle tip move with it). The servo-related ranges (output compliance, hysteresis, latency, speed, position loop) are in the servo datasheet ([../servo/servo_datasheet.md](../servo/servo_datasheet.md)).

