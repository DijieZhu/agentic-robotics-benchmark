# Customer brief: eight upgraded replicas of the lab's inspection quadruped

Clause ids in square brackets are the trace ids of your requirements sheet (`source_id`).

[B-01] Context. The customer orders 8 units that replicate the lab's reference unit: a 12-actuator, belt-reduced
quadruped (four legs with three revolute joints each, 12 joints in all, one actuator module per joint), shown in the
survey photos PH-01..PH-36 and described in the build log (BL). The replicas keep the reference unit's kinematics, its
printed parts (the nose box, the shoulder shells with their rings and the spacer rings included) and its printed
stops. They get NEW actuator modules, a new battery pack, new drivers and a new hub for the customer's heavier, longer
duty. The reference unit has left the lab on a long loan and cannot be measured again.

[B-02] Precedence. This brief > the build log > the lot reports > the datasheets. An erratum sheet overrides the
datasheet it corrects. Lot data describe the lot they were measured on; the pilot modules and packs come from the lots
named in the quote. The photos show the reference unit as it was when it left the lab.

[B-03] Units. Deliverables use SI units (m, rad, kg, s, N, N m, rad/s, A, V, Ah; `fraction`, `count` and `USD` for
ratios, counts and costs); the unit table of the schemas lists the accepted equivalents (mm, g, min, h, rpm, lbf.in,
mAh, %, ...). An item's unit must convert to the SI unit of its quantity.

[B-04] Payload. The replicas carry a payload of at least the bottom and at most the top of the payload band. The top
of the payload band is the mass of the customer's sensor pod on the kitchen scale in PH-35 (read the display); the
bottom of the band is 1.34 kg. The pod (with its own base plate, as weighed) mounts on the deck's mount plate (its
four inserts, PH-31), long side along the trunk, centred on the plate. Its centre of mass is where the customer marked
it: the two marks on its long sides. The replicas carry the pod as it is. Its box (PH-34..PH-36) is the pod as it
stands on its base plate, the plate included: the length of its long side, its width, and its height from the bottom
face of the plate to the top of the box.

[B-05] Stance. The trunk is level; the plane of the four hip_roll axes is 0.262 m above the ground; every foot point
lies directly under its hip_pitch axis, with its hip_roll joint at 0; the knees are in the configuration of
PH-01..PH-12.

[B-06] Stand-up. From the crouch, with the hip_roll-axis plane 0.147 m above the ground and the feet in their stance
positions, to the stance of B-05 within 1.0 s (quintic profile, B-14 c).

[B-07] Inspection crawl. The stop-and-go crawl of B-14 d with a step length of 0.138 m, a swing height of 0.067 m, a
weight shift of 0.95 s and a swing of 0.54 s per step (neither may take longer) and a sway fraction of 0.65;
continuous duty: the cycle repeats without rest.

[B-08] Proof. The four swing-apex poses of the crawl, each held static on three feet with a proof load of at least
1.25 x the top of the payload band on the pod mount (B-14 e).

[B-09] Battery and mission. One Ardent Cells pack of the 6S or 8S family per unit, carried in the reference unit's
pack bay (Appendix D); its outline must fit that bay. Mission: at least 51 min on one charge; at most 80 % of the
delivered capacity usable; 3.5 V per cell at the end of the mission.

[B-10] Actuators. From the vendor pages DS-* only. One module SKU per joint type on all four legs: the legs are
interchangeable (one qualification and one spare type per joint type). The module's can must fit the replicated
housing of its joint type (PH-25..PH-27). A vendor without an incoming-inspection report counts at no more than 70 %
of its typical (datasheet) peak torque, no-load speed and continuous torque, on the datasheet's straight-line
characteristic.

[B-11] Power. One pack, one driver type (count = 12 / channels per board) and one hub per unit; the harness drop of
BL-03 at every module; the tray current of BL-03; no current limiting during the duty.

[B-12] Cost. At most 1589 USD per unit for the modules, the pack, the drivers and the hub at the 10-99 pcs list price
(price_list.md).

[B-13] Engineering change. Any BOM line changed after the pilot acceptance costs 60 USD per unit (rework and
re-qualification).

[B-14] Acceptance procedure. The customer runs it on the pilot unit; it is public and exact. Its code is in
`grader_public/quad/req_info/q1/`: `quadmodel.py` builds the dynamics model from a replica model, `procedure.py` runs
the duty and the pass rules, `lotmodel.py` gives the capabilities; the code is the definition of the words below.
Coordinates are in the trunk frame of Appendix D; the world frame has z up and the ground at z = 0; the trunk never
rotates.
  a) Replica model. The procedure takes a replica model (`quad-model/1`, schema in `schemas/`): the Appendix-D
     geometry (with the spacer legs, the spacer thickness, the foot radius, the stops and the housing classes); every
     printed part and fixed component as a rigid inertial (mass, centre of mass, inertia tensor, attached to the trunk
     or to a link); the seats of the 12 modules; the pack bay's centre and inner box; the pod mount and the pod's CoM
     marks. `quadmodel.py` builds from it the dynamics model: a free joint at the trunk and 12 hinges (hip_roll about
     +x, hip_pitch and knee about +y, on every leg, no mirrored axes), with no armature, damping, friction, contacts or
     actuators. The BOM enters as point masses: each module at its seat, with the box inertia per kg of the can box
     of its joint type's housing class; the pack at the bay centre, with the bay's box inertia per kg; the pod
     (negligible inertia) at its marked centre of mass.
  b) Stance (B-05). The trunk is level with the hip_roll-axis plane h_s above the ground. Every foot point sits r_f
     (the foot radius) above the ground, directly under its hip_pitch axis with its hip_roll joint at 0: at
     (+-pitch_axis_spacing / 2, +-(roll_axis_spacing / 2 + leg_offset [+ hip_spacer on a spacer leg]), r_f - h_s) in
     the trunk frame. Leg inverse kinematics is analytic per leg: hip_roll from the lateral offset in the frontal
     plane, then the planar two-link solution for hip_pitch and knee in the knee configuration of B-05. Joint rates
     come from the analytic leg Jacobian J: qd = J^-1 (pd_foot - pd_trunk) and qdd = J^-1 (pdd_foot - pdd_trunk -
     Jd qd).
  c) Stand-up (B-06). In the crouch the trunk sits at x = y = 0 with the hip_roll-axis plane h_0 above the ground
     and the feet at their stance positions; it rises to the stance in T_up along s(u) = 10 u^3 - 15 u^4 + 6 u^5,
     u = t / T_up, on four stance feet. Sampled every DT = 0.005 s, both ends included.
  d) Inspection crawl (B-07). One steady cycle of four steps in the leg order rr, fr, rl, fl, repeated without rest.
     Before the first swing the feet sit at x-offsets from their stance positions of rr -s/2, fr -s/4, rl 0, fl +s/4
     (s: the step length). Each step is a shift, then a swing:
     - shift (T_sh, the quintic s(u), four stance feet): the trunk translates horizontally (level, at h_s) to
       c4 + k_sw (c3 - c4), where c4 is the mean (x, y) of the four feet and c3 the mean (x, y) of the three feet
       that stay down for the coming swing (k_sw: the sway fraction);
     - swing (T_sw, trunk still, the other three feet in stance): the swing foot moves from p0 to p0 + (s, 0, 0)
       along x(u) = s s(u), z(u) = s_lift sin^2(pi u), u = t / T_sw (s_lift: the swing height), and is a stance foot
       again at u = 1.
     The cycle is the steady one: the trunk starts at the shift target of the previous cycle's last step (the feet as
     above with fl one step further back), so the cycle ends at its start moved by (s, 0, 0). Every shift and swing is
     sampled every DT at u = i / n, i = 1..n, n = T / DT. The crawl speed is s / (4 (T_sh + T_sw)).
  e) Proof holds (B-08). The four swing-apex poses (u = 1/2) of the cycle, each held static (qd = qdd = 0) on its
     three stance feet with the pod at 1.25 x the top of the payload band.
  f) Loads and contact rule. For every sample: h = the 18 generalized forces of the rigid-body inverse dynamics
     (gravity 9.81 m/s^2 along -z, no contacts); J_i = the 3 x 18 translational Jacobian of each stance foot point
     and J_S the stack of them; f = pinv(J_Sb^T) h_b, the minimum-norm stance forces (world frame, at the foot points;
     pinv with rcond 1e-10; J_Sb = the six free-joint columns of J_S, h_b = the six free-joint rows of h); the joint
     load torques are tau = h_j - J_Sj^T f (the joint rows of h, the joint columns of J_S). With three or four stance
     feet, as in every sample of this duty, the base wrench is reproduced exactly; with three feet the normal forces
     are statically determinate. Forces act at the foot points; unilateral contact and friction are not enforced.
     tau excludes the joint's own output inertia, which the pass rule adds.
  g) Capability of a SKU: the weakest expected unit of the lot the pilot parts come from (B-02). The modules see
     V_s = n_cells x 3.5 V - V_drop (the end-of-mission cell voltage of B-09 times the cell count of the BOM's
     battery family, minus the harness drop of BL-03). Peak torque, no-load speed and peak phase current at V_s by
     linear interpolation or extrapolation between the page's two voltage points; a page with one voltage point gives
     the peak torque and the peak current as printed and the no-load speed in proportion to the voltage. A unit's
     ratio to the datasheet (measured / datasheet at the same voltage) is, for the peak torque, f_peak (1 + s z1); for
     the no-load speed, f_speed (1 + s z2); for the continuous torque, f_cont (1 + s z3); z1..z3 are independent
     standard normal per unit. The lot factors f_peak, f_speed, f_cont, the lot spread s and the lot droop d belong to
     one lot and apply to every SKU of its vendor. The weakest expected unit is z = -1.96 on every factor:
     tau_s = peak(V_s) f_peak (1 - 1.96 s), w0 = no-load(V_s) f_speed (1 - 1.96 s), tau_c = continuous f_cont
     (1 - 1.96 s). A unit's speed under the torque tau is w(tau) = w0 (1 - u) (1 - d u) with u = tau / tau_s (d = 0:
     the datasheet's straight line). A vendor without an inspection report counts at 0.7 x its typical peak
     torque, no-load speed and continuous torque, with d = 0 (B-10).
  h) Actuator joint j passes (12 items) iff, with that capability of the BOM's module of its type (J_out: the
     module's output inertia from its page):
     - move: at every stand-up and crawl sample |tau_j| + J_out |qdd_j| <= tau_s u*(|qd_j| / w0, d), where u*(x, d)
       is the u in [0, 1] with (1 - u) (1 - d u) = x, x clipped to [0, 1] (u* = 1 - x when d = 0): the torque the unit
       has at the speed |qd_j|;
     - proof: at every proof pose |tau_j| <= tau_s;
     - duty RMS: sqrt(mean over the crawl cycle of (tau_j + J_out qdd_j)^2) <= tau_c;
     - fit: the module's housing class equals the housing class of joint j's type;
     - voltage: the pack's window (its cell count times the cut-off and the full-charge cell voltage of its page)
       lies inside the module's voltage range.
  i) Battery passes iff its cell count is that of the BOM's battery family; its window lies inside every module's and
     the driver's input range; the peak bus current over the stand-up and the crawl <= its continuous discharge
     current; its outline fits the pack bay (the sorted outline dimensions <= the sorted inner dimensions of the bay);
     and the mission charge (the crawl cycle's mean bus current x the mission time) <= 0.8 x its rated capacity
     x g_lot. A delivered pack holds its rated capacity x g_mean (1 + s_g z), z standard normal per pack; g_mean and
     the spread s_g belong to the cell lot, and one cell lot serves every pack type of its vendor. g_lot = g_mean
     (1 - 1.96 s_g) is the weakest expected pack of the cell lot named in the quote; its incoming-inspection report
     gives the measured capacities.
  j) Currents. The phase current of joint j is i_j = |tau_j + J_out qdd_j| x I_pk(V_s) / tau_s (the weakest unit's
     current per unit torque). The bus current is I_bus = I_tray + sum over the 12 joints of [max(0, tau_j qd_j) +
     1.5 R_ph,j i_j^2] / (eta V_s), with I_tray the tray current of BL-03, R_ph,j the module's phase resistance and
     eta = 0.92, the drivers' efficiency at rated load (92 %, TK pages). Regenerated power is not credited.
  k) Drivers pass iff the BOM has one driver type with count = 12 / channels per board; its input range contains the
     pack's window; every joint's peak i_j over the stand-up and the crawl <= the channel's peak current; and every
     joint's crawl-cycle RMS of i_j <= the channel's continuous current.
  l) Hub, budget, acceptance. The hub passes iff its driver ports >= the driver count. The budget passes iff the
     BOM's list price (four modules of each joint type, the pack, the drivers and the hub) <= the B-12 ceiling. The
     unit is accepted iff all 16 items pass: the 12 actuator joints, the battery, the drivers, the hub and the budget.
     A failed item is fixed by an engineering change (B-13).
  m) Sampling and linearity. The stand-up has T_up / DT + 1 samples (201 here), the crawl cycle
     4 (T_sh + T_sw) / DT and the proof four poses. The loads and the normal forces are linear in the BOM masses:
     tau = tau0 + m_hip_roll B_hip_roll + m_hip_pitch B_hip_pitch + m_knee B_knee + m_pack b_pack + m_pod b_pod (in the
     proof poses b_pod carries the proof factor), from six inverse-dynamics passes; one BOM evaluation then takes
     about a millisecond.

[B-15] Conventions. Appendix D below is verbatim part of this brief.

[B-16] Deliverables. `requirements.yaml`, `bom.json` and `kinematic_sketch.json` (schemas in `schemas/`). The
requirements sheet traces every requirement to its source (`source_id`: a brief clause B-nn, a build-log entry BL-nn,
a photo PH-nn, a datasheet DS-..., a lot report LR-...). It also reports, with `source_id: derived` and a
`derivation` note, the 16 derived items of the acceptance duty for YOUR BOM installed on the replica, each as the
value the procedure of B-14 produces (worst leg of the joint type; the delivered lots' weakest expected units; SI
units):
- `proof_peak_torque:hip_roll`, `proof_peak_torque:hip_pitch`, `proof_peak_torque:knee`: the largest |tau| over the
  type's four joints and the four proof poses;
- `duty_rms_torque:hip_roll`, `duty_rms_torque:hip_pitch`, `duty_rms_torque:knee`: the largest crawl-cycle RMS of the
  load torque tau over the type's four joints (without J_out);
- `standup_peak_torque:knee`: the largest |tau| of the knees over the stand-up;
- `joint_peak_speed:hip_pitch`, `joint_peak_speed:knee`: the largest |qd| of the type over the stand-up and the crawl;
- `supply_peak_current`: the largest I_bus over the stand-up and the crawl (B-14 j);
- `mission_charge`: the crawl cycle's mean I_bus x the mission time (Ah);
- `drive_peak_current:knee`, `drive_rms_current:knee`: over the knees, the largest phase current over the stand-up
  and the crawl, and the largest crawl-cycle RMS phase current;
- `robot_mass`: the robot with your BOM installed, without the pod;
- `loaded_com_x`: the x of the whole system's centre of mass in the trunk frame at the B-05 stance, with your BOM and
  the pod at the top of the payload band;
- `peak_normal_force`: the largest normal stance force over the stand-up, the crawl and the proof holds.

## Appendix D. Conventions (verbatim part of the brief, B-15)

- **Trunk frame.**
  - x points forward (the arrow on the deck, toward the nose), y left, z up.
  - The origin lies on the plane of the four hip_roll axes (all parallel to x), midway between the left and right
    hip_roll axes and midway between the front and rear hip_pitch axes, with the hip_roll joints at 0.
  - Legs `fl, fr, rl, rr`.
- **Foot point.** The centre of the round foot, on its axis at mid-width: the contact rule's force point.

| quantity | definition |
|---|---|
| `pitch_axis_spacing` | distance between the front and the rear hip_pitch axes (parallel to y), hip_roll at 0 |
| `roll_axis_spacing` | distance between the left and the right hip_roll axes |
| `pitch_axis_drop` | distance of the hip_pitch axis below the hip_roll-axis plane, hip_roll at 0 |
| `leg_offset` | lateral (y) distance from a leg's hip_roll axis to its foot point, hip_roll at 0, for a leg without a spacer ring |
| `hip_spacer` | how much farther from the trunk's midplane, in y, the foot point of a leg with a spacer ring lies than a plain leg's (the ring's effective thickness) |
| spacer legs | the set of legs that carry spacer rings (may be empty) |
| `thigh_length` | distance between the parallel hip_pitch and knee axes |
| `shank_length` | distance from the knee axis to the foot point |
| `foot_radius` | outer radius of the foot's rubber tyre |
| pack bay | the bay of the trunk that carries the reference unit's pack (nose box or belly bay); B-14 a puts the replica's pack CoM at its centre |
| mount plate centre | the centre of the deck's four mount inserts |
| hip_roll angle | rotation about the leg's hip_roll axis; 0 when the leg plane is vertical; + = abduction (the foot moves away from the trunk's midplane) |
| hip_pitch angle | angle of the vector hip_pitch axis -> knee axis from straight down; + when the knee moves toward the trunk centre (front legs: backward; rear legs: forward) |
| knee angle | angle from the thigh vector to the shank vector (knee axis -> foot point); 0 = straight; + = folding |
| stops | `[lower, upper]` per joint type in these conventions: the printed limiters, identical on every leg (PH-13..PH-24) |
| housing class | per joint type: the class (R52, R38, R62) of the can that the housing takes as replicated; a ring printed in one piece with a shell is part of the housing |

In the deliverables the eight lengths use the names above (metres), "spacer legs" is `spacer_legs` (a list of leg
names), "pack bay" is `pack_bay` (`nose_box` or `belly_bay`), "housing class" is `housing_class` (`R52`, `R38` or
`R62` per joint type) and "stops" is `stops` (radians, the conventions above).
