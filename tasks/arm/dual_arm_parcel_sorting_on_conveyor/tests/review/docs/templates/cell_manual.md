# The label-facing station: the integrator's data sheet

This is the integrator's description of the station your program drives: the conveyor, the robot and its controller,
its grippers, the cameras, the parcels the line brings, how a shift is scored, the safety watch, the budgets and
commissioning. Everything stated here as exact is exact. The wire is described in `cell_api.md`, the public numbers are
collected in `cell_config.json`, and the model files of the robot and of the station are in `../robot/` and
`../station/`.

## 1. Frame and units

World frame: z up, the floor at z = 0. The robot's column stands at the origin and the robot faces +x. The belt runs
along y in front of it, and parcels travel towards +y (downstream). Units: metres, radians, seconds, newtons;
quaternions are written w, x, y, z. One control tick is {{tick_ms}} ms of simulated time ({{substeps}} physics steps of
{{phys_ms}} ms).

## 2. The station

A straight belt conveyor passes in front of the robot. Its parts, all rigid and fixed except the belt itself:

* **Belt surface.** Top at z = {{belt_top}} m, centre line at x = {{belt_cx}} m, {{belt_w}} m wide (x from {{belt_x0}}
  to {{belt_x1}} m). The belt runs on beyond both ends of the frame.
* **Frame (shell).** From y = -{{frame_L}} to +{{frame_L}} m ({{frame_len}} m long), its top {{shell_gap_mm}} mm below
  the belt surface, {{shell_over_mm}} mm wider than the belt on each side, {{shell_depth}} m deep.
* **Legs.** {{leg_n}} legs under the frame at y = {{leg_ys}} m, each {{leg_posts_text}}, up to {{leg_gap}} m below the
  belt surface.
* **Far rail.** Along the whole frame on the far edge (centre x = {{rail_far_x}} m), its top {{rail_far_above}} m above
  the belt surface.
* **Near rails.** On the near (robot) side outside the work window (centre x = {{rail_near_x}} m), their tops
  {{rail_near_above}} m above the belt surface.
* **Work window.** Along |y| < {{window_W}} m ({{window_len}} m) the near side has no rail, so the arms can reach in.
  A low lip runs along the near edge {{lip_extent}} (y from {{lip_y0}} to {{lip_y1}} m, centre x = {{lip_x}} m), its top
  {{lip_above_mm}} mm above the belt surface.
* **Hardware near the arms.** The infeed camera's housing, its mounting plate and its drop rod above the upstream
  belt, the belt's drive unit under the belt near its downstream end with its motor on the far side, the head camera's
  housing on the robot's head and the further station hardware in the table below are solid too. The collision shapes
  are the drawn parts (boxes around them where a hand or a parcel can meet them).
* **The robot's base.** The robot's pedestal (a box {{pedestal_half}} m in half size around the column, {{pedestal_top}}
  m tall) stands on the floor at the origin; the belt's near edge is at x = {{belt_x0}} m.

The collision shapes the cell uses (boxes; centre and half sizes in the world frame):

{{hardware_table}}

The lines across the belt, from upstream to downstream:

| line | y (m) | what happens there |
|---|---|---|
| set-down | {{set_y0}} to {{set_y1}} | each parcel of the list is set down on the belt's centre line here (section 7) |
| work window | -{{window_W}} to +{{window_W}} | the near side is open (above) |
| indexing zone | {{zone_y0}} to {{zone_y1}} | on an indexing line the belt stands while the parcel in the station has its centre here (section 3) |
| scan line | {{scan_y}} | each parcel is judged as its centre crosses this line (section 8) |
| belt end | {{exit_y}} | a parcel whose centre passes this line leaves the station |

A parcel that drops more than {{fall_drop}} m below the belt surface, or whose centre gets more than {{fall_xy}} m from
the column along x or along y, has left the line: it is removed and counts as fallen (section 8).

`../station/station.xml` holds this geometry as a MuJoCo model (the boxes above, the belt surface as a fixed box, the
`infeed` camera at the installation drawing's pose with the data sheet's field of view, and marker sites for the
lines; `station_with_robot.xml` adds the robot with its three cameras, section 6). It is geometry only: its
friction, contact and drive settings are not the cell's, and it holds no parcel.

## 3. The belt

* **Belt speed.** Constant through a shift and set per shift between {{v_min}} and {{v_max}} m/s. Your program does not
  command the belt, as on the real line. The belt's drive is stiff: parcels and the robot pressing on the belt do not
  change its speed measurably.
* **Running and indexing lines.** On a running line the belt runs through the whole shift. On an indexing line the
  belt stands while the centre of the parcel in the station (the one set down last and not yet settled, section 7) is
  between y = {{zone_y0}} and {{zone_y1}} m, and runs again, at the shift's speed, once that parcel's centre has left
  that zone (pushed on downstream, or taken back upstream); the drive checks it at the start of every tick. A parcel
  left in the zone keeps the belt standing, and the next parcel of the list does not come before it is settled. A
  shift's line is one or the other from its start to its end; which, is not on the wire. {{halt_text}}
* **Grip of the belt.** A rubber top cover (friction coefficient between {{belt_fr_lo}} and {{belt_fr_hi}} against a
  parcel, by the data sheet): a parcel
  lying on the belt rides at belt speed; a parcel held back or blocked slips on the belt, and the belt keeps pulling it
  downstream.
* **No line signals.** The robot cell is not wired to the line's own sensors: nothing on the wire gives the belt's
  speed or its mode, or tells that a parcel has arrived.
* **Arrivals in the infeed camera's view.** A parcel is set down with its centre between y = {{set_y0}} and {{set_y1}}
  m (section 7). The top face of the tallest parcel ({{infeed_top_h}} m above the belt) is inside the `infeed` camera's
  view from y = {{infeed_ytop0}} to {{infeed_ytop1}} m. The largest footprint at its longest along the belt (its
  diagonal, {{infeed_long}} m: a parcel may lie at any yaw) lies wholly inside the view while its centre is between y =
  {{infeed_whole_y0}} and {{infeed_whole_y1}} m: at the fastest belt ({{v_max}} m/s) that lasts {{infeed_whole_s}} s.
  Some part of a parcel is in that view for about {{infeed_part_s}} s after its set-down at the fastest belt, longer at
  a slower one.

## 4. The robot

A fixed dual-arm platform: a column on a pedestal with a head on top (z up to {{head_top}} m), and two arms of
{{n_arm}} joints each, their shoulders {{shoulder_y}} m to either side of the column's axis at z = {{shoulder_z}} m.
The left arm is on the +y side (downstream), the right arm on the -y side (upstream).

* **Joints.** {{n_joints}} joints, always in this order (left arm, then right arm, each from the shoulder out). The
  ranges and the torque limits:

{{joint_table}}

* **Mirror image.** The right arm is a mirror image of the left about the x-z plane, but not every joint axis is
  mirrored the same way: {{mirror_rule}}. The ranges are not symmetric (the table).
* **Commands.** Each tick the robot follows {{n_joints}} joint position targets and two gripper commands. A target is
  held until you replace it. The cell limits how fast the applied target moves: at most {{cmd_speed}} rad/s, that is
  {{cmd_step}} rad per tick for each joint, toward the commanded target. A target outside a joint's range is clipped
  to the range, never refused. `q_target` in the state is the applied target after that limit. The drives are position
  servos with velocity damping, limited to the torque limits: a small commanded step is followed smoothly and without
  overshoot, a moving target with a lag (the tool-pose level below gives it as measured), and under a steady load a
  joint settles off its target in proportion to the load (the servo's stiffness: the arm's compliance under load, as
  measured below). Their gains are not published; they are found in commissioning. `q` and `dq` are measured exactly;
  `tau` is the drives' torque reading, the motor torque with the sensor's noise: a zero-mean noise of
  {{tau_noise_pct}} % of the joint's torque limit (one sigma, independent per joint and per tick) and a slowly
  wandering offset of at most {{tau_offset_pct}} % of the limit. The drive's own data sheet allows {{vendor_speed}}
  rad/s; the cell's command limit is the lower one above.
* **Tool-pose level.** Instead of an arm's joint targets, a `step` may give that arm a target pose of its tool centre
  point in the world frame (`tool_pose`: a position and a unit quaternion, `cell_api.md`); the two arms may use
  different levels in one step, one arm only one. The station's controller then computes that arm's joint targets
  itself, every tick: it solves the arm's inverse kinematics by damped least squares from the arm's present solution
  (at most {{tp_iters}} iterations per tick, each kept only when it brings the tool closer), taking the smallest joint
  change that reduces the pose error, so the posture stays near the present one and the redundant joint does not
  drift; it keeps every joint {{tp_margin}} rad inside its range; it scales the joint step to at most {{tp_speed_pct}} %
  of the command speed limit per tick (the path's direction is kept and the limiter never has to cut it); and it adds
  a static gravity correction to the joint targets (the servos' sag under the arm's own weight), so that the measured
  tool pose, not only the computed one, settles at the target. A pose counts as reached when the computed solution is
  within {{tp_reached_mm}} mm and {{tp_reached_deg}} degrees of it. The solve weighs one radian of orientation error
  like {{tp_rot_m}} m of position error, so the orientation comes first: a target the arm cannot reach from its posture
  (beyond its reach, or beyond a joint's range) is approached in position as far as the arm goes with the tool's
  orientation kept close to the target's (the tool is not turned away to gain position); the arm then stays there,
  smoothly, without a stop, and the state says so: `tool_target.out_of_reach` turns true when a tick no longer brings
  the solution {{tp_stall_pct}} % closer and then stays true (it latches) until the target changes, the target is
  reached, or the solution's position error falls more than {{tp_unlatch_mm}} mm (or {{tp_unlatch_pct}} %, whichever is
  larger) below its value when the flag was set; `error_m` and `error_deg` give the remaining error. Where it stops
  short, the hand may also be displaced from the target in any direction, sideways included, by several centimetres,
  and so may a hand holding a parcel between the two hands, into that parcel: {{tp_beyond_text}}; `out_of_reach` and
  `error_m` report it at every step. The solve is local: it follows the present posture and does not search other
  postures. The
  controller avoids nothing: the tool moves towards the target whatever lies on the way (the safety watch still
  applies). A tool-pose target is held until replaced; joint targets for that arm, a protective stop or a lock-out end
  it (the arm then holds its posture). `q_target` in the state shows the joint targets the controller applied. Each
  tool's measured pose (forward kinematics of the measured joints) is `tool` in every state reply. As measured on this
  station: {{tp_measured_text}}
* **Start posture.** Every shift starts with the arms at `describe()["robot"]["start_posture"]` = {{start_posture}}:
  both hands parked beside the column, palms down and tilted, the grippers open, nothing touching. Before t = 0 the
  cell lets the arms settle there for {{settle_s}} s; the measured posture then differs from the commanded one by at
  most {{start_sag_mrad}} mrad (the servos' sag under gravity). Every part of the robot is then behind the belt's near
  edge (its largest x is {{start_xmax}} m, {{start_behind_mm}} mm behind the edge at x = {{belt_x0}} m), so parcels
  ride past untouched. The tool centre points are then at {{tcp_start_left}} (left) and {{tcp_start_right}} (right).
* **Tool centre point.** The grasp centre between the finger pads, {{tcp_off}} m from the wrist flange along the tool
  axis (the flange's z axis): `describe()["robot"]["T_flange_tcp"]`, with the flange sites `Left_flange` and
  `Right_flange` of the model.
* **Contacts.** The robot's links and grippers collide physically with each other, with the station and with the
  parcels; nothing passes through anything. The safety watch (section 9) watches the contacts with the station, between
  the two arms and with the belt.
* **Model files.** `../robot/` holds the robot with both grippers (`dual_arm_with_grippers.xml`, a MuJoCo model with
  the maker's data), the arm alone and the gripper alone, and their meshes. Exact in them: the kinematics, the
  collision shapes and the torque limits. Nominal: every link's mass and inertia and every joint's damping, friction
  loss and armature; the station's robot differs from these values by up to {{mass_tol_pct}} % (masses and inertias)
  and up to {{joint_tol_pct}} % (joint damping, friction loss and armature). Not in them: the servos' gains (each
  actuator there is a torque input at the drive's torque limit). They contain neither the station nor any parcel. Its
  `README.md` gives the frames and names.

## 5. The grippers

* **Gripper.** A 2F-85 class adaptive parallel gripper on each wrist flange (the MuJoCo Menagerie model of the Robotiq
  2F-85, BSD-2-Clause, licence in `../robot/gripper_2f85/LICENSE`): two linked fingers, a stroke of 0 to
  {{grip_max_mm}} mm, two high-friction pads on each finger (friction coefficient between {{pad_fr_lo}} and
  {{pad_fr_hi}} against a parcel, by the data sheet). A parcel is held by friction only.
* **Command.** `grip` = 0 (open) .. 1 (closed), continuous: the finger drive's target. From fully open to closed on
  nothing takes about {{grip_close_s}} s, from closed to fully open about {{grip_open_s}} s. The opening a command
  settles at, on nothing: {{grip_curve}}.
* **State.** As the real gripper reports it: `gripper.command` (the applied command), `gripper.opening_m` (the
  measured distance between the inner faces of the pads: {{grip_open_text}}, {{grip_closed_m}} m closed on
  nothing) and `gripper.object_detected` (true when the fingers have stopped short of the opening the command gives on
  nothing, by more than {{obj_short_mm}} mm, and stood still within {{obj_still_mm}} mm for {{obj_hold_ms}} ms: they
  hold something, or something blocks them). There is no force reading.
* **Grip strength.** Closed with command 1 on a block {{grip_block_mm}} mm wide, the pads press on it with about
  {{grip_force_N}} N in total (the sum of the normal forces on the block), and {{grip_detect_text}}. A pinch of that
  kind {{grip_hold_text}}.

## 6. The cameras

{{n_cameras_cap}} cameras, rendered by the cell: {{camera_names}}.

{{camera_table}}

* **`infeed`.** Fixed (by the drawing) at {{infeed_pos}}, {{infeed_above}} m above the belt surface over its centre line, looking
  straight down. At the belt surface it sees y from {{infeed_y0}} to {{infeed_y1}} m and x within about
  {{infeed_xhalf}} m of the camera's axis: the set-down range and the parcels' top faces as they arrive (section 3). Its
  image's top edge faces upstream, its right edge faces the robot. `T_world_cam` = {{T_world_infeed}}.
* **`head`.** Fixed on the robot's head (by the drawing) at {{head_pos}}, looking forward and {{head_pitch}} degrees down; its optical
  axis meets the belt surface at {{head_aim}}, {{head_aim_dist}} m away: the work window. Its image's right edge faces
  upstream. `T_world_cam` = {{T_world_head}}.
* **`wrist_left`, `wrist_right`.** Palm cameras: each sits flush in its gripper base's palm between the finger
  roots, behind a small dark window, {{wrist_ahead_mm}} mm ahead of the flange along the tool axis and
  {{wrist_side_mm}} mm beside it, looking along the tool axis between the fingers (its optical axis is
  {{wrist_axis_deg}} degrees off the tool axis). Nothing of the camera stands above the palm and it has no collision
  shape of its own. The image's horizontal axis is the fingers' closing direction; {{wrist_tcp_text}}. What the views
  show: {{wrist_views_text}} `T_flange_cam` = {{T_flange_cam}} (left) and {{T_flange_cam_right}} (right); a wrist frame
  carries its own `T_world_cam`: the drawing's mount composed with the flange's pose from the measured joints at the
  frame's exposure.
* **Calibration.** `describe()["cameras"]` publishes the data sheet's optics and the installation drawing's poses:
  fx = fy = (H/2) / tan(fovy/2), cx = W/2 - 0.5, cy = H/2 - 0.5, pixel (u, v) = (column, row) with row 0 at the top;
  OpenCV axes (x right, y down, z forward); no distortion. A pixel centre's ray in the camera frame is
  ((u - cx) / fx, (v - cy) / fy, 1). The installed cameras differ from the drawing: the `infeed` camera by up to
  {{tol_infeed_mm}} mm and {{tol_infeed_deg}} in the world frame, the `head` camera by up to {{tol_head_mm}} mm
  and {{tol_head_deg}} against the robot's base, each palm camera by up to {{tol_wrist_mm}} mm and
  {{tol_wrist_deg}} in its gripper's frame; each focal length is within {{tol_focal_pct}} % of the data
  sheet's. The installed poses are to be calibrated on site; the images, depth included, are taken by the installed
  cameras.
* **Timing.** All {{n_cameras_word}} cameras expose together every {{exp_ms}} ms of simulated time ({{fps}} frames per second),
  at t = 0 and every {{exp_s}} s after it; they run during the settle before t = 0 too, so a frame is available from
  your first request (its stamp may lie before t = 0). A frame is delivered {{age_ms}} ms after its exposure. `get_camera` returns
  the newest delivered frame, so a frame is {{age_ms}}-{{age_max_ms}} ms old when you receive it. Its `stamp` is its
  exposure time, and it shows the world exactly as it was then (a global shutter: no motion blur, no rolling shutter).

* **Frame budget.** Each new frame costs one frame of the shift's pooled budget (all {{n_cameras_word}} cameras together:
  section 10). Asking again for a frame already delivered (the same camera and the same exposure) is free and returns the same
  bytes, and adding depth to a frame already delivered is free too. A frame the cell fails to render answers
  `camera_unavailable` and is not charged. Rendering takes wall time but no program time and no simulated time.
* **Images.** Colour: an 8-bit RGB PNG with sensor noise of sigma {{rgb_sigma}} levels per channel, independent per
  pixel and per frame. Depth (on request): the planar distance along the optical axis as a 16-bit PNG in millimetres,
  0 = no reading; noise of {{depth_rel_pct}} % of the distance (one sigma), {{dropout_pct}} % of the pixels without a
  reading, and no reading outside the camera's depth range ({{fixed_depth}} m for the fixed cameras,
  {{wrist_depth}} m for the wrist cameras). Inside a camera's minimum range there is no reading either: a palm
  camera's depth gives no reading wherever the surface is nearer than {{wrist_depth_min_mm}} mm, which includes the
  fingers and a face pressed by the pads or against the palm (the tool centre point is {{wrist_tip_mm}} mm
  ahead of the camera): while a hand holds or pushes a parcel's face, its palm camera's depth image of that face is
  empty, and the colour image still shows it. Depth sees the same surfaces as the colour image: {{depth_surfaces_text}}.
* **The picture.** {{look_text}}

**What each camera resolves.** The waybill (section 7) as it appears in each camera. {{resolve_note}}

{{resolve_table}}

## 7. The parcels

* **Kinds.** Rigid parcels (cartons and boxes, flat ones included) and soft parcels (poly mailers: a filled bag,
  compressible, a compliant cover that yields under a squeeze; {{mailer_body_text}}). {{kind_text}}.
* **Sizes.** Full sizes along the parcel's own axes x, y, z, drawn independently on a millimetre grid (x may be the
  shorter side): rigid x {{carton_x}} m, y {{carton_y}} m, z {{carton_z}} m; soft x {{mailer_x}} m, y {{mailer_y}} m,
  z {{mailer_z}} m.
* **Mass.** {{mass}} kg, independent of the size; the centre of mass {{com_text}}.
* **Surface friction** (against the rails, the lip, the pads' rubber and other parcels): coefficient {{friction}} for
  rigid parcels, a little higher for soft ones (above). Against the belt the belt's cover sets the grip (section 3).
* **Set-down.** The line sets each parcel down at rest, lying flat (its own z axis up), {{drop_mm}} mm above the belt,
  its centre on the belt's centre line (x = {{belt_cx}} m) between y = {{set_y0}} and {{set_y1}} m, at any yaw (every
  heading on the full circle equally likely). The belt carries it away within a fraction of a second.
  {{landing_drift_text}}
* **The shift's list and its feed.** A shift brings a fixed list of parcels, one at a time. The first is set down at
  the shift's first tick; each next one at the first tick after the one before it is settled: its centre crossed the
  scan line going downstream (section 8), it fell off the belt (section 2), or it was damaged (section 9). So one
  parcel of the list is in the station at a time (one already settled may still ride on to the belt's end). If a
  parcel is still near the set-down line, the next one waits until that place is clear.
* **The station's rated cycle.** Each parcel is due at the scan line by its set-down time plus its free ride (from its
  set-down y to the scan line at the shift's belt speed) plus {{due_margin_s}} s: the station may delay a parcel by at
  most {{due_margin_s}} s against the belt's own ride, an indexing halt included; a parcel that crosses later is late
  and not readable (section 8). A shift's list is sized at {{rated_handling_s}} s per parcel: it holds as many parcels
  as fit into the shift's duration at the belt's ride over {{rated_ride_m}} m (from the middle of the set-down range
  to the scan line) plus {{rated_handling_s}} s, that is floor(duration / ({{rated_ride_m}} m / v + {{rated_handling_s}}
  s)) parcels at belt speed v: {{list_text}}. The list's length and the due times are not on the wire (they follow
  from the belt's speed, which your program measures).
* **The waybill.** Every parcel carries one shipping label (the waybill) on one face: any face whose two sides are both
  at least {{label_min_mm}} mm, each such face equally likely (the top or the bottom when no face qualifies). The
  waybill is a paper of {{papers}} (either with equal chance where both fit the face with a margin, else the one that
  fits: on a face {{label_min_mm}}-{{small_face_mm}} mm across, the smaller paper with an edge margin of
  {{min_margin_mm}} mm), in any of the four reading directions, a little skewed, its centre offset from the face's
  centre by up to {{uv_pct}} % of the free margin along each side. A soft parcel has no side wall: a waybill on one of
  its side faces wraps over the rounded flank onto its top or bottom film. The weight printed on it is
  not the parcel's. The label is **up** when its outward normal is within {{tilt}} degrees of
  vertical up. What it looks like: {{waybill_look}}

## 8. A shift and its score

* **The shift.** From t = 0 to at most `describe()["shift"]["duration_s"]`: {{shift_s}} s for every graded shift and
  every full development shift ({{bench_shift_s}} s for a bench shift), whatever its belt speed and its parcels. It
  ends earlier, at the end of the tick in which the last parcel of its list is settled (section 7).
* **The station's duty.** The station turns every parcel: {{duty_kinds}}, whatever face carries the label. The duty
  is published in `describe()["score"]["duty"]`: the kinds (`rigid`, `soft`) and the label positions in duty
  ({{duty_positions_codes}}; the positions are {{label_position_codes}}, the world direction of the label face's normal
  as the parcel lands).
* **Its class (W or U).** Fixed as the parcel is set down, from its label's position then; nothing the robot does
  changes it:
  * **W** (work): a parcel whose label is not up (more than {{tilt}} degrees from vertical);
  * **U** (up): a parcel, rigid or soft, that arrives with its label up.
* **Its judgement.** In the tick in which its centre first crosses the scan line (y = {{scan_y}} m) going downstream,
  a parcel is **readable** when all of these hold at the end of that tick:
  1. its label is up (outward normal within {{tilt}} degrees of vertical);
  2. it rests on the belt: it touched the belt surface at least once in the last {{rest_window}} s (that tick included)
     and its lowest point is at most {{rest_clear_mm}} mm above the belt surface, it turns slower than {{max_w}} rad/s,
     and it moves at less than {{max_v}} m/s relative to the belt (horizontal velocity). Contact is judged over that
     short window, never from one tick: a soft parcel riding untouched rocks on its soft underside and loses and regains
     contact with the belt for a few ticks at a time;
  3. no part of the robot touches it;
  4. no other parcel touches it;
  5. it has not been damaged (section 9);
  6. it crossed in time: at the latest at its due time (section 7).

  Only the first downstream crossing counts; a later one changes nothing.
* **Done, lost, not done.** A parcel of the list is done when it was judged at the scan line, fell (section 2) or was
  damaged (section 9). It is lost when it fell or was damaged, at any time. A parcel that is not done when the shift
  ends (never set down, or in the station and not across, whatever ended the shift) costs nothing and earns nothing. A
  parcel that crossed readable and is lost afterwards (damaged, or knocked off before the belt's end) keeps its reading
  and still counts as lost: net 0 for a W parcel, -1 for a U parcel.
* **The shift's score.** With W and U counted over the whole list:

  ```
  credit      = (W parcels readable) - (U parcels done and not readable) - (parcels lost)
  shift score = max(0, credit) / (number of W parcels in the shift's list)
  ```

  A W parcel turned label up and crossing calmly in time earns 1; a late one earns nothing. A U parcel left alone
  costs nothing; a U parcel that crosses unreadable (knocked over, still touched, sliding, late) costs 1. A lost parcel costs 1 more: knocked off the belt or
  damaged, a W parcel costs 1 and a U parcel 2. Every W parcel of the list counts in the divisor, reached or not. A
  program that leaves the belt alone scores 0.
* **How a shift ends.** Normally at the end of the tick in which the last parcel of its list is settled, or at its
  duration. It ends early when your program calls `finish()`, when `solve` returns or raises, when the program time is
  used up, a silence runs over its limit or the wire breaks (section 10), and when the cell itself fails (then the
  shift is run again, `cell_api.md`). Whatever ends it, the parcels already done keep their judgement and every other
  parcel is not done; there is no grace period at the end. The end's reason, as the cell, the wire and the
  commissioning service name it: `shift_end` (the list is settled, or the duration elapsed), `finished` (`finish()`, or `solve` returned or raised), `program_time` (the program-time budget
  is used up), `stalled` (a silence over its limit, the start window used up, or a reply not taken in time),
  `protocol` (a request line that is not one JSON request, or too long), `crashed` (the program's process ended while
  the shift ran), and the cell's own failures `physics_fault` and `internal_error`.
* **The graded shifts.** {{shifts_graded}} shifts drawn from the line's distribution above, nothing selected or left
  out by their contents: {{graded_text}}.

## 9. Safety and damage

The cell watches every contact of the robot at every tick (the normal force of each contact, per pair of bodies, at
the tick's last physics step), as a collaborative robot's collision detection does. Three classes:

| class | what the robot touches | protective stop above |
|---|---|---|
| structure | any part of the station that is not a parcel and not the moving belt surface: the frame, the legs, the rails, the lip, the floor | {{structure_N}} N |
| arm to arm | one arm touching the other arm, or an arm touching the robot's own column, head or pedestal | {{self_N}} N |
| belt | the moving belt surface (a light touch below the limit is allowed) | {{belt_N}} N |

* **Protective stop.** A contact above its class's limit for {{press_ticks}} consecutive ticks ({{press_ms}} ms)
  stops the robot: the joint targets freeze at the measured posture (clipped to the ranges), the grippers keep their
  command, and for {{stop_lock_s}} s of simulated time every `step` that carries a joint target or a gripper command
  is refused (`refused: "protective_stop"`; the world still advances, with the frozen targets). The belt runs on and
  the stop is counted. `stop` in the state gives `active`, `until` (the time the stop ends), `count`, `locked_out` and
  `link` (the robot link of the latest stop; when the two arms touch, the link of each arm joined by "/"; never what
  it touched).
  After the stop the robot holds the frozen posture until you command it again. A contact that was pressing when the
  stop came, or that comes during its {{stop_lock_s}} s, stops the robot again only once its force has dropped below
  the limit.
* **Lock-out.** Any of those contacts above {{damage_N}} N for {{press_ticks}} consecutive ticks, or the
  {{lockout_stop}} protective stop of a shift, locks the robot out for the rest of the shift: it holds its posture and
  every command is refused (`refused: "locked_out"`, `stop.locked_out` true). The belt runs on and the parcels are
  still judged as they cross.
* **Parcel damage.** When the robot presses on one parcel with more than {{squeeze_N}} N in total (the sum of the
  normal forces of all its contacts with that parcel) for {{squeeze_ticks}} consecutive ticks ({{squeeze_ms}} ms), the
  parcel is damaged: it is lost (section 8). Nothing on the wire reports it, but the cameras show it: from that tick
  on, every frame draws the parcel with a crushed look. Its faces turn darker and greyer and dark irregular creases
  cross them (a rigid parcel's board), or its film is scuffed and creased (a soft parcel); the waybill on it is creased
  too. On a crushed parcel's faces the mean grey level drops and the share of edge pixels (Canny edges of the grey
  image) rises; {{damage_look_text}} Its shape, its pose and its depth image do not
  change, and no text or mark is printed on it. An undamaged parcel never shows this look.

## 10. Time and budgets

The world advances only inside `step`: nothing moves and no time passes between your requests. Per shift (all in
`describe()["budgets"]`):

* **Program time.** {{pt_rule_cap}} ({{pt_shift}} s in a graded shift): the shift's wall clock minus the cell's own time inside `step` (simulating) and `get_camera`
  (rendering). It measures your program's own computing and its other requests. Its CPU time is capped at
  {{cpu_factor}} x the program-time budget (all threads together). `program_time_left` in the state counts it down;
  when it runs out, the shift ends.
* **Silence.** At most {{silence_s}} s of program time from a reply to your next request; a longer silence ends the
  shift.
* **Frames.** {{frames_per_s}} x the shift's duration in seconds, pooled over the {{n_cameras_word}} cameras ({{frames_shift}}
  frames in a graded shift). When they are used up, `get_camera` answers `no_frames_left` and the
  shift goes on.
* **Ticks per call.** One `step` runs at most {{max_ticks}} ticks ({{max_ticks_s}} s); a larger `n` is cut to that
  (the reply's `ticks` says how many ran).
* **After the end.** Once the shift is over, `step` answers with `ticks: 0` and nothing moves; the program may read its
  state and must return. The cell answers at most {{after_steps}} more steps, {{after_requests}} more requests and
  {{after_s}} s more, then the connection ends.
* **Start.** From the program's start to its first request at most {{start_s}} s (the interpreter's start and your
  imports; the start window), inside the program-time budget; after the first request the silence limit applies.
* **Replies and requests.** Your program must take in each reply within the silence limit (a camera reply with depth
  is up to about {{reply_mb}} MB); one request line holds at most {{line_mib}} MiB and must be one JSON request (the
  client writes them so). A reply not taken in, a malformed or longer request line, or the program's process ending
  while the shift runs ends the shift.

## 11. The client library

`~/given/lib/parcelcell/client.py` is the one part of the cell your program imports: the proxy (`Cell`), the frame
decoders, the program launcher the grader uses and the commissioning connection (section 12). The cell itself (the
physics, the robot's servos, the cameras, the scoring) runs in the grader and in the commissioning service, never in
your account. The launcher moves the wire off the standard streams before your program starts: `print()` and
anything a library writes to the standard output go to the standard error, never into the wire.

## 12. Commissioning on development shifts

A robot program is not handed over before it has run on the real line: yours can be commissioned before it runs the
shifts it is graded on. The commissioning service in your sandbox runs the station's cell on {{dev_shifts}}
development shifts drawn from the line's distribution above: {{bench_shifts}} bench shifts with {{bench_n}} parcels
each ({{bench_contents}}) and {{full_shifts}} full shifts ({{full_contents}}). None of them is a graded shift.

* **What it is.** The cell of the graded shifts (the robot with its servos and limits, the grippers, the safety watch,
  the cameras' frames, the belt, the budgets) on the wire of `cell_api.md`, plus `reset(site)`, which starts
  development shift `site` (1 to {{dev_shifts}}). A development shift is the same each time you start it. The
  `reset` answers once the shift's cell is built: {{shift_start_text}}.
* **What it is not.** A simulator you can read or query: it answers no ground truth (no parcel pose, size, kind or label
  face), no score and no debrief. How a shift went, you learn from what your program saw, logged and saved.
* **The window.** {{comm_shifts}} commissioning shifts in your session (a shift counts from its `reset`), at most
  {{comm_open}} at a time (while that many are open, a new connection's first request is answered `busy` and the
  connection is closed; no shift is used), each under the graded shift's budgets. When the service closes a shift for
  any reason but `finished` (section 8), it sends `{"id": null, "event": "shift_over", "reason": ...}` and then closes
  the connection. The count is in
  `/run/parcelcell/commission_status.json`. It does not grow.
* **Time.** A full development shift lasts as long as a graded one ({{shift_s}} s of simulated
  time, more in wall time with your program's own time and the frames).
* **How.** `parcelcell.client.connect_commissioning(site)` returns a `Cell` whose shift has begun (one connection, one
  shift): a script of your own that moves the arms, grabs frames and records what it sees is a commissioning
  engineer's teach-in. `/opt/venv/bin/python3 ~/commission/run_shift.py --site N` runs your whole program in a shift
  the way the grader runs it (a read-only copy of the folder, the grader's environment, launcher, CPU cap and
  no-network filter) and prints how the shift ended, the frames used, the protective stops and the refused commands
  (`--keep DIR` keeps what your program wrote in its HOME and TMPDIR, `--transcript FILE` the wire).
