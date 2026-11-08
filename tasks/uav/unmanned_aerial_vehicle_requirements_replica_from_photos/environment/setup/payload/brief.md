# Customer brief / revision E

B-01. Supply eight replicas of U-17 in its final configuration after the 25 September rebuild and subsequent survey.
Retain that configuration's fabricated guarded core, mounting positions and individual motor-axis orientations, battery cage and
ballast-pod position. Retain the PX4 / ROS 2 split, sensors and compute. Purchase only the seven BOM lines in B-03.

B-02. Fly with the replacement ballast pod shown on the scale in PH-12. The display is grams, tare zero.
Use the survey register and build history to establish its final installed position and box extents;
its centre of mass is its box centre. The photographs span more than one build stage.
The pod and scanner box edges are parallel to the vehicle axes. Their external outlines must be measured.
Every replica carries this mass throughout the duty. The old shipping note's 250 g is obsolete.

B-03. Four independent motor lines, stations FL/FR/RL/RR, one shared propeller line (4 units, two of each handedness),
one 4-in-1 ESC line, one 6S battery line. No stacking batteries, mixed propeller types or modification of the core.
Prices and delivery lot are fixed by Q-01. All quoted options may be evaluated, even if one cannot fit this unit.

B-04. Coordinates: x forward (toward scanner), y left, z up along the retained core's vertical direction.
The centres of the four upper circular motor-case faces are coplanar. Their midpoint is the origin and their
centre plane is z=0; the tilted circular faces themselves are not parallel. Exclude propeller hubs and nuts.
FL is x positive/y positive; FR positive/negative; RL negative/positive; RR negative/negative.
Spin is viewed toward each motor from its positive shaft axis. Reaction signs s in this order are -1,+1,+1,-1.
Install CCW propellers at FL/RR and CW at FR/RL. Full-row centre spacing is twice the corresponding half-span.
For station r=(x,y,0), u=r/|r| points radially outward and t=ez cross u. Before tilting, motor and
square top-land axes align with body x/y/z. Apply R=Rot(u,s*theta)*Rot(t,beta), with the right-hand
rotation applied first. Theta is tangential cant and beta is radial tilt; both use degrees in the input.
The thrust axis is n=cos(beta)*(cos(theta)*ez-s*sin(theta)*t)+sin(beta)*u. Recover both angles
independently at every final station; no equality or common tilt direction is prescribed.

B-05. Use the conservative projected swept-disc corridor: propeller diameter plus twice [2 mm tip clearance
+ max_i(h*sqrt(n_ix^2+n_iy^2))] must fit the smallest shaft-centred inscribed circle in any guard. Here h is the
qualified propeller-plane offset along its shaft from the upper motor-face centre (DS-02). The offset shifts
its projected centre; the full diameter bounds the tilted ellipse. Guard apertures can be offset from their
shaft centres; use the limiting shaft-to-inner-edge distance, not the guard outer diameter. Battery boxes must fit the cage's internal x/y/z dimensions without rotation.
Motors must share this unit's 16 mm mounting-hole pitch. Frame or cage rework is disallowed.

B-06. Use the delivered lot B. Lot A is retained for comparison only. For each motor SKU, retained thrust multiplier
is the minimum of its three normalized samples minus 0.006; retained electrical-power multiplier is the maximum
plus 0.008. The retained minimum-control-thrust multiplier is the maximum of its three normalized
low-output samples plus 0.010. This is a conservative LOWER endpoint (QL-1 on DS-07), independent of
the upper-thrust and power reductions. Battery capacity factor is minimum measured retained fraction minus 0.005; resistance factor is maximum
measured multiple plus 0.02. ESC limit factor is minimum measured retained fraction minus 0.01. The vendor-page
erratum takes precedence over the preliminary page. Apply lot factors once, after the erratum.

B-07. Evaluate a 60 s hover and all 64 three-second envelope states. For every six independent signs
(a,b,c,d,e,f) in {-1,+1}^6, use az=3.1 m/s^2, ax=5.4 m/s^2,
omega=1.20*(a*2.2,b*2.7,c*1.8) rad/s and alpha=1.20*(d*8.2,e*10.7,f*1.2) rad/s^2.
Each envelope state includes gust G=(-d*1.5,e*1,0) N acting at the payload box centre. Rate signs and
acceleration signs vary independently; gust signs follow acceleration. This is a bench qualification
state grid, not a time-integrated flight trajectory. Hover has zero acceleration/rate/gust.
No takeoff, cruise or reserve time is silently added. Auxiliary battery-side power is 38 W throughout;
g=9.80665 m/s^2. Energy reserve is 20% of retained battery energy.

B-08. Use the vendor bench fit at 22.2 V, not throttle percentages. Both retained upper and lower thrust endpoints scale with V^2.
The lower endpoint is minimum_thrust_ref_n * minimum_thrust_factor * prop thrust_factor * (V/22.2)^2;
use each state's actual solved loaded voltage. The qualified fixed control-output window must contain every
allocated thrust. Boosting collective or reducing requested moments is not an allowed repair of this contract.
Per-motor electrical power is k*T^(3/2), including the propeller and delivered-lot power factors; T is newtons,
k has units W/N^(3/2). Divide by ESC efficiency to obtain battery-side motor power; add B-07 auxiliary power.
At end-of-mission test open-circuit voltage, solve V=Voc-R*P/V using the HIGH-voltage root. No real root is failure.
Total battery current is P/V; each ESC-channel current excludes auxiliary current.

B-09. Contractual rigid-body reduction: reconstruct the final dry-core mass using BL-01, then use it as
a uniform box centred at (0,0,-18 mm). Use the scanner
as a uniform box with its photographed extents and centre (mass from DS-08), the payload as the B-02 box,
and the chosen battery/ESC as
their datasheet boxes. The ESC centre is (0,0,-22 mm). Motors plus propellers are point masses at the four shafts.
Compute the combined CoM and full inertia tensor ABOUT THAT CoM, including parallel-axis terms.
For each state, prescribe resultant force magnitude F=sqrt((m*(g+az))^2+(m*ax-Gx)^2+Gy^2) and angular demand
M=I*alpha + omega cross (I*omega) - (payload_centre-CoM) cross G. Use the B-04 thrust axes n_i. The moment
column for unit thrust at station i is (r_i-CoM) cross n_i + s_i*propeller_reaction_arm*n_i. The latter is
propeller reaction torque along the tilted shaft. Form A with first row n_i,z and the three moment rows.
Let a=A^-1*[0,M] and b=A^-1*[1,0,0,0]. With N having rows n_i, solve |N^T*(a+Fz*b)|^2=F^2 for Fz using
the plus-square-root quadratic branch, and set individual thrusts to a+Fz*b. A singular matrix, negative
discriminant, nonpositive Fz or negative individual thrust is failure. Shifting a thrust application point
along its own shaft to the propeller plane adds no moment. All vectors are expressed in the vehicle axes.
This four-input bench reduction constrains resultant magnitude and three moments, not an arbitrary six-axis
wrench. Aerodynamic drag, detailed CAD density and closed-loop flight stability are outside this contract.

B-10. Every allocated thrust must lie between its retained voltage-dependent lower and upper endpoints, inclusive. Check motor
current and ESC peak channel, time-weighted RMS channel and sum-of-channels board limits, applying the ESC lot
factor to all three ESC limits. Also check battery peak current, loaded voltage against both cutoff limits,
and sum(P*seconds)/3600 against 80% of retained energy. All 65 states must pass simultaneously.

B-11. Select the least-cost passing BOM. Economy uses the cheapest passing design's cost J*. For each passing
repair of your BOM, add 12 USD for each changed motor station, shared propeller line, ESC line or battery line.
Effective cost is the minimum of those repaired costs. Economy decreases linearly from 1 at J* to 0 at J*+12 USD.
For an already passing BOM, retaining it is an available zero-change repair. Fixed core costs cancel out.

B-12. Submit the three files using the schema and trace every reported quantity to clause/photo/page IDs.
Do not rely on catalogue prices or nominal upstream dimensions as measurements of this fabricated unit.
