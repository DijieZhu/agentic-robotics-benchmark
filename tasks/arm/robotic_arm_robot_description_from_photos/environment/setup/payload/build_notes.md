# Build notes (the builder's own notes; no dimension was recorded)

- The arm was printed from the published part files (the ten parts in `parts/`) with these changes: the two long beams
  (upper arm P05, forearm P06) were lengthened or shortened in their middle sections, the joint ends unchanged; the base
  shell and the shoulder parts may have been printed taller; the shoulder's offset from the yaw axis may have been changed;
  the wrist body may be longer or shorter; both fingers (fixed P09 and moving P10) were lengthened or shortened by the same
  amount. Printed spacer blocks may sit at the far end of the elbow (S3) and wrist-pitch (S4) servo pockets, which moves
  that servo, its horn axis and everything after it. Lightening windows and cable slots may have been cut into some parts.
- Material and slicer settings as for every build of the lab (the slicer's mass estimates of the parts AS PUBLISHED are in
  `masses.json`); the real infill of the printed parts (published or modified) is not known: the slicer figures are
  estimates, and the weighed total is the reference.
- The six servos (BUS-SERVO-12V-A) were fitted at an arbitrary middle position of their range: the design angle at which a
  servo reads bus angle 0 was never noted. Each servo's drive-mode bit is in the EEPROM dump. The EEPROM dump
  (`servo_config.json`) was read off the bus after the build; the firmware angle limits were set by hand to keep the arm
  clear of itself.
- Thin rectangular printed cover plates are clipped over the outer bracket faces at both ends of the shoulder, elbow and
  wrist-pitch axes (they hide the servo horns; not part of the model).
