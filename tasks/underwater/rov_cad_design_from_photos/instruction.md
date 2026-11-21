# Recover the CAD of an inspection-class ROV

The lab's inspection-class ROV (a remotely operated underwater vehicle with eight thrusters) has lost its editable CAD:
the models of all the parts that were made for it are gone, and only the files of its purchased modules remain. A
colleague photographed the complete vehicle, took it apart stage by stage while photographing every stage, and
photographed every made part on a cutting mat. The vehicle is in pieces now and cannot be measured again. Re-create its
mechanical design in FreeCAD from those photographs and the purchased modules' documents.

The vehicle has 31 made parts in 16 shapes (plastic plate, moulded polycarbonate, buoyancy foam, aluminium, stainless
steel sheet) around 20 purchased modules: an electronics enclosure, a battery enclosure, eight ducted thrusters (four
with a clockwise and four with a counter-clockwise propeller), four LED lights, a gripper, a Doppler velocity log and
four ballast weights. Infer every made part's shape, dimensions and installed position from the evidence.

## How to work

Use the FreeCAD MCP server to inspect and edit the running FreeCAD session. Its `execute_code` tool runs Python inside
that session. Other tools remain available for image measurement, calculations and scripts. The Python environment is
`/opt/venv/bin/python3`; the native headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`.

Save early and periodically. You have six hours. A single model response is limited to 64k output tokens; do not wait
until the end to create the deliverable.

Three facts about this workstation (the times were measured when the task was prepared):

- `execute_code` gives a call 90 s unless the call passes its `timeout` argument (at most 1800 s). Importing the
  electronics enclosure's STEP file (266 MB) took about 150 s in the FreeCAD window.
- A screenshot of the session took about 14 s with all 20 purchased modules shown and about 8 s with the electronics
  enclosure hidden. An object can be hidden while it is not needed in view.
- The display tessellation of imported shapes can be made coarser (the view properties `Deviation` and
  `AngularDeflection`); with that, a screenshot of all modules took about 9 s. It changes the display only, not the
  geometry.

## Given files (in `/home/user/`)

`photos/photo_001.jpg` to `photos/photo_156.jpg`, 156 photographs (64 teardown, 17 extra views, 35 close-ups, 8 of the
complete vehicle, 32 of the stages).

- Teardown photos show every made part next to its paper tag (P01 to P31), first one side up, then turned over. In the
  first of the two, a part lies the way up it sits on the vehicle wherever it can rest so: a part that lies flat on the
  vehicle shows the side that faces up there. A part that cannot rest so (it stands on edge on the vehicle) lies on one
  of its faces, and that says nothing about how it is installed. Parts with the same shape were photographed together,
  each with its own tag.
- Extra views show what a flat lay cannot: the inside of hollow parts, profiles, and the edges of plates, with a rule
  flat on the mat along the edge.
- Close-ups show groups of holes and other small details of the tagged parts, with a rule in view.
- The photos of the complete vehicle show it standing on the bench, from all around and from above, and turned over on
  wooden blocks.
- The stage photos show the vehicle as it was taken apart. Each caption says what had been removed so far: made parts by
  tag, purchased modules by name and number. The photos of the complete vehicle and of the stages show no tags.

The camera is perspective and uncalibrated, and file names do not encode its pose.

| photos | what they show | scale references |
|---|---|---|
| `photo_001`-`002` | P01; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_003`-`004` | P01; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_005`-`006` | P02; first side up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_007`-`008` | P02; turned over | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_009`-`010` | P03, P04; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_011`-`012` | P03, P04; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_013`-`014` | P05, P06; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_015`-`016` | P05, P06; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_017`-`018` | P07, P08; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_019`-`020` | P07, P08; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_021`-`022` | P09-P12; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_023`-`024` | P09-P12; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_025`-`026` | P13-P16; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_027`-`028` | P13-P16; turned over | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_029`-`030` | P17, P18; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule across P17, P18 |
| `photo_031`-`032` | P17, P18; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule across P17, P18 |
| `photo_033`-`034` | P19, P20; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_035`-`036` | P19, P20; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_037`-`038` | P21; first side up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_039`-`040` | P21; turned over | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_041`-`042` | P22; first side up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_043`-`044` | P22; turned over | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_045`-`046` | P23, P24; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_047`-`048` | P23, P24; turned over | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_049`-`050` | P25; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_051`-`052` | P25; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_053`-`054` | P26, P27; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_055`-`056` | P26, P27; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_057`-`058` | P28, P29; first side up | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_059`-`060` | P28, P29; turned over | cutting mat with a 10 mm grid under the parts, checker card (12 mm squares) flat on the mat, 300 mm steel rule flat on the mat |
| `photo_061`-`062` | P30, P31; first side up | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_063`-`064` | P30, P31; turned over | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_065`-`066` | P13 | cutting mat with a 10 mm grid under the parts |
| `photo_067`-`068` | P23 | cutting mat with a 10 mm grid under the parts |
| `photo_069` | P09 | cutting mat with a 10 mm grid under the parts |
| `photo_070` | P28 | cutting mat with a 10 mm grid under the parts |
| `photo_071` | P26 | cutting mat with a 10 mm grid under the parts |
| `photo_072` | P07 | cutting mat with a 10 mm grid under the parts |
| `photo_073` | P05 | cutting mat with a 10 mm grid under the parts |
| `photo_074` | P21 | cutting mat with a 10 mm grid under the parts |
| `photo_075` | P22 | cutting mat with a 10 mm grid under the parts |
| `photo_076`-`077` | P25 | cutting mat with a 10 mm grid under the parts |
| `photo_078` | P01 | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_079` | P17 | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_080` | P19 | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_081` | P30 | cutting mat with a 10 mm grid under the parts, 300 mm steel rule flat on the mat |
| `photo_082`-`083` | The complete vehicle on the bench | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, at about the same distance from the camera as the near face of the vehicle, 600 mm steel rule flat on the bench |
| `photo_084` | The complete vehicle on the bench | 300 mm steel rule flat on the bench |
| `photo_085` | The complete vehicle on the bench | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_086`-`087` | The complete vehicle on the bench | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_088` | The complete vehicle on the bench, seen from straight above | checker card (12 mm squares) flat on the bench, 300 mm steel rule flat on the bench |
| `photo_089` | The complete vehicle turned over, resting on two stacks of two wooden blocks | 150 mm steel rule flat on P01 |
| `photo_090` | After stage 1. Removed so far: 4 x LED light, 1500 lm. Seen from straight above | checker card (12 mm squares) flat on the bench, 300 mm steel rule flat on the bench |
| `photo_091` | After stage 1. Removed so far: 4 x LED light, 1500 lm | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_092` | After stage 1. Removed so far: 4 x LED light, 1500 lm | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, at about the same distance from the camera as the near face of the vehicle, 600 mm steel rule flat on the bench |
| `photo_093` | After stage 1. Removed so far: 4 x LED light, 1500 lm | 300 mm steel rule flat on the bench |
| `photo_094` | After stage 2. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29; 4 x LED light, 1500 lm; 1 x scanning sonar transducer. Seen from straight above | checker card (12 mm squares) flat on the bench, 300 mm steel rule flat on the bench |
| `photo_095` | After stage 2. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29; 4 x LED light, 1500 lm; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_096` | After stage 2. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29; 4 x LED light, 1500 lm; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_097` | After stage 2. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29; 4 x LED light, 1500 lm; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_098` | After stage 3. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_099` | After stage 3. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_100` | After stage 3. Removed so far: P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_101` | After stage 4. Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_102` | After stage 4. Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, at about the same distance from the camera as the near face of the vehicle, 600 mm steel rule flat on the bench |
| `photo_103` | After stage 4. Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P28, P29, P30, P31; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_104` | After stage 5 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | 300 mm steel rule flat on the bench |
| `photo_105` | After stage 5 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | 600 mm steel rule flat on the bench |
| `photo_106` | After stage 5 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) flat on the bench, 600 mm steel rule flat on the bench |
| `photo_107` | After stage 6 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) flat on the bench, 600 mm steel rule flat on the bench |
| `photo_108` | After stage 6 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | 300 mm steel rule flat on the bench |
| `photo_109` | After stage 6 (turned over, resting on two wooden blocks). Removed so far: P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) flat on the bench, 600 mm steel rule flat on the bench |
| `photo_110`-`111` | After stage 7. Removed so far: one of P05-P06, P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_112` | After stage 7. Removed so far: one of P05-P06, P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 2 x ducted thruster, counter-clockwise propeller; 2 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, at about the same distance from the camera as the near face of the vehicle, 600 mm steel rule flat on the bench |
| `photo_113` | After stage 8. Removed so far: one of P05-P06, P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_114` | After stage 8. Removed so far: one of P05-P06, P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_115` | After stage 8. Removed so far: one of P05-P06, P07, P08, P09-P12, P13-P16, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_116`-`117` | After stage 9. Removed so far: P02, one of P05-P06, P07, P08, P09-P12, P13-P16, P21, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 1 x single-axis gripper; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_118` | After stage 9. Removed so far: P02, one of P05-P06, P07, P08, P09-P12, P13-P16, P21, P22, P23, P24, P25, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 1 x single-axis gripper; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, at about the same distance from the camera as the near face of the vehicle, 600 mm steel rule flat on the bench |
| `photo_119`-`120` | After stage 10. Removed so far: P02, P03, P04, P05, P06, P07, P08, P09-P12, P13-P16, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 1 x single-axis gripper; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 600 mm steel rule flat on the bench |
| `photo_121` | After stage 10. Removed so far: P02, P03, P04, P05, P06, P07, P08, P09-P12, P13-P16, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31; 4 x ballast weight, stainless steel; 1 x battery enclosure, 3-inch acrylic tube, with battery pack; 1 x Doppler velocity log, four beams; 1 x electronics enclosure, 4-inch acrylic tube with dome; 1 x single-axis gripper; 4 x LED light, 1500 lm; 4 x ducted thruster, counter-clockwise propeller; 4 x ducted thruster, clockwise propeller; 1 x scanning sonar transducer | checker card (12 mm squares) standing on the bench, leaning back on a wooden block, 300 mm steel rule flat on the bench |
| `photo_122`-`123` | P17, P18; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_124` | P17, P18; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_125` | P17, P18; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_126` | P01; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_127` | P01; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_128` | P01; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_129` | P01; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_130`-`131` | P19, P20; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_132` | P19, P20; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_133`-`134` | P03, P04; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_135` | P03, P04; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_136`-`137` | P30, P31; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_138` | P26, P27; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_139` | P07, P08; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_140` | P07, P08; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_141` | P05, P06; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_142` | P13-P16; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_143`-`144` | P01; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_145`-`146` | P23, P24; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_147` | P21; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_148` | P22; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_149` | P22; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_150` | P25; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_151` | P25; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_152` | P19, P20; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_153` | P21; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_154` | P03, P04; turned over; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_155` | P22; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |
| `photo_156` | P01; first side up; close-up | cutting mat with a 10 mm grid under the parts, 150 mm steel rule flat on the mat |

Scale references in the photographs: the cutting mats have a 10 mm grid, every fifth line bold, the bold lines numbered
in centimetres along two margins; the checker card has ten by seven squares of 12 mm inside a 12 mm white border; the
steel rules (150 mm, 300 mm and 600 mm) are graduated in millimetres and numbered every centimetre; a paper tag is 34 x
18 mm.

`purchased/<part number>/` for each purchased module: `<part number>.step` (the module's geometry in its own frame; most
modules have several solids in one file), `<part number>_datasheet.md` (mass, frame, envelope and the module's own
interfaces) and `<part number>_drawing.png`.

| part number | module | in the vehicle | solids in the file |
|---|---|---:|---:|
| `EV-411` | electronics enclosure, 4-inch acrylic tube with dome | 1 | 422 |
| `BV-311` | battery enclosure, 3-inch acrylic tube, with battery pack | 1 | 40 |
| `TD-76-CW` | ducted thruster, clockwise propeller | 4 | 14 |
| `TD-76-CCW` | ducted thruster, counter-clockwise propeller | 4 | 14 |
| `LD-15` | LED light, 1500 lm | 4 | 23 |
| `GR-62` | single-axis gripper | 1 | 24 |
| `DV-4B` | Doppler velocity log, four beams | 1 | 7 |
| `BW-19` | ballast weight, stainless steel | 4 | 1 |

`parts_list.md`: the made parts by tag, one line per shape, with pieces, material and density: HDPE plate, black, 0.96
g/cm³ (P01, P03, P04, P17-P20, P30, P31); aluminium alloy, black anodised, 2.70 g/cm³ (P02, P05-P08, P21, P22, P26,
P27); rigid polyurethane buoyancy foam, 0.305 g/cm³ (P09-P12, P28, P29); polycarbonate, moulded, 1.20 g/cm³ (P13-P16,
P23, P24); stainless steel sheet, 8.00 g/cm³ (P25). The masses of the purchased modules are in their datasheets. No
drawing, dimension or hole table of a made part is given, and nothing states which tagged part sits where.

## Deliverable

Save `/home/user/submission/rov.FCStd`.

**P01 first: the whole vehicle is aligned on it.** P01 must be exactly one PartDesign Body that holds exactly one valid,
closed solid, and it must be in the Assembly exactly once. If P01 is missing, holds several solids or no valid closed
one, or is there more than once, the whole document scores 0 (the first case under "How it is graded", with the way to
check it).

The document holds:

1. One PartDesign Body for each tag P01 to P31. The **Label** of the Body itself is exactly the paper tag (the label of
   a link to the Body does not count), and the Body holds exactly one valid, closed solid. Use each tag once: one Body
   per tag, in the Assembly once, not reused through links. Parts photographed together have the same shape; give each
   its own Body and tag. Which of several identical parts sat where is not graded: the parts of one shape are matched to
   that shape's places. Parts of different shapes are different parts: a part is compared only with the original parts
   of its own tag's shape.
2. Every purchased module of the table above, imported unchanged from its STEP file and placed where it is installed: 20
   modules, 701 solids in all. Move a module only as a whole. Keep every solid of a file, in its place relative to the
   others: do not fuse, simplify, scale, repair or re-model the solids, and do not delete a single one (a module's score
   is multiplied by the share of its solids that are present; the small parts inside the electronics enclosure count
   like the large ones). How a file is imported is up to you: one object that holds all its solids, FreeCAD's own STEP
   import, or one imported object linked once per module are read the same. The two thruster part numbers differ in the
   hand of the propeller; which hand sits where is part of the design.
3. One native FreeCAD Assembly that contains all of these, with P01 grounded and every other tagged part and every
   purchased module connected to P01 through Fixed joints, directly or through other parts and modules. The vehicle is
   one rigid unit: nothing in it needs another kind of joint. Keep exactly one Assembly that is not nested in another
   one in the document.
4. The Assembly saved in the assembled state of the photographs of the complete vehicle. The origin and the overall
   orientation are free; units are millimetres. Only what is inside the Assembly is graded: leave trial imports and
   construction objects outside the Assembly, or delete them. A complete document holds 732 solids inside the Assembly
   (31 made parts and the 701 solids of the purchased modules); do not include a bench, a mat, a photo scene,
   alternative versions or duplicate solids.

Model every made part as the photographs show it. Do not adjust a made part to a purchased module's file where the two
do not seem to fit: overlaps that the original design has itself are not penalised.

Screws and nuts, the eight small angle brackets on the outside of the frame, the scanning sonar transducer (the round
sonar head; no file of it is given) and cables are outside the modelling scope: do not model them (anything extra counts
as an extra solid, see below). The screws inside a purchased module's file are part of that module and stay in it. The
photographed vehicle has no cables fitted. Only the saved document is evaluated.

## How it is graded

The verifier runs in a separate sandbox after your session. It opens the saved document with FreeCAD (nothing in it is
executed), exports every solid in world coordinates and reads the joints. It grades what is inside the Assembly: its
members, the members of App::Part containers and folders inside it, and links. Objects outside the Assembly are ignored.
Of several Assemblies that are not nested in another one, only the one that was created first is read. A document
without an Assembly is graded from its visible solids, with no credit for joints. The saved placements are graded as
they are; the Assembly is not solved again.

The whole document scores 0 in seven cases, and in no other:

1. **P01 is missing or cannot serve the alignment**: nothing labelled `P01` is inside the Assembly, `P01` is there more
   than once (a second object with that label, or a second link to its Body at another placement), it holds several
   solids, or it holds no valid closed solid. Check it before you finish, in the session: the Body labelled `P01` has
   `len(Shape.Solids) == 1`, `Shape.isValid()` and `Shape.Solids[0].isClosed()` are true, and the Assembly contains that
   Body, or one link to it, exactly once.
2. **There is no document**: `/home/user/submission/rov.FCStd` does not exist and `/home/user/submission` does not hold
   exactly one `.FCStd` file under another name either (a single document under another name is read; of two, none is);
   or `/home/user/submission` is a symbolic link or leads outside `/home/user`.
3. **The document cannot be read**: FreeCAD cannot open it; what is read of it (the content of the Assembly, or the
   visible objects of a document without one) has more than 4000 shape objects (a Body, a shape feature or a link to one
   counts as one) or more than 6000 solids; or opening and exporting it takes longer than 1500 s (25 minutes).
4. **More than 4000 solids are inside the Assembly.** A complete document holds 732.
5. **It holds no usable solid**: nothing inside the Assembly is a closed solid that can be read.
6. **The submission folder is still written to after your session has ended**, by a process that is still running, for
   instance.
7. **The submission folder holds an unchanged copy of a public CAD or mesh file of this vehicle.**

Everything else is local and reported. A tag that is not on a PartDesign Body of exactly one valid closed solid gets no
credit, and neither does a tag found on more than one object (a Body reused through links at different placements counts
as its tag used more than once); the solids concerned count as extra solids. Check every tag the way P01 is checked: the
object that carries the label has `TypeId == 'PartDesign::Body'` and `len(Shape.Solids) == 1`, and `Shape.isValid()` and
`Shape.Solids[0].isClosed()` are true. A shape built as another kind of object (a `Part::Feature` written by a script,
for instance) gets no credit under its tag, and when no tagged part is credited the shape floor leaves nothing of the
other criteria either. Invalid solids of your own, open shells and loose faces are not used. Extra solids are all solids
that are neither a credited tagged part nor part of a recognised purchased module: they count in the mass (a stray copy
of a purchased solid at its module's density, anything else at 0.96 g/cm³) and are obstacles in the fit and flow
clearance criteria.

Your document is aligned with the original rigidly, without scaling or mirroring, on P01 alone. When P01 by itself fits
in several ways, the way under which the other tagged parts and the modules land nearest their places is taken. After
that nothing is resized, turned or moved into place. Purchased modules are recognised by their full shape, never by a
label: each solid must be the supplied geometry (volume within 1 %, then a rigid registration), and the solids of one
file must keep their places relative to each other within 1 mm; a solid that does not is an extra solid and leaves its
module incomplete. Some solids of the purchased modules do not pass FreeCAD's validity check (`isValid()`), as supplied
or once they are placed; unchanged, they count all the same. The modules of one part number are matched to its installed
positions one-to-one, and so are the tagged parts of one shape (one line of `parts_list.md`): it does not matter which
of them carries which tag.

| Criterion | Weight | What is measured |
|---|---:|---|
| Shape | 0.30 | Each of the 31 made parts where it is installed, against the original part there. Surface distance, both ways, 90th percentile: full credit up to 0.35 mm, none from 5 mm (60 % of the part's score). Volume overlap (intersection over union): none up to 0.25, full from 0.95 (40 %). A missing part scores 0. Mean over the 31 parts. |
| Functional features | 0.25 | The round features of the made parts: holes and bores whose wall closes (every cylindrical step covers at least 300° about its axis) and half-open round walls (a cylindrical step that covers at least 175°: a slot end, a notch, a half-round seat), with a radius of 0.5 mm or more. Every coaxial cylindrical step of such a feature is scored: each step of a stepped hole, and in a half-open wall also a step that covers less than that. Each step is matched one-to-one to a concave cylindrical face of your part: axis offset (full up to 0.25 mm, none from 1.5 mm) × radius error (full up to 0.15 mm, none from 0.6 mm) × axis angle (full up to 2°, none from 10°). A step's score is then divided by one plus the number of other concave cylindrical faces of your part (the pieces of one cylinder count as one) that lie inside all three of those none-from limits of the step and that the original part does not have: a hole that the vehicle does not have lowers the score of the feature beside it, and what the original has, scored here or not, never does. Conical faces (countersinks, chamfers, draft) are not scored here: they belong to the shape. Mean per feature, then per part, then over the parts that have such features. |
| Installation | 0.20 | 80 %: each of the 20 purchased modules scores the position of its centroid (full up to 0.6 mm, none from 12 mm) × (0.7 + 0.3 × orientation, full up to 0.5°, none from 12°) × the share of its solids present. A module that is not found scores 0. A pose that is a symmetric equivalent of the original counts as exact. A thruster's hand is its part number: a thruster with the other propeller is another module and is matched only to the places of its own part number. All modules weigh the same. 20 %: the made parts (half) and the purchased modules (half) that are connected to the grounded part through unsuppressed Fixed joints. |
| Mass properties | 0.10 | Total mass: full up to 2 %, none from 25 % (30 %). Centre of mass: full up to 1 mm, none from 15 mm (35 %). Principal moments of inertia, largest relative error: full up to 4 %, none from 50 % (35 %). Made parts count at their material's density, purchased modules at their datasheet mass, extra solids as stated above. Screws, the angle brackets, the sonar transducer and cables are on neither side. |
| Component fit | 0.10 | 60 %: each purchased module scores its contact with what it rests on in the original (full up to 0.1 mm farther apart than there, none from 2 mm farther) × its penetration into made parts and extra solids (full up to 0.05 %, none from 5 % of its own volume). 40 %: each made part's penetration into the other made parts and extra solids, on the same band of its own volume. What the original design has itself of overlap between the same two items is exempt, and 0.5 mm³ more per pair. A missing part or module scores 0. |
| Thruster flow clearance | 0.05 | Each thruster's flow body: a cylinder on the propeller axis with the propeller's tip radius (38.1 mm), from 10 mm ahead of the thruster's nose-cone end to 25 mm beyond its duct exit. It moves with the thruster's pose in your document and is tested against everything else in the vehicle: made parts, the other modules, extra solids and the other thrusters' flow bodies. Overlap beyond what the original has there, and 0.5 mm³ more: full up to 0.05 mm³, none from 200 mm³. A missing thruster scores 0. Mean over the 8 thrusters. |

**Shape floor:** features, installation, mass properties, fit and flow clearance are multiplied by a factor that rises
linearly from 0 at a mean shape score of 0.15 to 1 at 0.55. Purchased modules placed around crude or missing made parts
therefore collect nothing beyond the shape credit. design_score = 0.30 shape + 0.25 features + 0.20 installation + 0.10
mass properties + 0.10 fit + 0.05 flow clearance; the reward is the design score.

Whether you work through the MCP server or the shell does not affect your score. Geometry and assembly correctness do.
