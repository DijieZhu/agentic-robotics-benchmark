# Paper: the free-flyer's propulsion nozzle mechanism (May 2020)

Re-typeset from the document's text layer, page by page; the text of each page is kept as it is laid out; pictures and drawings are not reproduced.

## Page 1

```text
                                 Cabinflyer Free-Flyer Nozzle Mechanism
                                               the paper's author

Abstract

This paper describes the development and design of the Cabinflyer Free-Flyer propulsion nozzle
assemblies. As will be illustrated in this paper, the Free-Flyer nozzles are thrust control devices used to
propel the Cabinflyer Free-Flyer robot autonomously inside the orbital station (STATION). The
development process, design evolution, and prototyping methods, are described. Key design features
are discussed in greater detail to highlight how a seemingly simple design can present surprisingly large
challenges. Several lessons learned are given.

Introduction

The Cabinflyer project provides free-flying robots (referred to as “Units”), charging station, software, and
ground control operations to the STATION for crew support and guest payload research. The Free-Flyers are
31.8 cm (12.5 inch) cubes, battery powered, air propelled free flying robots designed for use within the
STATION (Fig 1). Two Free-Flyers (Units Alder and Birch) were launched in 2019 along with the charging
station (called the “Dock”) and are installed in the second laboratory module (LABB) Labb. A third Unit,
Cedar, is also on station in storage. Working autonomously or via remote control by astronauts, flight
controllers, or researchers on the ground, the robots are designed to complete tasks such as taking
inventory, documenting onboard experiments with their built-in cameras or working together to move
cargo throughout the station. In addition, the system serves as a research platform that can be outfitted
and programmed to carry out experiments in microgravity [1]. Currently, Alder is going through its
commissioning phase and has successfully completed mapping the LABB in preparation for autonomous
operation.




                                 Figure 1 – Cabinflyer Free Flyers and Dock in the STATION
```

## Page 2

```text
Free-Flyer Propulsion System

The nozzles are part of the Free-Flyer propulsion system which is comprised of two modules located on
opposite sides of the Free-Flyer (Fig. 2). Air is drawn into the propulsion module by a centralized
centrifugal fan which pressurizes a plenum (~1.5 in-H2O). Thrust is produced when the pressurized air is
exhausted through one of its six nozzles that are located around the plenum. For a constant impeller
speed the thrust from each of the twelve nozzles has fixed direction with magnitude controlled by
adjusting the nozzle open area.




                                    Figure 2 - Free-Flyer Propulsion Module

The key propulsion requirements are: 1. provide holonomic control in 6-Degrees-Of-Freedom, i.e., the
ability to produce instantaneous thrust in any direction and torque about any axis, 2. produce 0.6 N max
thrust on at least one motion axis, and 3. keep noise below 65 dBA at max thrust.

The two propulsion modules can be replaced by astronauts and are designed to be interchangeable.
Their impellers rotate at the same speed and, because they are on opposite sides, their gyroscopic
moment and drag torques are cancelled. The impeller speed is adjustable to trade thrust performance
vs. reduced power and noise. The propulsion layout (dual impellers feeding pressurized air to multiple
thrust nozzles) is conceptually similar to what was employed by the AGENCY Facility Earlier Free Flyer (EFF) project in 2003.

Novel features of the Cabinflyer Free-Flyer propulsion system include low acoustic noise with high thrust,
asymmetric nozzle layout which reduces the total nozzle count while maintaining 6-DOF
maneuverability, modular design, and variable thrust control.
```

## Page 3

```text
Propulsion Development Process

During the early phases of the Cabinflyer project, several propulsion options were considered. The four
leading options were: 1. Onboard compressor. Essentially replicate the Facility PREDECESSOR cold-gas
propulsion system (ref. [2]) except replace the liquid CO2 tanks with compressed air tanks filled by an
onboard compressor. 2. Distributed fans. Use a system of six axial fans with reversible fan rotation, like a
multi-roto drone adapted for zero-gee, 3. Similar to option 2, but with Variable-Pitch Propellers (VPP) to
increase fan thrust responsiveness without changing the rotor RPM, and 4. Centralized fans. Use two
impellers to drive airflow to many independently controllable nozzles. Option 3, multi axial VPP fans,
was initially chosen for the Free-Flyer propulsion system. Testing of Prototype 2 is shown in Figure 3.




                                Figure 3- Cabinflyer Prototype 2 Multi-VPP fan Testing

After encountering difficulties with packaging the VPP system (including limited future payload carrying
capabilities) and growing concerns over fan performance and reliability, a new propulsion trade study
was started. This time, the ducted nozzles and reaction wheels of the original EFF centralized fan system
were redesigned to a single pressurized plenum which greatly simplified packaging. A propulsion system
proof-of-concept was quickly made using 3D printed and off-the-shelf parts (Fig. 4). The system
immediately proved capable and was chosen to be developed for flight.




                          Figure 4 - Centralized fan Proof-of-Concept Testing on Force Plate
```

## Page 4

```text
The basic elements of the centralized fan design are the centrifugal fan, plenum, and nozzles. The
propulsion system development was iterative because no single element was independent of, or linearly
related to, the other elements and they all needed to work together to meet the final requirements.
This meant that iteration occurred at different levels: Component Level, Module Level, and Free-Flyer
Level. The challenge was to obtain the data needed, then iterate and then test at the next higher level,
then iterate again, and so on. This required significant testing, modeling, and analyses to narrow down
the design space. Unfortunately, this started late in the project due to the earlier propulsion system
being chosen. Some of the more challenging propulsion system requirements are listed in Table 1.
                              Table 1 – Challenging Propulsion System Requirements

             Requirement                                              Value
 Thrust                                 Minimum of 0.6N in X-direction
 Acoustics                              Maximize NC-40 operation
                                        (lower noise allows longer operating time)
 Control                                Holonomic control
 Power (electrical) consumption         Operate flying ~ 2 hours without recharge
 Size and Mass                          Physical envelope and mass limits
 STATION Requirements                       Safety, materials, flammability, loads,..
 Other project requirements             - Replaceable components
                                        - Industrial Design
                                        - Expanded research options: LED signals, swappable skin


Component Level Development

Examples of component level tests are shown in Figure 5. Nozzle aerodynamic performance
characteristics (along with the impeller) were determined through test using the ANSI Fan Test Rig
within the AGENCY Facility Fluid Mechanics Lab [a colleague, Private Communication, 2019]. The test rig,
representative test articles and nozzle performance charts based on analysis and test are shown. In each
chart, there are multiple curves representing different fan sizes. The left chart shows combinations of
fan rotation rate (N) and nozzle open are (A) that achieve the required maximum thrust (0.3N). The right
chart shows the estimated Sound Pressure Level (SPL) for each combination.




                            Figure 5 – Examples of Component Level Tests and Analysis
```

## Page 5

```text
Module Level Development

Example module level test results and analyses are shown in Figure 6. The photo shows sound and
pressure measurements during proof-of-concept testing. Several CFD analyses were completed to study
grill effects and parameters such as spacing and depth dimensions. The chart provides measured SPL vs.
Force (Thrust) for two different nozzle areas and four impeller RPM’s. The target SPL and thrust are
highlighted to indicate operational regions (note, acceptable SPL levels for one propulsion module were
estimated to be -3dB of two modules in operation).




                                Figure 6 - Examples of Propulsion Level Test Results

Free-Flyer Level Development

Examples of Free-Flyer level test are shown in Figure 7. The granite table test allowed near friction-free
3-DOF testing (2 translational axis and 1 rotation) and was the only “flying” test prior to commissioning
on the STATION. As shown, the P4 prototype is mounted on top of a goniometer to test off-axis thrust
components. Along with EMI tests, early acoustic measurements were made in the Facility EMI test
chamber. Official EMI and Acoustic measurements were made at FCC.
```

## Page 6

```text
                                   Figure 7 – Example of Free-Flyer Level Tests

Nozzle Description

Thrust is controlled by varying the nozzle open area for a constant impeller speed. The nozzle area and
impeller are sized to provide the required thrust performance (Maximum 0.3 N for the x-nozzle and ≈
11m/s exit velocity). The forward and aft (X-direction) nozzles are physically larger to meet operational
thrust requirements. A propulsion module cross-section illustrating airflow and identifying a nozzle is
shown (Fig. 8).




                                     Figure 8 - Propulsion Module X-Section
```

## Page 7

```text
Nozzle open area is adjusted by two gear synchronized flappers that are driven by an off-the-shelf hobby
style RC servo (Fig. 9). The two flappers provide the benefit of dimensional compactness relative to a
single flapper design. The Nozzle grill and divider (made of one-piece 3D printed PEI 9085) provides
flow straightening. Nozzle open area is determined by the flapper angle as shown in Figure 10 (flapper
rotation from closed to full open is 64 degrees).




                                          Figure 9 - Free-Flyer Nozzle




                                          Figure 10 - Nozzle Open Area

Figure 11 shows a few of the nozzle design iterations with left to right going from the earliest prototype
to final flight design respectively. The initial proof of concept nozzle incorporated a single flapper which
provided the desired thrust performance but was physically too large in the flow direction to fit the
available space. To address that issue a geared dual flapper design was developed to minimize thickness
while providing an acceptable level of flow efficiency. To reduce development time, the first two nozzles
were made from 3D printed ABS material. The middle nozzle was an attempt to make a 3D printed
“Flight” nozzle (shown is an X-Nozzle), which proved problematic (see Lesson’s Learned).
```

## Page 8

```text
                                           Figure 11 - Nozzle Iterations

Nozzle Development Issues

Several nozzle development issues are expanded on.

Nozzle Issue: Flapper Slipping

Once calibrated, the flapper drive elements (consisting of clamps, gears and shafts) must not slip over
the life of Cabinflyer. However, slipping was observed during early prototype tests. The required flapper
shaft torque is 10 in-lb (which includes a 5X slip factor). Slipping can occur at two joints (Fig. 12): 1. the
flapper to the shaft, and 2. the press fit gear to the shaft. In torque tests, the integrated flapper clamps
worked well and were not considered the problem. The press-fit gear did slip at low torques.




                                         Figure 12 - Flapper slip locations

Options to increase the torque limits of the titanium shaft to bronze gear included:

    1. Increase press-fit interference
```

## Page 9

```text
       The nominal fit is ANSI FN2 providing 1 mil interference. Torque values as low as 3.68 in-lb were
       measured in tests. The interference was increased to the point that bronze shavings were
       observed but with little additional torque benefit.
    2. Add adhesives
       Adhesive 7649 primer with Adhesive 609 retaining compound were tried. Even though the measured
       torque values were higher than those without adhesive, the decision was made to include
       option 3 for consistently higher results.
    3. Knurl the shaft
       Knurls are commonly used for small plastic gears on metal shafts (The cold flow of plastic
       material into various topographical feature can result in a more constant or increase in torsional
       strength over time [3]). Also, referencing a common home remedy for slipping where a center
       punch is used to provide mechanical interlock, a straight knurl on the shaft was considered as an
       option. The bronze gear would not plastically flow into the knurl, but it could essentially be
       broached while press-fit to provide an additional mechanical interlock. Several different knurls
       were tested along with and without Adhesive adhesives. The measured torque values were all
       above 12 in-lb (highest being 21 in-lb). Because of the consistent results, the knurl and Adhesive
       adhesives are used on the flight nozzles. Figure 13 shows the two shafts and stock gears prior
       to assembly. The gear in the right image was first press-fit onto the flapper shaft and then
       removed to show how the bronze gear is cold worked by the shaft knurl.




                                 Figure 13 - Knurled shafts for improved torque

Nozzle Issue: Acoustic Noise and Vibrations

Initial nozzle prototypes generated noise that was noticeably loud and annoying. The Free-Flyer’s
operating time on the STATION is limited by the maximum noise produced, so it was critical to minimize all
acoustic noise sources. The following tasks were completed to reduce nozzle generated noise:

    1. Select the best servo
       The servo was quickly discovered to be the primary noise source. The noise occurred as the
       servo feedback continuously adjusted to hold a position, with the noise being louder under load.
       It was observed the noise was different for different servos. The following test was used to
```

## Page 10

```text
   select the best servo: With the servo stalled (by commanding it past its limits) measure a)
   housing temperature, b) current draw, c) time before failure, and d) a qualitative noise
   judgement. Seven different servos were considered with the Orrin SV95 being the clear winner.
   The Orrin SV95 had the lowest noise and was still functional after being stalled for 15 minutes.
2. Remove nozzle backlash springs
   The initial nozzle prototypes used backlash springs to apply a small closing torque to eliminate
   gear backlash (Fig. 14). Unfortunately this had an unintended consequence, in that the servo in
   overcoming the backlash spring torque produced additional noise. See lessons learned for
   comments on removing the backlash springs.




                                    Figure 14 - Back-lash Springs

3. Isolate and dampen nozzle vibrations
   Qualitatively, the servo noise was amplified when mounted to a propulsion module. Also, the
   small vibrations could excite surrounding structure and be a motion jitter source for the Free-
   Flyer. Vibration measurements of the propulsion system plenum structure, nozzle frame, and
   servo body were made for an operating nozzle (Fig. 15). Tests with the nozzle hard mounted and
   isolated, along with the servo mounted with aluminum and nylon standoffs were made.
   Vibration levels were lowest for the isolated nozzle using aluminum standoffs. The final isolated
   design is shown in Figure 16.




                                   Figure 15 - Nozzle Vibration Test
```

## Page 11

```text
                                     Figure 16 - Nozzle Isolation Design

Nozzle Issue: Bearing Installation

Aligning and pressing the 6mm and 12mm O.D. nozzle bearings by hand (and with other simple tools)
was time consuming and often resulted in damaged bearings. Including all the flight and ground unit
nozzles, over 800 bearings are installed. To improve the bearing installation process, two custom
bearing tools were designed and fabricated that proved reliable and capable of rework when necessary.
The installation steps are given in Table 2 and illustrated in Figure 17.
                                     Table 2 - Bearing Installation Steps

     Step       Description
       1        Locate Bearing and Nozzle Frame
                The frame and bearing are radially located by the guide pin and sliding locator (the
                locator slides on the guide pin and sits on top of the spring).
       2        Press Bearing
                The guide pin centers the Ram such that only the bearing outer race is contacted as
                shown in the closeup. The bearing and frame are now concentrically located and can
                be pressed together using an arbor press and overcoming the soft spring. The bearing
                axial position is determined when the Ram bottoms against the frame.
       3        Remove
                Remove the Ram and nozzle frame. Inspect bearing movement.


Re-work was necessary when the frame was not held down to the bearing tool allowing a slight angular
press. When this occurred, the bearing was pressed through the frame and re-pressed. A better method
to securely clamp the frame to the tool would be an improvement.
```

## Page 12

```text
                                      Figure 17 - Bearing Installation Tool

Lessons Learned

Several lessons learned are expanded on.

Lessons Learned: Limitation of 3D printed plastic parts

Significant use was made of 3D printed plastic parts during nozzle development and Cabinflyer
development in general. 3D printed materials make up 32% of a flight Free-Flyer dry mass (the materials
include: PEI 9085, Composite XT 2.0, RESIN Watershed XC 11122). 3D printed test articles and early
prototypes reduced development time and helped catch errors in the design.

Partly due to the success the team had using 3D printed materials, and their lightweight properties, the
initial flight nozzles were designed to incorporate 3D printed flappers and nozzle frames (Fig. 11).
However, when significant issues with outside vendor part quality and assembly reliability issues were
discovered, those parts were changed to a more traditional machined aluminum. The inlet and grill
dividers are the only flight nozzle parts made from 3D printed plastic.

The Cabinflyer hardware is classified as non-critical. That meant the 3D printed material strength
requirements are driven by the projects goal to have a reliable and long-lasting robot, and not the more
rigorous fracture critical material requirements. There are no released AGENCY standards for materials and
parts made of 3D printed plastic (A committee is working on AGENCY-STD-6030 for Additive
Manufacturing, which included 3D printed plastic, but it is not released).

A significant effort was required to develop 3D printed plastic standard processes, analysis assumptions,
and best practices to assure a level of strength and quality. And in some cases, alternatives to 3D printed
plastic should have been explored. A good example is the Free-Flyer painted bezel shown in Figure 1 and
Figure 18. A simple design change would have allowed the part to be easily machined thereby
eliminating the additional labor cost necessary to fill and smooth the PEI 9085 raw surface in
preparation for painting.
```

## Page 13

```text
                                 Figure 18 - Free-Flyer 3D printed plastic painting

Lessons Learned: Backlash Testing

Gear backlash affects nozzle open area and associated thrust for a given command input. It’s worse for
small nozzle openings since the backlash error is proportionally greater for a smaller open area. It wasn’t
clear how accurate the open area needed to be during the early development phases. The aerodynamic
forces acting on the flappers should pre-loading the gear train and reduce backlash. However, flapper
binding was observed on some early nozzle prototypes (which could act in the opposite direction of the
aerodynamic forces) and the decision was made to add a nozzle backlash spring to eliminate all sources
of error. A custom 0.012” diameter 302/304 CRES torsion spring with 0.04 in-lb of torque was added as
shown in Figure 14. The spring torque is applied to close the flappers and help minimize leakage when
fully closed (minimizing leaks was another early concern). The torque direction would minimize the
required servo torque since the plenum and airflow pressures are trying to open the flappers.

Two issues developed that focused attention on the backlash springs. The first was the nozzle servo
noise as discussed earlier. The backlash spring made the servo noise much worse. The second was the
discovery of broken backlash springs on the Prototype 4 (P4) Granite Table Testing shown in Figure 7.
Small 0.012” diameter spring pieces floating in the STATION cabin would be an obvious safety concern. The
decision was made to try operating P4 without the nozzle springs. The P4 testing had been successful
and going on long enough that any adverse changes to the Free-Flyer could be measured. The tests
showed no change without the springs and the servo noise was noticeably lower. The backlash spring
was removed from the design (however, the flapper still has a notch for the spring due to fabrication
schedule constraints).

The backlash springs were added because of uncertainty in the required nozzle opening accuracy and
nozzle function. As it turned out, the springs were added to solve a theoretical concern that wound up
causing a real problem. It was fortunate that the design was not impacted when they were removed.
```

## Page 14

```text
Concluding Remarks

This paper tries to describe how complex the nozzle design was for something that overall is rather
simple. It’s typical that designs quickly become boxed in by competing requirements and practicality.
But the added dependency of full system test to feedback the next design iteration made this more
difficult and schedule critical. The original project plan included four (4) granite table “flying” prototypes
(Free-Flyer Level prototypes). The prototype (iterative) philosophy was a good choice for this
development. However, because the centralized fan wasn’t the original propulsion system choice, it
wasn’t until Prototype 4 (P4) that it could be fully tested and provide valuable data for the next
iteration.

The centralized fan propulsion system has proven itself capable and a high performance option for intra-
vehicular robots (IVR). Future centralized fan propulsion designs can be scaled and optimized. The
Cabinflyer Free-Flyers serve both as a feasibility demonstration and reference design for future mission-
critical IVR systems.

References

[1] website https://www.agency.gov/cabinflyer

[2] website https://www.agency.gov/predecessor

[3] the textbook's author, Plastic Part Design for Injection Molding, Hanser/Gardner Publications, Cincinnati,
OH (1994).
```
