# Slides: system overview (24 November 2019)

Re-typeset from the document's text layer, page by page; the text of each page is kept as it is laid out; pictures and drawings are not reproduced.

## Page 1

```text
Cabinflyer Overview




        System, Status,
 Performance, and specification
     November 24th, 2019
```

## Page 2

```text
                Cabinflyer Objectives
• Provide a microgravity
  robotic research facility in
  the STATION laboratory segment
  (LABSEG), which will replace
  the existing PREDECESSOR
  facility
• Provide remotely operated
  mobile camera views of the
  STATION LABSEG to enhance the
  situation awareness of
  mission control
• Perform mobile sensor
  tasks in the STATION LABSEG
July 15, 2019                         2
```

## Page 3

```text
                System Description - Hardware
       Speaker/Microphone                                                  Terminate Button
                               NavigationCamera   Nozzle (x 12)                                         Battery (x 4)
  Laser Pointer      ImagingCamera                                     VelocityCamera


                                                                                                                           Arm

      FrontdepthCamera

Touch Screen


                                                        Impeller
                                                                                                                BackdepthCamera
 Power Switch                                                                                                   Dock cup (x 2)
                                                                                                                Dock 20-pin
                                                                                                                connector
   Forward Flashlight
                                                                                                    BackCamera
          Wake Button                           Signal Lights
                     Status LEDs                                                               Aft Flashlight
                                            Bumper (x 8)



  •     Free flying robot inside the STATION     • Three smartphone computers
  •     32 cm wide, ~9.1kg (2 batt., no arm) • Three payload bays for expansion
  •     All electric + fan-based propulsion  • Microphone not currently enabled
  •     Robot arm for “perching”, ~1kg
July 15, 2019                                           TEAM Mini-talk                                                            3
```

## Page 4

```text
                System Description - Communications
                                           STATION       Free Flyers   Crew Control
• Communicates                                                        Station
  through STATION WiFi
  when flying                                                                        STATION NAS
                                                                                     Storage
• Single
  telemetry/video                       Dock
  stream to ground                                         STATION Payload LAN
• Multiple ground                      Relay
  stations can connect                 Site                   Ku-Band

  through server                        Cabinflyer Ground                    STATION Payload Ops
• Large file transfers                    Relay (PCC)                   Integration Center
                                                                                (PCC)
  and software updates                 Multi-Mission Ops
  through Ethernet on                   Center (FAC)                     STATION Mission Control
  the dock                                                                   Center (FCC)

                                         Cabinflyer Data                       Guest Scientist
                                         Archive (FAC)                        Institutions


                                        Cabinflyer communications path

July 15, 2019                  TEAM Mini-talk                                                   4
```

## Page 5

```text
                System Description - Navigation
• Vision-based navigation
       • Compares features with on-
         board a priori map
       • Incorporates inertial
         measurements
• Fiducials used for
  autonomous docking
       • Requires approximately 1 cm
         position accuracy                        Feature map of the LABB-PM
• Visual odometry
       • Robot can continue to navigate where no map features are
         recognized



July 15, 2019                     TEAM Mini-talk                               5
```

## Page 6

```text
                System Description - Perching Arm
                              Camera View Direction

• Designed to grasp
  handrails
• Stows completely in
  payload bay
• Acts as a pan-tilt unit      Cabinflyer Perching Arm pan motion
  while perched
                          Camera View Direction
• Flexible and back-
  drivable
• May be perched                    -30°
  manually                                                   +90°


                                              Cabinflyer Perching Arm tilt motion

July 15, 2019                 TEAM Mini-talk                                       6
```

## Page 7

```text
                System Description - Docking Station
                                                      Cooling fan Ethernet Power
                                 Circuit                intake      port    port
                    Cooling fan breakers
                                         Handle
                     exhaust




                    Berth
                  lance (x 4)            AR target
                          Berth 20-pin    panel                  Adjustable rear
                         connector (x 2)                        mounting bracket

• 85 cm x 38 cm x 28 cm                               • Fiducials used for visual
• Berths for 2 free flyers                              servoing to autonomously dock
• Provides power and Ethernet                         • Magnets provide retention force
July 15, 2019                                   TEAM Mini-talk                         7
```

## Page 8

```text
                  System Description - Ground Data System

• Cabinflyer Control Station
       • Sortie planning tool
       • Execution monitoring
                • Live telemetry
                • Image and video streams
                • 3D virtual display
       • Supervisory control (run plans or
         single commands)
       • Typically used by ground operators
• Crew Control Station runs on an
  crew laptop Computer (CLC)
• Server for archiving and distributing
  Cabinflyer data
• Suite of engineering tools to
  support maintenance and software
  upgrades


July 15, 2019                               TEAM Mini-talk   8
```

## Page 9

```text
                Cabinflyer Control Station




July 15, 2019             TEAM Mini-talk    9
```

## Page 10

```text
                          Basic Conops
• When an Cabinflyer is idle, it charges in its dock
• Cabinflyers can execute complex plans with full autonomy
  and no astronauts present
       • Including undock, traverse multiple modules, return to dock
• However, Cabinflyers run with ground operator oversight
       • When an anomaly occurs, an Cabinflyer generally stops and
         waits for operator intervention
       • It can continue operating during communication outages until
         it encounters an anomaly
• The operator can always take over and teleoperate
• Astronauts can also be operators, but this is will likely be a
  rare occurance (minimize crew time)


July 15, 2019                      TEAM Mini-talk                        10
```

## Page 11

```text
                      Hardware Status
• Docking Station
       • Launched on CF-10 – 11/17/18
       • Installed on 2/15/19
• 2 Cabinflyers (Alder & Birch) and Dock Spares Kit
       • Launched on CF-11 – 4/17/19
       • Checkout of Alder began 4/30/19
       • Checkout of Birch around 7/25/19
• 1 Cabinflyer (Cedar) and 3 Perching Arms
       • Launching on CF-18 – 7/21/19
       • All hardware delivered and verifications closed
• Cabinflyer Spares Kit and Port Tester
       • Launching on CF-12 – 10/19/19
       • On-dock 8/6/19
July 15, 2019                    TEAM Mini-talk             11
```

## Page 12

```text
                Cabinflyer Commissioning Activities
1. Checkout
       • Crew inspects the robot
       • Functional tests of all hardware
2. Calibration & Mapping
       • Crew collects camera and IMU calibration data
       • Crew collects mapping data of Labb
3. Localization & Mobility
       • Verify robot localization within Labb
       • Robot performs increasingly complex motions to test mobility system
4. Checkout & Calibration
       • Combination of 1 & 2 without mapping



July 15, 2019                         TEAM Mini-talk                        12
```

## Page 13

```text
                Cabinflyer Commissioning Activities

6. Ops Demo
       • Demonstrate an operational mission scenario
7. Payload Installation
       • Crew installs payload (perching arm)
       • Functional tests of payload
8. Payload Demo
       • Operational demonstration of payload (perching arm)
9. Performance Characterization
       • TBD tests to further characterize Cabinflyer performance
10. PREDECESSOR/Cabinflyer Hand-off
       • Symbolic passing of the torch from PREDECESSOR to Cabinflyer
       • Completion of commissioning


July 15, 2019                         TEAM Mini-talk               13
```

## Page 14

```text
                Commissioning Schedule
                           Completed | Scheduled | Desired
Activity                                       Robot         Date
Checkout                                       Alder        4/30/19
Calibration & Mapping                          Alder        5/13/19
Additional Mapping                             Alder        5/23/19
Localization & Mobility                        Alder        6/14/19
Localization & Mobility                        Alder        7/9/19
                                                             7/12/19
Localization
Checkout   & Calibration
             & Mobility                        Alder
                                               Birch         7/23/19
                                                             7/25/19
Checkout & Calibration
Localization & Mobility                        Birch
                                               Alder        7/30/19
Localization & Mobility                        Birch         8/6/19
                                                             8/13/19
Localization
Checkout   & Calibration
             & Mobility                        Birch
                                               Cedar         8/13/19
                                                             8/20/19
Localization & Mobility                        Birch         8/27/19


July 15, 2019                          TEAM Mini-talk                   14
```

## Page 15

```text
                Alder Checkout




July 15, 2019         TEAM Mini-talk   15
```

## Page 16

```text
                IMU Calibration




July 15, 2019         TEAM Mini-talk   16
```

## Page 17

```text
                1st Flight




July 15, 2019      TEAM Mini-talk   17
```

## Page 18

```text
                Stopping Ability




July 15, 2019         TEAM Mini-talk   18
```

## Page 19

```text
                1 Autonomous Undocking
                 st




July 15, 2019            TEAM Mini-talk   19
```

## Page 20

```text
                1 Autonomous Docking
                 st




July 15, 2019           TEAM Mini-talk   20
```

## Page 21

```text
                Near-Term Guest payload Payloads
• AGENCY FCC’s Tag Survey (TAGREADER-2)
       • RFID reader allows inventory and searches for logistics reduction
       • Launch on CF-12
• SPONSOR’s Acoustic Survey (Robotics Partner/Industry Partner)
       • Microphone array that creates a sound map used to monitor health of on-board
         systems
       • Launch on CF-12
• Adhesive Gripper (University B)
       • Gecko-inspired end-effector for gripping of smooth surfaces
       • Launch on CF-18
• Student League (UNIV)
       • Middle school and high school STEM robotics competition
• PARTNER Cabinflyer-PartnerRobot Joint activity, Labb-RPC
       • Asia-region STEM robotics competition
• Guest Experiment A (University C)
       • Software-only payload investigating propellant-less propulsion using a manipulator

July 15, 2019                                 TEAM Mini-talk                                   21
```

## Page 22

```text
                       Fun with Robots




                Questions?

July 15, 2019                TEAM Mini-talk   22
```

## Page 23

```text
Cabinflyer System Overview




     Facility Research Centre, AGENCY
       the maker's site
       November 25th, 2019
```

## Page 24

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 25

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 26

```text
        Cabinflyer System
• Main components
                             Cabinflyer


                                                             Ground Data
     Free Flyer                Dock
                                                               System




                  a regional space forum
November, 2019                                                             4
                         (RSF-26), the forum's host city
```

## Page 27

```text
                Cabinflyer System: Free-flyer
                                                                                  Battery
  Speaker/Microphone            NavigationCamera                   Terminate Button
                                              Nozzle
                       ImagingCamera                              VelocityCamera
Laser Pointer

                                                                                                             Arm
   FrontdepthCamera


 Touch Screen



  Power Switch                                                                                           BackdepthCamera
                                                                                               BackCamera
Forward Flashlight                       Signal Lights                                      Aft Flashlight
                 Wake Button                                               Corner Bumpers
                                                          Impeller
                                Status LEDs

                                       a regional space forum
     November, 2019                                                                                             5
                                              (RSF-26), the forum's host city
```

## Page 28

```text
        Cabinflyer System:
        Free-flyer’s Hardware Payloads




                 a regional space forum
November, 2019                                              6
                        (RSF-26), the forum's host city
```

## Page 29

```text
             Cabinflyer System: Dock                                  Cooling Fan
                     Subsystem Breakers
                                              Main Power Breaker           Cooling Fan Screen
Air Vent Deflector


                                                                                  RJ-45 Connector




                                                                                  Power Connector




Free Flyer Berth         AR Target
                         a regional space forum
    November, 2019                                                                       7
                                (RSF-26), the forum's host city
```

## Page 30

```text
        Cabinflyer System: Ground Data
        System
• Provides control over the Cabinflyer free-flyer
      • by the astronaut
      • by ground controllers
      • by Guest Scientists (GS)
• Enables data exchange between Cabinflyer and
  ground
• GDS interfaces navigation computer’s (NC)
  Executive
• GDS communicates through Executive with guest computer (GC) where GS Android Apps are
  deployed
                       a regional space forum
November, 2019                                                    8
                              (RSF-26), the forum's host city
```

## Page 31

```text
              Cabinflyer System: Data Flow
                                                                                                  Link Legend
                              Cedar    STATION
                                                                                                  Ethernet/LAN
                               Birch       Dock
                                Alder    Internal               Payload       Crew
                                               Switch              LAN        Control                Ethernet:
               Internal                                           Switch      Station               Internal IP
                Switch           CC
                                               Dock
                                               CPU                                      Ku-Band      Ethernet:
                                                                                                   Internal and
                                NC                                                                Payload LAN
                                                                 5GHz
                                                                                 NAS
                                                                 WAP
                                                                                                      WiFi:
                                GC                                                                Payload LAN
              Payload                                                     SLAN
                                                                      Payload LAN                     USB
                                                                                        Relay
                                                                                        Site      Other/LAN

               Storage    Operator/     Operator        Ground
 Operator                                                                   Operator
                          Engineer       Control         Relay
  Control                                                                    Control
                           Control       (POC)         (PSC)

    GS
Institution     FAC FOR                      PCC                         FCC FCR
                                       a regional space forum
   November, 2019                                                                                        9
                                              (RSF-26), the forum's host city
```

## Page 32

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 33

```text
        Cabinflyer Robot Software
                                                  Cabinflyer




                                                                                Ground Data
        Free Flyer                                  Dock
                                                                                  System




                 Users Development
                      Testbed                   Cabinflyer Robot
                                                Software Scope
                                     a regional space forum
November, 2019                                                                                11
                                            (RSF-26), the forum's host city
```

## Page 34

```text
        Cabinflyer Robot Software: Overview

• Cabinflyer Robot Software (ARS) is deployed on 4 cell phone
  type processors (Cabinflyer + Dock) running Linux and
  Android
• Cabinflyer contains 7 distinct microprocessors with custom
  firmware + several microprocessors with Components Off-
  The-Shelf (COTS) firmware
• Software deliverables includes:
      – Custom firmware(s)
      – Custom tailored Linux kernels
      – Linux and Android Operating Systems
      – ARS dependencies (third-party libraries)
      – ARS Source Code      Open Source

                          a regional space forum
November, 2019                                                       12
                                 (RSF-26), the forum's host city
```

## Page 35

```text
        Cabinflyer Robot Software: Overview




                 a regional space forum
November, 2019                                              13
                        (RSF-26), the forum's host city
```

## Page 36

```text
        Cabinflyer Robot Software: Overview




                 a regional space forum
November, 2019                                              14
                        (RSF-26), the forum's host city
```

## Page 37

```text
        Cabinflyer Robot Software: Features

• Manage Cabinflyer sensing and actuation
• Localize and Navigate within the STATION
• Perform autonomous docking
  (+ return to dock)
• Perform autonomous perching
• Support teleoperation from ground
• Support plan based automated tasks
• Support “Guest payload” operations
• Support multi Cabinflyers communication
• Support hardware for multisensory human
  interaction

                        a regional space forum
November, 2019                                                     15
                               (RSF-26), the forum's host city
```

## Page 38

```text
           Cabinflyer Robot Software: Components

•   Communication Framework (ROS + DDS based)                        Number or ROS nodes for a simulation:                 ~36
•   Localization                                                     Number of ROS nodes on Cabinflyer:                      ~48
       •   Marker less Flying                                        (not counting GC)
       •   Docking
       •   Perching
•   Offline mapping for localization
•   Pose Estimation + Propulsion Control (GNC)
                                                              --------------------------------------------------
•   Executive                                                 Language                 files comment              code
       •   Mode Management                                    --------------------------------------------------
       •   Sequencer (Plan Execution)                         C++                        263       11078         45023
                                                              Lua and config               76        2179        14380
•   Mobility                                                  C/C++ Header                158        4232         8209
       •   Coordinates required subsystem                     Python                       58        1798          6706
       •   Generates and validates trajectories               CMake                       150        3631          5624
       •   Performs collision detection                       IDL MSG
                                                              SRV ACTION                   148            0         5139
•   Fault Management                                          XML                         163          309          4271
•   Guest payload                                             Java                          70        1764          3822
                                                              Bourne (Again) Shell 83                  853          2522
•   User Interfaces Support                                   YAML                            4         29           294
•   Simulator                                                 make                            9         75           238
                                                              --------------------------------------------------
•   Platform Management and development tools                 SUM:                        1182 25948               96228
                                                              --------------------------------------------------
                                           a regional space forum
November, 2019                                                                                                                   16
                                                  (RSF-26), the forum's host city
```

## Page 39

```text
        Cabinflyer Robot Software: ROS

• ARS makes extensive use of the
  open-source Robot Operating
  System (ROS):
      • Communication framework linking
        all “nodes” running on the target
        platform
      • Try to maximize the re-use of
        existing ROS messages benefit
        from existing ROS packages
      • Use ROS introspections tools to
        rapid debugging
      • Use ROS facilities to
        record/replay/analyze data
      • Use some ROS/Simulator
        components for the     simulator
                           a regional space forum
November, 2019                                                        17
                                      (RSF-26), the forum's host city
```

## Page 40

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 41

```text
        Simulator
• ARS can run on:
      • actual hardware on Cabinflyer
        (camera, IMU, PM)
      • simulated robot hardware on
        a PC
      • both simultaneously
• Simulation is possible by
  enabling/disabling ARS
  elements
• Simulator may be thought as
  a “software-in-the-loop”
  simulation of the free-flyer’s
  hardware drivers.
• Hardware drivers are
  simulated via Simulator Plug-
  ins


                           a regional space forum
November, 2019                                                        19
                                  (RSF-26), the forum's host city
```

## Page 42

```text
        Simulator
• ARS can run on:
      • actual hardware on Cabinflyer
        (camera, IMU, PM)
      • simulated robot hardware on
        a PC
      • both simultaneously
• Simulation is possible by
  enabling/disabling ARS
  elements
• Simulator may be thought as
  a “software-in-the-loop”
  simulation of the free-flyer’s
  hardware drivers.
• Hardware drivers are
  simulated via Simulator Plug-
  ins


                           a regional space forum
November, 2019                                                        20
                                  (RSF-26), the forum's host city
```

## Page 43

```text
        Simulator

• Contains custom
  propulsion system and
  some localization
  sensors
• Simulator-based dynamics
  model, imagers, arm,
  lights and STATION model
• Can run all nodes on
  desktop or some nodes
  on target development                                        Simulator Viewer (ROS visualization) used to debug ARS

  board

                    a regional space forum
November, 2019                                                                                       21
                           (RSF-26), the forum's host city
```

## Page 44

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 45

```text
        Cabinflyer Robot Software:
        Visualizers
• ARS supports 3 main visualizers: Viewer, Gviz, and Sviz
• Simulation can run headless (without any
  visualizer) or with any combination of them
• Viewer serves as an internal debugging tool from the
  robot’s perspective
• Gviz describes several data streams (e.g. EKF, PM,
  etc.) to determine attitude/localization
• Sviz shows a 3D view of the STATION, the Cabinflyer free-
  flyer, its arm, and obstacles

                  a regional space forum
November, 2019                                               23
                         (RSF-26), the forum's host city
```

## Page 46

```text
        Cabinflyer Robot Software:
        Visualizers
• Viewer




                 a regional space forum
November, 2019                                              24
                        (RSF-26), the forum's host city
```

## Page 47

```text
        Cabinflyer Robot Software:
        Visualizers
• Gviz




                 a regional space forum
November, 2019                                              25
                        (RSF-26), the forum's host city
```

## Page 48

```text
        Cabinflyer Robot Software:
        Visualizers
• Sviz




                 a regional space forum
November, 2019                                              26
                        (RSF-26), the forum's host city
```

## Page 49

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 50

```text
        Ground Data System (GDS)
• Ground communication
  uses DDS middleware to
  command Cabinflyer and
  receive telemetry
• Data Distribution Service
  (DDS): standard enabling
  secure real-time
  information exchange,
  modular application
  development and rapid
  integration
  (www.rti.com/products/dds
  -standard)

                  a regional space forum
November, 2019                                               28
                         (RSF-26), the forum's host city
```

## Page 51

```text
        Ground Data System (GDS)
• Commands:
      • Commands are defined
        using XP-JSON schema
      • ARS defined a “ROS
        Command” mirroring the
        DDS command structure
      • Onboard Cabinflyer Guest
        payload or Ground
        Applications share the
        same command
        dictionary with either DDS
        or ROS transport
• Telemetry:
      • Internal uses ROS
        Messages (using ROS
        messages when possible)
      • External uses DDS                                            Simplified image for illustrative purposes
        Messages (subset only)

                          a regional space forum
November, 2019                                                                                 29
                                 (RSF-26), the forum's host city
```

## Page 52

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 53

```text
        Executive
• The executive is
  responsible for keeping
  track of the operating and
  mobility states of the
  robot.
• Responsible for forwarding
  accepted commands to
  the correct nodes and
  informing the ground and
  system on the status of
  the command.
• Communicates with
  ground via DDS-ROS
  bridge
• Interfaces ground with the
  guest computer (GC)


                       a regional space forum
November, 2019                                                    31
                              (RSF-26), the forum's host city
```

## Page 54

```text
        Executive
• The executive is
  responsible for keeping
  track of the operating and
  mobility states of the
  robot.
• Responsible for forwarding
  accepted commands to
  the correct nodes and
  informing the ground and
  system on the status of
  the command.
• Communicates with
  ground via DDS-ROS
  bridge
• Interfaces ground with the
  guest computer (GC)


                       a regional space forum
November, 2019                                                    32
                              (RSF-26), the forum's host city
```

## Page 55

```text
        Executive
• The executive is
  responsible for keeping
  track of the operating and
  mobility states of the
  robot.
• Responsible for forwarding
  accepted commands to
  the correct nodes and
  informing the ground and
  system on the status of
  the command.
• Communicates with
  ground via DDS-ROS
  bridge
• Interfaces ground with the
  guest computer (GC)


                       a regional space forum
November, 2019                                                    33
                              (RSF-26), the forum's host city
```

## Page 56

```text
        Executive
• The executive is                                           Cabinflyer                                     Ground
  responsible for keeping
  track of the operating and                  GC                         NC
  mobility states of the
  robot.                                              data                               cmd
• Responsible for forwarding              GS App                 cmd Executive                     data      GDS
  accepted commands to                                  cmd                                      cmd
  the correct nodes and
  informing the ground and
  system on the status of
  the command.                                                                data

• Communicates with
  ground via DDS-ROS
  bridge                                                      Teleoperation                        Ground Control Ops

• Interfaces ground with the                                  Android Guest payload commanding
                                                                                                   Guest payload Ops
  guest computer (GC)                                  Android Cabinflyer API commanding




                       a regional space forum
November, 2019                                                                                             34
                              (RSF-26), the forum's host city
```

## Page 57

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 58

```text
        Command Dictionary

• Command Dictionary is a set of
  functions either planned to be
  or already implemented on
  Cabinflyer.
• Its functions are implemented
  in the Executive




                 a regional space forum
November, 2019                                              36
                        (RSF-26), the forum's host city
```

## Page 59

```text
        Command Dictionary

• These functions
      • help control the robot via
        commands
      • allow guest scientist to start/stop
        their Android apps from the
        ground




                       a regional space forum
November, 2019                                                    37
                              (RSF-26), the forum's host city
```

## Page 60

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 61

```text
        Libraries
• Cabinflyer API Library
      • Java API/library wraps Executive’s
        Dictionary commands and send them to
        the robot.

      • Found at GitHub Non-AGENCY users
        Cabinflyer Android repository

      • If a command is not being implemented
        yet, the Cabinflyer Executive system will
        return a failed command message.

      • Provide means to get data from the robot

      • Utilized by the Guest payload Code
        (Android App) at the GC


                          a regional space forum
November, 2019                                                       39
                                 (RSF-26), the forum's host city
```

## Page 62

```text
        Libraries

• Guest payload Library
      • Java library that enables GS
        applications to be
        commanded from the GDS

      • Sets a standard on how to
        design and implement GS
        apps

      • Examples can be found at
        GitHub Non-AGENCY users
        repository

      • Utilized by the Guest payload
        Service (Android App) at the
        GC
                        a regional space forum
November, 2019                                                     40
                               (RSF-26), the forum's host city
```

## Page 63

```text
       Libraries
                                          Cabinflyer                                            Ground
                         GC                                 NC
                 App
                        API Impl.          data                            cmd
                   <cabinflyer_api>                 cmd
                                                      Executive                      data       GDS
                              data / cmd                             cmd


                   GS Code
                             data / cmd

                  GS Service                      cmd

                       <gs_library>



                                                              data


                                       Teleoperation                             Ground Control Ops

                                        Android Guest payload commanding
                                                                                 Guest payload Ops
                                        Android Cabinflyer API commanding
                                        Android App internal calls
                                      a regional space forum
November, 2019
                                             (RSF-26), the forum's host city
```

## Page 64

```text
    Contents
• Cabinflyer System
• Cabinflyer Robot Software
• Simulator
• Visualizers
• Ground Data System
• Executive
• Command Dictionary
• Libraries
• Guest payload Applications
```

## Page 65

```text
        Guest payload Applications
• Guest payload Applications
      • Two types:
            • Primary (commands robot, gets data, internal computation,
              payload communication)
            • Secondary (gets data, internal computation, payload
              communication)
      • Standard components:
            • GS API implementation (cabinflyer_api)
            • GS Service implementation (gs_science_lib)
      • Remote capabilities from GDS:
            • Starting/Stopping
            • Custom commanding and data feedback

                           a regional space forum
November, 2019                                                            43
                                  (RSF-26), the forum's host city
```

## Page 66

```text
        Guest payload Applications
• Guest payload Manager
      • GS Manager is an Android App of its own (Android
        Service)
      • Manage the commands and states of the GS Apps
      • Each GS App needs to provide an Android framework for
        the GS Manager to use
      • In charge of interfacing GS Android Apps and the GDS
        through the Executive system and the DDS-ROS Bridge




                      a regional space forum
November, 2019                                                   44
                             (RSF-26), the forum's host city
```

## Page 67

```text
        Guest payload Applications
                                          Cabinflyer                                                       Ground
                         GC                                 NC
                 App
                        API Impl.          data    cmd
                   <cabinflyer_api>                        Executive
                              data / cmd                                                                    GDS
                                                                                                           data
                   GS Code
                             data / cmd                      ROS
                                                  cmd




                                                                         cmd
                                                            Nodes
                  GS Service
                       <gs_library>

                             data / cmd
                                                                                  cmd
                                                         DDS-ROS
                  GS Manager                      data
                                                          Bridge
                                                                                *Teleoperation is not represented in this data/cmd flow diagram

                                      Android Guest payload commanding
                                                                           Guest payload Ops
                                      Android Cabinflyer API commanding
                                      Android App internal calls

                                      a regional space forum
November, 2019                                                                                                                 45
                                             (RSF-26), the forum's host city
```

## Page 68

```text
        Guest payload Applications




                 a regional space forum
November, 2019                                              46
                        (RSF-26), the forum's host city
```
