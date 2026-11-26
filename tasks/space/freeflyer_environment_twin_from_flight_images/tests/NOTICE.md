# NOTICE (grader side; never staged into the agent's home)

The instruction, the givens and the check tool name neither the robot whose flight this task uses, nor its makers,
nor the station: they say "the cabin free-flyer" and "the module". The given frames are left as recorded and show
lettering that a person can read: the module's agency name on a rack's logo and on labels, a row of decals with a
national flag above the far hatch, and on a round emblem the station's name and a facility's name. This file carries
the attribution, what the hidden side is made of, what the truth rests on and what the certification of the grader did
not exercise.

## Sources

- **The flight.** Every given of the flight `flight_a` (the wide-camera frames, `frames.csv`, `imu.csv`,
  `onboard_estimate.csv`) and the withheld depth record from which the truth is built come from one recording of
  NASA's Astrobee free-flying robot inside the Japanese Experiment Module (Kibo) of the International Space Station:
  sequence `iva_kibo_trans` of the Astrobee ISS dataset (https://astrobee-iss-dataset.github.io/); its page and paper
  state that the work was authored by contractor employees under a United States Government contract and that the
  Government holds a licence to it, and name no other licence. The frames are
  the recorded images, renamed in recording order and written as plain grey PNG; the times are the recorded stamps
  relative to the first frame; the depth record is used only on the grader side.
- **The maker's files.** The module shell (`module_shell/shell.obj`) is the geometry of the Kibo module mesh of the
  Astrobee media (github.com/nasa/astrobee_media, `astrobee_iss/meshes/jpm.dae`, placed at the origin of its
  collision element in `astrobee_iss/urdf/model.urdf`; its readme: "All multimedia is covered by NASAs Media Usage
  Guidelines"). The wide camera's model and mounting, the inertial unit's mounting, the hazard mapper's parameters and
  the zone record (the keep-out file and the keep-in boxes of this module, each with its own dates) come from the
  Astrobee Robot Software (github.com/nasa/astrobee), Apache License 2.0.
- **The maintainers' own.** The depth camera's ray table (measured from the robot's recorded depth point clouds),
  its transform to the wide camera (measured on the robot's recorded calibration flights) and the time relation
  between the two cameras (measured on this flight's records); the maker's published two-camera transform and depth
  correction are not used. The truth (`tests/gt/`), the reference twin (`solution/answer/`; `tests/gt/reference/`
  holds the same scene and mesh pieces, and its camera file holds the entries of the 24 scored moments only, where
  the answer's holds one for each of the 1068 given frames), the grader, the check tool and its library, and the
  documents are this task's own.
- **The reconstruction tools in the image.** The task image holds the unmodified command-line tools of **OpenMVS
  2.4.0** (https://github.com/cdcseacave/openMVS: `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`,
  `RefineMesh`, `TextureMesh`, `TransformScene`), licensed under the **GNU Affero General Public License version 3**
  (AGPL-3.0), installed from the conda-forge CPU build `openmvs 2.4.0 cpu_h59d1794_202` with only the shared
  libraries its tools load (their licences: `/opt/openmvs/PACKAGES.tsv` and `/opt/openmvs/licenses/` in the image;
  the licence text of OpenMVS itself in `/opt/openmvs/licenses/openmvs/`). The repository holds none of its code:
  `environment/Dockerfile` installs it when the image is built, from the exact package list
  `environment/openmvs-conda-explicit.txt` (download URLs and checksums), which the image keeps as
  `/opt/openmvs/conda-explicit.txt`. The pointer to the corresponding source is `/opt/openmvs/README.md` (a copy of
  `environment/openmvs_README.md`): the OpenMVS v2.4.0 source release built by the conda-forge recipe
  `openmvs-feedstock` at that build, and the other packages' sources on conda-forge under the names and versions in
  `PACKAGES.tsv`. The image runs these programs unmodified; nothing else in it links to them.

## Neutral names (agent-visible) and what they stand for

| neutral name | real counterpart |
|---|---|
| the cabin free-flyer | NASA's Astrobee free flyer |
| the module | the Japanese Experiment Module (Kibo) pressurised module of the International Space Station |
| the wide camera | the robot's navigation camera (NavCam) |
| the depth camera | the robot's hazard camera (HazCam), a time-of-flight depth camera |
| the on-board estimate | the robot's own localisation pose with its filter's health fields |
| `flight_a` | the dataset's sequence `iva_kibo_trans` |

## What the truth rests on

- The selection of key frames, scored pixels and scored moves as registered before it ran
  (`tests/review/SELECTION_RULE.json`), with its two amendments.
  - In that copy the words "the registered value" stand where the registration names an entry of the maintainers'
    table of numbers or a rule by its id, and the selection code is the maintainers' and does not ship: its sha256
    are listed there for the record.
- The reference twin inside the contract's caps (2000 convex pieces): a pixel or a move is scored only
  where this twin reproduces the record under the grader.
- Three rules added after the record was looked at, each before any twin or model was scored on the truth: returns
  that the rest of the flight's depth record does not confirm are no obstacles; a move whose outcome would change if
  returns that the record cannot decide were put back is not scored; no key frame gives more than
  8 moves of a class.
- A fourth rule (rule BC), added after the first model submission measured on this task had been scored on the
  truth and its lost moves looked at: the hazard term must not punish what the depth term accepts in full, so a
  scored move is removed when the record itself does not earn full credit on it once the returns its hazard check
  reads are moved along their own rays by the depth term's full band, nearer or farther (provided every class
  keeps its minimum and the reference twin still scores exactly 1.0). The rule was written after that look and
  after a first census at smaller shifts on a few returns; its shifts and what follows from its result were fixed
  before the census of all moves at those shifts was run. It reads the record only, and that submission was the
  only sample. It removed 16 of the 205 scored moves; 189 are scored.
- The reading of rule P in force (a scored moment is lost when one of its scored moves scores 0, not when one
  earns less than full credit) was chosen after that submission's scores under both readings were known: it is
  not a registration before numbers, and that submission was the only sample.
- Rule P counts only the scored moments that keep at least one scored move: 20 of the 24, because rule BC's removals
  took every scored move from 3 scored moments and 1 had none before. This count was not foreseen when rule BC was
  written and was ruled after the numbers were known; that submission was still the only sample. The moments without
  a scored move stay scored moments of the depth term, and the list of scored moves itself follows the branch of
  rule BC that was fixed before its census ran.

## Not exercised by the certification

- No twin of the certification makes the simulator warn while it compiles: the contract's sentence that such a
  warning is reported and does not zero the flight is implemented in the gate but was not exercised by any case.
