# NOTICE (grader-side; never staged into the agent's home)

The robot of this task is the OpenArm 2.0 two-arm pair of Enactic, Inc. on its pedestal, exactly as the maker publishes
it: https://github.com/enactic/openarm_mujoco, folder v2/, commit ed2046f30981398fc610688f08beac0ebd08f972 (Apache
License 2.0). The grader's simulated unit (code/shared/openarm_unit/) carries the maker's model files unchanged under
base/ (each pinned by sha256 in base/MANIFEST.json, with the repository's licence in base/LICENSE); the unit files it
writes at run time are modified versions of them, as code/shared/openarm_unit/NOTICE states. The ZED Mini mesh of the
maker's pedestal file is Stereolabs' (Apache-2.0, https://github.com/stereolabs/zed-ros2-description).

The givens under environment/setup/payload/given/maker/ are byte copies of the maker's files at the commits that
given/maker/NOTICE.txt names, each with its repository's licence. The joint modules' data pages, the integrator's
sheet, the delivery-record format, the site's sheets, the acceptance specification and the controller's interface are
written for this task from the vendor's published figures and the task's own constants.

What is hidden here: the three units of the acceptance (hidden/registry: each unit's drawn state, its site and its
shift), the frozen set with its band ends and anchor (hidden/frozen_set.json), the oracle token's digest, the shared
unit's law parameters and the grader. None of it reaches the agent.
