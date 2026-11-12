"""Load and transport a narrow rigid snack box using a whole-body side grasp."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transport_core", HERE / "loaded_tray_transport.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
BOX_XML = (
    Path(core.robocasa.__file__).resolve().parent
    / "models/assets/objects/lightwheel/tupperware/Tupperware036/model.xml"
)


class LoadedBoxKitchen(core.LoadedTrayKitchen):
    def _get_obj_cfgs(self):
        configs = super()._get_obj_cfgs()
        configs[0]["obj_groups"] = str(BOX_XML)
        # Registry scale 0.8 times 0.65: approximately 94 x 68 x 46 mm.
        configs[0]["object_scale"] = 0.65
        configs[1]["object_scale"] = 0.50
        configs[1]["max_size"] = (0.038, 0.038, 0.045)
        return configs


class LoadedBoxController(core.LoadedTrayController):
    def loading_offset(self, payload_half):
        # Centre the load near the grasp and combined centre of mass.
        return np.zeros(3)

    def carrier_grasp(self):
        m = self.monitor
        local = (m.local_low + m.local_high) / 2
        grasp = m.position("tray") + m.rotation("tray") @ local
        grasp[2] -= 0.004
        # Open fingers span the whole short dimension of the box. Both pads
        # squeeze opposite walls, rather than cantilevering from one thin rim.
        short_axis = m.rotation("tray")[:, 1]
        yaw = np.arctan2(short_axis[1], short_axis[0])
        down = Rotation.from_euler("xyz", [np.pi, 0, yaw]).as_matrix()
        return grasp, down, 0

    def payload_release_clearance(self):
        return 0.020

    def retreat_offset(self):
        return np.array([0, -0.18, 0.22])

    def release_into_carrier(self, release, over_carrier, down):
        # Open only enough to let go inside the narrow mouth; fully opening
        # before withdrawing would strike the container's opposite walls.
        self.stage = "release_payload"
        self.hold(release, down, -1, 5)
        self.hold(release, down, 0, 20)
        self.move("clear_payload", over_carrier, down, 0)
        self.hold(over_carrier, down, -1, 25)


if __name__ == "__main__":
    raise SystemExit(core.run_trial(
        core.parse_args(), env_class=LoadedBoxKitchen,
        controller_class=LoadedBoxController, carrier_xml=BOX_XML,
        carrier_scale=0.65, experiment="loaded_box", entry_script=Path(__file__),
    ))
