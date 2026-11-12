"""Isolate loaded-container transport; contents are placed during scene setup."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("box_test", HERE / "loaded_box_transport.py")
box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box)


class PreloadedBoxKitchen(box.LoadedBoxKitchen):
    def _get_obj_cfgs(self):
        configs = super()._get_obj_cfgs()
        # A flat-bottomed package is a better calibrated load than a small
        # rounded fruit, which can roll to the mouth during initialization.
        # Fix the geometry for the mass sweep: a 23 x 59 x 36 mm package.
        # At 200 g the bounding volume corresponds to approximately 4.2 g/cm3,
        # so this represents dense ballast, not an ordinary snack's material.
        # A category sample with incompatible size limits can loop forever in
        # RoboCasa's rejection sampler. Seed variation remains in scene setup.
        configs[1]["obj_groups"] = str(
            Path(box.core.robocasa.__file__).resolve().parent
            / "models/assets/objects/objaverse/boxed_food/boxed_food_0/model.xml"
        )
        configs[1]["object_scale"] = 0.45
        configs[1]["max_size"] = (0.030, 0.065, 0.040)
        configs[1].pop("reset_region", None)
        configs[1]["placement"] = dict(
            size=(0.002, 0.002), rotation=0,
            ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            sample_args=dict(reference="tray", on_top=True),
        )
        return configs

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        # Define the initial scene before _reset_internal applies placements.
        # Keep the sampler's XY jitter and orientation; start the package just
        # above the box floor instead of dropping it from above the rim. This
        # changes only the reset specification, never live qpos during a trial.
        carrier_pos, _, carrier = self.object_placements["tray"]
        position, quaternion, payload = self.object_placements["obj"]
        position = list(position)
        position[2] = (carrier_pos[2] + carrier.bottom_offset[-1]
                       - payload.bottom_offset[-1] + 0.008)
        self.object_placements["obj"] = (position, quaternion, payload)


class PreloadedBoxMonitor(box.core.TransportMonitor):
    def __init__(self, env):
        super().__init__(env)
        # A preloaded trial must retain its contents throughout approach and
        # grasp too, not only after the controller starts lifting the carrier.
        self.armed = True


class PreloadedBoxController(box.LoadedBoxController):
    def load_payload(self, orientation):
        # The ordinary scene sampler initializes contents before the trial.
        # No robot loading ability is claimed by this component test.
        m = self.monitor
        if not m.contained() or not self.env.check_contact(self.env.objects["obj"], self.env.objects["tray"]):
            raise RuntimeError("Scene setup did not produce a valid preloaded container")
        m.loaded_verified = True


if __name__ == "__main__":
    raise SystemExit(box.core.run_trial(
        box.core.parse_args(), env_class=PreloadedBoxKitchen,
        controller_class=PreloadedBoxController, carrier_xml=box.BOX_XML,
        carrier_scale=0.65, experiment="preloaded_box", entry_script=Path(__file__),
        monitor_class=PreloadedBoxMonitor,
        extra_sources=(HERE / "loaded_box_transport.py",),
    ))
