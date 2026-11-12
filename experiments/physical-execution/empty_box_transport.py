"""Isolated empty-container grasp and transport check; not a loaded test."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("box_test", HERE / "loaded_box_transport.py")
box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box)


class EmptyBoxMonitor(box.core.TransportMonitor):
    require_payload = False

    def metrics(self):
        result = super().metrics()
        result["trial_kind"] = "empty_container_grasp_and_transport"
        result["payload_contained"] = None
        result["payload_tray_contact"] = None
        result["empty_transport_verified"] = result.pop("loaded_transport_verified")
        result.pop("loaded_verified")
        return result


class EmptyBoxController(box.LoadedBoxController):
    def load_payload(self, orientation):
        # Deliberately leave the payload on the source counter in this test.
        pass


if __name__ == "__main__":
    args = box.core.parse_args()
    if args.payload_mass_g is not None:
        raise SystemExit("This is an empty-container check; omit --payload-mass-g")
    raise SystemExit(box.core.run_trial(
        args, env_class=box.LoadedBoxKitchen, controller_class=EmptyBoxController,
        carrier_xml=box.BOX_XML, carrier_scale=0.65, experiment="empty_box",
        entry_script=Path(__file__), monitor_class=EmptyBoxMonitor,
        extra_sources=(HERE / "loaded_box_transport.py",),
    ))
