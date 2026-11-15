import copy
import unittest
from fpga_bench.config import DEFAULTS
from fpga_bench import hardware


class HLSSettingsTests(unittest.TestCase):
    def test_layer_defaults_follow_requested_settings_and_explicit_overrides_win(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["hls"].update(
            reuse_factor=8,
            strategy="Resource",
            layer_overrides={"dense": {"ReuseFactor": 4}},
        )
        generated = {
            "Model": {"ReuseFactor": 1},
            "LayerName": {
                "dense": {"ReuseFactor": 1, "Strategy": "Latency"},
                "norm": {"ReuseFactor": 1},
            },
        }
        self.assertTrue(
            hasattr(hardware, "configure_hls"), "Layer configuration helper is missing"
        )
        actual = hardware.configure_hls(generated, cfg)
        self.assertEqual(actual["LayerName"]["dense"]["ReuseFactor"], 4)
        self.assertEqual(actual["LayerName"]["norm"]["ReuseFactor"], 8)
        self.assertEqual(actual["LayerName"]["dense"]["Strategy"], "Resource")
        self.assertEqual(generated["LayerName"]["norm"]["ReuseFactor"], 1)

    def test_clock_is_loaded_before_synthesis(self):
        script = hardware.implementation_tcl(True, 5)
        self.assertIn("read_xdc", script)
        self.assertLess(script.index("read_xdc"), script.index("synth_design"))
        self.assertIn("create_clock", script)
        self.assertIn("general.maxThreads", script)
