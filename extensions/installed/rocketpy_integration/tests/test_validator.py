"""
Tests for config validator.

Exercises every validation rule with crafted configs.
No simulations are executed.
"""

import os
import sys
import unittest

_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

from core.validator import validate_config, format_validation_report, ValidationMessage


def _base_config():
    """Return a minimal valid config."""
    return {
        "rocket": {
            "dry_mass": 50.0,
            "propellant_mass": 10.0,
            "diameter": 0.3,
            "length": 5.0,
            "thrust_curve": [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]],
            "burn_time": 3.0,
            "drag_coefficient": 0.5,
        },
        "environment": {
            "gravity": 9.81,
            "air_density": 1.225,
        },
        "simulation": {},
    }


class TestValidateConfig(unittest.TestCase):

    def test_valid_config_no_errors(self):
        msgs = validate_config(_base_config())
        errors = [m for m in msgs if m.level == "error"]
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")

    def test_missing_dry_mass(self):
        cfg = _base_config()
        del cfg["rocket"]["dry_mass"]
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("MISSING_DRY_MASS", codes)

    def test_missing_thrust_curve(self):
        cfg = _base_config()
        del cfg["rocket"]["thrust_curve"]
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("MISSING_THRUST_CURVE", codes)

    def test_missing_burn_time(self):
        cfg = _base_config()
        del cfg["rocket"]["burn_time"]
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("MISSING_BURN_TIME", codes)

    def test_tvc_warning(self):
        cfg = _base_config()
        cfg["simulation"]["tvc_enabled"] = True
        msgs = validate_config(cfg)
        tvc_msgs = [m for m in msgs if m.code == "TVC_LIMITED"]
        self.assertTrue(len(tvc_msgs) >= 1)
        self.assertEqual(tvc_msgs[0].level, "warning")

    def test_wind_model_info(self):
        cfg = _base_config()
        cfg["rocketpy"] = {"atmospheric_model": "Forecast"}
        msgs = validate_config(cfg)
        wind_msgs = [m for m in msgs if m.code == "WIND_MODEL"]
        self.assertTrue(len(wind_msgs) >= 1)

    def test_missing_rocket_section(self):
        cfg = {"environment": {}, "simulation": {}}
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        # Should at minimum flag missing required fields
        self.assertTrue(any("MISSING" in c for c in codes))

    def test_high_drag_warning(self):
        cfg = _base_config()
        cfg["rocket"]["drag_coefficient"] = 1.5
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("HIGH_CD", codes)

    def test_low_mass_ratio_warning(self):
        cfg = _base_config()
        cfg["rocket"]["dry_mass"] = 1.0
        cfg["rocket"]["propellant_mass"] = 100.0
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("LOW_MASS_RATIO", codes)

    def test_solid_motor_grain_validation(self):
        cfg = _base_config()
        cfg["rocketpy"] = {
            "motor_type": "solid",
            "grains": [],  # empty grains list
        }
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("SOLID_NO_GRAINS", codes)

    def test_parachute_missing_cd_s(self):
        cfg = _base_config()
        cfg["rocketpy"] = {
            "parachutes": [
                {"name": "Drogue"},  # missing cd_s
            ],
        }
        msgs = validate_config(cfg)
        codes = [m.code for m in msgs]
        self.assertIn("PARACHUTE_NO_CDS", codes)


class TestValidationMessage(unittest.TestCase):

    def test_repr(self):
        m = ValidationMessage("error", "TEST_CODE", "Something went wrong", "Fix it")
        s = repr(m)
        self.assertIn("error", s)
        self.assertIn("TEST_CODE", s)

    def test_str(self):
        m = ValidationMessage("warning", "W1", "Watch out")
        self.assertIn("warning", str(m).lower())


class TestFormatReport(unittest.TestCase):

    def test_empty_list(self):
        report = format_validation_report([])
        self.assertIn("pass", report.lower())

    def test_with_messages(self):
        msgs = [
            ValidationMessage("error", "E1", "Bad config"),
            ValidationMessage("warning", "W1", "Minor issue"),
        ]
        report = format_validation_report(msgs)
        self.assertIn("E1", report)
        self.assertIn("W1", report)


if __name__ == "__main__":
    unittest.main()
