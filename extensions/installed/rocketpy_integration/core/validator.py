"""
Config Validator for RocketPy Integration
===========================================

Validates a Vortex configuration dict for compatibility with RocketPy,
identifies unsupported features, and suggests parameter adjustments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ValidationMessage:
    """A single validation finding."""

    LEVEL_INFO = "info"
    LEVEL_WARNING = "warning"
    LEVEL_ERROR = "error"

    def __init__(self, level: str, code: str, message: str,
                 suggestion: str = ""):
        self.level = level
        self.code = code
        self.message = message
        self.suggestion = suggestion

    def __repr__(self) -> str:
        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(self.level, "?")
        text = f"{prefix} [{self.code}] {self.message}"
        if self.suggestion:
            text += f"  → {self.suggestion}"
        return text

    def to_dict(self) -> Dict[str, str]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[ValidationMessage]]:
    """
    Validate a Vortex config for RocketPy compatibility.

    Returns:
        (is_valid, list_of_messages)

    ``is_valid`` is False only if there are ERROR-level messages that would
    prevent simulation execution.  Warnings and info messages are advisory.
    """
    msgs: List[ValidationMessage] = []

    rocket = config.get("rocket", {})
    env = config.get("environment", {})
    sim = config.get("simulation", {})
    rpy = config.get("rocketpy", {})

    # --- Required fields ---
    if not rocket:
        msgs.append(ValidationMessage(
            "error", "MISSING_ROCKET",
            "No 'rocket' block in configuration.",
        ))
    else:
        if "dry_mass" not in rocket:
            msgs.append(ValidationMessage(
                "error", "MISSING_DRY_MASS",
                "rocket.dry_mass is required.",
            ))
        if "propellant_mass" not in rocket:
            msgs.append(ValidationMessage(
                "error", "MISSING_PROP_MASS",
                "rocket.propellant_mass is required.",
            ))

        # Thrust curve
        curve = rocket.get("thrust_curve", [])
        if not curve or len(curve) < 2:
            msgs.append(ValidationMessage(
                "error", "BAD_THRUST_CURVE",
                "thrust_curve must have at least 2 data points.",
            ))

        # Geometry
        if "diameter" not in rocket:
            msgs.append(ValidationMessage(
                "warning", "NO_DIAMETER",
                "rocket.diameter not set; defaulting to 0.3 m.",
                "Add 'diameter' to rocket config for accurate aerodynamics.",
            ))

        if "length" not in rocket:
            msgs.append(ValidationMessage(
                "warning", "NO_LENGTH",
                "rocket.length not set; defaulting to 5.0 m.",
                "Add 'length' to rocket config for accurate inertia estimation.",
            ))

    # --- TVC — not natively supported in RocketPy ---
    tvc_keys = [k for k in rocket if k.startswith("tvc_")]
    if tvc_keys:
        msgs.append(ValidationMessage(
            "warning", "TVC_NOT_SUPPORTED",
            "TVC (Thrust Vector Control) is not natively supported by RocketPy. "
            f"Keys found: {', '.join(tvc_keys[:5])}",
            "TVC parameters will be ignored. Results will differ from native Vortex simulation "
            "for controlled flight. Consider using Vortex Native backend for TVC scenarios.",
        ))

    # --- Wind model ---
    wind_model = env.get("wind_model", "constant")
    if wind_model not in ("constant", "altitude_varying", "gusts"):
        msgs.append(ValidationMessage(
            "warning", "UNKNOWN_WIND_MODEL",
            f"Wind model '{wind_model}' may not map cleanly to RocketPy.",
            "Use 'constant' or configure wind in the rocketpy config block.",
        ))

    # --- Ascent simulation ---
    if sim.get("simulate_ascent"):
        msgs.append(ValidationMessage(
            "info", "ASCENT_ALWAYS",
            "RocketPy always simulates the full flight (ascent + descent). "
            "The 'simulate_ascent' flag is redundant.",
        ))

    # --- RocketPy-specific validation ---
    if "motor_type" in rpy:
        motor_type = rpy["motor_type"]
        if motor_type == "solid":
            needed = ["grain_number", "grain_outer_radius",
                      "grain_initial_inner_radius", "grain_initial_height"]
            missing = [k for k in needed if k not in rpy]
            if missing:
                msgs.append(ValidationMessage(
                    "warning", "SOLID_MOTOR_PARAMS",
                    f"Solid motor type specified but missing grain parameters: {', '.join(missing)}",
                    "Defaults will be used. For accurate results, provide grain geometry.",
                ))

    # Check for rocketpy-only advanced options
    if "parachutes" in rpy:
        for i, chute in enumerate(rpy["parachutes"]):
            if "cd_s" not in chute:
                msgs.append(ValidationMessage(
                    "warning", "PARACHUTE_NO_CDS",
                    f"Parachute #{i+1} missing 'cd_s' (drag coefficient × area).",
                    "Add 'cd_s' to each parachute definition.",
                ))

    if "fins" in rpy:
        fins = rpy["fins"]
        needed_fin = ["root_chord", "span"]
        missing_fin = [k for k in needed_fin if k not in fins]
        if missing_fin:
            msgs.append(ValidationMessage(
                "warning", "FINS_INCOMPLETE",
                f"Fin definition missing: {', '.join(missing_fin)}",
                "Provide complete fin geometry for accurate stability calculation.",
            ))

    # --- Mass sanity checks ---
    dry = rocket.get("dry_mass", 0)
    prop = rocket.get("propellant_mass", 0)
    if dry > 0 and prop > 0:
        if prop > dry * 2:
            msgs.append(ValidationMessage(
                "warning", "HIGH_MASS_RATIO",
                f"Propellant mass ({prop} kg) is more than 2× dry mass ({dry} kg). "
                "This is unusual for most rockets.",
                "Double-check mass values.",
            ))

    # --- Drag coefficient ---
    cd = env.get("drag_coefficient", 0.5)
    if cd > 2.0:
        msgs.append(ValidationMessage(
            "warning", "HIGH_DRAG",
            f"Drag coefficient ({cd}) seems very high.",
            "Typical values are 0.3–0.8 for rockets.",
        ))

    # --- Determine overall validity ---
    has_errors = any(m.level == "error" for m in msgs)
    return (not has_errors), msgs


def format_validation_report(messages: List[ValidationMessage]) -> str:
    """Format validation messages as a human-readable text report."""
    if not messages:
        return "✅ Configuration is valid for RocketPy simulation."

    lines = ["RocketPy Configuration Validation Report",
             "=" * 42, ""]

    errors = [m for m in messages if m.level == "error"]
    warns = [m for m in messages if m.level == "warning"]
    infos = [m for m in messages if m.level == "info"]

    if errors:
        lines.append("ERRORS (must fix):")
        for m in errors:
            lines.append(f"  ❌ [{m.code}] {m.message}")
            if m.suggestion:
                lines.append(f"     → {m.suggestion}")
        lines.append("")

    if warns:
        lines.append("WARNINGS:")
        for m in warns:
            lines.append(f"  ⚠️ [{m.code}] {m.message}")
            if m.suggestion:
                lines.append(f"     → {m.suggestion}")
        lines.append("")

    if infos:
        lines.append("INFO:")
        for m in infos:
            lines.append(f"  ℹ️ [{m.code}] {m.message}")
            if m.suggestion:
                lines.append(f"     → {m.suggestion}")
        lines.append("")

    summary = "PASS" if not errors else "FAIL"
    lines.append(f"Summary: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info — {summary}")
    return "\n".join(lines)
