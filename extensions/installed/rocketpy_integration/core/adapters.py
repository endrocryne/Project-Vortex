"""
RocketPy <-> Vortex Parameter Adapters
========================================

Converts between Vortex JSON configuration format and RocketPy object API.

Vortex config layout (config_ideal.json / config_realistic.json):
    {
        "rocket": { dry_mass, propellant_mass, length, diameter, thrust_curve, ... },
        "environment": { gravity, air_density, temperature, wind_*, drag_*, ... },
        "simulation": { altimeter_error, velocity_sensor_error, simulate_ascent, ... },
        "rocketpy": { ... extended RocketPy-specific overrides ... }
    }

The "rocketpy" block is optional — when omitted, the adapter synthesises
sensible RocketPy objects from the standard Vortex parameters.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# We guard the import so the module can be loaded even when rocketpy is not
# installed (e.g.  for config validation or test mocking).
try:
    import rocketpy
    from rocketpy import Environment, Flight, Rocket, SolidMotor, GenericMotor
    HAS_ROCKETPY = True
except ImportError:
    HAS_ROCKETPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thrust_curve_to_tuples(curve: list) -> list:
    """Normalise Vortex thrust_curve (list of [t, F] pairs) to list of tuples."""
    return [(float(pt[0]), float(pt[1])) for pt in curve]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Vortex config -> RocketPy objects
# ---------------------------------------------------------------------------

class VortexConfigAdapter:
    """
    Builds RocketPy Environment, Motor, Rocket and Flight objects from a
    Vortex configuration dictionary.

    Usage:
        adapter = VortexConfigAdapter(config)
        env = adapter.build_environment()
        motor = adapter.build_motor()
        rocket = adapter.build_rocket(motor)
        flight = adapter.build_flight(rocket, env)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.rocket_cfg = self.config.get("rocket", {})
        self.env_cfg = self.config.get("environment", {})
        self.sim_cfg = self.config.get("simulation", {})
        self.rpy_cfg = self.config.get("rocketpy", {})

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def build_environment(self) -> "rocketpy.Environment":
        """Create a RocketPy Environment from Vortex config."""
        if not HAS_ROCKETPY:
            raise ImportError("rocketpy is not installed.  pip install rocketpy")

        env = Environment()

        # Elevation (metres above sea level) — Vortex sims default to 0
        elevation = _safe_float(self.rpy_cfg.get("elevation",
                                self.env_cfg.get("elevation", 0.0)))
        env.set_elevation(elevation)

        # Latitude / Longitude — used for Earth model + wind API
        latitude = _safe_float(self.rpy_cfg.get("latitude", 0.0))
        longitude = _safe_float(self.rpy_cfg.get("longitude", 0.0))
        env.set_location(latitude, longitude)

        # Atmospheric model — default to standard_atmosphere
        atm_model = self.rpy_cfg.get("atmospheric_model", "standard_atmosphere")
        if atm_model == "standard_atmosphere":
            env.set_atmospheric_model(type="standard_atmosphere")
        elif atm_model == "custom_atmosphere":
            # Build from Vortex constants
            pressure_at_sea = _safe_float(self.rpy_cfg.get("pressure_at_sea_level", 101325.0))
            temperature_k = _safe_float(self.env_cfg.get("temperature", 288.15))
            env.set_atmospheric_model(
                type="custom_atmosphere",
                pressure=pressure_at_sea,
                temperature=temperature_k,
            )
        elif atm_model in ("wyoming_sounding", "forecast", "reanalysis"):
            # Requires date — just set standard as fallback
            date_info = self.rpy_cfg.get("date")
            if date_info:
                env.set_date(tuple(date_info))
            env.set_atmospheric_model(type=atm_model)
        else:
            env.set_atmospheric_model(type="standard_atmosphere")

        # Wind — RocketPy supports wind profiles; map Vortex constant wind
        wind_speed = _safe_float(self.env_cfg.get("wind_speed", 0.0))
        wind_dir = _safe_float(self.env_cfg.get("wind_direction", 0.0))
        if wind_speed > 0:
            # RocketPy uses wind in x/y from meteorological convention
            wind_u = wind_speed * math.sin(math.radians(wind_dir))
            wind_v = wind_speed * math.cos(math.radians(wind_dir))
            try:
                env.set_atmospheric_model(
                    type="custom_atmosphere",
                    wind_u=wind_u,
                    wind_v=wind_v,
                )
            except Exception:
                pass  # Not all RocketPy versions support this cleanly

        # Gravity override if provided
        gravity = self.env_cfg.get("gravity")
        if gravity is not None:
            env.gravity = _safe_float(gravity, 9.81)

        return env

    # ------------------------------------------------------------------
    # Motor
    # ------------------------------------------------------------------

    def build_motor(self) -> Any:
        """Create a RocketPy motor from Vortex rocket config."""
        if not HAS_ROCKETPY:
            raise ImportError("rocketpy is not installed.  pip install rocketpy")

        # RocketPy-specific motor override
        if "motor_type" in self.rpy_cfg:
            return self._build_rocketpy_motor()

        # Default: build GenericMotor from Vortex thrust curve
        thrust_curve = self.rocket_cfg.get("thrust_curve", [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]])
        thrust_tuples = _thrust_curve_to_tuples(thrust_curve)
        burn_time = _safe_float(self.rocket_cfg.get("burn_time", thrust_tuples[-1][0]))
        propellant_mass = _safe_float(self.rocket_cfg.get("propellant_mass", 10.0))

        # Nozzle / grain geometry — estimate from diameter if not supplied
        diameter = _safe_float(self.rocket_cfg.get("diameter", 0.3))
        nozzle_radius = _safe_float(self.rpy_cfg.get("nozzle_radius", diameter / 2 * 0.4))
        throat_radius = _safe_float(self.rpy_cfg.get("throat_radius", nozzle_radius * 0.4))

        # Dry-mass of motor casing
        casing_mass = _safe_float(self.rocket_cfg.get("ascent_motor_casing_mass",
                                  self.rpy_cfg.get("motor_dry_mass", 2.0)))

        # Use GenericMotor for maximum compatibility — it accepts a thrust curve
        # and total impulse directly without grain geometry.
        motor = GenericMotor(
            thrust_source=thrust_tuples,
            burn_time=burn_time,
            propellant_initial_mass=propellant_mass,
            dry_mass=casing_mass,
            dry_inertia=(0.05, 0.05, 0.01),  # Approximate Ixx, Iyy, Izz
            nozzle_radius=nozzle_radius,
            center_of_dry_mass_position=_safe_float(self.rpy_cfg.get("motor_cg_position", 0.0)),
            nozzle_position=_safe_float(self.rpy_cfg.get("nozzle_position", 0.0)),
        )

        return motor

    def _build_rocketpy_motor(self) -> Any:
        """Build motor from explicit rocketpy config block."""
        motor_type = self.rpy_cfg.get("motor_type", "solid")
        rp = self.rpy_cfg
        thrust_curve = rp.get("thrust_source", self.rocket_cfg.get("thrust_curve", [[0, 0], [3, 1000], [3.1, 0]]))
        thrust_tuples = _thrust_curve_to_tuples(thrust_curve)
        burn_time = _safe_float(rp.get("burn_time", self.rocket_cfg.get("burn_time", 3.0)))
        propellant_mass = _safe_float(rp.get("propellant_initial_mass",
                                      self.rocket_cfg.get("propellant_mass", 10.0)))

        if motor_type == "solid":
            grain_count = int(rp.get("grain_number", 1))
            grain_outer = _safe_float(rp.get("grain_outer_radius", 0.05))
            grain_inner = _safe_float(rp.get("grain_initial_inner_radius", 0.02))
            grain_height = _safe_float(rp.get("grain_initial_height", 0.15))
            grain_sep = _safe_float(rp.get("grain_separation", 0.005))
            nozzle_radius = _safe_float(rp.get("nozzle_radius", 0.04))
            throat_radius = _safe_float(rp.get("throat_radius", 0.02))
            casing_mass = _safe_float(rp.get("motor_dry_mass",
                                      self.rocket_cfg.get("ascent_motor_casing_mass", 2.0)))

            motor = SolidMotor(
                thrust_source=thrust_tuples,
                burn_time=burn_time,
                dry_mass=casing_mass,
                dry_inertia=(
                    _safe_float(rp.get("Ixx", 0.05)),
                    _safe_float(rp.get("Iyy", 0.05)),
                    _safe_float(rp.get("Izz", 0.01)),
                ),
                center_of_dry_mass_position=_safe_float(rp.get("motor_cg_position", 0.0)),
                nozzle_radius=nozzle_radius,
                throat_radius=throat_radius,
                grain_number=grain_count,
                grain_density=_safe_float(rp.get("grain_density", 1700.0)),
                grain_outer_radius=grain_outer,
                grain_initial_inner_radius=grain_inner,
                grain_initial_height=grain_height,
                grain_separation=grain_sep,
                nozzle_position=_safe_float(rp.get("nozzle_position", 0.0)),
            )
            return motor

        # Fallback to GenericMotor for hybrid / liquid / unknown
        nozzle_radius = _safe_float(rp.get("nozzle_radius", 0.04))
        casing_mass = _safe_float(rp.get("motor_dry_mass",
                                  self.rocket_cfg.get("ascent_motor_casing_mass", 2.0)))
        motor = GenericMotor(
            thrust_source=thrust_tuples,
            burn_time=burn_time,
            propellant_initial_mass=propellant_mass,
            dry_mass=casing_mass,
            dry_inertia=(0.05, 0.05, 0.01),
            nozzle_radius=nozzle_radius,
            center_of_dry_mass_position=_safe_float(rp.get("motor_cg_position", 0.0)),
            nozzle_position=_safe_float(rp.get("nozzle_position", 0.0)),
        )
        return motor

    # ------------------------------------------------------------------
    # Rocket
    # ------------------------------------------------------------------

    def build_rocket(self, motor: Any) -> "rocketpy.Rocket":
        """Create a RocketPy Rocket object and attach the motor."""
        if not HAS_ROCKETPY:
            raise ImportError("rocketpy is not installed.  pip install rocketpy")

        mass = _safe_float(self.rocket_cfg.get("dry_mass", 50.0))
        length = _safe_float(self.rocket_cfg.get("length", 5.0))
        diameter = _safe_float(self.rocket_cfg.get("diameter", 0.3))
        radius = diameter / 2.0

        # Inertia — estimate from solid cylinder if not provided
        rp = self.rpy_cfg
        Ixx = _safe_float(rp.get("Ixx", mass * (3 * radius ** 2 + length ** 2) / 12.0))
        Iyy = _safe_float(rp.get("Iyy", Ixx))
        Izz = _safe_float(rp.get("Izz", mass * radius ** 2 / 2.0))

        rocket = Rocket(
            mass=mass,
            radius=radius,
            inertia=(Ixx, Iyy, Izz),
            power_off_drag=_safe_float(self.env_cfg.get("drag_coefficient", 0.5)),
            power_on_drag=_safe_float(rp.get("power_on_drag",
                                      self.env_cfg.get("drag_coefficient", 0.5))),
            center_of_mass_without_motor=_safe_float(rp.get("center_of_mass_without_motor",
                                                     length * 0.5)),
        )

        # Add motor at nozzle position
        motor_position = _safe_float(rp.get("motor_position", 0.0))
        rocket.add_motor(motor, position=motor_position)

        # Nose cone
        nose_length = _safe_float(rp.get("nose_length", length * 0.2))
        nose_kind = rp.get("nose_kind", "Von Karman")
        nose_position = _safe_float(rp.get("nose_position", length))
        try:
            rocket.add_nose(length=nose_length, kind=nose_kind, position=nose_position)
        except Exception:
            pass  # Some RocketPy versions have different API

        # Fins — optional, from rocketpy config block
        if "fins" in rp:
            fins_cfg = rp["fins"]
            n_fins = int(fins_cfg.get("number", 4))
            root_chord = _safe_float(fins_cfg.get("root_chord", 0.15))
            tip_chord = _safe_float(fins_cfg.get("tip_chord", 0.05))
            span = _safe_float(fins_cfg.get("span", 0.1))
            fin_position = _safe_float(fins_cfg.get("position", 0.1))
            try:
                rocket.add_trapezoidal_fins(
                    n=n_fins,
                    root_chord=root_chord,
                    tip_chord=tip_chord,
                    span=span,
                    position=fin_position,
                )
            except Exception:
                pass

        # Parachutes — optional
        if "parachutes" in rp:
            for chute_cfg in rp["parachutes"]:
                chute_name = chute_cfg.get("name", "Main")
                cd_s = _safe_float(chute_cfg.get("cd_s", 10.0))
                trigger_alt = _safe_float(chute_cfg.get("trigger_altitude", 500.0))
                trigger_type = chute_cfg.get("trigger", "apogee")

                if trigger_type == "apogee":
                    trigger_fn = "apogee"
                elif trigger_type == "altitude":
                    trigger_fn = trigger_alt
                else:
                    trigger_fn = "apogee"

                try:
                    rocket.add_parachute(
                        name=chute_name,
                        cd_s=cd_s,
                        trigger=trigger_fn,
                    )
                except Exception:
                    pass

        # Rail buttons — optional
        if "rail_buttons" in rp:
            rb = rp["rail_buttons"]
            try:
                rocket.add_rail_buttons(
                    upper_button_position=_safe_float(rb.get("upper_position", length * 0.7)),
                    lower_button_position=_safe_float(rb.get("lower_position", length * 0.3)),
                )
            except Exception:
                pass

        return rocket

    # ------------------------------------------------------------------
    # Flight
    # ------------------------------------------------------------------

    def build_flight(self, rocket: "rocketpy.Rocket", env: "rocketpy.Environment",
                     inclination: float = 90.0, heading: float = 0.0,
                     rail_length: float = 5.0,
                     initial_solution: Optional[list] = None,
                     terminate_on_apogee: bool = False,
                     max_time: float = 600.0) -> "rocketpy.Flight":
        """Run a RocketPy flight simulation and return the Flight object."""
        if not HAS_ROCKETPY:
            raise ImportError("rocketpy is not installed.  pip install rocketpy")

        rp = self.rpy_cfg
        inclination = _safe_float(rp.get("inclination", inclination))
        heading = _safe_float(rp.get("heading", heading))
        rail_length = _safe_float(rp.get("rail_length", rail_length))
        max_time = _safe_float(rp.get("max_time", max_time))
        terminate_on_apogee = rp.get("terminate_on_apogee", terminate_on_apogee)

        flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
            terminate_on_apogee=terminate_on_apogee,
            max_time=max_time,
        )

        return flight

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def build_all(self, inclination: float = 90.0, heading: float = 0.0,
                  rail_length: float = 5.0,
                  terminate_on_apogee: bool = False,
                  max_time: float = 600.0) -> Tuple[Any, Any, Any, Any]:
        """
        Convenience: build env, motor, rocket, and flight in one call.

        Returns:
            (environment, motor, rocket, flight)
        """
        env = self.build_environment()
        motor = self.build_motor()
        rocket = self.build_rocket(motor)
        flight = self.build_flight(rocket, env,
                                   inclination=inclination,
                                   heading=heading,
                                   rail_length=rail_length,
                                   terminate_on_apogee=terminate_on_apogee,
                                   max_time=max_time)
        return env, motor, rocket, flight


# ---------------------------------------------------------------------------
# State vector conversion
# ---------------------------------------------------------------------------

class StateConverter:
    """Convert between RocketPy flight state and Vortex 14-element state vector.

    Vortex state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, mass]
    """

    @staticmethod
    def rocketpy_flight_to_vortex_states(flight: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract full trajectory from a RocketPy Flight object as Vortex-format
        arrays.

        Returns:
            (times, states) where states has shape (14, N)
        """
        try:
            times = np.array(flight.time)
        except Exception:
            times = np.array(flight.solution[:, 0]) if hasattr(flight, 'solution') else np.array([0.0])

        n = len(times)
        states = np.zeros((14, n))

        # Position: RocketPy x, y, z (inertial frame, metres)
        try:
            states[0, :] = np.array(flight.x(times))
            states[1, :] = np.array(flight.y(times))
            states[2, :] = np.array(flight.z(times))
        except Exception:
            try:
                sol = np.array(flight.solution)
                states[0, :] = sol[:, 1]
                states[1, :] = sol[:, 2]
                states[2, :] = sol[:, 3]
            except Exception:
                pass

        # Velocity
        try:
            states[3, :] = np.array(flight.vx(times))
            states[4, :] = np.array(flight.vy(times))
            states[5, :] = np.array(flight.vz(times))
        except Exception:
            try:
                sol = np.array(flight.solution)
                states[3, :] = sol[:, 4]
                states[4, :] = sol[:, 5]
                states[5, :] = sol[:, 6]
            except Exception:
                pass

        # Quaternion — RocketPy stores Euler parameters (e0, e1, e2, e3)
        # which map to (qw, qx, qy, qz) in Vortex's scalar-first convention
        try:
            states[6, :] = np.array(flight.e0(times))
            states[7, :] = np.array(flight.e1(times))
            states[8, :] = np.array(flight.e2(times))
            states[9, :] = np.array(flight.e3(times))
        except Exception:
            # Fallback: generate quaternion from velocity vector (nose-first)
            for i in range(n):
                vel = states[3:6, i]
                speed = np.linalg.norm(vel)
                if speed > 0.1:
                    # Align body z-axis with velocity vector
                    direction = vel / speed
                    # Quaternion that rotates [0,0,1] to direction
                    z_axis = np.array([0.0, 0.0, 1.0])
                    cross = np.cross(z_axis, direction)
                    dot = np.dot(z_axis, direction)
                    s = np.sqrt(2.0 * (1.0 + dot))
                    if s > 1e-6:
                        states[6, i] = s / 2.0
                        states[7, i] = cross[0] / s
                        states[8, i] = cross[1] / s
                        states[9, i] = cross[2] / s
                    else:
                        states[6, i] = 1.0
                else:
                    states[6, i] = 1.0  # Identity quaternion

        # Angular velocity
        try:
            states[10, :] = np.array(flight.w1(times))
            states[11, :] = np.array(flight.w2(times))
            states[12, :] = np.array(flight.w3(times))
        except Exception:
            pass  # Zeros (already initialised)

        # Mass — RocketPy tracks this through motor burn
        try:
            # RocketPy stores rocket mass as function of time
            states[13, :] = np.array(flight.rocket.total_mass(times))
        except Exception:
            try:
                dry = float(flight.rocket.mass)
                prop = float(flight.rocket.motor.propellant_initial_mass)
                bt = float(flight.rocket.motor.burn_time)
                for i in range(n):
                    t = times[i]
                    remaining = max(0.0, prop * (1.0 - t / bt)) if t < bt else 0.0
                    states[13, i] = dry + remaining
            except Exception:
                states[13, :] = 50.0  # Fallback constant mass

        return times, states

    @staticmethod
    def vortex_state_to_initial_solution(state: np.ndarray, t0: float = 0.0) -> list:
        """
        Convert a Vortex 14-element state vector to a RocketPy initial_solution
        list suitable for Flight(..., initial_solution=...).

        RocketPy initial_solution: [t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        (14 elements — no mass, mass is handled by motor)
        """
        sol = [
            t0,
            float(state[0]),   # x
            float(state[1]),   # y
            float(state[2]),   # z
            float(state[3]),   # vx
            float(state[4]),   # vy
            float(state[5]),   # vz
            float(state[6]),   # e0 / qw
            float(state[7]),   # e1 / qx
            float(state[8]),   # e2 / qy
            float(state[9]),   # e3 / qz
            float(state[10]),  # w1
            float(state[11]),  # w2
            float(state[12]),  # w3
        ]
        return sol


# ---------------------------------------------------------------------------
# Reverse adapter: RocketPy Flight -> Vortex-style config dict
# ---------------------------------------------------------------------------

def flight_to_vortex_config(flight: Any) -> Dict[str, Any]:
    """
    Extract a Vortex-compatible config dict from a completed RocketPy Flight.
    Useful for logging and recording what parameters were actually used.
    """
    result: Dict[str, Any] = {
        "rocket": {},
        "environment": {},
        "simulation": {},
        "rocketpy": {"source": "rocketpy_flight_extraction"},
    }

    try:
        r = flight.rocket
        result["rocket"]["dry_mass"] = float(r.mass)
        result["rocket"]["diameter"] = float(r.radius * 2)
        result["rocket"]["length"] = 0.0  # RocketPy doesn't store length directly

        m = r.motor
        result["rocket"]["propellant_mass"] = float(m.propellant_initial_mass)
        result["rocket"]["burn_time"] = float(m.burn_time)
        result["rocketpy"]["motor_type"] = type(m).__name__
    except Exception:
        pass

    try:
        e = flight.env
        result["environment"]["gravity"] = float(e.gravity(0)) if callable(e.gravity) else float(e.gravity)
    except Exception:
        result["environment"]["gravity"] = 9.81

    result["simulation"]["backend"] = "rocketpy"
    return result
