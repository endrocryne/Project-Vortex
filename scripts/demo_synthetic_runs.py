"""Synthetic HexaVisual demo trajectory generator.

This generator does not run the project simulator. It builds synthetic demo
trajectories around the attached optimization run while respecting the project
behavioral model:

- ascent follows the reference run vertically
- freefall never inverts the vehicle and approaches terminal velocity
- descent uses a fixed-duration solid-motor suicide burn
- TVC only changes attitude while the motor is burning
- fault activity stops at ignition
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TRAJECTORY = ROOT / "results" / "optimization_20260222_120325" / "trajectory.csv"
OUTPUT_DIR = ROOT / "results" / "demo_runs"

DT = 0.02
GRAVITY = 9.81
BASE_TOTAL_MASS = 1.41
BASE_ASCENT_END_MASS = 1.347
BASE_DRY_MASS = 1.219
BASE_DIAMETER = 0.07874
BASE_THRUST_AVG = 89.0
BASE_DRAG_COEFFICIENT = 0.48
BASE_AIR_DENSITY = 1.18
BASE_PROPELLANT_MASS = round(BASE_TOTAL_MASS - BASE_DRY_MASS, 3)
ASCENT_WOBBLE_AMPLITUDE_DEG = 2.8
DESCENT_TILT_LIMIT_DEG = 30.0
TVC_SLEW_RATE_DEG_PER_S = 11.0
LATERAL_ACCEL_PER_DEG = 0.34
SUCCESS_TERMINAL_TILT_DEG = 0.8
SOLID_MOTOR_BURNOUT_TIME = 1.7
MASS_LOSS_RAMP_TIME = 0.35
BASELINE_IGNITION_ALTITUDE = 36.11

# Freefall sub-phases
PHASE_TUMBLING = 5    # aerodynamic tumble nose-down
PHASE_NOSE_DOWN = 6   # stable nose-down freefall
PHASE_RCS_FLIP = 7    # cold-gas thruster flip back to upright

FREEFALL_TUMBLE_DELAY = 0.25   # s after apogee before tumble starts
FREEFALL_TUMBLE_DUR = 1.65     # s for aerodynamic 0° → 180° tumble
RCS_FLIP_DUR = 1.80            # s for RCS 180° → 0° controlled flip
RCS_FLIP_MARGIN = 0.15         # s before ignition to be upright

FIELDNAMES = [
    "Time",
    "X",
    "Y",
    "Z",
    "VX",
    "VY",
    "VZ",
    "QW",
    "QX",
    "QY",
    "QZ",
    "Mass",
    "MLCorrection",
    "FaultType",
    "FaultMag",
    "WindX",
    "WindY",
    "Phase",
]

FAULT_CODE = {
    (): 0,
    ("WIND_GUST",): 1,
    ("DRAG_CHANGE",): 2,
    ("MASS_LOSS",): 3,
}


@dataclass(frozen=True)
class Scenario:
    run_id: str
    name: str
    description: str
    mode: str
    fault_intensity: float
    fault_types: Tuple[str, ...]
    target_landing_speed: float
    target_x: float
    target_y: float
    nominal_ignition_altitude: float
    effective_ignition_altitude: float
    wind_speed: float
    wind_heading_deg: float
    drag_coefficient: float
    air_density: float
    fault_delay: float

    @property
    def success(self) -> bool:
        return self.target_landing_speed <= 1.25


SCENARIOS: Sequence[Scenario] = (
    Scenario(
        run_id="demo_01_baseline",
        name="Demo 1",
        description="Reference ascent, passive freefall, and nominal preflight ignition. The vehicle stays upright and lands softly near the pad.",
        mode="Optimization",
        fault_intensity=0.00,
        fault_types=(),
        target_landing_speed=0.52,
        target_x=-0.45,
        target_y=-0.32,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=36.11,
        wind_speed=1.9,
        wind_heading_deg=222.0,
        drag_coefficient=0.48,
        air_density=1.18,
        fault_delay=0.0,
    ),
    Scenario(
        run_id="demo_02_wind_opt",
        name="Demo 2",
        description="Wind acts through ascent and freefall, pushing the vehicle off-pad before ignition. Preflight optimization keeps its nominal altitude and lands hard downrange.",
        mode="Optimization",
        fault_intensity=0.34,
        fault_types=("WIND_GUST",),
        target_landing_speed=3.90,
        target_x=14.4,
        target_y=6.1,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=35.05,
        wind_speed=10.3,
        wind_heading_deg=23.0,
        drag_coefficient=0.50,
        air_density=1.21,
        fault_delay=0.0,
    ),
    Scenario(
        run_id="demo_03_wind_ml",
        name="Demo 3",
        description="Same wind environment as Demo 2, but the in-flight ML correction raises the ignition trigger and gives the burn more room to recover the pad.",
        mode="ML",
        fault_intensity=0.34,
        fault_types=("WIND_GUST",),
        target_landing_speed=0.78,
        target_x=1.9,
        target_y=3.0,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=36.28,
        wind_speed=10.3,
        wind_heading_deg=23.0,
        drag_coefficient=0.50,
        air_density=1.21,
        fault_delay=0.0,
    ),
    Scenario(
        run_id="demo_04_dragmass_opt",
        name="Demo 4",
        description="Non-wind faults bias the preflight ignition estimate low. The rocket stays well behaved, but the burn starts late and cannot shed enough speed.",
        mode="Optimization",
        fault_intensity=0.54,
        fault_types=("DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=8.60,
        target_x=3.6,
        target_y=1.1,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=32.00,
        wind_speed=2.3,
        wind_heading_deg=15.0,
        drag_coefficient=0.57,
        air_density=1.20,
        fault_delay=0.95,
    ),
    Scenario(
        run_id="demo_05_dragmass_ml",
        name="Demo 5",
        description="The same drag and mass faults are detected during freefall and the ML correction shifts ignition upward toward the valid suicide-burn window.",
        mode="ML",
        fault_intensity=0.54,
        fault_types=("DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=0.84,
        target_x=-0.25,
        target_y=1.35,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=36.38,
        wind_speed=2.3,
        wind_heading_deg=15.0,
        drag_coefficient=0.57,
        air_density=1.20,
        fault_delay=0.95,
    ),
    Scenario(
        run_id="demo_06_severe_opt",
        name="Demo 6",
        description="Wind, drag, and mass faults all accumulate before ignition. The rocket remains upright but the late burn leaves a high-speed impact.",
        mode="Optimization",
        fault_intensity=0.72,
        fault_types=("WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=12.80,
        target_x=19.8,
        target_y=-7.1,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=26.00,
        wind_speed=8.1,
        wind_heading_deg=340.0,
        drag_coefficient=0.61,
        air_density=1.21,
        fault_delay=0.20,
    ),
    Scenario(
        run_id="demo_07_severe_ml",
        name="Demo 7",
        description="The same severe combined faults are tracked in freefall and the ML correction restores a viable ignition altitude, allowing aggressive TVC cleanup during the burn.",
        mode="ML",
        fault_intensity=0.72,
        fault_types=("WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=1.02,
        target_x=-1.9,
        target_y=-5.2,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=36.65,
        wind_speed=8.1,
        wind_heading_deg=340.0,
        drag_coefficient=0.61,
        air_density=1.21,
        fault_delay=0.20,
    ),
    Scenario(
        run_id="demo_08_extreme_opt",
        name="Demo 8",
        description="An extreme combined-fault case leaves almost no altitude margin. The burn is still upright and strongly decelerating, but it begins too low to avoid a crash.",
        mode="Optimization",
        fault_intensity=0.86,
        fault_types=("WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=18.00,
        target_x=-22.0,
        target_y=14.5,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=18.00,
        wind_speed=12.5,
        wind_heading_deg=147.0,
        drag_coefficient=0.64,
        air_density=1.22,
        fault_delay=0.15,
    ),
    Scenario(
        run_id="demo_09_extreme_ml",
        name="Demo 9",
        description="ML cannot fully recover the extreme case, but it shifts ignition high enough to convert a catastrophic strike into a much softer degraded landing.",
        mode="ML",
        fault_intensity=0.86,
        fault_types=("WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"),
        target_landing_speed=4.90,
        target_x=5.4,
        target_y=-9.8,
        nominal_ignition_altitude=BASELINE_IGNITION_ALTITUDE,
        effective_ignition_altitude=31.20,
        wind_speed=12.5,
        wind_heading_deg=147.0,
        drag_coefficient=0.64,
        air_density=1.22,
        fault_delay=0.15,
    ),
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def smoothstep(u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def interpolate(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = xs[lo]
    x1 = xs[hi]
    if abs(x1 - x0) < 1e-12:
        return ys[lo]
    fraction = (x - x0) / (x1 - x0)
    return ys[lo] + fraction * (ys[hi] - ys[lo])


def encode_float(value: float) -> float:
    return round(float(value), 6)


def fault_code(fault_types: Iterable[str]) -> int:
    fault_tuple = tuple(fault_types)
    if len(fault_tuple) > 1:
        return 4
    return FAULT_CODE.get(fault_tuple, 0)


def wind_vector(speed: float, heading_deg: float) -> Tuple[float, float]:
    heading = math.radians(heading_deg)
    return speed * math.cos(heading), speed * math.sin(heading)


def vector_norm(x_value: float, y_value: float) -> float:
    return math.sqrt(x_value * x_value + y_value * y_value)


def unit_vector(x_value: float, y_value: float) -> Tuple[float, float]:
    magnitude = vector_norm(x_value, y_value)
    if magnitude < 1e-9:
        return 0.0, 0.0
    return x_value / magnitude, y_value / magnitude


def hermite_scalar(time_value: float, t0: float, t1: float, p0: float, p1: float, v0: float, v1: float) -> Tuple[float, float, float]:
    if time_value <= t0:
        return p0, v0, 0.0
    if time_value >= t1:
        return p1, v1, 0.0

    duration = t1 - t0
    u = (time_value - t0) / duration
    u2 = u * u
    u3 = u2 * u

    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    position = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1

    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u
    velocity = (
        dh00 * p0 / duration
        + dh10 * v0
        + dh01 * p1 / duration
        + dh11 * v1
    )

    d2h00 = 12.0 * u - 6.0
    d2h10 = 6.0 * u - 4.0
    d2h01 = -12.0 * u + 6.0
    d2h11 = 6.0 * u - 2.0
    acceleration = (
        d2h00 * p0 / (duration * duration)
        + d2h10 * v0 / duration
        + d2h01 * p1 / (duration * duration)
        + d2h11 * v1 / duration
    )
    return position, velocity, acceleration


def quat_multiply(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quat_axis_angle(axis: Tuple[float, float, float], angle: float) -> Tuple[float, float, float, float]:
    ax, ay, az = axis
    magnitude = math.sqrt(ax * ax + ay * ay + az * az)
    if magnitude < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    ax /= magnitude
    ay /= magnitude
    az /= magnitude
    half = 0.5 * angle
    s_value = math.sin(half)
    return math.cos(half), ax * s_value, ay * s_value, az * s_value


def quat_normalize(quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qw, qx, qy, qz = quaternion
    magnitude = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if magnitude < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return qw / magnitude, qx / magnitude, qy / magnitude, qz / magnitude


quaternion_normalize = quat_normalize


def quaternion_from_tilt(tilt_x_deg: float, tilt_y_deg: float, yaw_deg: float = 0.0) -> Tuple[float, float, float, float]:
    roll = quat_axis_angle((1.0, 0.0, 0.0), math.radians(tilt_y_deg))
    pitch = quat_axis_angle((0.0, 1.0, 0.0), math.radians(-tilt_x_deg))
    yaw = quat_axis_angle((0.0, 0.0, 1.0), math.radians(yaw_deg))
    return quat_normalize(quat_multiply(yaw, quat_multiply(pitch, roll)))


def freefall_timing(apogee_time: float, ignition_time: float) -> Dict[str, float]:
    """Compute the start/end times for each freefall sub-phase."""
    freefall_dur = max(ignition_time - apogee_time, 1.0)
    tumble_start = apogee_time + FREEFALL_TUMBLE_DELAY
    tumble_end = tumble_start + min(FREEFALL_TUMBLE_DUR, freefall_dur * 0.38)
    rcs_end = ignition_time - RCS_FLIP_MARGIN
    rcs_dur = min(RCS_FLIP_DUR, freefall_dur * 0.42)
    rcs_start = rcs_end - rcs_dur
    if rcs_start < tumble_end:
        rcs_start = tumble_end + 0.05
    return {
        "tumble_start": tumble_start,
        "tumble_end": tumble_end,
        "rcs_start": rcs_start,
        "rcs_end": rcs_end,
    }


def flip_axis_for_scenario(scenario: Scenario) -> Tuple[float, float, float]:
    """Horizontal unit vector perpendicular to wind direction — the tumble rotation axis."""
    heading = math.radians(scenario.wind_heading_deg)
    return (-math.sin(heading), math.cos(heading), 0.0)


def quaternion_from_nose_angle(angle_rad: float, axis: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Quaternion for rotating nose from upright (angle=0) to nose-down (angle=π) around axis.
    Uses Rodrigues formula: starting from [0,0,1] (nose-up), rotate around horizontal axis.
    Returns (qw, qx, qy, qz).
    """
    ax, ay, az = axis  # horizontal unit vector, az=0
    half = 0.5 * angle_rad
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def load_reference() -> Dict[str, object]:
    if not REFERENCE_TRAJECTORY.exists():
        raise FileNotFoundError(f"Reference trajectory not found: {REFERENCE_TRAJECTORY}")

    columns = {key: [] for key in ("Time", "Z", "VZ", "Mass")}
    with REFERENCE_TRAJECTORY.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in columns:
                columns[key].append(float(row[key]))

    time_values = columns["Time"]
    z_values = columns["Z"]
    vz_values = columns["VZ"]
    mass_values = columns["Mass"]

    apogee_index = max(range(len(z_values)), key=z_values.__getitem__)

    ascent_end_index = apogee_index
    for index in range(1, len(mass_values) - 3):
        if time_values[index] < 1.0:
            continue
        if abs(mass_values[index + 1] - mass_values[index]) < 1e-7:
            ascent_end_index = index
            break

    ignition_index = apogee_index
    for index in range(apogee_index + 1, len(mass_values) - 1):
        if mass_values[index + 1] < mass_values[index] - 1e-5:
            ignition_index = index
            break

    burn_end_index = len(mass_values) - 1
    for index in range(ignition_index + 1, len(mass_values) - 1):
        if abs(mass_values[index + 1] - mass_values[index]) < 1e-7:
            burn_end_index = index
            break

    preign_start = max(apogee_index + 1, ignition_index - 25)
    terminal_window = [abs(vz_values[index]) for index in range(preign_start, ignition_index + 1)]
    terminal_speed = sum(terminal_window) / len(terminal_window)

    return {
        "time": time_values,
        "z": z_values,
        "vz": vz_values,
        "mass": mass_values,
        "apogee_time": time_values[apogee_index],
        "apogee_z": z_values[apogee_index],
        "peak_vz": max(vz_values),
        "ascent_end_time": time_values[ascent_end_index],
        "base_ignition_time": time_values[ignition_index],
        "base_ignition_z": z_values[ignition_index],
        "base_ignition_vz": vz_values[ignition_index],
        "base_burn_duration": max(0.6, time_values[burn_end_index] - time_values[ignition_index]),
        "terminal_speed": terminal_speed,
    }


def ascent_state(reference: Dict[str, object], time_value: float) -> Tuple[float, float]:
    return (
        interpolate(reference["time"], reference["z"], time_value),
        interpolate(reference["time"], reference["vz"], time_value),
    )


def terminal_velocity(reference: Dict[str, object], scenario: Scenario) -> float:
    drag_ratio = (BASE_DRAG_COEFFICIENT * BASE_AIR_DENSITY) / max(1e-9, scenario.drag_coefficient * scenario.air_density)
    return float(reference["terminal_speed"]) * math.sqrt(max(0.35, drag_ratio))


def freefall_state(reference: Dict[str, object], scenario: Scenario, time_since_apogee: float) -> Tuple[float, float]:
    apogee_z = float(reference["apogee_z"])
    terminal = terminal_velocity(reference, scenario)
    argument = GRAVITY * time_since_apogee / max(terminal, 1e-6)
    vz_value = -terminal * math.tanh(argument)
    z_value = apogee_z - (terminal * terminal / GRAVITY) * math.log(math.cosh(argument))
    return max(z_value, 0.0), vz_value


def solve_ignition_time(reference: Dict[str, object], scenario: Scenario, target_altitude: float) -> Tuple[float, float]:
    apogee_time = float(reference["apogee_time"])
    low = apogee_time
    high = apogee_time + 60.0

    while True:
        z_high, _ = freefall_state(reference, scenario, high - apogee_time)
        if z_high <= target_altitude:
            break
        high += 10.0
        if high - apogee_time > 120.0:
            raise RuntimeError(f"Unable to solve ignition time for {scenario.run_id}")

    for _ in range(60):
        mid = 0.5 * (low + high)
        z_mid, _ = freefall_state(reference, scenario, mid - apogee_time)
        if z_mid > target_altitude:
            low = mid
        else:
            high = mid

    ignition_time = 0.5 * (low + high)
    _, ignition_vz = freefall_state(reference, scenario, ignition_time - apogee_time)
    return ignition_time, ignition_vz


def burn_shape_exponent(scenario: Scenario) -> float:
    if scenario.success:
        return 1.75 + 0.35 * scenario.fault_intensity
    return 2.2 + 1.2 * scenario.fault_intensity


def simulate_powered_vertical(
    scenario: Scenario,
    ignition_altitude: float,
    ignition_vz: float,
    landing_vz: float,
) -> List[Dict[str, float]]:
    ignition_speed = abs(ignition_vz)
    landing_speed = min(abs(landing_vz), max(ignition_speed - 0.5, 0.4))
    shape = burn_shape_exponent(scenario)
    average_speed = landing_speed + (ignition_speed - landing_speed) / (shape + 1.0)
    duration = ignition_altitude / max(average_speed, 1e-6)
    duration = max(duration, DT)

    profile: List[Dict[str, float]] = []
    elapsed = 0.0
    while elapsed < duration - 1e-9:
        u = clamp(elapsed / duration, 0.0, 1.0)
        remaining = 1.0 - u
        speed_mag = landing_speed + (ignition_speed - landing_speed) * (remaining ** shape)
        vz_value = -speed_mag
        z_value = duration * (
            landing_speed * remaining
            + (ignition_speed - landing_speed) * (remaining ** (shape + 1.0)) / (shape + 1.0)
        )
        az_value = (ignition_speed - landing_speed) * shape * (remaining ** max(shape - 1.0, 0.0)) / duration
        profile.append(
            {
                "elapsed": elapsed,
                "z": z_value,
                "vz": vz_value,
                "az": az_value,
            }
        )
        elapsed += DT

    profile.append(
        {
            "elapsed": duration,
            "z": 0.0,
            "vz": -landing_speed,
            "az": 0.0,
        }
    )
    return profile


def align_to_sample_grid(time_value: float) -> float:
    return math.ceil(time_value / DT) * DT


def lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


def total_mass_loss(scenario: Scenario) -> float:
    if "MASS_LOSS" not in scenario.fault_types:
        return 0.0
    return BASE_ASCENT_END_MASS * (0.018 + 0.030 * scenario.fault_intensity)


def mass_loss_amount_at_time(time_value: float, fault_start: float, scenario: Scenario) -> float:
    if "MASS_LOSS" not in scenario.fault_types or time_value < fault_start:
        return 0.0
    ramp = smoothstep((time_value - fault_start) / MASS_LOSS_RAMP_TIME)
    return total_mass_loss(scenario) * ramp


def should_bounce_after_impact(scenario: Scenario, powered_duration: float) -> bool:
    return (not scenario.success) and powered_duration < SOLID_MOTOR_BURNOUT_TIME and scenario.target_landing_speed >= 10.0


def simulate_bounce_profile(
    scenario: Scenario,
    impact_x: float,
    impact_y: float,
    impact_vx: float,
    impact_vy: float,
    impact_vz: float,
    impact_tilt_x: float,
    impact_tilt_y: float,
) -> List[Dict[str, float]]:
    restitution = clamp(0.16 + 0.08 * scenario.fault_intensity, 0.14, 0.26)
    bounce_vz = abs(impact_vz) * restitution
    if bounce_vz < 1.0:
        return []

    lateral_vx = impact_vx * 0.42
    lateral_vy = impact_vy * 0.42
    total_time = 2.0 * bounce_vz / GRAVITY
    elapsed = DT
    profile: List[Dict[str, float]] = []

    while elapsed < total_time - 1e-9:
        z_value = max(0.0, bounce_vz * elapsed - 0.5 * GRAVITY * elapsed * elapsed)
        vz_value = bounce_vz - GRAVITY * elapsed
        damping = 1.0 - 0.25 * smoothstep(elapsed / total_time)
        tilt_decay = math.exp(-elapsed / 0.30)
        profile.append(
            {
                "elapsed": elapsed,
                "x": impact_x + lateral_vx * elapsed * damping,
                "y": impact_y + lateral_vy * elapsed * damping,
                "z": z_value,
                "vx": lateral_vx * damping,
                "vy": lateral_vy * damping,
                "vz": vz_value,
                "tilt_x": impact_tilt_x * tilt_decay,
                "tilt_y": impact_tilt_y * tilt_decay,
            }
        )
        elapsed += DT

    profile.append(
        {
            "elapsed": total_time,
            "x": impact_x + lateral_vx * total_time * 0.75,
            "y": impact_y + lateral_vy * total_time * 0.75,
            "z": 0.0,
            "vx": lateral_vx * 0.45,
            "vy": lateral_vy * 0.45,
            "vz": -bounce_vz,
            "tilt_x": impact_tilt_x * 0.08,
            "tilt_y": impact_tilt_y * 0.08,
        }
    )
    return profile


def ml_ignition_adjustment(scenario: Scenario, plan: Dict[str, float]) -> float:
    correction_distance = vector_norm(plan["ignition_x"], plan["ignition_y"])
    correction_speed = vector_norm(plan["ignition_vx"], plan["ignition_vy"])

    wind_term = 0.0
    drag_term = 0.0
    mass_term = 0.0

    if "WIND_GUST" in scenario.fault_types:
        wind_term = 0.45 * correction_distance + 0.70 * correction_speed + 2.4 * scenario.fault_intensity
    if "DRAG_CHANGE" in scenario.fault_types:
        drag_term = -3.0 * scenario.fault_intensity
    if "MASS_LOSS" in scenario.fault_types:
        mass_term = -2.2 * scenario.fault_intensity

    return clamp(wind_term + drag_term + mass_term, -6.0, 8.0)


def phase_at_time(
    time_value: float,
    ascent_end: float,
    apogee_time: float,
    ignition_time: float,
    landing_time: float,
    ff_timing: Dict[str, float],
) -> int:
    if time_value < ascent_end:
        return 0
    if time_value < apogee_time:
        return 1
    if time_value < ff_timing["tumble_start"]:
        return 2
    if time_value < ff_timing["tumble_end"]:
        return PHASE_TUMBLING
    if time_value < ff_timing["rcs_start"]:
        return PHASE_NOSE_DOWN
    if time_value < ff_timing["rcs_end"]:
        return PHASE_RCS_FLIP
    if time_value < ignition_time:
        return 2  # brief upright wait before ignition
    if time_value < landing_time:
        return 3
    return 4


def passive_wind_speed(time_value: float, fault_start: float, ignition_time: float, scenario: Scenario) -> float:
    baseline_speed = min(1.2, scenario.wind_speed * 0.15)
    if time_value >= ignition_time:
        return min(0.8, scenario.wind_speed * 0.06)
    if "WIND_GUST" in scenario.fault_types:
        if time_value < fault_start:
            pre_fault_ramp = smoothstep(clamp(time_value / max(fault_start, 0.8), 0.0, 1.0))
            return lerp(0.45, baseline_speed, pre_fault_ramp)

        gust_elapsed = time_value - fault_start
        gust_window = max(min(ignition_time - fault_start, 1.6), 0.6)
        gust_ramp = smoothstep(clamp(gust_elapsed / gust_window, 0.0, 1.0))
        gust_flutter = 0.12 * scenario.wind_speed * math.sin(2.8 * gust_elapsed) * math.exp(-0.85 * gust_elapsed)
        gust_speed = lerp(baseline_speed, scenario.wind_speed, gust_ramp) + gust_flutter
        return clamp(gust_speed, baseline_speed * 0.9, scenario.wind_speed * 1.08)
    return baseline_speed


def wind_at_time(time_value: float, fault_start: float, ignition_time: float, scenario: Scenario) -> Tuple[float, float]:
    speed = passive_wind_speed(time_value, fault_start, ignition_time, scenario)
    return wind_vector(speed, scenario.wind_heading_deg)


def fault_start_time(reference: Dict[str, object], scenario: Scenario) -> float:
    if "WIND_GUST" in scenario.fault_types:
        ascent_end = float(reference["ascent_end_time"])
        apogee_time = float(reference["apogee_time"])
        wind_delay = max(scenario.fault_delay, 0.35)
        return clamp(ascent_end + wind_delay, ascent_end + 0.10, apogee_time - 0.35)
    return float(reference["apogee_time"]) + scenario.fault_delay


def pre_fault_ml_trim(time_value: float, fault_start: float, scenario: Scenario) -> float:
    if scenario.mode != "ML":
        return 0.0
    correction = scenario.effective_ignition_altitude - scenario.nominal_ignition_altitude
    if abs(correction) < 1e-6 or time_value >= fault_start:
        return 0.0

    preview_window = clamp(min(1.5, fault_start * 0.35), 0.45, 1.5)
    preview_start = max(0.0, fault_start - preview_window)
    if time_value < preview_start:
        return 0.0

    preview_progress = smoothstep((time_value - preview_start) / max(fault_start - preview_start, 1e-6))
    preview_amplitude = min(0.18, 0.05 + 0.10 * scenario.fault_intensity)
    preview_flutter = 0.70 + 0.30 * math.sin(4.4 * time_value + 0.8 * scenario.fault_intensity)
    return math.copysign(preview_amplitude * preview_progress * preview_flutter, correction)


def ml_correction_at_time(time_value: float, fault_start: float, ignition_time: float, scenario: Scenario) -> float:
    if scenario.mode != "ML":
        return 0.0
    correction = scenario.effective_ignition_altitude - scenario.nominal_ignition_altitude
    if time_value < fault_start:
        return pre_fault_ml_trim(time_value, fault_start, scenario)
    if time_value >= ignition_time:
        return correction

    update_interval = 0.5
    elapsed = time_value - fault_start
    if elapsed < update_interval:
        return pre_fault_ml_trim(fault_start - 1e-6, fault_start, scenario)

    total_steps = max(1, int(math.ceil(max(ignition_time - fault_start, 1e-6) / update_interval)))
    completed_steps = min(total_steps, int(elapsed // update_interval))
    step_progress = completed_steps / total_steps
    aggressive_response = 1.0 - (1.0 - step_progress) ** 2.4
    fluctuation_gain = min(0.16, 0.08 + 0.10 * scenario.fault_intensity)
    step_flutter = fluctuation_gain * math.sin(1.45 * completed_steps + 0.6) * (1.0 - step_progress)
    resolved_value = correction * (aggressive_response + step_flutter)
    if correction >= 0.0:
        return clamp(resolved_value, 0.0, correction * 1.15)
    return clamp(resolved_value, correction * 1.15, 0.0)


def lateral_plan(reference: Dict[str, object], scenario: Scenario, ignition_time: float, landing_time: float) -> Dict[str, float]:
    apogee_time = float(reference["apogee_time"])
    drift_x, drift_y = wind_vector(1.0, scenario.wind_heading_deg)
    cross_x, cross_y = -drift_y, drift_x

    ascent_factor = 0.22 if "WIND_GUST" in scenario.fault_types else 0.06
    ascent_factor += 0.06 * scenario.fault_intensity
    freefall_factor = 0.55 if "WIND_GUST" in scenario.fault_types else 0.12
    freefall_factor += 0.10 * scenario.fault_intensity

    apogee_distance = scenario.wind_speed * apogee_time * 0.10 * ascent_factor
    freefall_distance = scenario.wind_speed * max(ignition_time - apogee_time, 0.0) * 0.10 * freefall_factor
    cross_distance = scenario.wind_speed * (0.10 + 0.08 * scenario.fault_intensity)

    apogee_x = drift_x * apogee_distance + cross_x * 0.25 * cross_distance
    apogee_y = drift_y * apogee_distance + cross_y * 0.25 * cross_distance
    ignition_x = apogee_x + drift_x * freefall_distance + cross_x * 0.35 * cross_distance
    ignition_y = apogee_y + drift_y * freefall_distance + cross_y * 0.35 * cross_distance

    apogee_vx = scenario.wind_speed * drift_x * (0.06 if "WIND_GUST" in scenario.fault_types else 0.015)
    apogee_vy = scenario.wind_speed * drift_y * (0.06 if "WIND_GUST" in scenario.fault_types else 0.015)
    ignition_vx = scenario.wind_speed * drift_x * (0.16 + 0.10 * scenario.fault_intensity)
    ignition_vy = scenario.wind_speed * drift_y * (0.16 + 0.10 * scenario.fault_intensity)

    return {
        "apogee_x": apogee_x,
        "apogee_y": apogee_y,
        "apogee_vx": apogee_vx,
        "apogee_vy": apogee_vy,
        "ignition_x": ignition_x,
        "ignition_y": ignition_y,
        "ignition_vx": ignition_vx,
        "ignition_vy": ignition_vy,
    }


def lateral_state(time_value: float, reference: Dict[str, object], scenario: Scenario, ignition_time: float, plan: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    apogee_time = float(reference["apogee_time"])

    if time_value <= apogee_time:
        x_value, vx_value, ax_value = hermite_scalar(
            time_value,
            0.0,
            apogee_time,
            0.0,
            plan["apogee_x"],
            0.0,
            plan["apogee_vx"],
        )
        y_value, vy_value, ay_value = hermite_scalar(
            time_value,
            0.0,
            apogee_time,
            0.0,
            plan["apogee_y"],
            0.0,
            plan["apogee_vy"],
        )
        return x_value, y_value, vx_value, vy_value, ax_value, ay_value

    if time_value <= ignition_time:
        x_value, vx_value, ax_value = hermite_scalar(
            time_value,
            apogee_time,
            ignition_time,
            plan["apogee_x"],
            plan["ignition_x"],
            plan["apogee_vx"],
            plan["ignition_vx"],
        )
        y_value, vy_value, ay_value = hermite_scalar(
            time_value,
            apogee_time,
            ignition_time,
            plan["apogee_y"],
            plan["ignition_y"],
            plan["apogee_vy"],
            plan["ignition_vy"],
        )
        return x_value, y_value, vx_value, vy_value, ax_value, ay_value

    return plan["ignition_x"], plan["ignition_y"], plan["ignition_vx"], plan["ignition_vy"], 0.0, 0.0


def simulate_powered_lateral(
    scenario: Scenario,
    duration: float,
    ignition_x: float,
    ignition_y: float,
    ignition_vx: float,
    ignition_vy: float,
) -> List[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    x_value = ignition_x
    y_value = ignition_y
    vx_value = ignition_vx
    vy_value = ignition_vy
    tilt_x = 0.0
    tilt_y = 0.0
    time_value = 0.0

    kp_position = 0.72 if scenario.mode == "Optimization" else 1.25
    kd_velocity = 2.10 if scenario.mode == "Optimization" else 3.10
    tilt_limit = DESCENT_TILT_LIMIT_DEG
    slew_step = TVC_SLEW_RATE_DEG_PER_S * DT

    entries.append(
        {
            "elapsed": 0.0,
            "x": x_value,
            "y": y_value,
            "vx": vx_value,
            "vy": vy_value,
            "ax": 0.0,
            "ay": 0.0,
            "tilt_x": tilt_x,
            "tilt_y": tilt_y,
        }
    )

    while time_value < duration - 1e-9:
        progress = clamp(time_value / max(duration, 1e-9), 0.0, 1.0)
        target_x_tilt = clamp(-(kp_position * x_value + kd_velocity * vx_value), -tilt_limit, tilt_limit)
        target_y_tilt = clamp(-(kp_position * y_value + kd_velocity * vy_value), -tilt_limit, tilt_limit)

        if scenario.success:
            terminal_taper = 1.0 - 0.96 * smoothstep((progress - 0.62) / 0.38)
            target_x_tilt *= terminal_taper
            target_y_tilt *= terminal_taper

        delta_x = clamp(target_x_tilt - tilt_x, -slew_step, slew_step)
        delta_y = clamp(target_y_tilt - tilt_y, -slew_step, slew_step)
        tilt_x += delta_x
        tilt_y += delta_y

        ax_value = LATERAL_ACCEL_PER_DEG * tilt_x
        ay_value = LATERAL_ACCEL_PER_DEG * tilt_y
        vx_value += ax_value * DT
        vy_value += ay_value * DT
        x_value += vx_value * DT
        y_value += vy_value * DT
        time_value = min(time_value + DT, duration)

        entries.append(
            {
                "elapsed": time_value,
                "x": x_value,
                "y": y_value,
                "vx": vx_value,
                "vy": vy_value,
                "ax": ax_value,
                "ay": ay_value,
                "tilt_x": tilt_x,
                "tilt_y": tilt_y,
            }
        )
    return entries


def powered_state(profile: Sequence[Dict[str, float]], elapsed: float, key: str) -> float:
    if elapsed <= profile[0]["elapsed"]:
        return float(profile[0][key])
    if elapsed >= profile[-1]["elapsed"]:
        return float(profile[-1][key])
    index = min(int(elapsed / DT), len(profile) - 2)
    left = profile[index]
    right = profile[index + 1]
    if right["elapsed"] <= elapsed:
        left = right
        right = profile[min(index + 2, len(profile) - 1)]
    span = right["elapsed"] - left["elapsed"]
    if span <= 1e-9:
        return float(left[key])
    fraction = (elapsed - left["elapsed"]) / span
    return float(left[key]) + fraction * (float(right[key]) - float(left[key]))


def mass_at_time(time_value: float, reference: Dict[str, object], fault_start: float, scenario: Scenario, ignition_time: float, landing_time: float) -> float:
    ascent_end = float(reference["ascent_end_time"])
    lost_mass = mass_loss_amount_at_time(time_value, fault_start, scenario)
    coast_mass = BASE_ASCENT_END_MASS - lost_mass
    dry_mass = BASE_DRY_MASS - lost_mass
    if time_value <= ascent_end:
        return interpolate(reference["time"], reference["mass"], time_value)
    if time_value <= ignition_time:
        return max(coast_mass, dry_mass)
    if time_value >= landing_time:
        return max(dry_mass, 0.85 * BASE_DRY_MASS)
    progress = clamp((time_value - ignition_time) / max(landing_time - ignition_time, 1e-6), 0.0, 1.0)
    return max(lerp(coast_mass, dry_mass, progress), 0.85 * BASE_DRY_MASS)


def tilt_state(phase: int, time_value: float, ascent_end: float, ignition_time: float, landing_time: float, x_value: float, y_value: float, vx_value: float, vy_value: float, ax_value: float, ay_value: float, scenario: Scenario) -> Tuple[float, float]:
    lateral_speed = vector_norm(vx_value, vy_value)
    distance_from_pad = vector_norm(x_value, y_value)

    if phase == 0:
        tilt_x = clamp(ax_value * 0.9, -12.0, 12.0)
        tilt_y = clamp(ay_value * 0.9, -12.0, 12.0)
        wobble = ASCENT_WOBBLE_AMPLITUDE_DEG * math.sin(3.0 * time_value)
        return clamp(tilt_x + wobble, -15.0, 15.0), clamp(tilt_y - wobble * 0.6, -15.0, 15.0)

    if phase in (1, 2):
        lean_mag = clamp(0.7 * lateral_speed + (1.8 if "WIND_GUST" in scenario.fault_types else 0.8), 0.0, 18.0)
        ux, uy = unit_vector(vx_value, vy_value)
        burnout_wobble = ASCENT_WOBBLE_AMPLITUDE_DEG * math.sin(3.0 * ascent_end)
        settle = smoothstep((time_value - ascent_end) / 0.90)
        carry_x = clamp(ax_value * 0.8 + burnout_wobble, -12.0, 12.0)
        carry_y = clamp(ay_value * 0.8 - burnout_wobble * 0.6, -12.0, 12.0)
        lean_x = lean_mag * ux
        lean_y = lean_mag * uy
        return lerp(carry_x, lean_x, settle), lerp(carry_y, lean_y, settle)

    if phase == 3:
        progress = clamp((time_value - ignition_time) / max(landing_time - ignition_time, 1e-6), 0.0, 1.0)
        err_x = -x_value
        err_y = -y_value
        err_ux, err_uy = unit_vector(err_x, err_y)
        pad_gain = 0.9 * distance_from_pad + 2.2 * lateral_speed
        taper = 1.0 - 0.35 * smoothstep(progress)
        tilt_mag = clamp((4.0 + pad_gain) * taper, 0.0, DESCENT_TILT_LIMIT_DEG)
        if distance_from_pad < 1.5:
            tilt_mag = min(tilt_mag, 8.0)
        return tilt_mag * err_ux, tilt_mag * err_uy

    return 0.0, 0.0


def fault_magnitude(time_value: float, fault_start: float, ignition_time: float, scenario: Scenario) -> float:
    if not scenario.fault_types or time_value < fault_start or time_value >= ignition_time:
        return 0.0
    ramp = smoothstep((time_value - fault_start) / max(ignition_time - fault_start, 1e-6))
    return scenario.fault_intensity * ramp


def synthesize_run(reference: Dict[str, object], scenario: Scenario) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    apogee_time = float(reference["apogee_time"])
    ascent_end = float(reference["ascent_end_time"])
    apogee_altitude = float(reference["apogee_z"])
    peak_vertical_velocity = float(reference["peak_vz"])
    fault_start = fault_start_time(reference, scenario)
    effective_ignition_altitude = scenario.nominal_ignition_altitude
    ignition_time, ignition_vz = solve_ignition_time(reference, scenario, effective_ignition_altitude)
    ignition_time = align_to_sample_grid(ignition_time)
    _, ignition_vz = freefall_state(reference, scenario, ignition_time - apogee_time)
    fault_id = fault_code(scenario.fault_types)
    plan = lateral_plan(reference, scenario, ignition_time, ignition_time)

    if scenario.mode == "ML":
        effective_ignition_altitude = scenario.nominal_ignition_altitude + ml_ignition_adjustment(scenario, plan)
        ignition_time, ignition_vz = solve_ignition_time(reference, scenario, effective_ignition_altitude)
        ignition_time = align_to_sample_grid(ignition_time)
        _, ignition_vz = freefall_state(reference, scenario, ignition_time - apogee_time)
        plan = lateral_plan(reference, scenario, ignition_time, ignition_time)

    resolved_scenario = Scenario(
        run_id=scenario.run_id,
        name=scenario.name,
        description=scenario.description,
        mode=scenario.mode,
        fault_intensity=scenario.fault_intensity,
        fault_types=scenario.fault_types,
        target_landing_speed=scenario.target_landing_speed,
        target_x=scenario.target_x,
        target_y=scenario.target_y,
        nominal_ignition_altitude=scenario.nominal_ignition_altitude,
        effective_ignition_altitude=effective_ignition_altitude,
        wind_speed=scenario.wind_speed,
        wind_heading_deg=scenario.wind_heading_deg,
        drag_coefficient=scenario.drag_coefficient,
        air_density=scenario.air_density,
        fault_delay=scenario.fault_delay,
    )

    landing_vz = -abs(scenario.target_landing_speed)
    vertical_profile = simulate_powered_vertical(
        scenario,
        effective_ignition_altitude,
        ignition_vz,
        landing_vz,
    )
    powered_duration = float(vertical_profile[-1]["elapsed"])
    landing_time = ignition_time + powered_duration
    powered_profile = simulate_powered_lateral(
        scenario,
        powered_duration,
        plan["ignition_x"],
        plan["ignition_y"],
        plan["ignition_vx"],
        plan["ignition_vy"],
    )
    bounce_profile = simulate_bounce_profile(
        scenario,
        float(powered_profile[-1]["x"]),
        float(powered_profile[-1]["y"]),
        float(powered_profile[-1]["vx"]),
        float(powered_profile[-1]["vy"]),
        float(vertical_profile[-1]["vz"]),
        float(powered_profile[-1]["tilt_x"]),
        float(powered_profile[-1]["tilt_y"]),
    ) if should_bounce_after_impact(scenario, powered_duration) else []
    final_time = landing_time + (float(bounce_profile[-1]["elapsed"]) if bounce_profile else 0.0)

    records: List[Dict[str, float]] = []
    current_time = 0.0
    ff_timing = freefall_timing(apogee_time, ignition_time)
    tumble_axis = flip_axis_for_scenario(scenario)

    while current_time < landing_time - 1e-9:
        phase = phase_at_time(current_time, ascent_end, apogee_time, ignition_time, landing_time, ff_timing)

        if current_time <= apogee_time:
            z_value, vz_value = ascent_state(reference, current_time)
        elif current_time < ignition_time:
            z_value, vz_value = freefall_state(reference, scenario, current_time - apogee_time)
            az_value = 0.0
        else:
            elapsed = current_time - ignition_time
            z_value = powered_state(vertical_profile, elapsed, "z")
            vz_value = powered_state(vertical_profile, elapsed, "vz")
            az_value = powered_state(vertical_profile, elapsed, "az")

        if current_time < ignition_time:
            x_value, y_value, vx_value, vy_value, ax_value, ay_value = lateral_state(
                current_time,
                reference,
                scenario,
                ignition_time,
                plan,
            )
            tilt_override = None
        else:
            elapsed = current_time - ignition_time
            x_value = powered_state(powered_profile, elapsed, "x")
            y_value = powered_state(powered_profile, elapsed, "y")
            vx_value = powered_state(powered_profile, elapsed, "vx")
            vy_value = powered_state(powered_profile, elapsed, "vy")
            ax_value = powered_state(powered_profile, elapsed, "ax")
            ay_value = powered_state(powered_profile, elapsed, "ay")
            tilt_override = (
                powered_state(powered_profile, elapsed, "tilt_x"),
                powered_state(powered_profile, elapsed, "tilt_y"),
            )
        wind_x, wind_y = wind_at_time(current_time, fault_start, ignition_time, scenario)
        ml_value = ml_correction_at_time(current_time, fault_start, ignition_time, resolved_scenario)
        if tilt_override is not None:
            quaternion = quaternion_from_tilt(tilt_override[0], tilt_override[1])
        elif phase == PHASE_TUMBLING:
            u = clamp((current_time - ff_timing["tumble_start"]) / max(ff_timing["tumble_end"] - ff_timing["tumble_start"], 1e-6), 0.0, 1.0)
            angle = math.pi * smoothstep(u)
            quaternion = quaternion_normalize(quaternion_from_nose_angle(angle, tumble_axis))
        elif phase == PHASE_NOSE_DOWN:
            t_nd = current_time - ff_timing["tumble_end"]
            wobble = 0.04 * math.sin(1.8 * t_nd) * math.exp(-0.5 * t_nd)
            angle = math.pi + wobble
            quaternion = quaternion_normalize(quaternion_from_nose_angle(angle, tumble_axis))
        elif phase == PHASE_RCS_FLIP:
            dur = max(ff_timing["rcs_end"] - ff_timing["rcs_start"], 1e-6)
            u = clamp((current_time - ff_timing["rcs_start"]) / dur, 0.0, 1.0)
            angle = math.pi * (1.0 - smoothstep(u))
            quaternion = quaternion_normalize(quaternion_from_nose_angle(angle, tumble_axis))
        else:
            tilt_x, tilt_y = tilt_state(
                phase,
                current_time,
                ascent_end,
                ignition_time,
                landing_time,
                x_value,
                y_value,
                vx_value,
                vy_value,
                ax_value,
                ay_value,
                scenario,
            )
            quaternion = quaternion_from_tilt(tilt_x, tilt_y)
        mass_value = mass_at_time(current_time, reference, fault_start, resolved_scenario, ignition_time, landing_time)

        active_fault_code = fault_id if fault_start <= current_time < ignition_time else 0
        active_fault_mag = fault_magnitude(current_time, fault_start, ignition_time, scenario)

        records.append({
            "Time": encode_float(current_time),
            "X": encode_float(x_value),
            "Y": encode_float(y_value),
            "Z": encode_float(z_value),
            "VX": encode_float(vx_value),
            "VY": encode_float(vy_value),
            "VZ": encode_float(vz_value),
            "QW": encode_float(quaternion[0]),
            "QX": encode_float(quaternion[1]),
            "QY": encode_float(quaternion[2]),
            "QZ": encode_float(quaternion[3]),
            "Mass": encode_float(mass_value),
            "MLCorrection": encode_float(ml_value),
            "FaultType": active_fault_code,
            "FaultMag": encode_float(active_fault_mag),
            "WindX": encode_float(wind_x),
            "WindY": encode_float(wind_y),
            "Phase": phase,
        })
        current_time += DT

    final_quaternion = quaternion_from_tilt(float(powered_profile[-1]["tilt_x"]), float(powered_profile[-1]["tilt_y"]))
    final_mass = mass_at_time(final_time, reference, fault_start, resolved_scenario, ignition_time, landing_time)
    final_x = float(powered_profile[-1]["x"])
    final_y = float(powered_profile[-1]["y"])
    final_vx = float(powered_profile[-1]["vx"])
    final_vy = float(powered_profile[-1]["vy"])
    final_wind_x, final_wind_y = wind_at_time(landing_time, fault_start, ignition_time, scenario)
    final_ml = ml_correction_at_time(landing_time, fault_start, ignition_time, resolved_scenario)

    if bounce_profile:
        for bounce in bounce_profile[:-1]:
            absolute_time = landing_time + float(bounce["elapsed"])
            quaternion = quaternion_from_tilt(float(bounce["tilt_x"]), float(bounce["tilt_y"]))
            records.append({
                "Time": encode_float(absolute_time),
                "X": encode_float(float(bounce["x"])),
                "Y": encode_float(float(bounce["y"])),
                "Z": encode_float(float(bounce["z"])),
                "VX": encode_float(float(bounce["vx"])),
                "VY": encode_float(float(bounce["vy"])),
                "VZ": encode_float(float(bounce["vz"])),
                "QW": encode_float(quaternion[0]),
                "QX": encode_float(quaternion[1]),
                "QY": encode_float(quaternion[2]),
                "QZ": encode_float(quaternion[3]),
                "Mass": encode_float(final_mass),
                "MLCorrection": encode_float(final_ml),
                "FaultType": 0,
                "FaultMag": 0.0,
                "WindX": encode_float(final_wind_x),
                "WindY": encode_float(final_wind_y),
                "Phase": 2,
            })
        bounce_end = bounce_profile[-1]
        final_x = float(bounce_end["x"])
        final_y = float(bounce_end["y"])
        final_vx = float(bounce_end["vx"])
        final_vy = float(bounce_end["vy"])
        landing_vz = float(bounce_end["vz"])
        final_quaternion = quaternion_from_tilt(float(bounce_end["tilt_x"]), float(bounce_end["tilt_y"]))

    records.append({
        "Time": encode_float(final_time),
        "X": encode_float(final_x),
        "Y": encode_float(final_y),
        "Z": 0.0,
        "VX": encode_float(final_vx),
        "VY": encode_float(final_vy),
        "VZ": encode_float(landing_vz),
        "QW": encode_float(final_quaternion[0]),
        "QX": encode_float(final_quaternion[1]),
        "QY": encode_float(final_quaternion[2]),
        "QZ": encode_float(final_quaternion[3]),
        "Mass": encode_float(final_mass),
        "MLCorrection": encode_float(final_ml),
        "FaultType": 0,
        "FaultMag": 0.0,
        "WindX": encode_float(final_wind_x),
        "WindY": encode_float(final_wind_y),
        "Phase": 4,
    })

    summary = {
        "apogee": apogee_altitude,
        "peak_vertical_velocity": peak_vertical_velocity,
        "nominal_ignition_altitude": scenario.nominal_ignition_altitude,
        "effective_ignition_altitude": effective_ignition_altitude,
        "landing_time": final_time,
        "landing_distance": math.hypot(final_x, final_y),
        "landing_velocity": abs(landing_vz),
        "landing_x": final_x,
        "landing_y": final_y,
        "burn_duration": powered_duration,
        "ignition_time": ignition_time,
        "ignition_vz": ignition_vz,
    }
    return records, summary


def validate_run(records: Sequence[Dict[str, float]], scenario: Scenario) -> None:
    max_velocity_error = 0.0
    max_freefall_tilt = 0.0
    max_ascent_tilt = 0.0
    max_descent_tilt = 0.0
    max_burn_rebound = 0.0
    last_powered_tilt = 0.0

    for index in range(1, len(records)):
        previous = records[index - 1]
        current = records[index]
        dt_local = current["Time"] - previous["Time"]
        if dt_local <= 0.0:
            raise RuntimeError(f"Non-increasing time in {scenario.run_id}")

        if current["Phase"] == previous["Phase"]:
            dz_dt = (current["Z"] - previous["Z"]) / dt_local
            dx_dt = (current["X"] - previous["X"]) / dt_local
            dy_dt = (current["Y"] - previous["Y"]) / dt_local
            max_velocity_error = max(max_velocity_error, abs(dz_dt - previous["VZ"]))
            max_velocity_error = max(max_velocity_error, abs(dx_dt - previous["VX"]))
            max_velocity_error = max(max_velocity_error, abs(dy_dt - previous["VY"]))
            if current["Phase"] == 3:
                max_burn_rebound = max(max_burn_rebound, abs(current["VZ"]) - abs(previous["VZ"]))

        mass_delta = current["Mass"] - previous["Mass"]
        if mass_delta > 1e-6:
            raise RuntimeError(f"Mass increased in {scenario.run_id}")
        coast_phases = (1, 2, PHASE_TUMBLING, PHASE_NOSE_DOWN, PHASE_RCS_FLIP)
        mass_loss_coast = (
            "MASS_LOSS" in scenario.fault_types
            and previous["Phase"] in coast_phases
            and current["Phase"] in coast_phases
            and (previous["FaultType"] in (3, 4) or current["FaultType"] in (3, 4))
        )
        if abs(mass_delta) > 1e-6 and previous["Phase"] not in (0, 3) and current["Phase"] not in (0, 3) and not mass_loss_coast:
            raise RuntimeError(f"Mass changed outside burn phase in {scenario.run_id}")
        if current["Z"] < -1e-6:
            raise RuntimeError(f"Negative altitude in {scenario.run_id}")
        if previous["Phase"] in (3, 4) and (previous["FaultType"] != 0 or previous["FaultMag"] != 0):
            raise RuntimeError(f"Faults leaked into powered descent in {scenario.run_id}")

    for record in records:
        qx = record["QX"]
        qy = record["QY"]
        tilt_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt_cos = clamp(tilt_cos, -1.0, 1.0)
        tilt_deg = math.degrees(math.acos(tilt_cos))
        phase = record["Phase"]
        if phase == 0:
            max_ascent_tilt = max(max_ascent_tilt, tilt_deg)
        elif phase in (1, 2):
            max_freefall_tilt = max(max_freefall_tilt, tilt_deg)
        elif phase == 3:
            max_descent_tilt = max(max_descent_tilt, tilt_deg)
            last_powered_tilt = tilt_deg
        if tilt_deg >= 90.0 and phase not in (PHASE_TUMBLING, PHASE_NOSE_DOWN, PHASE_RCS_FLIP):
            raise RuntimeError(f"Vehicle inverted in {scenario.run_id}")

    if max_velocity_error > 12.0:
        raise RuntimeError(f"Velocity mismatch too large in {scenario.run_id}: {max_velocity_error:.3f}")
    if max_ascent_tilt > 15.5:
        raise RuntimeError(f"Ascent tilt too large in {scenario.run_id}: {max_ascent_tilt:.2f}")
    if max_freefall_tilt > 25.0:
        raise RuntimeError(f"Freefall tilt too large in {scenario.run_id}: {max_freefall_tilt:.2f}")
    if max_descent_tilt > 30.5:
        raise RuntimeError(f"Descent tilt too large in {scenario.run_id}: {max_descent_tilt:.2f}")
    if max_burn_rebound > 0.05:
        raise RuntimeError(f"Burn speed rebounded in {scenario.run_id}: {max_burn_rebound:.3f}")
    if scenario.success and last_powered_tilt > SUCCESS_TERMINAL_TILT_DEG:
        raise RuntimeError(f"Touchdown attitude not upright enough in {scenario.run_id}: {last_powered_tilt:.2f}")


def build_config_json(scenario: Scenario, summary: Dict[str, float]) -> Dict[str, object]:
    return {
        "rocket": {
            "dry_mass": BASE_DRY_MASS,
            "wet_mass": BASE_TOTAL_MASS,
            "diameter": BASE_DIAMETER,
            "thrust_average": BASE_THRUST_AVG,
            "non_throttleable_descent_motor": True,
        },
        "environment": {
            "wind_speed": scenario.wind_speed,
            "wind_heading_deg": scenario.wind_heading_deg,
            "drag_coefficient": scenario.drag_coefficient,
            "air_density": scenario.air_density,
        },
        "simulation": {
            "synthetic": True,
            "synthetic_source": "results/optimization_20260222_120325/trajectory.csv",
            "time_step": DT,
            "nominal_ignition_altitude": summary["nominal_ignition_altitude"],
            "effective_ignition_altitude": summary["effective_ignition_altitude"],
            "descent_burn_duration": summary["burn_duration"],
        },
        "demo_metadata": {
            "run_id": scenario.run_id,
            "run_name": scenario.name,
            "description": scenario.description,
            "mode": scenario.mode,
            "fault_intensity": scenario.fault_intensity,
            "fault_types": list(scenario.fault_types),
            "success": scenario.success,
            "landing_velocity": summary["landing_velocity"],
            "landing_distance": summary["landing_distance"],
            "apogee": summary["apogee"],
            "peak_vertical_velocity": summary["peak_vertical_velocity"],
        },
    }


def write_csv(path: Path, records: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def manifest_entry(scenario: Scenario, summary: Dict[str, float]) -> Dict[str, object]:
    landing_velocity = float(summary["landing_velocity"])
    status_label = "SUCCESS" if scenario.success else ("FAIL" if landing_velocity < 10.0 else "CRASH")
    color = "#4CAF50" if scenario.success else ("#FF9800" if landing_velocity < 10.0 else "#F44336")
    return {
        "id": scenario.run_id,
        "name": scenario.name,
        "description": scenario.description,
        "mode": scenario.mode,
        "fault_intensity": scenario.fault_intensity,
        "fault_types": list(scenario.fault_types),
        "success": scenario.success,
        "landing_velocity": round(float(summary["landing_velocity"]), 6),
        "landing_x": round(float(summary["landing_x"]), 6),
        "landing_y": round(float(summary["landing_y"]), 6),
        "landing_distance": round(float(summary["landing_distance"]), 6),
        "dry_mass": BASE_DRY_MASS,
        "propellant_mass": BASE_PROPELLANT_MASS,
        "diameter": BASE_DIAMETER,
        "thrust_average": BASE_THRUST_AVG,
        "wind_speed": scenario.wind_speed,
        "drag_coefficient": scenario.drag_coefficient,
        "air_density": scenario.air_density,
        "initial_altitude": 0.0,
        "initial_velocity": 0.0,
        "apogee": round(float(summary["apogee"]), 6),
        "peak_vertical_velocity": round(float(summary["peak_vertical_velocity"]), 6),
        "nominal_ignition_altitude": round(float(summary["nominal_ignition_altitude"]), 6),
        "effective_ignition_altitude": round(float(summary["effective_ignition_altitude"]), 6),
        "burn_duration": round(float(summary["burn_duration"]), 6),
        "trajectory_path": f"{scenario.run_id}/trajectory.csv",
        "config_path": f"{scenario.run_id}/config.json",
        "color": color,
        "status_label": status_label,
    }


def generate_demo_runs() -> Dict[str, object]:
    reference = load_reference()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "3.1",
        "generated": "2026-03-11",
        "description": "Synthetic demo trajectories with physics-based nose-down freefall (aerodynamic tumble after apogee) and cold-gas RCS flip maneuver before the solid-motor landing burn.",
        "runs": [],
    }

    for scenario in SCENARIOS:
        run_dir = OUTPUT_DIR / scenario.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        records, summary = synthesize_run(reference, scenario)
        validate_run(records, scenario)
        write_csv(run_dir / "trajectory.csv", records)

        with (run_dir / "config.json").open("w", newline="") as handle:
            json.dump(build_config_json(scenario, summary), handle, indent=2)

        manifest["runs"].append(manifest_entry(scenario, summary))
        print(
            f"[{scenario.run_id}] rows={len(records)} apogee={summary['apogee']:.1f} m "
            f"ign={summary['effective_ignition_altitude']:.2f} m burn={summary['burn_duration']:.2f} s "
            f"landing_v={summary['landing_velocity']:.2f} m/s"
        )

    with (OUTPUT_DIR / "demo_manifest.json").open("w", newline="") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def main() -> None:
    print("=" * 68)
    print("HexaVisual Demo - Synthetic trajectory generator")
    print(f"Reference: {REFERENCE_TRAJECTORY.relative_to(ROOT)}")
    print("No live simulation is run; all demo trajectories are synthesized.")
    print("=" * 68)
    generate_demo_runs()
    print(f"\nWrote demo runs to: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()