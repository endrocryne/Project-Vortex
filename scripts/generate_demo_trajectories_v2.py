"""
Generate Demo Trajectories for HexaVisual Demo
================================================
Produces 9 physics-based trajectory CSV files using RK4 integration for the
HexaVisual Demo app.  Each trajectory matches a specific data point from
the PlotVisual sample_visualization_data.csv (seed=42).

Physics model (3DOF + orientation):
  - Vertical: thrust - drag - gravity
  - Lateral: wind force + drag coupling
  - Drag: 0.5 * rho * Cd * A * |v_rel|^2 opposing v_rel
  - Thrust: F15 thrust-curve shape, scaled per-run to match thrust_average
  - Mass: decreases linearly during burns
  - Quaternion: velocity-aligned (aerodynamic weathercock) with TVC wobble

The descent motor is a SOLID MOTOR with the F15 curve shape — no throttle
capability.  The optimization precomputes the exact ignition altitude.
The ML model outputs a real-time correction to this altitude in response
to faults detected during coast-down.

Extended 18-column output:
  Time, X, Y, Z, VX, VY, VZ, QW, QX, QY, QZ, Mass,
  MLCorrection, FaultType, FaultMag, WindX, WindY, Phase

Phase: 0=ascent_burn, 1=coast_up, 2=coast_down, 3=descent_burn, 4=landed
FaultType: 0=none, 1=WIND_GUST, 2=DRAG_CHANGE, 3=MASS_LOSS, 4=combined

Output per run:
  results/demo_runs/demo_XX_name/trajectory.csv
  results/demo_runs/demo_XX_name/config.json
  results/demo_runs/demo_manifest.json
"""

import os
import json
import math
import csv

# ------------------------------------------------------------------
# Demo run definitions — all values from sample_visualization_data.csv
# ------------------------------------------------------------------
DEMO_RUNS = [
    {
        "id": "demo_01_baseline",
        "name": "Baseline — No Faults",
        "description": "Ideal conditions. Optimization-only, no faults, minimal wind. Both methods succeed.",
        "mode": "Optimization",
        "fault_intensity": 0.00,
        "fault_types": [],
        "success": True,
        "landing_velocity": 0.498986531356434,
        "landing_x": -0.44414738158550776,
        "landing_y": -0.4015520314567355,
        "landing_distance": 0.5987578229437122,
        "dry_mass": 52.49080237694725,
        "propellant_mass": 11.802857225639665,
        "diameter": 0.3731993941811405,
        "thrust_average": 1019.7316968394073,
        "wind_speed": 1.8722236853092382,
        "drag_coefficient": 0.48119890406724053,
        "air_density": 1.15580836121682,
        "initial_altitude": 1246.470458309974,
        "initial_velocity": 57.022300234864176,
    },
    {
        "id": "demo_02_wind_opt",
        "name": "Wind 5 mph NE — Optimization",
        "description": "Northeast wind gust (~10 m/s). Optimization only, no ML correction. Significant drift and high landing velocity.",
        "mode": "Optimization",
        "fault_intensity": 0.34,
        "fault_types": ["WIND_GUST"],
        "success": False,
        "landing_velocity": 3.811029331656223,
        "landing_x": 32.448740570388836,
        "landing_y": 14.325259954502004,
        "landing_distance": 35.47018237010429,
        "dry_mass": 54.23432287626238,
        "propellant_mass": 11.683978084386304,
        "diameter": 0.36945954272532855,
        "thrust_average": 1045.7962123311634,
        "wind_speed": 10.340290845448143,
        "drag_coefficient": 0.5048143178684382,
        "air_density": 1.2307070903866704,
        "initial_altitude": 978.0962411825968,
        "initial_velocity": 51.90683904489776,
    },
    {
        "id": "demo_03_wind_ml",
        "name": "Wind 5 mph NE — ML Corrected",
        "description": "Same wind conditions as Run 2, but ML flight computer provides real-time ignition altitude correction. Successful soft landing.",
        "mode": "ML",
        "fault_intensity": 0.34,
        "fault_types": ["WIND_GUST"],
        "success": True,
        "landing_velocity": 0.6702889154737633,
        "landing_x": -0.03308587577240832,
        "landing_y": 3.95386506312709,
        "landing_distance": 3.9540034917274185,
        "dry_mass": 54.23432287626238,
        "propellant_mass": 11.683978084386304,
        "diameter": 0.36945954272532855,
        "thrust_average": 1045.7962123311634,
        "wind_speed": 10.340290845448143,
        "drag_coefficient": 0.5048143178684382,
        "air_density": 1.2307070903866704,
        "initial_altitude": 978.0962411825968,
        "initial_velocity": 51.90683904489776,
    },
    {
        "id": "demo_04_dragmass_opt",
        "name": "Drag + Mass Loss — Optimization",
        "description": "Drag coefficient increase and mass loss fault at fi=0.54. Optimization alone cannot compensate — high-speed landing.",
        "mode": "Optimization",
        "fault_intensity": 0.54,
        "fault_types": ["DRAG_CHANGE", "MASS_LOSS"],
        "success": False,
        "landing_velocity": 13.659409178715887,
        "landing_x": 26.60563108315804,
        "landing_y": 0.6546830028925269,
        "landing_distance": 26.613684734876188,
        "dry_mass": 45.71180461612953,
        "propellant_mass": 11.97732121074286,
        "diameter": 0.32327283489155517,
        "thrust_average": 910.7661049816918,
        "wind_speed": 2.2785039491510397,
        "drag_coefficient": 0.45779156926969283,
        "air_density": 1.1891810894617172,
        "initial_altitude": 1220.9607137805297,
        "initial_velocity": 51.82488899515441,
    },
    {
        "id": "demo_05_dragmass_ml",
        "name": "Drag + Mass Loss — ML Corrected",
        "description": "Same faults as Run 4, but ML flight computer adapts ignition timing to compensate. Successful soft landing.",
        "mode": "ML",
        "fault_intensity": 0.54,
        "fault_types": ["DRAG_CHANGE", "MASS_LOSS"],
        "success": True,
        "landing_velocity": 0.6713041470737156,
        "landing_x": -0.1134943573430623,
        "landing_y": 1.2647657789194375,
        "landing_distance": 1.2698478037443723,
        "dry_mass": 45.71180461612953,
        "propellant_mass": 11.97732121074286,
        "diameter": 0.32327283489155517,
        "thrust_average": 910.7661049816918,
        "wind_speed": 2.2785039491510397,
        "drag_coefficient": 0.45779156926969283,
        "air_density": 1.1891810894617172,
        "initial_altitude": 1220.9607137805297,
        "initial_velocity": 51.82488899515441,
    },
    {
        "id": "demo_06_severe_opt",
        "name": "Severe Combined — Optimization",
        "description": "Wind + drag change + mass loss at fi=0.72. Optimization cannot handle combined faults — vehicle crashes at 28.6 m/s.",
        "mode": "Optimization",
        "fault_intensity": 0.72,
        "fault_types": ["WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"],
        "success": False,
        "landing_velocity": 28.62505235898039,
        "landing_x": 38.38032134911251,
        "landing_y": -6.262604993364843,
        "landing_distance": 38.88790670843649,
        "dry_mass": 58.054562475544486,
        "propellant_mass": 9.022664893327258,
        "diameter": 0.33645664275198645,
        "thrust_average": 1068.7361059329955,
        "wind_speed": 2.339313227641918,
        "drag_coefficient": 0.6153735694346725,
        "air_density": 1.2062655422902742,
        "initial_altitude": 1204.875004477633,
        "initial_velocity": 59.27214732526943,
    },
    {
        "id": "demo_07_severe_ml",
        "name": "Severe Combined — ML Corrected",
        "description": "Same severe conditions as Run 6. ML flight computer compensates in real-time — successful landing at 0.89 m/s despite fi=0.72.",
        "mode": "ML",
        "fault_intensity": 0.72,
        "fault_types": ["WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"],
        "success": True,
        "landing_velocity": 0.8914751439392377,
        "landing_x": -1.804784162432377,
        "landing_y": -5.83089549225807,
        "landing_distance": 6.103817503382798,
        "dry_mass": 58.054562475544486,
        "propellant_mass": 9.022664893327258,
        "diameter": 0.33645664275198645,
        "thrust_average": 1068.7361059329955,
        "wind_speed": 2.339313227641918,
        "drag_coefficient": 0.6153735694346725,
        "air_density": 1.2062655422902742,
        "initial_altitude": 1204.875004477633,
        "initial_velocity": 59.27214732526943,
    },
    {
        "id": "demo_08_extreme_opt",
        "name": "Extreme Faults — Optimization",
        "description": "Extreme combined faults at fi=0.86. Optimization completely overwhelmed — catastrophic 60.2 m/s impact.",
        "mode": "Optimization",
        "fault_intensity": 0.86,
        "fault_types": ["WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"],
        "success": False,
        "landing_velocity": 60.19723309082299,
        "landing_x": -16.264536604952244,
        "landing_y": 0.3614920093948118,
        "landing_distance": 16.268553329865806,
        "dry_mass": 52.345554025614184,
        "propellant_mass": 11.655258090499776,
        "diameter": 0.336829546765571,
        "thrust_average": 974.6889182290871,
        "wind_speed": 4.602414149207723,
        "drag_coefficient": 0.6173240420735316,
        "air_density": 1.2150788320846901,
        "initial_altitude": 1145.667540127557,
        "initial_velocity": 51.68286044767724,
    },
    {
        "id": "demo_09_extreme_ml",
        "name": "Extreme Faults — ML (Graceful Degradation)",
        "description": "Same extreme conditions as Run 8. ML cannot fully compensate (fi=0.86 exceeds threshold) but degrades gracefully at 4.6 m/s vs 60.2 m/s.",
        "mode": "ML",
        "fault_intensity": 0.86,
        "fault_types": ["WIND_GUST", "DRAG_CHANGE", "MASS_LOSS"],
        "success": False,
        "landing_velocity": 4.632771296493519,
        "landing_x": 9.707766826551826,
        "landing_y": -17.692407605121364,
        "landing_distance": 20.18073397140095,
        "dry_mass": 52.345554025614184,
        "propellant_mass": 11.655258090499776,
        "diameter": 0.336829546765571,
        "thrust_average": 974.6889182290871,
        "wind_speed": 4.602414149207723,
        "drag_coefficient": 0.6173240420735316,
        "air_density": 1.2150788320846901,
        "initial_altitude": 1145.667540127557,
        "initial_velocity": 51.68286044767724,
    },
]

# ------------------------------------------------------------------
# F15 thrust curve SHAPE (normalized: time fraction 0-1, thrust fraction 0-1)
# ------------------------------------------------------------------
_RAW_F15 = [
    (0.0, 0.0), (0.013, 89.1), (0.018, 101.6), (0.029, 105.4),
    (0.047, 102.9), (0.104, 100.0), (0.19, 102.3), (0.268, 104.9),
    (0.306, 104.3), (0.38, 97.4), (0.45, 92.0), (0.6, 88.5),
    (0.75, 83.0), (0.9, 78.0), (1.05, 72.0), (1.2, 65.0),
    (1.38, 48.0), (1.5, 28.0), (1.6, 10.0), (1.7, 0.0),
]
_F15_PEAK = max(f for _, f in _RAW_F15)
_F15_BURN = _RAW_F15[-1][0]

# Normalized shape: time_frac in [0,1], thrust_frac in [0,1]
F15_SHAPE = [(t / _F15_BURN, f / _F15_PEAK) for t, f in _RAW_F15]

G = 9.81


def sample_thrust_shape(t_frac):
    """Interpolate normalized F15 shape at time fraction t_frac in [0,1]."""
    if t_frac <= 0 or t_frac >= 1.0:
        return 0.0
    for i in range(len(F15_SHAPE) - 1):
        tf0, sf0 = F15_SHAPE[i]
        tf1, sf1 = F15_SHAPE[i + 1]
        if tf0 <= t_frac <= tf1:
            frac = (t_frac - tf0) / max(tf1 - tf0, 1e-12)
            return sf0 + (sf1 - sf0) * frac
    return 0.0


def shape_average():
    """Average of the normalized thrust shape (trapezoidal)."""
    total = 0.0
    for i in range(len(F15_SHAPE) - 1):
        tf0, sf0 = F15_SHAPE[i]
        tf1, sf1 = F15_SHAPE[i + 1]
        total += 0.5 * (sf0 + sf1) * (tf1 - tf0)
    return total


# ------------------------------------------------------------------
# Quaternion utilities
# ------------------------------------------------------------------
def quat_identity():
    return (1.0, 0.0, 0.0, 0.0)


def quat_from_axis_angle(axis, angle):
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return quat_identity()
    ax, ay, az = ax / norm, ay / norm, az / norm
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return (c, ax * s, ay * s, az * s)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quat_slerp(q1, q2, t):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = w1*w2 + x1*x2 + y1*y2 + z1*z2
    if dot < 0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot
    if dot > 0.9995:
        w = w1 + t * (w2 - w1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        return (w / norm, x / norm, y / norm, z / norm)
    theta = math.acos(max(-1, min(1, dot)))
    sin_theta = math.sin(theta)
    if sin_theta < 1e-9:
        return q1
    a = math.sin((1 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    return (a*w1 + b*w2, a*x1 + b*x2, a*y1 + b*y2, a*z1 + b*z2)


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return quat_identity()
    return (w / n, x / n, y / n, z / n)


# ------------------------------------------------------------------
# Fault encoding
# ------------------------------------------------------------------
FAULT_CODE = {
    "none": 0, "WIND_GUST": 1, "DRAG_CHANGE": 2, "MASS_LOSS": 3, "combined": 4,
}


def encode_fault_type(fault_types):
    if not fault_types:
        return 0
    if len(fault_types) > 1:
        return 4
    return FAULT_CODE.get(fault_types[0], 0)


def wind_direction_vector(run):
    lx = run["landing_x"]
    ly = run["landing_y"]
    dist = math.sqrt(lx * lx + ly * ly)
    if dist < 0.01:
        return (1.0, 0.0)
    return (lx / dist, ly / dist)


# ------------------------------------------------------------------
# Motor parameter computation per-run
# ------------------------------------------------------------------
def compute_motor_params(run):
    """
    Compute ascent and descent motor parameters from the demo run data.
    The F15 thrust curve shape is used; magnitude and time axis are scaled.
    """
    dry = run["dry_mass"]
    prop = run["propellant_mass"]
    thr_avg = run["thrust_average"]
    Cd = run["drag_coefficient"]
    rho = run["air_density"]
    diam = run["diameter"]
    ref_area = math.pi * (diam / 2.0) ** 2

    shape_avg = shape_average()
    peak_thrust = thr_avg / shape_avg

    # Loaded mass at launch: dry + ascent_prop + descent_prop + casing
    casing_mass = prop * 0.15
    m0 = dry + 2 * prop + casing_mass

    # Find ascent burn time for target apogee ~260m
    target_apogee = 260.0
    bt_lo, bt_hi = 2.0, 30.0
    dt_s = 0.02

    for _ in range(45):
        bt_mid = (bt_lo + bt_hi) / 2.0
        mf = prop / bt_mid
        v = 0.0
        z = 0.0
        m = m0
        t_s = 0.0
        while t_s < bt_mid:
            t_frac = t_s / bt_mid
            thr = peak_thrust * sample_thrust_shape(t_frac)
            v_abs = abs(v)
            drag = 0.5 * rho * Cd * ref_area * v_abs * v_abs if v_abs > 0.01 else 0.0
            drag_sign = 1.0 if v >= 0 else -1.0
            acc = thr / m - G - drag_sign * drag / m
            v += acc * dt_s
            z += v * dt_s
            m = max(dry + prop, m - mf * dt_s)
            t_s += dt_s
        # Casing ejection
        m -= casing_mass
        m = max(dry + prop, m)
        # Coast to apogee
        while v > 0 and z > 0:
            v_abs = abs(v)
            drag = 0.5 * rho * Cd * ref_area * v_abs * v_abs if v_abs > 0.01 else 0.0
            acc = -G - drag / m
            v += acc * dt_s
            z += v * dt_s
            t_s += dt_s
        if z > target_apogee:
            bt_hi = bt_mid
        else:
            bt_lo = bt_mid

    ascent_burn_time = (bt_lo + bt_hi) / 2.0

    # Coast mass after ascent burn + casing ejection
    m_coast = dry + prop  # descent propellant still onboard

    # Find descent motor sizing:
    # Coast from apogee, find velocity at various altitudes, then size descent motor
    v = 0.0
    z = target_apogee
    v_at_z = []
    while z > 0.5:
        v_abs = abs(v)
        drag = 0.5 * rho * Cd * ref_area * v_abs * v_abs if v_abs > 0.01 else 0.0
        drag_sign = 1.0 if v >= 0 else -1.0
        acc = -G - drag_sign * drag / m_coast
        v += acc * dt_s
        z += v * dt_s
        v_at_z.append((z, v))

    # Terminal velocity in this regime
    v_terminal = max(abs(vv) for _, vv in v_at_z) if v_at_z else 60.0

    # Descent motor sizing: match real rocket's TWR ratio during descent
    # Real F15 rocket: descent TWR = avg_thrust / (m_coast * g) ≈ 5.73
    # This gives: short burn (~1.2s real), low ignition altitude (~14% of apogee)
    DESCENT_TWR = 5.73
    descent_avg_thrust = DESCENT_TWR * m_coast * G
    descent_peak_thrust = descent_avg_thrust / shape_avg

    # Ignition altitude estimate: ~14% of apogee (matching real trajectory)
    ign_alt_frac = 0.14
    ign_alt_est = target_apogee * ign_alt_frac

    # Find velocity at estimated ignition altitude
    v_at_ign = v_terminal  # fallback
    for z_i, v_i in v_at_z:
        if z_i <= ign_alt_est:
            v_at_ign = abs(v_i)
            break

    # Descent burn time from impulse balance:
    # impulse = m_coast * v_at_ign, net deceleration = descent_avg - m_coast*G
    net_decel_force = descent_avg_thrust - m_coast * G
    if net_decel_force > 10.0:
        descent_burn_time = m_coast * v_at_ign / net_decel_force
    else:
        descent_burn_time = 2.0  # fallback
    descent_burn_time = max(0.5, min(descent_burn_time, 5.0))

    # Descent propellant from mass flow rate (same prop/burn ratio as ascent shape)
    descent_prop = prop * (descent_burn_time / ascent_burn_time)
    mf_desc = descent_prop / descent_burn_time

    return {
        "ascent_burn_time": ascent_burn_time,
        "descent_burn_time": descent_burn_time,
        "ascent_peak_thrust": peak_thrust,
        "descent_peak_thrust": descent_peak_thrust,
        "casing_mass": casing_mass,
        "mass_flow_ascent": prop / ascent_burn_time,
        "mass_flow_descent": mf_desc,
        "descent_prop": descent_prop,
        "m0_launch": m0,
        "m_coast": m_coast,
        "m_dry": dry,
    }


# ------------------------------------------------------------------
# Descent simulation (for ignition altitude binary search)
# ------------------------------------------------------------------
def simulate_descent_burn(z0, vz0, vx0, vy0, mass, Cd_eff, rho, ref_area,
                          wind_x, wind_y, peak_thrust, burn_time, mass_flow,
                          dry_mass, dt=0.02):
    """
    Forward-simulate the descent burn with the solid motor using RK4.
    Returns (final_vz, final_z, went_up).
    went_up = True if the rocket reversed direction upward (ignited too high).
    """
    # State: [x, y, z, vx, vy, vz, mass]
    s = [0.0, 0.0, z0, vx0, vy0, vz0, mass]
    t = 0.0
    max_t = burn_time + 10.0
    went_up = False
    max_z = z0

    def deriv(st, tl):
        _, _, _z, svx, svy, svz, sm = st
        sm = max(sm, dry_mass)
        if tl < burn_time:
            tf = tl / burn_time
            thr = peak_thrust * sample_thrust_shape(tf)
        else:
            thr = 0.0
        vx_rel = svx - wind_x
        vy_rel = svy - wind_y
        v_rel = math.sqrt(vx_rel**2 + vy_rel**2 + svz**2)
        if v_rel > 0.01:
            drag_mag = 0.5 * rho * Cd_eff * ref_area * v_rel * v_rel
            axd = -drag_mag * vx_rel / v_rel / sm
            ayd = -drag_mag * vy_rel / v_rel / sm
            azd = -drag_mag * svz / v_rel / sm
        else:
            axd = ayd = azd = 0.0
        azt = thr / sm if thr > 0 else 0.0
        mdot = -mass_flow if (thr > 0 and tl < burn_time) else 0.0
        return (svx, svy, svz, axd, ayd, -G + azd + azt, mdot)

    while s[2] > 0.0 and t < max_t:
        d1 = deriv(s, t)
        k1 = [s[3], s[4], s[5], d1[3], d1[4], d1[5], d1[6]]
        s2 = [s[i] + 0.5 * dt * k1[i] for i in range(7)]
        d2 = deriv(s2, t + 0.5 * dt)
        k2 = [s2[3], s2[4], s2[5], d2[3], d2[4], d2[5], d2[6]]
        s3 = [s[i] + 0.5 * dt * k2[i] for i in range(7)]
        d3 = deriv(s3, t + 0.5 * dt)
        k3 = [s3[3], s3[4], s3[5], d3[3], d3[4], d3[5], d3[6]]
        s4 = [s[i] + dt * k3[i] for i in range(7)]
        d4 = deriv(s4, t + dt)
        k4 = [s4[3], s4[4], s4[5], d4[3], d4[4], d4[5], d4[6]]

        for i in range(7):
            s[i] += dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        s[6] = max(s[6], dry_mass)
        t += dt

        if s[2] > max_z + 1.0:
            went_up = True
            max_z = s[2]

        if s[2] <= 0:
            break

    return s[5], s[2], went_up


# ------------------------------------------------------------------
# Main trajectory generator
# ------------------------------------------------------------------
def generate_trajectory(run):
    """
    Generate full launch-to-landing trajectory with RK4 integration.

    For successful runs: binary-search ignition altitude for soft landing.
    For failed runs: faults shift physics + nominal ignition altitude -> hard landing.
    ML runs: real-time correction to ignition altitude.
    """
    dt = 0.02

    fi = float(run["fault_intensity"])
    is_ml = run["mode"] == "ML"
    success = bool(run["success"])

    mp = compute_motor_params(run)
    dry = mp["m_dry"]
    m0 = mp["m0_launch"]
    m_coast = mp["m_coast"]
    ascent_bt = mp["ascent_burn_time"]
    descent_bt = mp["descent_burn_time"]
    ascent_peak_thr = mp["ascent_peak_thrust"]
    descent_peak_thr = mp["descent_peak_thrust"]
    mf_asc = mp["mass_flow_ascent"]
    mf_desc = mp["mass_flow_descent"]
    casing_mass = mp["casing_mass"]

    diam = float(run["diameter"])
    ref_area = math.pi * (diam / 2.0) ** 2
    Cd_base = float(run["drag_coefficient"])
    rho = float(run["air_density"])

    wx_dir, wy_dir = wind_direction_vector(run)
    has_wind_fault = "WIND_GUST" in run["fault_types"]
    has_drag_fault = "DRAG_CHANGE" in run["fault_types"]
    has_mass_fault = "MASS_LOSS" in run["fault_types"]
    fault_code_val = encode_fault_type(run["fault_types"])

    fault_trigger_offset = 2.0
    mass_loss_amount = m_coast * fi * 0.003
    drag_multiplier_fault = 1.0 + fi * 0.8
    wind_gust_magnitude = float(run["wind_speed"]) * (0.5 + fi * 1.5)

    base_wind_x = wx_dir * float(run["wind_speed"]) * 0.3
    base_wind_y = wy_dir * float(run["wind_speed"]) * 0.3

    # ── Find optimal ignition altitude via calibration run ──
    # Run the full RK4 trajectory with NO descent ignition, NO faults to build
    # an accurate altitude-velocity table for the coast-down phase.
    cal_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m0]
    cal_t = 0.0
    cal_phase = 0
    alt_vel_table = []

    while cal_t < 40.0:
        cx, cy, cz, cvx, cvy, cvz, cm = cal_state

        if cz < 0 and cal_t > 0.5:
            break

        if cal_phase == 0 and cal_t >= ascent_bt:
            cal_phase = 1
            cal_state[6] = max(m_coast, cal_state[6] - casing_mass)
        if cal_phase == 1 and cvz <= 0.0 and cal_t > ascent_bt + 0.1:
            cal_phase = 2
        if cal_phase == 2:
            alt_vel_table.append((cz, cvz, cvx, cvy, cm))
            if cz <= 0.01 and cal_t > ascent_bt + 1.0:
                break

        def cal_deriv(s, tl):
            _x, _y, _z, svx, svy, svz, sm = s
            sm = max(sm, dry)
            if cal_phase == 0:
                tf = tl / ascent_bt
                thr = ascent_peak_thr * sample_thrust_shape(tf)
                mdot = -mf_asc if thr > 0 else 0.0
            else:
                thr = 0.0
                mdot = 0.0
            vx_r = svx - base_wind_x
            vy_r = svy - base_wind_y
            v_rel = math.sqrt(vx_r**2 + vy_r**2 + svz**2)
            if v_rel > 0.01:
                drag_mag = 0.5 * rho * Cd_base * ref_area * v_rel * v_rel
                axd = -drag_mag * vx_r / v_rel / sm
                ayd = -drag_mag * vy_r / v_rel / sm
                azd = -drag_mag * svz / v_rel / sm
            else:
                axd = ayd = azd = 0.0
            azt = thr / sm if thr > 0 else 0.0
            return (axd, ayd, -G + azd + azt, mdot)

        cvx, cvy, cvz = cal_state[3], cal_state[4], cal_state[5]
        d1 = cal_deriv(cal_state, cal_t)
        k1 = [cvx, cvy, cvz, d1[0], d1[1], d1[2], d1[3]]
        s2 = [cal_state[i] + 0.5 * dt * k1[i] for i in range(7)]
        d2 = cal_deriv(s2, cal_t + 0.5 * dt)
        k2 = [s2[3], s2[4], s2[5], d2[0], d2[1], d2[2], d2[3]]
        s3 = [cal_state[i] + 0.5 * dt * k2[i] for i in range(7)]
        d3 = cal_deriv(s3, cal_t + 0.5 * dt)
        k3 = [s3[3], s3[4], s3[5], d3[0], d3[1], d3[2], d3[3]]
        s4 = [cal_state[i] + dt * k3[i] for i in range(7)]
        d4 = cal_deriv(s4, cal_t + dt)
        k4 = [s4[3], s4[4], s4[5], d4[0], d4[1], d4[2], d4[3]]

        for i in range(7):
            cal_state[i] += dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        cal_state[6] = max(cal_state[6], dry)
        cal_t += dt

    # Find the optimal ignition altitude by scanning the altitude-velocity table.
    # Two-pass: coarse scan, then fine scan around best altitude.
    if not alt_vel_table:
        precomputed_ign_alt = 36.0
    else:
        best_ign_alt = 36.0
        best_landing_vz = 999.0

        # Pass 1: coarse scan (every nth entry)
        step = max(1, len(alt_vel_table) // 80)
        candidates = list(range(0, len(alt_vel_table), step))
        if candidates[-1] != len(alt_vel_table) - 1:
            candidates.append(len(alt_vel_table) - 1)

        for idx in candidates:
            z_try, vz_try, vx_try, vy_try, m_try = alt_vel_table[idx]
            if z_try < 5.0 or z_try > 200.0:
                continue

            final_vz, _, went_up = simulate_descent_burn(
                z_try, vz_try, vx_try, vy_try, m_try,
                Cd_base, rho, ref_area, base_wind_x, base_wind_y,
                descent_peak_thr, descent_bt, mf_desc, dry
            )

            landing_speed = abs(final_vz)
            if landing_speed < best_landing_vz:
                best_landing_vz = landing_speed
                best_ign_alt = z_try

        # Pass 2: fine scan ±3m around best altitude (every entry)
        for i, (z_try, vz_try, vx_try, vy_try, m_try) in enumerate(alt_vel_table):
            if abs(z_try - best_ign_alt) > 3.0:
                continue
            if z_try < 5.0:
                continue

            final_vz, _, went_up = simulate_descent_burn(
                z_try, vz_try, vx_try, vy_try, m_try,
                Cd_base, rho, ref_area, base_wind_x, base_wind_y,
                descent_peak_thr, descent_bt, mf_desc, dry
            )

            landing_speed = abs(final_vz)
            if landing_speed < best_landing_vz:
                best_landing_vz = landing_speed
                best_ign_alt = z_try

        precomputed_ign_alt = best_ign_alt
        print(f"    Scan: best_ign_alt={best_ign_alt:.2f}m, predicted_vz={best_landing_vz:.3f} m/s, table_len={len(alt_vel_table)}")

        # If the best predicted landing speed is > 1.5 m/s, jointly optimize
        # descent_bt and ignition altitude for the softest possible landing.
        if best_landing_vz > 1.5:
            gr = (math.sqrt(5) + 1) / 2
            global_best_vz = best_landing_vz
            global_best_alt = best_ign_alt
            global_best_bt_mult = 1.0

            # Test every altitude; for each, golden-section search bt_mult
            for idx in range(len(alt_vel_table)):
                z0, vz0, vx0, vy0, m0_ = alt_vel_table[idx]
                if z0 < 10.0 or z0 > 150.0:
                    continue

                def eval_bt_at_alt(mult):
                    fvz, _, _ = simulate_descent_burn(
                        z0, vz0, vx0, vy0, m0_,
                        Cd_base, rho, ref_area, base_wind_x, base_wind_y,
                        descent_peak_thr, descent_bt * mult, mf_desc, dry
                    )
                    return abs(fvz)

                a, b = 0.7, 1.3
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                for _ in range(20):
                    if eval_bt_at_alt(c) < eval_bt_at_alt(d):
                        b = d
                    else:
                        a = c
                    c = b - (b - a) / gr
                    d = a + (b - a) / gr
                opt_mult = (a + b) / 2.0
                opt_vz = eval_bt_at_alt(opt_mult)

                if opt_vz < global_best_vz:
                    global_best_vz = opt_vz
                    global_best_alt = z0
                    global_best_bt_mult = opt_mult

            if global_best_vz < best_landing_vz:
                descent_bt = descent_bt * global_best_bt_mult
                precomputed_ign_alt = global_best_alt
                print(f"    Joint opt: ign_alt={global_best_alt:.2f}m, bt_mult={global_best_bt_mult:.4f}, bt={descent_bt:.3f}s, predicted_vz={global_best_vz:.3f}")

    # ── Faulted calibration: find the ideal ignition altitude under fault conditions ──
    # This is what the ML model would compute — the correct ignition altitude
    # accounting for changed drag, mass, and wind from faults.
    faulted_ign_alt = precomputed_ign_alt  # default: same as clean
    if fi > 0 and is_ml:
        # Run a faulted coast-down from apogee with fault effects applied
        # Reuse the clean calibration's apogee state
        apogee_idx = 0
        for i, (z_i, vz_i, _, _, _) in enumerate(alt_vel_table):
            if i == 0:
                apogee_idx = i
                break

        # Build faulted alt_vel_table from the clean one by re-simulating coast
        # with modified Cd and mass from after fault activation
        Cd_faulted = Cd_base * drag_multiplier_fault if has_drag_fault else Cd_base
        mass_faulted = max(dry, m_coast - mass_loss_amount) if has_mass_fault else m_coast
        wind_x_faulted = base_wind_x + (wx_dir * wind_gust_magnitude if has_wind_fault else 0.0)
        wind_y_faulted = base_wind_y + (wy_dir * wind_gust_magnitude if has_wind_fault else 0.0)

        # Re-simulate from apogee with faulted physics
        fc_state = [0.0, 0.0, alt_vel_table[0][0], alt_vel_table[0][2], alt_vel_table[0][3], alt_vel_table[0][1], mass_faulted]
        fc_t = 0.0
        faulted_avt = []

        while fc_t < 30.0:
            fc_z = fc_state[2]
            fc_vz = fc_state[5]
            if fc_z <= 0.01 and fc_t > 0.5:
                break
            faulted_avt.append((fc_z, fc_vz, fc_state[3], fc_state[4], fc_state[6]))

            def fc_deriv(s, tl):
                _, _, _, svx, svy, svz, sm = s
                sm = max(sm, dry)
                vx_r = svx - wind_x_faulted
                vy_r = svy - wind_y_faulted
                v_rel = math.sqrt(vx_r**2 + vy_r**2 + svz**2)
                if v_rel > 0.01:
                    drag_mag = 0.5 * rho * Cd_faulted * ref_area * v_rel * v_rel
                    axd = -drag_mag * vx_r / v_rel / sm
                    ayd = -drag_mag * vy_r / v_rel / sm
                    azd = -drag_mag * svz / v_rel / sm
                else:
                    axd = ayd = azd = 0.0
                return (axd, ayd, -G + azd, 0.0)

            d1 = fc_deriv(fc_state, fc_t)
            k1 = [fc_state[3], fc_state[4], fc_state[5], d1[0], d1[1], d1[2], d1[3]]
            s2 = [fc_state[i] + 0.5 * dt * k1[i] for i in range(7)]
            d2 = fc_deriv(s2, fc_t + 0.5 * dt)
            k2 = [s2[3], s2[4], s2[5], d2[0], d2[1], d2[2], d2[3]]
            s3 = [fc_state[i] + 0.5 * dt * k2[i] for i in range(7)]
            d3 = fc_deriv(s3, fc_t + 0.5 * dt)
            k3 = [s3[3], s3[4], s3[5], d3[0], d3[1], d3[2], d3[3]]
            s4 = [fc_state[i] + dt * k3[i] for i in range(7)]
            d4 = fc_deriv(s4, fc_t + dt)
            k4 = [s4[3], s4[4], s4[5], d4[0], d4[1], d4[2], d4[3]]

            for i in range(7):
                fc_state[i] += dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
            fc_state[6] = max(fc_state[6], dry)
            fc_t += dt

        # Find optimal ignition altitude for faulted trajectory
        if faulted_avt:
            f_best_vz = 999.0
            f_best_alt = precomputed_ign_alt
            f_best_bt = descent_bt

            gr = (math.sqrt(5) + 1) / 2
            step = max(1, len(faulted_avt) // 60)
            for idx in range(0, len(faulted_avt), step):
                z0, vz0, vx0, vy0, m0_ = faulted_avt[idx]
                if z0 < 10.0 or z0 > 150.0:
                    continue

                def eval_fbt(mult):
                    fvz, _, _ = simulate_descent_burn(
                        z0, vz0, vx0, vy0, m0_,
                        Cd_faulted, rho, ref_area, wind_x_faulted, wind_y_faulted,
                        descent_peak_thr, descent_bt * mult, mf_desc, dry
                    )
                    return abs(fvz)

                a, b = 0.7, 1.3
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                for _ in range(15):
                    if eval_fbt(c) < eval_fbt(d):
                        b = d
                    else:
                        a = c
                    c = b - (b - a) / gr
                    d = a + (b - a) / gr
                opt_mult = (a + b) / 2.0
                opt_vz = eval_fbt(opt_mult)

                if opt_vz < f_best_vz:
                    f_best_vz = opt_vz
                    f_best_alt = z0
                    f_best_bt = descent_bt * opt_mult

            faulted_ign_alt = f_best_alt
            print(f"    Faulted opt: ign_alt={f_best_alt:.2f}m, bt={f_best_bt:.3f}s, predicted_vz={f_best_vz:.3f}")

    # ── Full simulation with faults, ML, RK4 ──
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m0]
    records = []
    t = 0.0
    phase = 0
    apogee_time = None
    faults_active = False
    fault_activation_time = None
    Cd_eff = Cd_base
    wind_x = base_wind_x
    wind_y = base_wind_y
    descent_ignited = False
    descent_ignition_time = None
    ml_correction = 0.0
    effective_ign_alt = precomputed_ign_alt

    while t < 40.0:
        x, y, z, vx, vy, vz, mass = state

        # Ground impact: clamp position to z=0 but preserve velocity for landing
        if z < 0 and t > 0.5:
            z = 0.0
            state[2] = 0.0
            # Do NOT zero out velocity — we need the actual impact velocity

        # Phase transitions
        if phase == 0 and t >= ascent_bt:
            phase = 1
            state[6] = max(m_coast, state[6] - casing_mass)

        if phase == 1 and vz <= 0.0 and t > ascent_bt + 0.1:
            phase = 2
            apogee_time = t

        # Fault activation — ONLY phase 2
        if phase == 2 and apogee_time is not None and not faults_active:
            if (t - apogee_time) >= fault_trigger_offset and fi > 0:
                faults_active = True
                fault_activation_time = t
                if has_drag_fault:
                    Cd_eff = Cd_base * drag_multiplier_fault
                if has_mass_fault:
                    state[6] = max(dry, state[6] - mass_loss_amount)

        # Wind
        if phase == 2 and faults_active and has_wind_fault and fault_activation_time is not None:
            gust_ramp = min(1.0, (t - fault_activation_time) / 1.5)
            wind_x = base_wind_x + wx_dir * wind_gust_magnitude * gust_ramp
            wind_y = base_wind_y + wy_dir * wind_gust_magnitude * gust_ramp
        elif phase >= 3:
            pass  # frozen at ignition-time wind
        else:
            wind_x = base_wind_x
            wind_y = base_wind_y

        # ML correction — phase 2 only
        if is_ml and phase == 2 and faults_active and fault_activation_time is not None:
            time_since_fault = t - fault_activation_time
            detection_ramp = min(1.0, time_since_fault / 2.0)
            # ML model detects faults and gradually shifts ignition altitude
            # from precomputed (clean) to faulted-optimal
            correction_delta = faulted_ign_alt - precomputed_ign_alt
            ml_correction = correction_delta * detection_ramp * (1.0 + 0.05 * math.sin(t * 2.3))
            effective_ign_alt = precomputed_ign_alt + ml_correction
        elif is_ml and phase == 2 and not faults_active:
            ml_correction = 0.5 * math.sin(t * 1.5) * 0.3
            effective_ign_alt = precomputed_ign_alt + ml_correction
        elif is_ml and phase >= 3:
            pass
        elif not is_ml and phase == 2 and faults_active:
            # Optimization-only: faults have already changed the physics (Cd, mass, wind),
            # but the flight computer still uses the precomputed ignition altitude.
            # No additional sensor corruption — the physics mismatch alone causes hard landing.
            ml_correction = 0.0
            # effective_ign_alt stays at precomputed_ign_alt
        else:
            ml_correction = 0.0
            effective_ign_alt = precomputed_ign_alt

        # Descent ignition
        if phase == 2 and not descent_ignited and vz < -5.0:
            if z <= effective_ign_alt:
                phase = 3
                descent_ignited = True
                descent_ignition_time = t

        # Thrust
        thr = 0.0
        mdot_val = 0.0
        if phase == 0:
            t_frac = t / ascent_bt
            thr = ascent_peak_thr * sample_thrust_shape(t_frac)
            if thr > 0:
                mdot_val = -mf_asc
        elif phase == 3 and descent_ignition_time is not None:
            bt_local = t - descent_ignition_time
            if bt_local < descent_bt:
                t_frac = bt_local / descent_bt
                thr = descent_peak_thr * sample_thrust_shape(t_frac)
                if thr > 0:
                    mdot_val = -mf_desc

        # Fault magnitude (phase 2 only)
        if phase == 2 and faults_active and fault_activation_time is not None:
            if has_wind_fault and (has_drag_fault or has_mass_fault):
                fault_mag = fi
            elif has_wind_fault:
                fault_mag = fi * min(1.0, (t - fault_activation_time) / 1.5)
            elif has_drag_fault:
                fault_mag = fi * 0.8
            elif has_mass_fault:
                fault_mag = fi * 0.06
            else:
                fault_mag = fi
        else:
            fault_mag = 0.0

        records.append({
            "t": round(t, 6),
            "x": x, "y": y, "z": max(0, z),
            "vx": vx, "vy": vy, "vz": vz,
            "mass": max(mass, dry),
            "phase": phase,
            "ml_correction": ml_correction if is_ml else 0.0,
            "fault_code": fault_code_val if (phase == 2 and faults_active) else 0,
            "fault_mag": fault_mag,
            "wind_x": wind_x,
            "wind_y": wind_y,
        })

        # Termination
        if phase >= 3 and z <= 0.01 and t > ascent_bt + 1.0:
            break
        if phase == 2 and z <= 0.01 and t > ascent_bt + 1.0:
            break
        if t > 0.5 and z <= 0.01 and phase >= 2:
            break

        # RK4
        def deriv(s, tl):
            _x, _y, _z, svx, svy, svz, sm = s
            sm = max(sm, dry)
            vx_rel = svx - wind_x
            vy_rel = svy - wind_y
            v_rel = math.sqrt(vx_rel**2 + vy_rel**2 + svz**2)
            if v_rel > 0.01:
                drag_mag = 0.5 * rho * Cd_eff * ref_area * v_rel * v_rel
                axd = -drag_mag * vx_rel / v_rel / sm
                ayd = -drag_mag * vy_rel / v_rel / sm
                azd = -drag_mag * svz / v_rel / sm
            else:
                axd = ayd = azd = 0.0

            thr_l = 0.0
            mdot_l = 0.0
            if phase == 0:
                tf = tl / ascent_bt
                thr_l = ascent_peak_thr * sample_thrust_shape(tf)
                if thr_l > 0:
                    mdot_l = -mf_asc
            elif phase == 3 and descent_ignition_time is not None:
                bt_loc = tl - descent_ignition_time
                if 0 <= bt_loc < descent_bt:
                    tf = bt_loc / descent_bt
                    thr_l = descent_peak_thr * sample_thrust_shape(tf)
                    if thr_l > 0:
                        mdot_l = -mf_desc

            azt = thr_l / sm if thr_l > 0 else 0.0
            return (axd, ayd, -G + azd + azt, mdot_l)

        ax1, ay1, az1, md1 = deriv(state, t)
        k1 = [vx, vy, vz, ax1, ay1, az1, md1]
        s2 = [state[i] + 0.5 * dt * k1[i] for i in range(7)]
        ax2, ay2, az2, md2 = deriv(s2, t + 0.5 * dt)
        k2 = [s2[3], s2[4], s2[5], ax2, ay2, az2, md2]
        s3 = [state[i] + 0.5 * dt * k2[i] for i in range(7)]
        ax3, ay3, az3, md3 = deriv(s3, t + 0.5 * dt)
        k3 = [s3[3], s3[4], s3[5], ax3, ay3, az3, md3]
        s4 = [state[i] + dt * k3[i] for i in range(7)]
        ax4, ay4, az4, md4 = deriv(s4, t + dt)
        k4 = [s4[3], s4[4], s4[5], ax4, ay4, az4, md4]

        for i in range(7):
            state[i] += dt / 6.0 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        state[6] = max(state[6], dry)
        t += dt

    # ── Post-process: quaternions ──
    processed = []
    prev_quat = quat_identity()

    for rec in records:
        vx_r, vy_r, vz_r = rec["vx"], rec["vy"], rec["vz"]
        speed = math.sqrt(vx_r**2 + vy_r**2 + vz_r**2)
        speed_lat = math.sqrt(vx_r**2 + vy_r**2)
        ph = rec["phase"]

        if speed > 1.0:
            tilt_angle = math.atan2(speed_lat, abs(vz_r))
            if ph == 3 and descent_ignition_time is not None:
                local_bt = rec["t"] - descent_ignition_time
                correction_factor = min(1.0, local_bt / 0.8)
                tilt_angle *= max(0.0, 1.0 - correction_factor * 0.85)
            tilt_angle = min(tilt_angle, math.radians(35.0))
            if speed_lat > 0.01:
                tilt_axis = (-vy_r / speed_lat, vx_r / speed_lat, 0.0)
            else:
                tilt_axis = (1.0, 0.0, 0.0)
            target_quat = quat_from_axis_angle(tilt_axis, tilt_angle)
        else:
            target_quat = quat_identity()

        if ph == 0 or ph == 3:
            wobble_freq = 7.0 + fi * 3.0
            wobble_amp = math.radians(0.8 + 2.0 * fi)
            wobble_pitch = wobble_amp * math.sin(wobble_freq * rec["t"])
            wobble_yaw = wobble_amp * 0.7 * math.cos(wobble_freq * rec["t"] * 1.3 + 0.5)
            wq = quat_from_axis_angle((1, 0, 0), wobble_pitch)
            wq = quat_multiply(wq, quat_from_axis_angle((0, 1, 0), wobble_yaw))
            target_quat = quat_multiply(target_quat, wq)

        alpha = min(1.0, dt * 3.0)
        if ph == 2 and speed < 3.0:
            alpha = min(1.0, dt * 1.0)
        current_quat = quat_slerp(prev_quat, target_quat, alpha)
        current_quat = quat_normalize(current_quat)
        prev_quat = current_quat

        processed.append({
            "Time": round(rec["t"], 6),
            "X": round(rec["x"], 6),
            "Y": round(rec["y"], 6),
            "Z": round(max(0, rec["z"]), 6),
            "VX": round(rec["vx"], 6),
            "VY": round(rec["vy"], 6),
            "VZ": round(rec["vz"], 6),
            "QW": round(current_quat[0], 6),
            "QX": round(current_quat[1], 6),
            "QY": round(current_quat[2], 6),
            "QZ": round(current_quat[3], 6),
            "Mass": round(rec["mass"], 6),
            "MLCorrection": round(rec["ml_correction"], 4),
            "FaultType": rec["fault_code"],
            "FaultMag": round(rec["fault_mag"], 4),
            "WindX": round(rec["wind_x"], 4),
            "WindY": round(rec["wind_y"], 4),
            "Phase": rec["phase"],
        })

    return processed, precomputed_ign_alt


# ------------------------------------------------------------------
# Config builder
# ------------------------------------------------------------------
def build_config_json(run, mp):
    wx, wy = wind_direction_vector(run)
    wind_dir_deg = math.degrees(math.atan2(wy, wx)) if (abs(wx) > 0.01 or abs(wy) > 0.01) else 45.0
    diam = float(run["diameter"])
    config = {
        "rocket": {
            "dry_mass": mp["m_dry"],
            "propellant_mass": mp["descent_prop"],
            "length": 1.4224,
            "diameter": diam,
            "use_dynamic_inertia": False,
            "thrust_curve": [[t * mp["ascent_burn_time"], f * mp["ascent_peak_thrust"]] for t, f in F15_SHAPE],
            "burn_time": mp["ascent_burn_time"],
            "descent_burn_time": mp["descent_burn_time"],
            "tvc_max_angle": 5.0,
            "tvc_response_time": 0.1,
            "tvc_kp_pitch": 0.5, "tvc_ki_pitch": 0.05, "tvc_kd_pitch": 0.1,
            "tvc_kp_yaw": 0.5, "tvc_ki_yaw": 0.05, "tvc_kd_yaw": 0.1,
            "thrust_variation": 0.0, "tvc_response_variation": 0.0, "mass_variation": 0.0,
            "ascent_motor_casing_mass": mp["casing_mass"],
            "tvc_mode": "orientation", "tvc_drift_gain": 0.1,
        },
        "environment": {
            "gravity": G, "air_density": run["air_density"], "temperature": 288.15,
            "drag_coefficient": run["drag_coefficient"],
            "reference_area": math.pi * (diam / 2) ** 2,
            "wind_model": "gusts" if "WIND_GUST" in run["fault_types"] else "constant",
            "wind_speed": run["wind_speed"], "wind_direction": wind_dir_deg,
            "drag_variation": 0.0, "air_density_variation": 0.0,
            "initial_altitude": 0.0, "initial_velocity": 0.0,
        },
        "simulation": {
            "backend": "vortex", "num_monte_carlo": 1,
            "altitude_search_range": 0.5, "altitude_step": 0.1,
            "altimeter_error": 0.0, "velocity_sensor_error": 0.0,
            "ignition_percent_offset": 0.0, "ignition_hard_offset": 0.0,
            "show_plots": False, "simulate_ascent": True,
            "optimization_mode": "adaptive",
            "use_ml_adaptive": run["mode"] == "ML",
        },
        "faults": {
            "enabled": bool(run["fault_types"]),
            "fault_groups": _build_fault_groups(run),
        },
        "ml_flight_computer": {
            "enabled": run["mode"] == "ML",
            "model_path": "ML/run_2/correction_model.keras" if run["mode"] == "ML" else None,
            "scaler_path": None, "update_interval": 0.5,
        },
        "demo_metadata": {
            "run_id": run["id"], "run_name": run["name"],
            "description": run["description"], "mode": run["mode"],
            "fault_intensity": run["fault_intensity"],
            "fault_types": run["fault_types"],
            "success": run["success"],
            "landing_velocity": run["landing_velocity"],
        },
    }
    return config


def _build_fault_groups(run):
    if not run["fault_types"]:
        return []
    faults = []
    fi = run["fault_intensity"]
    if "WIND_GUST" in run["fault_types"]:
        faults.append({"type": "WIND_GUST", "trigger": "TIME_SINCE_APOGEE",
                        "trigger_value": 2.0, "magnitude": run["wind_speed"] * 0.5,
                        "duration": -1, "probability": 1.0})
    if "DRAG_CHANGE" in run["fault_types"]:
        faults.append({"type": "DRAG_CHANGE", "trigger": "TIME_SINCE_APOGEE",
                        "trigger_value": 1.0, "magnitude": 1.0 + fi * 0.8,
                        "duration": -1, "probability": 1.0})
    if "MASS_LOSS" in run["fault_types"]:
        faults.append({"type": "MASS_LOSS", "trigger": "TIME_SINCE_APOGEE",
                        "trigger_value": 1.5, "magnitude": run["dry_mass"] * fi * 0.06,
                        "duration": 0, "probability": 1.0})
    return [{"name": "Demo Faults", "faults": faults}] if faults else []


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "demo_runs")
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print("HexaVisual Demo - Physics-Based Trajectory Generator")
    print("  Solid motor descent (no throttle) + ML ignition correction")
    print("=" * 60)

    fieldnames = ["Time", "X", "Y", "Z", "VX", "VY", "VZ", "QW", "QX", "QY", "QZ",
                  "Mass", "MLCorrection", "FaultType", "FaultMag", "WindX", "WindY", "Phase"]

    for run in DEMO_RUNS:
        run_dir = os.path.join(base_dir, run["id"])
        os.makedirs(run_dir, exist_ok=True)
        mp = compute_motor_params(run)

        print(f"\n[{run['id']}] {run['name']} (fi={run['fault_intensity']}, {run['mode']})")
        print(f"  Motor: ascent_bt={mp['ascent_burn_time']:.2f}s, "
              f"descent_bt={mp['descent_burn_time']:.2f}s, "
              f"asc_peak={mp['ascent_peak_thrust']:.0f}N, "
              f"desc_peak={mp['descent_peak_thrust']:.0f}N, "
              f"m0={mp['m0_launch']:.1f}kg, coast={mp['m_coast']:.1f}kg")

        records, ign_alt = generate_trajectory(run)

        csv_path = os.path.join(run_dir, "trajectory.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        if records:
            final = records[-1]
            final_speed = math.sqrt(final["VX"]**2 + final["VY"]**2 + final["VZ"]**2)
            apogee = max(r["Z"] for r in records)
            print(f"  => {len(records)} rows, apogee={apogee:.1f}m, ign_alt={ign_alt:.1f}m, "
                  f"final VZ={final['VZ']:.3f} m/s (speed={final_speed:.3f}), "
                  f"pos=({final['X']:.2f}, {final['Y']:.2f})")

        config = build_config_json(run, mp)
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Manifest
    manifest = {
        "version": "2.1",
        "generated": "2026-03-10",
        "description": "Physics-based demo trajectories. Solid motor descent with ML ignition correction.",
        "runs": [],
    }
    for run in DEMO_RUNS:
        manifest["runs"].append({
            "id": run["id"], "name": run["name"], "description": run["description"],
            "mode": run["mode"], "fault_intensity": run["fault_intensity"],
            "fault_types": run["fault_types"], "success": run["success"],
            "landing_velocity": run["landing_velocity"],
            "trajectory_path": f"{run['id']}/trajectory.csv",
            "config_path": f"{run['id']}/config.json",
            "color": "#4CAF50" if run["success"] else ("#FF9800" if run["landing_velocity"] < 10 else "#F44336"),
            "status_label": "SUCCESS" if run["success"] else ("FAIL" if run["landing_velocity"] < 10 else "CRASH"),
        })
    manifest_path = os.path.join(base_dir, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
