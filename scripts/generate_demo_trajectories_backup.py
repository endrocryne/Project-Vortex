"""
Generate Demo Trajectories for HexaVisual Demo
================================================
Produces 9 procedurally-generated trajectory CSV files for the locked-down
HexaVisual Demo app.  Each trajectory matches a specific data point from
the PlotVisual sample_visualization_data.csv (seed=42) and represents
one point on the Velocity vs. Fault Intensity graph.

The trajectories are SYNTHETIC — analytically constructed to look realistic:
  Phase 1 (coast):  smooth free-fall curve with drag shaping
  Phase 2 (burn):   smooth deceleration to target landing velocity
  Lateral drift:    sigmoidal buildup matching target landing X/Y
  Quaternions:      nose follows velocity, tilt during burn
  Mass:             decreases linearly during burn only

No actual physics simulation is run. All curves are shaped analytically to
produce correct-looking animations in HexaVisual with exact target endpoints.

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
        "description": "Same wind conditions as Run 2, but ML flight computer provides real-time TVC corrections. Successful soft landing.",
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
        "description": "Same faults as Run 4, but ML flight computer adapts ignition timing and TVC to compensate. Successful soft landing.",
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

ROCKET_BASE_CONFIG = {
    "dry_mass": 1.219,
    "propellant_mass": 0.063,
    "length": 1.4224,
    "diameter": 0.07874,
    "burn_time": 1.7,
    "thrust_curve": [
        [0.0, 0.0],
        [0.013, 89.1],
        [0.018, 101.6],
        [0.029, 105.4],
        [0.047, 102.9],
        [0.104, 100.0],
        [0.19, 102.3],
        [0.268, 104.9],
        [0.306, 104.3],
        [0.38, 97.4],
        [0.45, 92.0],
        [0.6, 88.5],
        [0.75, 83.0],
        [0.9, 78.0],
        [1.05, 72.0],
        [1.2, 65.0],
        [1.38, 48.0],
        [1.5, 28.0],
        [1.6, 10.0],
        [1.7, 0.0],
    ],
}
BURN_TIME = ROCKET_BASE_CONFIG["burn_time"]


def smoothstep(t):
    """Hermite smoothstep: 3t^2 - 2t^3, clamped to [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    """Perlin smootherstep: 6t^5 - 15t^4 + 10t^3."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def quat_from_axis_angle(axis, angle):
    """Return (w, x, y, z) quaternion from axis-angle."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax / norm, ay / norm, az / norm
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return (c, ax * s, ay * s, az * s)


def generate_trajectory(run):
    """Generate synthetic launch-to-landing trajectory with smooth, physical-looking motion."""
    dt = 0.02

    fi = float(run["fault_intensity"])
    is_ml = run["mode"] == "ML"
    success = bool(run["success"])
    v_land = float(run["landing_velocity"])
    target_lx = float(run["landing_x"])
    target_ly = float(run["landing_y"])
    wind_speed = float(run["wind_speed"])

    m_dry = 1.219
    m_prop = 0.063
    m_total = m_dry + m_prop

    apogee_base = 262.0
    apogee_penalty = 70.0 * fi * (1.0 if not is_ml else 0.45)
    apogee = max(170.0, min(280.0, apogee_base - apogee_penalty))

    ascent_burn_time = 1.7
    landing_burn_time = 1.7

    t_burnout = ascent_burn_time
    t_apogee = t_burnout + 5.0 + 0.8 * fi + (0.35 if not is_ml else 0.10)
    t_ignite = t_apogee + 4.2 + 2.1 * fi + (0.45 if (not is_ml and fi > 0.65) else 0.0)
    t_touch = t_ignite + landing_burn_time

    z_burnout = apogee * (0.32 if not is_ml else 0.35)
    z_ignite = 38.0 + 42.0 * fi + (9.0 if not success else 0.0)
    z_ignite = max(20.0, min(apogee * 0.72, z_ignite))

    v_burnout = 72.0 - 8.0 * fi + (3.0 if is_ml else 0.0)
    v_ignite = -(24.0 + 24.0 * fi + (8.0 if not is_ml else 0.0))

    prop_ascent = 0.028
    prop_landing = m_prop - prop_ascent

    def hermite(z0, v0, z1, v1, u, duration):
        h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
        h10 = u ** 3 - 2.0 * u ** 2 + u
        h01 = -2.0 * u ** 3 + 3.0 * u ** 2
        h11 = u ** 3 - u ** 2

        dh00 = 6.0 * u ** 2 - 6.0 * u
        dh10 = 3.0 * u ** 2 - 4.0 * u + 1.0
        dh01 = -6.0 * u ** 2 + 6.0 * u
        dh11 = 3.0 * u ** 2 - 2.0 * u

        z_val = h00 * z0 + h10 * duration * v0 + h01 * z1 + h11 * duration * v1
        v_val = (dh00 * z0 + dh10 * duration * v0 + dh01 * z1 + dh11 * duration * v1) / max(duration, 1e-6)
        return z_val, v_val

    times = []
    xs = []
    ys = []
    zs = []
    masses = []

    n_steps = int(t_touch / dt) + 1
    wind_start = t_apogee + 0.7
    wind_has_fault = "WIND_GUST" in run["fault_types"]

    for i in range(n_steps + 1):
        t = min(i * dt, t_touch)

        if t <= t_burnout:
            u = t / max(t_burnout, 1e-6)
            z, _ = hermite(0.0, 0.0, z_burnout, v_burnout, u, t_burnout)
        elif t <= t_apogee:
            u = (t - t_burnout) / max(t_apogee - t_burnout, 1e-6)
            z, _ = hermite(z_burnout, v_burnout, apogee, 0.0, u, t_apogee - t_burnout)
        elif t <= t_ignite:
            u = (t - t_apogee) / max(t_ignite - t_apogee, 1e-6)
            z, _ = hermite(apogee, 0.0, z_ignite, v_ignite, u, t_ignite - t_apogee)
        else:
            u = (t - t_ignite) / max(t_touch - t_ignite, 1e-6)
            z, _ = hermite(z_ignite, v_ignite, 0.0, -v_land, u, t_touch - t_ignite)

        z = max(0.0, z)

        p = min(1.0, t / max(t_touch, 1e-6))
        drift_progress = smootherstep(p)
        x_base = target_lx * drift_progress
        y_base = target_ly * drift_progress

        wind_ramp = 0.0
        if wind_has_fault and t > wind_start:
            wind_ramp = smootherstep((t - wind_start) / 1.0)

        end_damp = math.sin(math.pi * p)
        end_damp = max(0.0, end_damp) ** 1.4

        disturbance_gain = (0.35 + 3.0 * fi * fi) * (1.0 + min(wind_speed / 10.0, 1.5) * 0.5)
        disturbance_gain *= (0.85 if is_ml else 1.4)
        disturbance_gain *= (0.30 + 0.70 * wind_ramp)
        apogee_damp = 1.0 - 0.92 * math.exp(-((t - t_apogee) / 0.65) ** 2)
        disturbance_gain *= apogee_damp

        x_dist = disturbance_gain * end_damp * (
            0.75 * math.sin(1.2 * t + fi * 2.3) + 0.25 * math.sin(2.5 * t + 1.0)
        )
        y_dist = disturbance_gain * end_damp * (
            0.70 * math.cos(1.1 * t + 0.7) + 0.30 * math.sin(2.7 * t + 0.5)
        )

        x = x_base + x_dist
        y = y_base + y_dist

        settle_start = max(0.0, t_touch - 0.9)
        if t >= settle_start:
            settle_u = smootherstep((t - settle_start) / max(t_touch - settle_start, 1e-6))
            x = lerp(x, target_lx, settle_u)
            y = lerp(y, target_ly, settle_u)

        if t <= t_burnout:
            mass = m_total - prop_ascent * min(t / max(t_burnout, 1e-6), 1.0)
        elif t <= t_ignite:
            mass = m_total - prop_ascent
        else:
            burn_frac = min((t - t_ignite) / max(t_touch - t_ignite, 1e-6), 1.0)
            mass = m_total - prop_ascent - prop_landing * burn_frac

        times.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        masses.append(max(mass, m_dry))

    if times:
        xs[-1] = target_lx
        ys[-1] = target_ly
        zs[-1] = 0.0
        masses[-1] = m_dry

    # Recompute velocities from smooth positions to avoid endpoint spikes/jitter.
    vx_arr = [0.0] * len(times)
    vy_arr = [0.0] * len(times)
    vz_arr = [0.0] * len(times)
    if len(times) >= 3:
        for i in range(len(times)):
            if i == 0:
                dt_i = times[1] - times[0]
                vx_arr[i] = (xs[1] - xs[0]) / max(dt_i, 1e-6)
                vy_arr[i] = (ys[1] - ys[0]) / max(dt_i, 1e-6)
                vz_arr[i] = (zs[1] - zs[0]) / max(dt_i, 1e-6)
            elif i == len(times) - 1:
                dt_i = times[i] - times[i - 1]
                vx_arr[i] = (xs[i] - xs[i - 1]) / max(dt_i, 1e-6)
                vy_arr[i] = (ys[i] - ys[i - 1]) / max(dt_i, 1e-6)
                vz_arr[i] = (zs[i] - zs[i - 1]) / max(dt_i, 1e-6)
            else:
                dt_i = times[i + 1] - times[i - 1]
                vx_arr[i] = (xs[i + 1] - xs[i - 1]) / max(dt_i, 1e-6)
                vy_arr[i] = (ys[i + 1] - ys[i - 1]) / max(dt_i, 1e-6)
                vz_arr[i] = (zs[i + 1] - zs[i - 1]) / max(dt_i, 1e-6)

    # Enforce endpoint velocity targets
    if vz_arr:
        vz_arr[-1] = -v_land
    if vx_arr and vy_arr:
        vx_arr[-1] = 0.0
        vy_arr[-1] = 0.0

    records = []
    for i, t in enumerate(times):
        vx = vx_arr[i]
        vy = vy_arr[i]
        vz = vz_arr[i]

        speed_lat = math.sqrt(vx * vx + vy * vy)
        speed_tot = math.sqrt(vx * vx + vy * vy + vz * vz)

        if speed_tot > 0.35 and speed_lat > 0.05:
            # Orient nose along velocity trend (visual tilt follows trajectory)
            tilt_axis = (-vy / speed_lat, vx / speed_lat, 0.0)
            tilt = math.atan2(speed_lat, max(abs(vz), 0.25))
            if t <= t_burnout or t >= t_ignite:
                wobble = math.radians((0.7 + 2.8 * fi) * math.sin(7.0 * t + fi * 3.0))
                tilt = max(0.0, tilt + wobble)
            tilt = max(0.0, min(math.radians(42.0), tilt))
            qw, qx_q, qy_q, qz_q = quat_from_axis_angle(tilt_axis, tilt)
        else:
            qw, qx_q, qy_q, qz_q = (1.0, 0.0, 0.0, 0.0)

        records.append({
            "Time": round(t, 6),
            "X": round(xs[i], 6),
            "Y": round(ys[i], 6),
            "Z": round(zs[i], 6),
            "VX": round(vx, 6),
            "VY": round(vy, 6),
            "VZ": round(vz, 6),
            "QW": round(qw, 6),
            "QX": round(qx_q, 6),
            "QY": round(qy_q, 6),
            "QZ": round(qz_q, 6),
            "Mass": round(masses[i], 6),
        })

    return records


def wind_direction_vector(run):
    """Normalized vector from origin toward landing point."""
    lx = run["landing_x"]
    ly = run["landing_y"]
    dist = math.sqrt(lx ** 2 + ly ** 2)
    if dist < 0.01:
        return (1.0, 0.0)
    return (lx / dist, ly / dist)


def build_config_json(run):
    """Build a config.json for the demo run, matching the project schema."""
    m_dry = ROCKET_BASE_CONFIG["dry_mass"]
    m_prop = ROCKET_BASE_CONFIG["propellant_mass"]
    diameter = ROCKET_BASE_CONFIG["diameter"]
    thrust_curve = ROCKET_BASE_CONFIG["thrust_curve"]

    wx, wy = wind_direction_vector(run)
    wind_dir_deg = math.degrees(math.atan2(wy, wx)) if (abs(wx) > 0.01 or abs(wy) > 0.01) else 45.0

    config = {
        "rocket": {
            "dry_mass": m_dry,
            "propellant_mass": m_prop,
            "length": ROCKET_BASE_CONFIG["length"],
            "diameter": diameter,
            "use_dynamic_inertia": False,
            "thrust_curve": thrust_curve,
            "burn_time": BURN_TIME,
            "tvc_max_angle": 5.0,
            "tvc_response_time": 0.1,
            "tvc_kp_pitch": 0.5,
            "tvc_ki_pitch": 0.05,
            "tvc_kd_pitch": 0.1,
            "tvc_kp_yaw": 0.5,
            "tvc_ki_yaw": 0.05,
            "tvc_kd_yaw": 0.1,
            "thrust_variation": 0.0,
            "tvc_response_variation": 0.0,
            "mass_variation": 0.0,
            "ascent_motor_casing_mass": 0.065,
            "tvc_mode": "orientation",
            "tvc_drift_gain": 0.1,
        },
        "environment": {
            "gravity": 9.81,
            "air_density": run["air_density"],
            "temperature": 288.15,
            "drag_coefficient": run["drag_coefficient"],
            "reference_area": math.pi * (diameter / 2) ** 2,
            "wind_model": "gusts" if "WIND_GUST" in run["fault_types"] else "constant",
            "wind_speed": run["wind_speed"],
            "wind_direction": wind_dir_deg,
            "drag_variation": 0.0,
            "air_density_variation": 0.0,
            "initial_altitude": 0.0,
            "initial_velocity": 0.0,
        },
        "simulation": {
            "backend": "vortex",
            "num_monte_carlo": 1,
            "altitude_search_range": 0.5,
            "altitude_step": 0.1,
            "altimeter_error": 0.0,
            "velocity_sensor_error": 0.0,
            "ignition_percent_offset": 0.0,
            "ignition_hard_offset": 0.0,
            "show_plots": False,
            "simulate_ascent": True,
            "optimization_mode": "adaptive",
            "opt_max_iterations": 3,
            "opt_samples_per_step": 7,
            "opt_target_step": 0.01,
            "use_ml_adaptive": run["mode"] == "ML",
            "ascent_initial_pitch": 0.0,
            "ascent_initial_yaw": 0.0,
            "ascent_initial_roll": 0.0,
        },
        "faults": {
            "enabled": bool(run["fault_types"]),
            "fault_groups": _build_fault_groups(run),
        },
        "ml_flight_computer": {
            "enabled": run["mode"] == "ML",
            "model_path": "ML/run_2/correction_model.keras" if run["mode"] == "ML" else None,
            "scaler_path": None,
            "update_interval": 0.5,
        },
        "demo_metadata": {
            "run_id": run["id"],
            "run_name": run["name"],
            "description": run["description"],
            "mode": run["mode"],
            "fault_intensity": run["fault_intensity"],
            "fault_types": run["fault_types"],
            "success": run["success"],
            "landing_velocity": run["landing_velocity"],
            "landing_x": run["landing_x"],
            "landing_y": run["landing_y"],
            "landing_distance": run["landing_distance"],
        },
    }
    return config


def _build_fault_groups(run):
    """Build fault_groups array for config.json."""
    if not run["fault_types"]:
        return []

    faults = []
    fi = run["fault_intensity"]

    if "WIND_GUST" in run["fault_types"]:
        faults.append({
            "type": "WIND_GUST",
            "trigger": "TIME_SINCE_APOGEE",
            "trigger_value": 2.0,
            "magnitude": run["wind_speed"] * 0.5,
            "duration": -1,
            "probability": 1.0,
        })

    if "DRAG_CHANGE" in run["fault_types"]:
        faults.append({
            "type": "DRAG_CHANGE",
            "trigger": "TIME_SINCE_APOGEE",
            "trigger_value": 1.0,
            "magnitude": 1.0 + fi * 0.8,
            "duration": -1,
            "probability": 1.0,
        })

    if "MASS_LOSS" in run["fault_types"]:
        faults.append({
            "type": "MASS_LOSS",
            "trigger": "TIME_SINCE_APOGEE",
            "trigger_value": 1.5,
            "magnitude": run["dry_mass"] * fi * 0.06,
            "duration": 0,
            "probability": 1.0,
        })

    return [{"name": "Demo Faults", "faults": faults}] if faults else []


def main():
    """Generate all 9 demo trajectories (procedural/synthetic)."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_runs")
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print("HexaVisual Demo - Synthetic Trajectory Generator")
    print("=" * 60)

    for run in DEMO_RUNS:
        run_dir = os.path.join(base_dir, run["id"])
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{run['id']}] {run['name']} (fi={run['fault_intensity']}, {run['mode']})")

        records = generate_trajectory(run)

        csv_path = os.path.join(run_dir, "trajectory.csv")
        fieldnames = ["Time", "X", "Y", "Z", "VX", "VY", "VZ", "QW", "QX", "QY", "QZ", "Mass"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        final_vz = records[-2]["VZ"] if len(records) > 1 else 0
        print(f"  => {len(records)} rows, final VZ={final_vz:.3f} m/s "
              f"(target={-run['landing_velocity']:.3f}), "
              f"pos=({records[-1]['X']:.1f}, {records[-1]['Y']:.1f})")

        config = build_config_json(run)
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Write demo manifest
    manifest = {
        "version": "1.0",
        "generated": "2026-03-02",
        "description": "Pre-computed demo trajectories for HexaVisual Demo app. "
                       "Each run corresponds to a data point on the PlotVisual "
                       "Velocity vs. Fault Intensity graph (sample data, seed=42).",
        "runs": [],
    }
    for run in DEMO_RUNS:
        manifest["runs"].append({
            "id": run["id"],
            "name": f"Demo {len(manifest['runs']) + 1}",
            "description": f"Demo {len(manifest['runs']) + 1}",
            "mode": run["mode"],
            "fault_intensity": run["fault_intensity"],
            "fault_types": run["fault_types"],
            "success": run["success"],
            "landing_velocity": run["landing_velocity"],
            "landing_x": run["landing_x"],
            "landing_y": run["landing_y"],
            "landing_distance": run["landing_distance"],
            "dry_mass": ROCKET_BASE_CONFIG["dry_mass"],
            "propellant_mass": ROCKET_BASE_CONFIG["propellant_mass"],
            "diameter": ROCKET_BASE_CONFIG["diameter"],
            "thrust_average": 89.0,
            "wind_speed": run["wind_speed"],
            "drag_coefficient": run["drag_coefficient"],
            "air_density": run["air_density"],
            "initial_altitude": 0.0,
            "initial_velocity": 0.0,
            "trajectory_path": f"{run['id']}/trajectory.csv",
            "config_path": f"{run['id']}/config.json",
            "color": "#4CAF50" if run["success"] else ("#FF9800" if run["landing_velocity"] < 10.0 else "#F44336"),
            "status_label": "SUCCESS" if run["success"] else ("FAIL" if run["landing_velocity"] < 10.0 else "CRASH"),
        })

    manifest_path = os.path.join(base_dir, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")
    print("\nDone! All 9 demo trajectories generated.")


if __name__ == "__main__":
    main()
