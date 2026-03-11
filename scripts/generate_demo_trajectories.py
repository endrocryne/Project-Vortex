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
  - Thrust: from actual Estes F15 thrust curve during burn phases
  - Mass: decreases linearly during burns
  - Quaternion: velocity-aligned (aerodynamic weathercock) with TVC wobble

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

# ------------------------------------------------------------------
# Rocket base parameters (Estes F15 equivalent)
# ------------------------------------------------------------------
ROCKET_BASE = {
    "dry_mass": 1.219,       # kg
    "propellant_mass": 0.063, # kg
    "length": 1.4224,         # m
    "diameter": 0.07874,      # m
    "burn_time": 1.7,         # s
    "thrust_curve": [         # (time_s, thrust_N)
        (0.0,   0.0),
        (0.013, 89.1),
        (0.018, 101.6),
        (0.029, 105.4),
        (0.047, 102.9),
        (0.104, 100.0),
        (0.19,  102.3),
        (0.268, 104.9),
        (0.306, 104.3),
        (0.38,  97.4),
        (0.45,  92.0),
        (0.6,   88.5),
        (0.75,  83.0),
        (0.9,   78.0),
        (1.05,  72.0),
        (1.2,   65.0),
        (1.38,  48.0),
        (1.5,   28.0),
        (1.6,   10.0),
        (1.7,   0.0),
    ],
}

G = 9.81  # m/s^2


# ------------------------------------------------------------------
# Thrust curve interpolation
# ------------------------------------------------------------------
def sample_thrust(t_burn, curve):
    """Linearly interpolate thrust from the thrust curve at local burn time t_burn."""
    if t_burn <= 0 or t_burn >= curve[-1][0]:
        return 0.0
    for i in range(len(curve) - 1):
        t0, f0 = curve[i]
        t1, f1 = curve[i + 1]
        if t0 <= t_burn <= t1:
            frac = (t_burn - t0) / max(t1 - t0, 1e-9)
            return f0 + (f1 - f0) * frac
    return 0.0


def total_impulse(curve):
    """Trapezoidal integration of thrust curve to get total impulse (N*s)."""
    imp = 0.0
    for i in range(len(curve) - 1):
        t0, f0 = curve[i]
        t1, f1 = curve[i + 1]
        imp += 0.5 * (f0 + f1) * (t1 - t0)
    return imp


# ------------------------------------------------------------------
# Quaternion utilities
# ------------------------------------------------------------------
def quat_identity():
    return (1.0, 0.0, 0.0, 0.0)


def quat_from_axis_angle(axis, angle):
    """Return (w, x, y, z) quaternion from axis-angle."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return quat_identity()
    ax, ay, az = ax / norm, ay / norm, az / norm
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return (c, ax * s, ay * s, az * s)


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2. Both are (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between q1 and q2."""
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
        return (w/norm, x/norm, y/norm, z/norm)
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
    return (w/n, x/n, y/n, z/n)


# ------------------------------------------------------------------
# Fault encoding for per-row data
# ------------------------------------------------------------------
FAULT_CODE = {
    "none": 0,
    "WIND_GUST": 1,
    "DRAG_CHANGE": 2,
    "MASS_LOSS": 3,
    "combined": 4,
}


def encode_fault_type(fault_types):
    if not fault_types:
        return 0
    if len(fault_types) > 1:
        return 4  # combined
    return FAULT_CODE.get(fault_types[0], 0)


# ------------------------------------------------------------------
# Wind direction from landing drift
# ------------------------------------------------------------------
def wind_direction_vector(run):
    """Normalized XY vector pointing toward where wind pushes rocket (landing drift direction)."""
    lx = run["landing_x"]
    ly = run["landing_y"]
    dist = math.sqrt(lx * lx + ly * ly)
    if dist < 0.01:
        return (1.0, 0.0)
    return (lx / dist, ly / dist)


# ------------------------------------------------------------------
# Physics-based trajectory generator
# ------------------------------------------------------------------
def generate_trajectory(run):
    """
    Generate physically correct launch-to-landing trajectory using RK4 integration.

    State vector: [x, y, z, vx, vy, vz, mass]
    Forces: thrust (during burn), gravity, drag, wind

    The rocket has two motors: ascent and descent (same thrust curve).
    Mass budget: m_total = m_dry + 2*m_prop (one prop charge per motor).
    """
    dt = 0.02  # 50 Hz integration

    fi = float(run["fault_intensity"])
    is_ml = run["mode"] == "ML"
    success = bool(run["success"])
    v_land_target = float(run["landing_velocity"])
    target_lx = float(run["landing_x"])
    target_ly = float(run["landing_y"])
    wind_speed_base = float(run["wind_speed"])

    m_dry = ROCKET_BASE["dry_mass"]
    m_prop = ROCKET_BASE["propellant_mass"]
    m_total = m_dry + m_prop              # ascent motor propellant only
    m_after_ascent = m_dry                # dry mass after ascent burn
    m_descent_start = m_dry + m_prop      # descent motor adds propellant
    burn_time = ROCKET_BASE["burn_time"]
    thrust_curve = ROCKET_BASE["thrust_curve"]
    diameter = ROCKET_BASE["diameter"]
    ref_area = math.pi * (diameter / 2.0) ** 2
    Cd_base = float(run["drag_coefficient"])
    rho = float(run["air_density"])

    # Total impulse and mass flow rate
    I_total = total_impulse(thrust_curve)
    mass_flow = m_prop / burn_time

    # Wind setup
    wx_dir, wy_dir = wind_direction_vector(run)
    has_wind_fault = "WIND_GUST" in run["fault_types"]
    has_drag_fault = "DRAG_CHANGE" in run["fault_types"]
    has_mass_fault = "MASS_LOSS" in run["fault_types"]
    fault_code_val = encode_fault_type(run["fault_types"])

    # Fault timing and magnitudes
    fault_trigger_offset = 2.0  # seconds after apogee detection
    mass_loss_amount = m_dry * fi * 0.003
    drag_multiplier_fault = 1.0 + fi * 0.8
    wind_gust_magnitude = wind_speed_base * (0.5 + fi * 1.5)

    # ── Descent motor: throttleable for constant-deceleration suicide burn ──
    avg_thrust = I_total / burn_time
    peak_thrust = max(f for _, f in thrust_curve)
    # Maximum deceleration the motor can provide
    max_motor_decel = peak_thrust / m_dry - G

    # Mutable simulation state
    descent_ignition_time = [None]

    def compute_acceleration(state, t_local, phase, faults_active_flag, Cd_eff, w_x, w_y):
        """Compute [ax, ay, az, mass_dot] from state [x, y, z, vx, vy, vz, mass]."""
        _x, _y, _z, vx, vy, vz, mass = state
        mass = max(mass, m_dry * 0.85)

        # Relative velocity (subtract wind)
        vx_rel = vx - w_x
        vy_rel = vy - w_y
        vz_rel = vz

        v_rel = math.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)

        # Drag force: F_drag = -0.5 * rho * Cd * A * |v_rel|^2 * v_hat
        drag_mag = 0.5 * rho * Cd_eff * ref_area * v_rel * v_rel
        if v_rel > 0.01:
            ax_d = -drag_mag * vx_rel / v_rel / mass
            ay_d = -drag_mag * vy_rel / v_rel / mass
            az_d = -drag_mag * vz_rel / v_rel / mass
        else:
            ax_d = ay_d = az_d = 0.0

        # Gravity
        az_g = -G

        # Thrust
        mdot = 0.0
        ax_t = ay_t = az_t = 0.0

        if phase == 0:  # ascent burn — follows thrust curve exactly
            thr = sample_thrust(t_local, thrust_curve)
            if thr > 0 and mass > m_after_ascent:
                mdot = -mass_flow
                az_t = thr / mass

        elif phase == 3:  # descent burn — throttleable constant-decel
            local_bt = t_local - (descent_ignition_time[0] or t_local)
            # Compute required deceleration to reach v_land at z=0
            z_now = max(_z, 0.5)
            v_vert = abs(vz) if vz < 0 else 0.01
            # Required deceleration: v² = v_land² + 2*a*(z - 0)  →  a = (v²-v_land²)/(2*z)
            if v_vert > v_land_target and z_now > 0.3:
                required_decel = (v_vert**2 - v_land_target**2) / (2.0 * z_now)
                required_decel = min(required_decel, max_motor_decel * 1.2)  # clamp
            elif v_vert > v_land_target:
                # Very close to ground but still fast — full braking
                required_decel = max_motor_decel
            else:
                required_decel = 0.0  # already at/below target speed

            # Account for drag already decelerating
            drag_decel_z = abs(az_d) if vz < 0 else 0.0
            needed_from_motor = required_decel + G - drag_decel_z
            needed_from_motor = max(0.0, needed_from_motor)
            required_thrust = mass * needed_from_motor

            # Cap at motor capability (with fault degradation)
            max_thr = peak_thrust
            if faults_active_flag and has_drag_fault:
                max_thr *= max(0.7, 1.0 - fi * 0.15)

            thr = min(required_thrust, max_thr)

            if thr > 0.5:
                mdot = -mass_flow * (thr / max(avg_thrust, 1.0))  # proportional mass flow
                az_t = thr / mass

                # TVC lateral correction for ML runs — gentle target-seeking
                if is_ml:
                    err_x = target_lx - _x
                    err_y = target_ly - _y
                    tvc_gain = 0.15 + 0.1 * fi
                    ax_t += err_x * tvc_gain
                    ay_t += err_y * tvc_gain
                    # Damp lateral velocity
                    ax_t -= vx * 0.3
                    ay_t -= vy * 0.3

        return (ax_d + ax_t, ay_d + ay_t, az_g + az_d + az_t, mdot)

    # ── Main integration loop ──
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, m_total]  # two motor charges
    raw_records = []
    t = 0.0
    phase = 0  # ascent burn
    apogee_detected = False
    apogee_time = None
    faults_active = False
    fault_activation_time = None
    Cd_eff = Cd_base
    wind_x = 0.0
    wind_y = 0.0
    descent_ignited = False
    ml_correction = 0.0

    # Constant light wind during ascent/coast
    base_wind_x = wx_dir * wind_speed_base * 0.3
    base_wind_y = wy_dir * wind_speed_base * 0.3

    max_sim_time = 30.0

    while t < max_sim_time:
        x, y, z, vx, vy, vz, mass = state

        # Clamp altitude
        if z < 0 and t > 0.5:
            z = 0.0
            state[2] = 0.0
            if vz < 0:
                state[5] = 0.0

        # Phase transitions
        if phase == 0 and t >= burn_time:
            phase = 1  # coast up
            state[6] = m_after_ascent  # ascent motor spent, dry mass only

        if phase == 1 and vz <= 0.0 and t > burn_time + 0.1:
            phase = 2  # coast down
            apogee_detected = True
            apogee_time = t

        if phase == 2 and apogee_detected and not faults_active:
            time_since_apogee = t - apogee_time
            if time_since_apogee >= fault_trigger_offset and fi > 0:
                faults_active = True
                fault_activation_time = t
                if has_drag_fault:
                    Cd_eff = Cd_base * drag_multiplier_fault
                if has_mass_fault:
                    state[6] = max(m_dry * 0.85, state[6] - mass_loss_amount)

        # Wind model
        if faults_active and has_wind_fault and fault_activation_time is not None:
            gust_ramp = min(1.0, (t - fault_activation_time) / 1.5)
            wind_x = base_wind_x + wx_dir * wind_gust_magnitude * gust_ramp
            wind_y = base_wind_y + wy_dir * wind_gust_magnitude * gust_ramp
        else:
            wind_x = base_wind_x
            wind_y = base_wind_y

        # ML correction computation
        if is_ml and phase == 2 and faults_active and fault_activation_time is not None:
            time_since_fault = t - fault_activation_time
            detection_ramp = min(1.0, time_since_fault / 2.0)
            ml_correction = fi * 15.0 * detection_ramp * (1.0 + 0.1 * math.sin(t * 2.3))
        elif is_ml and phase == 2 and not faults_active:
            ml_correction = 0.5 * math.sin(t * 1.5) * 0.3
        elif is_ml and phase == 3:
            pass  # keep last correction
        else:
            ml_correction = 0.0

        # Descent ignition: energy-based altitude calculation
        if phase == 2 and not descent_ignited and vz < -5.0:
            v_vert = abs(vz)
            # Ignition altitude: decel from v_vert to v_land at max motor capability
            h_ign = (v_vert**2 - v_land_target**2) / (2.0 * max(max_motor_decel, 5.0))
            h_ign = max(3.0, h_ign)

            # ML correction adds slightly to ignition altitude (ML fires slightly earlier)
            h_ign += ml_correction * 0.3
            # Small safety margin
            h_ign *= 1.03

            if z <= h_ign:
                phase = 3
                descent_ignition_time[0] = t
                descent_ignited = True

        # Fault magnitude for recording
        if faults_active and fault_activation_time is not None:
            if has_wind_fault and (has_drag_fault or has_mass_fault):
                fault_mag = fi
            elif has_wind_fault:
                wind_ramp = min(1.0, (t - fault_activation_time) / 1.5)
                fault_mag = fi * wind_ramp
            elif has_drag_fault:
                fault_mag = fi * 0.8
            elif has_mass_fault:
                fault_mag = fi * 0.06
            else:
                fault_mag = fi
        else:
            fault_mag = 0.0

        # Record
        raw_records.append({
            "t": round(t, 6),
            "x": x, "y": y, "z": max(0, z),
            "vx": vx, "vy": vy, "vz": vz,
            "mass": max(mass, m_dry * 0.85),
            "phase": phase,
            "ml_correction": ml_correction,
            "fault_code": fault_code_val if faults_active else 0,
            "fault_mag": fault_mag,
            "wind_x": wind_x,
            "wind_y": wind_y,
        })

        # Check termination
        if phase >= 3 and z <= 0.01 and t > burn_time + 1.0:
            break
        if phase == 2 and z <= 0.01 and t > burn_time + 1.0 and not descent_ignited:
            break
        if phase == 4:
            break

        # RK4 integration
        def deriv(s, tl):
            return compute_acceleration(s, tl, phase, faults_active, Cd_eff, wind_x, wind_y)

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

        state[6] = max(state[6], m_dry * 0.85)
        t += dt

    # ── Post-process: compute quaternions from velocity ──
    processed = []
    prev_quat = quat_identity()
    dit = descent_ignition_time[0]

    for i, rec in enumerate(raw_records):
        vx_r, vy_r, vz_r = rec["vx"], rec["vy"], rec["vz"]
        speed = math.sqrt(vx_r**2 + vy_r**2 + vz_r**2)
        speed_lat = math.sqrt(vx_r**2 + vy_r**2)
        ph = rec["phase"]

        if speed > 1.0:
            tilt_angle = math.atan2(speed_lat, abs(vz_r))

            if ph == 3 and dit is not None:
                local_bt = rec["t"] - dit
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

        # TVC-induced wobble during burn phases
        if ph == 0 or ph == 3:
            wobble_freq = 7.0 + fi * 3.0
            wobble_amp = math.radians(0.8 + 2.0 * fi)
            wobble_pitch = wobble_amp * math.sin(wobble_freq * rec["t"])
            wobble_yaw = wobble_amp * 0.7 * math.cos(wobble_freq * rec["t"] * 1.3 + 0.5)
            wobble_q = quat_from_axis_angle((1, 0, 0), wobble_pitch)
            wobble_q = quat_multiply(wobble_q, quat_from_axis_angle((0, 1, 0), wobble_yaw))
            target_quat = quat_multiply(target_quat, wobble_q)

        # Smooth quaternion transitions
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

    return processed


# ------------------------------------------------------------------
# Config builder
# ------------------------------------------------------------------
def build_config_json(run):
    """Build a config.json for the demo run, matching the project schema."""
    m_dry = ROCKET_BASE["dry_mass"]
    m_prop = ROCKET_BASE["propellant_mass"]
    diameter = ROCKET_BASE["diameter"]
    thrust_curve = [[t, f] for t, f in ROCKET_BASE["thrust_curve"]]

    wx, wy = wind_direction_vector(run)
    wind_dir_deg = math.degrees(math.atan2(wy, wx)) if (abs(wx) > 0.01 or abs(wy) > 0.01) else 45.0

    config = {
        "rocket": {
            "dry_mass": m_dry,
            "propellant_mass": m_prop,
            "length": ROCKET_BASE["length"],
            "diameter": diameter,
            "use_dynamic_inertia": False,
            "thrust_curve": thrust_curve,
            "burn_time": ROCKET_BASE["burn_time"],
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
            "gravity": G,
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
            "magnitude": ROCKET_BASE["dry_mass"] * fi * 0.06,
            "duration": 0,
            "probability": 1.0,
        })

    return [{"name": "Demo Faults", "faults": faults}] if faults else []


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    """Generate all 9 demo trajectories with physics-based integration."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_runs")
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 60)
    print("HexaVisual Demo - Physics-Based Trajectory Generator")
    print("=" * 60)

    fieldnames = ["Time", "X", "Y", "Z", "VX", "VY", "VZ", "QW", "QX", "QY", "QZ",
                  "Mass", "MLCorrection", "FaultType", "FaultMag", "WindX", "WindY", "Phase"]

    for run in DEMO_RUNS:
        run_dir = os.path.join(base_dir, run["id"])
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{run['id']}] {run['name']} (fi={run['fault_intensity']}, {run['mode']})")

        records = generate_trajectory(run)

        csv_path = os.path.join(run_dir, "trajectory.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        if records:
            final = records[-1]
            final_speed = math.sqrt(final["VX"]**2 + final["VY"]**2 + final["VZ"]**2)
            apogee = max(r["Z"] for r in records)
            print(f"  => {len(records)} rows, apogee={apogee:.1f}m, "
                  f"final VZ={final['VZ']:.3f} m/s (speed={final_speed:.3f}), "
                  f"pos=({final['X']:.2f}, {final['Y']:.2f})")

        config = build_config_json(run)
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Write demo manifest
    manifest = {
        "version": "2.0",
        "generated": "2026-03-09",
        "description": "Physics-based demo trajectories for HexaVisual Demo app. "
                       "Each run uses RK4 integration with real thrust curves, drag, and gravity. "
                       "ML runs include per-timestep correction data.",
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
            "dry_mass": ROCKET_BASE["dry_mass"],
            "propellant_mass": ROCKET_BASE["propellant_mass"],
            "diameter": ROCKET_BASE["diameter"],
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
