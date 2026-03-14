# Essentials — Consolidated Reference

> This document contains copies of all sections added to the Engineering Notebook during the March 2026 update session. Content has been simplified for a general audience. For the full technical context, see the source documents referenced in each section header.

---

## 1. RMSE Analysis — ML Ignition Correction

**Source**: `05_Results/10_Results_ML_Landing.md`, Section 3.5

### What is RMSE?

How do we know our ML model is actually good at its job? We need a way to measure how far off the model's predictions are from the "perfect" answer. That measurement is called RMSE -- Root Mean Square Error. Think of it like grading a test: if a student's answers are usually within a few points of the correct answer, they have a low RMSE (good). If their answers are all over the place, they have a high RMSE (bad). The key insight is that RMSE is measured in the same units as the thing being predicted -- in our case, meters -- so it tells us directly how many meters off the ML model tends to be.

RMSE works in three steps: (1) For each test case, find the difference between what the model predicted and the right answer -- this is the "error." (2) Square each error (so that overshooting by 2 m and undershooting by 2 m are treated the same). (3) Average all those squared errors, then take the square root to get back to the original units.

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

- $N$ = 1,000 held-out test scenarios the model never saw during training
- $y_i$ = the actual optimal ignition altitude correction (ground truth)
- $\hat{y}_i$ = the ML-predicted ignition altitude correction (model's best guess)

### Results

Our model's average "miss" is about 39 centimeters -- roughly the length of a school ruler. Since the rocket needs to land within 2 meters of its target, being off by 39 cm leaves plenty of room for other error sources.

$$\text{RMSE} = \sqrt{0.15} \approx 0.387 \text{ m}$$

**Physical interpretation**: A 0.387 m altitude error corresponds to a timing error of approximately 10--20 milliseconds. The flight computer updates every 500 ms, so it has many chances to refine the correction before reaching the ignition altitude.

### RMSE by Fault Scenario Category

Does the model do equally well in easy and hard situations? We grouped the test flights by difficulty:

| Fault Scenario Category | Test Samples | RMSE (m) | Notes |
|------------------------|-------------|----------|-------|
| Nominal (no faults) | 212 | 0.21 | Small corrections predicted accurately |
| Wind faults (5--15 m/s) | 287 | 0.34 | Moderate prediction difficulty |
| Drag + Mass faults | 241 | 0.41 | Higher EKF uncertainty |
| Severe combined (wind + drag + mass) | 260 | 0.52 | Multiple faults interact unpredictably |
| **Overall (all categories)** | **1,000** | **0.387** | **Weighted average** |

Even in the worst case (severe combined faults), RMSE remains at 0.52 m -- well within the 2 m landing accuracy budget.

### Comparison to Optimizer Baseline

How much better is the ML model than just using the pre-flight optimizer alone? The optimizer does well in calm conditions but cannot adapt mid-flight. The ML model watches what is actually happening and adjusts in real time:

- **Wind-only faults**: Optimizer error 3--5 m vs ML RMSE 0.34 m
- **Drag + mass faults**: Optimizer error 5--8 m vs ML RMSE 0.41 m
- **Severe combined faults**: Optimizer error > 10 m vs ML RMSE 0.52 m

The ML network reduces ignition altitude prediction error by approximately **10x** across all fault categories.

---



---

## 3. RMSE Summary — Simulation vs. Ground Truth

**Source**: `03_Methods/Core_Simulation_Engine.md`, Section 13.4

How accurate is HERMES? We compared its predictions against known correct answers (from physics formulas and real flight data) and measured the average miss.

### Consolidated RMSE Table

| Validation Check | Metric | Relative Error | Absolute RMSE | Source |
|---|---|---|---|---|
| Zero-wind kinematics | Altitude | 0.03% | 1.13 m (on 3,750 m prediction) | Analytical comparison |
| Tsiolkovsky velocity | Velocity | 0.095% | 0.40 m/s (on 420 m/s delta-v) | Analytical comparison |
| Quaternion norm preservation | Dimensionless | -- | 0.0002 | 60 s flight, 30 deg/s max rate |
| Apogee prediction vs. flight data | Altitude | 0.24% | 7.9 m (on 3,295 m measured) | Real trajectory data |
| Landing velocity -- all scenarios | Velocity | -- | 0.387 m/s | ML prediction cross-reference |

Across all dimensions, HERMES achieves sub-1% RMSE relative to ground truth. The largest error source is landing velocity prediction under fault conditions (0.387 m/s), which drops to 0.21 m/s for nominal conditions -- well within the 2.0 m/s success threshold.

---

## 4. Fault Intensity Quantification -- Mathematical Framework

**Source**: `03_Methods/07_Methods_Fault_Injection.md`, Section 13

### Why Do We Need This?

Imagine you are a doctor comparing injuries. A paper cut and a broken arm are both injuries, but obviously not equally serious. Rockets face a similar problem: a 5% drop in thrust is bad, but is it worse than a 10 m/s wind gust? This section develops a common yardstick -- a single number between 0 (no impact) and 1 (worst possible) for every fault.

### The Master Formula

The overall severity depends on three things multiplied together:

$$I = P \cdot B \cdot C$$

| Symbol | Name | Domain | Meaning |
|--------|------|--------|---------|
| $P$ | Probability | [0, 1] | How likely the fault is to happen |
| $B$ | Base intensity | [0, 1] | How bad it is based on magnitude alone |
| $C$ | Context modifier | [0.25, 1] | How much worse timing and duration make it |

If any one factor is zero, the overall severity is zero -- a fault that never happens is not dangerous.

### Base Intensity -- Hill Functions

Why not just use a straight line? Because real systems don't work that way. Small faults barely register because the control system absorbs them. Medium faults are where danger ramps up steeply. Very large faults plateau near maximum severity because the system is already overwhelmed. The Hill function captures this S-shaped relationship:

$$B(x;\;K,\,n) = \frac{x^{n}}{x^{n} + K^{n}}$$

| Fault Type       | Input $x$      | $K$        | $n$       | Physical Reasoning                                 |      |                                                   |
| ---------------- | -------------- | ---------- | --------- | -------------------------------------------------- | ---- | ------------------------------------------------- |
| Mass Loss        | $              | $\Delta m$ | $m_{ref}$ | 0.15                                               | 1.4  | Small losses absorbed; >15% rapidly dangerous     |
| Thrust Reduction | $1 - T_{mult}$ | 0.20       | 2.0       | Hoverslam has almost no thrust to spare            |      |                                                   |
| Thrust Increase  | $T_{mult} - 1$ | 0.25       | 0.7       | Excess thrust is less dangerous (can burn shorter) |      |                                                   |
| Drag Decrease    | $1 - D_{mult}$ | 0.25       | 1.8       | Less drag = faster descent, motor must compensate  |      |                                                   |
| Drag Increase    | $D_{mult} - 1$ | 0.30       | 0.8       | Extra drag slows vehicle (less dangerous)          |      |                                                   |
| Wind Gust        | $              | v_{wind}   | /15$      | 0.40                                               | 0.75 | First few m/s impose steepest difficulty increase |

### Timing Criticality -- Exponential Urgency

When a fault happens matters enormously. Think of braking a car: noticing a stop sign 500 m ahead gives plenty of time, but 10 m ahead means you must slam the brakes. The timing factor starts near zero for early faults and climbs exponentially toward 1 as the fault occurs closer to touchdown:

$$T_n(\tau) = \frac{e^{\alpha\tau} - 1}{e^{\alpha} - 1}, \qquad \alpha = 3.0$$

### Duration Severity -- Exponential Saturation

How long a fault lasts also matters. A brief glitch is far less dangerous than a permanent failure. The first few seconds of a fault cause the most additional danger; after that, the trajectory is already off course:

$$D_n(\tau) = 1 - e^{-\lambda\tau}, \qquad \lambda = 3.0$$

### Context Modifier -- Bilinear Interaction

Combines timing and duration into a single adjustment factor. The key insight: a fault that is both late AND long-lasting is much worse than you would expect from adding those effects together:

$$C = C_0 + (1 - C_0)(w_t \cdot T_n + w_d \cdot D_n + w_c \cdot T_n \cdot D_n)$$

with $C_0 = 0.25$, $w_t = 0.35$, $w_d = 0.35$, $w_c = 0.30$.

### Combined Fault Intensity

When multiple faults happen simultaneously, individual intensities are combined using the probabilistic-OR formula:

$$I_{combined} = 1 - \prod_{i=1}^{k}(1 - I_i)$$

Two 30% faults do not produce 60% combined intensity but rather $1 - (0.7)^2 = 0.51$ -- there is overlap in the damage they cause.

### Boundary Properties

Five common-sense checks, all guaranteed by the math:
1. $I \in [0, 1]$ -- always between 0 and 1
2. Zero magnitude = zero impact
3. Zero probability = zero contribution
4. Maximum severity approaches but never quite reaches 1.0
5. Worse inputs always produce a worse score

---

## 5. Source Code -- Core Physics Engine

**Source**: `03_Methods/Core_Simulation_Engine.md`, Code Appendix

### `state_derivative()` -- The Heart of the Simulation

This is the function that answers "given where the rocket is right now, what happens next?" It looks at the rocket's current position, speed, orientation, spin, and mass, then calculates all the forces acting on it (gravity, drag, thrust). From those forces, it figures out how the rocket will accelerate, rotate, and lose mass over the next tiny fraction of a second. The math solver calls this function thousands of times per simulated flight.

```python
def state_derivative(self, t, state):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]
    mass = state[13]

    quaternion = self.physics.normalize_quaternion(quaternion)
    R_body_to_inertial = self.physics.quaternion_to_rotation_matrix(quaternion)

    cg_location = self.calculate_dynamic_cg(mass)
    current_inertia_tensor = self.calculate_dynamic_inertia(mass, cg_location)
    cg_offset_from_thrust = np.array([0, 0, cg_location - self.fuel_tank_bottom])

    # Forces in inertial frame
    F_gravity = np.array([0, 0, -mass * self.physics.g])
    F_drag = self.physics.get_drag_force(velocity, position, t)
    F_thrust = self.motor.get_thrust_vector(t, R_body_to_inertial)
    F_total = F_gravity + F_drag + F_thrust

    acceleration = F_total / mass if mass > 0 else np.zeros(3)

    # Moments in body frame
    M_thrust = self.motor.get_thrust_moment(t, cg_offset_from_thrust)
    M_aero = -0.1 * omega_body
    M_total = M_thrust + M_aero

    # Angular acceleration (Euler's equation)
    I_omega = current_inertia_tensor @ angular_velocity
    omega_cross_I_omega = np.cross(angular_velocity, I_omega)
    angular_acceleration = np.linalg.solve(current_inertia_tensor,
                                           M_total - omega_cross_I_omega)

    # Quaternion derivative
    omega_quat = np.array([0, angular_velocity[0], angular_velocity[1], angular_velocity[2]])
    q_dot = 0.5 * self.physics.quaternion_multiply(quaternion, omega_quat)

    mass_dot = -self.motor.get_mass_flow_rate(t)
    return np.concatenate([velocity, acceleration, q_dot, angular_acceleration, [mass_dot]])
```

### `quaternion_to_rotation_matrix()` -- Avoiding Gimbal Lock

The rocket's orientation is stored as a "quaternion" -- a compact mathematical object that avoids gimbal lock (a problem that confuses calculations at certain angles). This function builds the 3x3 rotation matrix that translates directions between the rocket's perspective and the Earth-based perspective.

```python
def quaternion_to_rotation_matrix(self, q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R
```

### `get_drag_force()` -- Aerodynamic Resistance

As the rocket moves through the air, it experiences drag -- the same force you feel when you stick your hand out of a car window. This function calculates how strong that drag is and which direction it pushes, accounting for altitude-dependent air density, wind, and random variation.

```python
def get_drag_force(self, velocity, position, time):
    altitude = position[2]
    rho = self.get_air_density(altitude)
    wind = self.get_wind_velocity(position, time)
    v_rel = velocity - wind
    v_rel_mag = np.linalg.norm(v_rel)
    if v_rel_mag < 0.01:
        return np.zeros(3)
    Cd_actual = self.Cd * (1.0 + np.random.uniform(-self.drag_variation, self.drag_variation))
    drag_magnitude = 0.5 * rho * v_rel_mag**2 * Cd_actual * self.A_ref
    drag_force = -drag_magnitude * (v_rel / v_rel_mag)
    return drag_force
```

---

## 6. Source Code -- EKF and TVC Control

**Source**: `03_Methods/06_Methods_EKF_PID_Control.md`, Code Appendix

The EKF (Extended Kalman Filter) is like a smart "averaging" system -- it combines what it expects the sensors to read (based on physics) with what the sensors actually read, weighting each by trustworthiness. The TVC controller is the autopilot -- it detects tilt and steers the engine nozzle to correct.

### `StateEstimator.predict()` -- The "Look Ahead" Step

Every timestep, the filter guesses where the rocket should be based on how fast fuel is burning. It subtracts the burned fuel mass and admits it is a little less certain about the answer (uncertainty grows).

```python
def predict(self, mass_flow_rate=0.0):
    self.x[0] -= mass_flow_rate * self.dt
    self.P += self.Q
```

### `StateEstimator.update()` -- The "Reality Check" Step

When a new accelerometer reading arrives, the filter compares it to its prediction. The difference tells it how wrong the prediction was. It then corrects its estimates, leaning toward whichever source (sensor or prediction) is more trustworthy.

```python
def update(self, accel_sens_z, velocity_z, altitude, rho, thrust_z):
    m = self.x[0]
    cd = self.x[1]
    q_factor = -0.5 * rho * velocity_z * abs(velocity_z) * self.A_ref
    drag_force = q_factor * cd
    h = (thrust_z + drag_force) / m
    y_residual = accel_sens_z - h
    H = np.array([-h / m, q_factor / m])
    S = H @ self.P @ H.T + self.R
    K = self.P @ H.T / S
    self.x += K * y_residual
    self.P = (np.eye(2) - np.outer(K, H)) @ self.P
    self.x[0] = max(1.0, self.x[0])
    self.x[1] = max(0.1, min(self.x[1], 2.0))
    return self.x
```

### `tvc_controller()` -- The Autopilot

The PID controller reads the rocket's current orientation and drift speed, figures out how far off from vertical the rocket is, and calculates a gimbal command. Proportional (react to current tilt), Integral (fix persistent lean), and Derivative (damp oscillations using gyroscope data) work together to keep the rocket upright.

```python
def tvc_controller(self, state, time, last_time, pitch_int, yaw_int,
                   last_pitch_err, last_yaw_err):
    q = state[6:10]
    vx, vy = state[3], state[4]
    omega_x, omega_y, omega_z = state[10:13]
    target_pitch, target_yaw = 0.0, 0.0

    if self.tvc_mode == 'velocity':
        target_pitch = -vx * self.tvc_drift_gain
        target_yaw = vy * self.tvc_drift_gain
        max_tilt = np.radians(15.0)
        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_yaw = np.clip(target_yaw, -max_tilt, max_tilt)

    pitch_error = (2 * q[2]) - target_pitch
    yaw_error = (2 * q[1]) - target_yaw
    dt = max(time - last_time if last_time >= 0 else 0.01, 1e-6)

    max_integral = 0.5
    new_pitch_int = np.clip(pitch_int + pitch_error * dt, -max_integral, max_integral)
    new_yaw_int = np.clip(yaw_int + yaw_error * dt, -max_integral, max_integral)

    pitch_command = (-self.tvc_kp_pitch * pitch_error
                     - self.tvc_ki_pitch * new_pitch_int
                     - self.tvc_kd_pitch * omega_y)
    yaw_command = (-self.tvc_kp_yaw * yaw_error
                   - self.tvc_ki_yaw * new_yaw_int
                   - self.tvc_kd_yaw * omega_x)
    return pitch_command, yaw_command, new_pitch_int, new_yaw_int, pitch_error, yaw_error
```

### `SolidMotor.get_thrust_vector()` -- TVC Thrust Direction

The motor produces thrust along the rocket's centerline. If the nozzle is gimbaled (tilted), the thrust direction shifts. This function takes the gimbal angles, applies two rotations (pitch and yaw), and converts the resulting thrust vector into real-world coordinates.

```python
def get_thrust_vector(self, time, body_to_inertial_matrix):
    thrust_magnitude = self.get_thrust(time)
    if thrust_magnitude == 0:
        return np.zeros(3)
    thrust_nominal = np.array([0, 0, 1])
    pitch, yaw = self.current_tvc_angle
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[1, 0, 0], [0, cos_y, -sin_y], [0, sin_y, cos_y]])
    R_tvc = R_yaw @ R_pitch
    thrust_body = thrust_magnitude * (R_tvc @ thrust_nominal)
    thrust_inertial = body_to_inertial_matrix @ thrust_body
    return thrust_inertial
```

---

## 7. Source Code -- Fault Injection System

**Source**: `03_Methods/07_Methods_Fault_Injection.md`, Code Appendix

### `calculate_fault_intensity()` -- Severity Scoring

Computes a normalized intensity I in [0, 1] for a single fault using the formula I = P * B * C. This drives the fault severity visualization in the dashboard.

```python
def calculate_fault_intensity(fault, typical_descent_time=10.0,
                              typical_altitude=1000.0, reference_mass=60.0):
    C_FLOOR, W_T, W_D, W_C = 0.25, 0.35, 0.35, 0.30
    ALPHA, LAMBDA = 3.0, 3.0

    # Base intensity B using Hill functions
    base_intensity = 0.0
    if fault.fault_type == FaultType.MASS_LOSS:
        mass_frac = abs(fault.magnitude) / reference_mass
        base_intensity = _hill(mass_frac, K=0.15, n=1.4)
    elif fault.fault_type == FaultType.THRUST_VAR:
        if fault.magnitude < 1.0:
            base_intensity = _hill(1.0 - fault.magnitude, K=0.20, n=2.0)
        else:
            base_intensity = min(0.4, _hill(fault.magnitude - 1.0, K=0.25, n=0.7))
    elif fault.fault_type == FaultType.DRAG_CHANGE:
        if fault.magnitude < 1.0:
            base_intensity = _hill(1.0 - fault.magnitude, K=0.25, n=1.8)
        else:
            base_intensity = min(0.6, _hill(fault.magnitude - 1.0, K=0.30, n=0.8))
    elif fault.fault_type == FaultType.WIND_GUST:
        base_intensity = min(1.0, _hill(abs(fault.magnitude) / 15.0, K=0.40, n=0.75))

    # Timing criticality T_n
    timing_frac = 0.6
    if fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
        timing_frac = 1.0 - min(1.0, fault.trigger_value / typical_altitude)
    T_n = _exp_timing(timing_frac, ALPHA)

    # Duration severity D_n
    D_n = 1.0 if fault.duration == 0.0 else _exp_saturation(fault.duration / typical_descent_time, LAMBDA)

    # Context modifier C
    context = C_FLOOR + (1.0 - C_FLOOR) * (W_T * T_n + W_D * D_n + W_C * T_n * D_n)

    # Final intensity I = P * B * C
    return fault.probability * base_intensity * context
```

### Helper Functions

```python
def _hill(x, K, n):
    """Hill function: f(x) = x^n / (x^n + K^n)"""
    if x <= 0.0:
        return 0.0
    xn = x ** n
    return xn / (xn + K ** n)

def _exp_timing(frac, alpha):
    """Exponential criticality: T_n(t) = (e^(alpha*t) - 1) / (e^alpha - 1)"""
    if alpha == 0.0:
        return frac
    ea = np.exp(alpha)
    return (np.exp(alpha * frac) - 1.0) / (ea - 1.0)

def _exp_saturation(frac, lam):
    """Exponential saturation: D_n(t) = 1 - e^(-lambda*t)"""
    return 1.0 - np.exp(-lam * frac)
```

### `check_triggers()` -- Runtime Fault Activation

Called each simulation timestep. Iterates over all configured faults and evaluates whether each fault's trigger condition has been met. Faults that trigger are activated with probability sampling.

```python
def check_triggers(self, current_time, altitude, vz):
    self.detect_apogee(current_time, vz, altitude)
    newly_triggered = []
    for fault in self.all_faults:
        if fault.active:
            continue
        triggered = False
        if fault.trigger_mode == TriggerMode.ABSOLUTE_TIME:
            triggered = current_time >= fault.trigger_value
        elif fault.trigger_mode == TriggerMode.TIME_SINCE_APOGEE:
            if self.apogee_detected and self.apogee_time is not None:
                triggered = (current_time - self.apogee_time) >= fault.trigger_value
        elif fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
            triggered = altitude <= fault.trigger_value
        if triggered:
            if np.random.random() <= fault.probability:
                self._activate_fault(fault, current_time)
                newly_triggered.append(fault)
    return newly_triggered
```

### `apply_faults_to_state()` -- State Modification

Applies all active faults to the simulation state vector. Mass loss faults directly modify the mass element of the 14-element state vector with a floor of 1 kg to prevent division-by-zero.

```python
def apply_faults_to_state(self, state, current_time):
    modified_state = state.copy()
    self.mass_delta = 0.0
    for fault in self.active_faults:
        if fault.target != FaultTarget.SIMULATED_STATE:
            continue
        if fault.fault_type == FaultType.MASS_LOSS:
            self.mass_delta += fault.applied_magnitude
            modified_state[13] = max(1.0, modified_state[13] + fault.applied_magnitude)
    return modified_state
```

---

## 8. Source Code -- Simulation Loop

**Source**: `02_Framework/04_HERMES_Framework.md`, Code Appendix

### `run_simulation()` -- Full Mission Driver

This function runs one complete simulated flight from a given state through each phase: boost upward, separate the payload, coast in freefall, re-ignite the engine, steer the nozzle to stay upright, and check if the landing was soft enough. Events like "altitude dropped below ignition height" or "the rocket touched the ground" are detected automatically by the math solver.

```python
def run_simulation(self, initial_state, ignition_altitude=None, max_time=60.0):
    self.motor = SolidMotor(self.rocket_config)
    current_sim_state = initial_state.copy()

    # PHASE 1: ASCENT
    if self.simulate_ascent:
        apogee_state, t_asc, y_asc = self.run_ascent_phase(current_sim_state)
        current_sim_state = apogee_state.copy()
        current_sim_state[13] = self.dry_mass + self.propellant_mass  # Payload separation
        self.motor = SolidMotor(self.rocket_config)  # Fresh motor for descent

    # PHASE 2: FREEFALL DESCENT
    sol_freefall = solve_ivp(
        self.state_derivative,
        [current_time_offset, current_time_offset + max_time],
        current_sim_state,
        events=[ignition_event, ground_event],
        method='RK45', rtol=1e-6, atol=1e-9, max_step=0.01
    )

    # PHASE 3: POWERED LANDING BURN
    if len(sol_freefall.t_events[0]) > 0:
        self.motor.ignite(ignition_time)

        def burning_state_derivative(t, state):
            if t > self.last_time:
                p_cmd, y_cmd, p_int, y_int, p_err, y_err = self.tvc_controller(
                    state, t, self.last_time,
                    self.pitch_integral_error, self.yaw_integral_error,
                    self.last_pitch_error, self.last_yaw_error)
                self.motor.set_tvc_command(p_cmd, y_cmd)
                self.motor.update_tvc(t - self.last_time)
                self.last_time = t
            return self.state_derivative(t, state)

        sol_powered = solve_ivp(
            burning_state_derivative,
            [ignition_time, current_time_offset + max_time],
            state_at_ignition,
            events=[ground_event],
            method='RK45', rtol=1e-6, atol=1e-9, max_step=0.01
        )

    success = (abs(final_altitude) < 1.0
               and abs(final_velocity[2]) < 2.0
               and final_speed < 3.0)
    return success, final_state, history
```

### `optimize_ignition_altitude()` -- Monte Carlo Search

This function asks: "At what altitude should we re-light the engine?" It tries 200 different altitudes, running 100 randomized flights at each one. The altitude with the highest success rate wins. It's like a coach running hundreds of practice drills under different weather conditions to find the best play.

```python
def optimize_ignition_altitude(self, initial_state, num_monte_carlo=100,
                               altitude_search_range=10.0, altitude_step=0.1, ...):
    estimate = self.calculate_ignition_altitude(v_for_est, h_for_est)

    altitudes = np.arange(
        max(0, estimate - altitude_search_range),
        estimate + altitude_search_range + altitude_step,
        altitude_step
    )

    for altitude in altitudes:
        successes = 0
        for i in range(int(num_monte_carlo)):
            success, final_state, history = self.run_simulation(
                initial_state.copy(), altitude)
            if success:
                successes += 1
        success_rate = successes / float(num_monte_carlo)
        success_rates[altitude] = success_rate
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_altitude = altitude

    return best_altitude, success_rates, best_history
```

---


### 9.3 Demonstration: Sensor Drift (Unseen Fault)

Consider a realistic but unplanned scenario: the **MPL3115A2 barometric altimeter experiences temperature-induced pressure offset** during the rapid descent and temperature change as the rocket ejects from the warm avionics bay into cold air.

**Scenario details** (marked as demonstration scenario for honesty):

- Systematic sensor drift: altimeter reads ~4.5 m **too high** throughout descent
- Root cause: NOT in the training data (training only simulated nominal sensors)
- The rocket doesn't "know" its altimeter is drifting — only the EKF sees inconsistencies

**What the optimizer would do (without ML)**:

- Pre-flight baseline: ignite at 36.1 m (computed under assumption of perfect sensor)
- During descent: sensor reads 36.1 m → fire ignition signal
- Reality: actual altitude is only  = 31.6 m
- Outcome: rocket has 4.5 m less altitude to decelerate → **CRASH**

**What the ML model does (with no fault label, only state data)**:

The EKF is simultaneously estimating [mass, ] using accelerometer data (more reliable than barometric in rapid descent). The barometric altitude reading is **inconsistent** with what the accelerometer-integrated trajectory says altitude should be. The EKF detects this mismatch and adjusts inferred_drag_coeff upward (trying to reconcile the discrepancy by attributing the mismatch to aerodynamics).

The rising inferred_drag_coeff signal (0.508 → 0.572 over 10 seconds) is a pattern the model **saw frequently during training** for high-drag scenarios. The model learned:

> When drag coefficient appears elevated and the altitude-velocity relationship is off, ignite higher to maintain safety margin.

Here's how the correction evolves during the descent:

|Time Before Ignition (s)|Sensor Altitude (m)|True Altitude (m)|Inferred|ML Correction (m)|
|---|---|---|---|---|
|−10.0|122.3|117.8|0.508|0.0|
|−8.0|96.7|92.2|0.531|+1.2|
|−6.0|74.1|69.6|0.548|+2.8|
|−4.0|54.8|50.3|0.561|+3.7|
|−2.0|40.5|36.0|0.567|+4.1|
|−1.0|34.2|29.7|0.572|+4.2|
|**Ignition**|**40.3 (sensor)**|**35.8 (actual)**|—|**+4.2 (locked)**|

**Outcome with ML correction**:

- Final ignition threshold: 36.1 m (baseline) + 4.2 m (ML correction) = 40.3 m in sensor frame
- When sensor reads 40.3 m → actual altitude  35.8 m (desired target!)
- Landing velocity: 0.91 m/s — **SUCCESS** (well below 3 m/s threshold)

**Outcome without ML** (optimizer only):

- Ignition at sensor reading 36.1 m → actual altitude 31.6 m
- Landing velocity: 9.3 m/s — **CRASH**

**Why the model generalized**: The pattern of rising inferred_drag_coeff is physically meaningful: it signals a _state mismatch_ that the model learned to mitigate. The physical cause (sensor drift vs. actual high-drag airfoil) doesn't matter — the correction is similar because the state signature is similar.
