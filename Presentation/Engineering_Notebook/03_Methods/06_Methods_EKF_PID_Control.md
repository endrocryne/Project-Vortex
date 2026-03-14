# Methods: Extended Kalman Filter and PID Attitude Control

## 1. Overview: Observation and Action

The landing phase of Project Vortex presents a control problem with two coupled challenges:

**Observation Challenge:** "What is the true state of the rocket right now?"
- We have noisy sensor data (accelerometer, barometer, gyroscope)
- We don't know the rocket's actual mass (propellant is being burned)
- We don't know the actual drag coefficient (changes with surface condition and angle of attack)
- These unknowns directly affect the optimal ignition altitude

**Action Challenge:** "How do we keep the rocket pointing straight down?"
- Wind pushes the rocket laterally
- The SRM produces thrust; without steering, this thrust doesn't oppose the velocity vector
- We have a limited steering authority (TVC gimbal ±5°)
- Poor attitude alignment wastes fuel and causes lateral drift at landing

**Solution:** Two systems working together:
- **Extended Kalman Filter (EKF):** Fuses noisy sensor data to estimate unknown state variables (mass, drag coefficient)
- **PID Controller:** Uses the estimated state to compute corrective TVC commands that maintain attitude alignment

---

## 2. Why State Estimation Is Needed

### The Sensor Noise Problem

A naive approach: read the altimeter, read the accelerometer, compute position and velocity directly.

**Problem:** Sensors are noisy.
- Barometric altimeter: ±10 m noise (standard deviation), ±50 m possible
- Accelerometer: ±0.1 m/s² bias, ±0.05 m/s² noise
- Gyroscope: ±1–2°/s drift over 10 minutes

If we integrate noisy acceleration twice to get position, errors accumulate rapidly:
- After 1 second: small error (cm-scale)
- After 10 seconds: significant error (meters)
- After 100 seconds (a long descent): error is meters to tens of meters

The Extended Kalman Filter solves this by using a **physics-based model** to predict where the rocket "should" be, then correcting that prediction using sensor measurements.

### The Unknown Parameters Problem

Two critical parameters affect the ignition decision:

1. **Actual vehicle mass m(t):** We know the initial mass (60 kg) and the burn rate. But the actual propellant grain might burn slightly faster or slower than predicted. By descent, the true mass could be 58–62 kg instead of the nominal 60 kg. A 2 kg error changes the deceleration rate ($a = T/m - g$) by 3%, which translates to 1–2 m error in ignition altitude.

2. **Actual drag coefficient $C_d$:** Nominal is 0.5, but the rocket's surface might be rougher or smoother than expected, the ablation layer might erode unevenly, or angle-of-attack variations might effectively change $C_d$. Range: 0.45–0.65. A 10% change in $C_d$ changes descent velocity by ~5%, which is significant over 100 seconds of descent.

**Solution:** The EKF continuously estimates mass and $C_d$ from sensor observations, allowing the ignition altitude to be adjusted in real-time based on actual vehicle properties.

### How the Observation Helps

When the rocket is descending, the accelerometer measures:

$a_{measured} = T_{thrust} / m + a_{drag} / m - g$

where $a_{drag}$ is the drag deceleration:

$a_{drag} = F_{drag} / m = (\frac{1}{2} \times \rho \times C_d \times A \times v^2) / m$

The EKF observes this acceleration, compares it to the predicted acceleration (which depends on mass and $C_d$), and uses the difference to refine its mass and $C_d$ estimates.

**Example:**
- Predicted acceleration: $a_{pred} = (0 \text{ N} + 2 \text{ N}) / 50 \text{ kg} - 9.81 \text{ m/s}^2 = 0.04 - 9.81 = -9.77 \text{ m/s}^2$ (downward)
- Measured acceleration: $a_{meas} = -9.90 \text{ m/s}^2$ (slightly more downward than predicted)
- Difference: $\Delta a = -9.90 - (-9.77) = -0.13 \text{ m/s}^2$

A larger-than-expected downward acceleration suggests the drag deceleration is lower than predicted, which could mean:
- $C_d$ is smaller than expected, or
- Mass is larger than expected (less mass → less drag force, but our model assumed 50 kg)

The EKF uses the difference to update its estimate of these parameters.

---

## 3. The Extended Kalman Filter

### Background: Kalman Filter vs Extended Kalman Filter

**Kalman Filter:** Optimal estimator for linear Gaussian systems. Given a model:
- $x_{k+1} = A \times x_k + \text{noise}$
- $z_k = C \times x_k + \text{noise}$

The Kalman filter computes the minimum-variance estimate.

**Extended Kalman Filter:** Handles nonlinear systems by linearizing around the current estimate. Given:
- $x_{k+1} = f(x_k) + \text{noise}$ [nonlinear state evolution]
- $z_k = h(x_k) + \text{noise}$ [nonlinear measurement]

The EKF approximates f and h locally using Taylor series, then applies Kalman filter logic.

### HERMES EKF: State Vector

The EKF in HERMES tracks only **two state variables:**

```
x_ekf = [mass, Cd]
```

**Why only two?** The other 12 variables (position, velocity, attitude) are estimated directly from sensors:
- Altitude from barometer
- Velocity from IMU integration
- Attitude from BNO055 (has onboard sensor fusion)

We only need to estimate the unknowns: mass and $C_d$.

### EKF Prediction Step (Time Update)

The prediction step runs at 100 Hz and advances the state by 10 ms.

#### State Transition Model

During descent (before ignition), the rocket is unpowered, so thrust T = 0. The mass decreases due to slow residual propellant outgassing (negligible during coat phase), so:

**m_{k+1} = m_k - ṁ_residual $\times$ $\Delta$t $pprox$ m_k**

(Mass is approximately constant until ignition; once ignition occurs, ṁ = T / (Isp $\times$ $g_0$) = 0.0102 kg/s)

**$C_d$_{k+1} = $C_d$_k**

(Drag coefficient is assumed constant between updates, though it could slowly increase due to ablation)

#### Covariance Propagation

The EKF maintains a covariance matrix P that represents uncertainty in the state estimate:

```
P = [
  P_mm  P_m,Cd
  P_Cd,m  P_Cd,Cd
]
```

where $P_{mm}$ = variance in mass estimate, $P_{C_d,C_d}$ = variance in $C_d$ estimate, and $P_{m,C_d}$ = covariance (correlation) between the two.

The covariance is propagated using the Jacobian of the state transition:

**Jacobian $F = \partial f / \partial x$:**

```
F = [
  [1, 0],           # ∂m/∂m = 1, ∂m/∂C_d = 0
  [0, 1]            # ∂C_d/∂m = 0, ∂C_d/∂C_d = 1
]
```

(Since mass and $C_d$ are approximately constant, the Jacobian is identity.)

**Covariance update:**

$P_{k+1|k} = F \times P_{k|k} \times F^T + Q$

where Q is the process noise covariance (models our uncertainty about how mass and $C_d$ evolve):

```
Q = diag([0.01, 0.001])
```

$Q_{mm} = 0.01$ means we expect mass to change by ~±0.1 kg over 10 ms (from uncertainties not captured by our model). $Q_{C_d,C_d} = 0.001$ is smaller because we expect $C_d$ to be more stable.

### EKF Update Step (Measurement Update)

The update step runs whenever an accelerometer measurement arrives (100 Hz on HERMES).

#### Predicted Measurement

We have a measurement model: the accelerometer should measure:

$$h_{pred} = (T + F_{drag}) / m - g$$

where:
- T = thrust (0 during descent, known value during burn)
- $F_{drag} = \frac{1}{2} \times \rho \times C_d \times A \times v^2$ (depends on $C_d$ and measured v)
- m = vehicle mass (being estimated)
- g = gravity (known)

Since the accelerometer measures in the vehicle's body frame, we also account for the gravity component along the acceleration axis and rotate velocity to body frame. Simplifying for the vertical channel:

$$h_{pred} = F_{drag} / m - g$$

where $F_{drag}$ depends on the state [mass, $C_d$]:

$$F_{drag} = \frac{1}{2} \times \rho \times C_d \times A \times v^2$$

The measurement h_{meas} is the accelerometer's reported acceleration in the vertical direction.

#### Innovation (Measurement Residual)

The innovation is the difference between predicted and measured acceleration:

$$y = h_{meas} - h_{pred} = h_{meas} - (F_{drag} / m - g)$$

If y $pprox$ 0, the prediction matches the measurement (good). If |y| is large, the prediction is wrong, and we need to update our estimate.

#### Measurement Jacobian

The Jacobian H relates changes in [mass, $C_d$] to changes in the predicted acceleration:

**H = $\partial h_{pred} / \partial x$:**

```
H = [
  ∂(F_{drag} / m - g) / ∂m,
  ∂(F_{drag} / m - g) / ∂C_d
]

H = [
  -F_{drag} / m²,                      # Increasing mass decreases drag decel
  -\frac{1}{2} × ρ × A × v² / m        # Increasing C_d increases drag decel
]
```

These partial derivatives are computed numerically:

```python
F_drag = 0.5 * rho * Cd * A_ref * v**2
H[0] = -F_drag / m**2
H[1] = -0.5 * rho * A_ref * v**2 / m
```

#### Kalman Gain

The Kalman gain K determines how much weight to give the measurement vs the prediction:

$$K = P \times H^T \times (H \times P \times H^T + R)^{-1}$$

where R = 0.1 is the measurement noise variance of the accelerometer.

**Interpretation:**
- If R is large (noisy accelerometer), K is small (trust prediction more)
- If R is small (clean accelerometer), K is large (trust measurement more)
- If P is large (uncertain state), K is large (update state significantly)
- If P is small (confident state), K is small (make minimal adjustments)

#### State Update

The state is updated proportionally to the innovation:

$$x_{k|k} = x_{k|k-1} + K \times y$$

$$m_{updated} = m_{predicted} + K[0] \times (h_{meas} - h_{pred})$$

$$C_d_{updated} = C_d_{predicted} + K[1] \times (h_{meas} - h_{pred})$$

Example:
- Predicted acceleration: $-9.77$ m/s²
- Measured acceleration: $-9.90$ m/s² (0.13 m/s² more downward)
- If K[0] = 0.01 s²/kg, then $\Delta m = 0.01 \times (-0.13) = -0.0013$ kg (mass estimate decreases slightly)
- If K[1] = 0.02, then $\Delta C_d = 0.02 \times (-0.13) = -0.0026$ ($C_d$ estimate decreases slightly)

Decreasing mass and $C_d$ estimates makes sense: if the rocket is accelerating downward faster than predicted, either it's lighter (less drag force per unit mass) or the drag coefficient is smaller.

#### Covariance Update

After incorporating the measurement, our uncertainty decreases:

$$P_{k|k} = (I - K \times H) \times P_{k|k-1}$$

The term $(I - K \times H)$ is always positive semi-definite and smaller than I, so covariance shrinks with each measurement.

### Safety Clamps

To prevent physically impossible estimates, the EKF applies bounds:

```python
m = clamp(m, 1.0, 60.0)      # Mass between 1 kg and 60 kg
Cd = clamp(Cd, 0.1, 2.0)     # Drag coefficient 0.1 to 2.0
```

If the EKF tries to estimate mass > 60 kg or $C_d$ < 0.1, these values are forced back into valid ranges. This prevents the filter from diverging due to modeling errors.

### Practical Effect on Ignition Altitude

Suppose the EKF estimates actual $C_d$ = 0.7 (vs nominal 0.5). This means the rocket experiences 40% more drag during descent. The descent velocity at a given altitude is lower than predicted. For the ignition logic:

**Original $h_{ign}$ = 36.1 m** (computed assuming $C_d$ = 0.5)

**Adjusted $h_{ign}$ = 36.1 - f($C_d$_estimated) = 36.1 - 2.0 = 34.1 m** (if EKF detects higher $C_d$)

The correction function f(·) is learned by the ML flight computer or hand-tuned based on simulation. The EKF provides the observation; the correction is applied elsewhere in the flight logic.

---

## 4. The PID Attitude Controller

### Control Objective

**Goal:** Keep the rocket pointing straight down (pitch = 0°, yaw = 0°).

**Why?** If the rocket is tilted, the thrust vector isn't aligned with the velocity vector. Misalignment causes:
- Thrust component perpendicular to velocity → lateral acceleration
- Inefficient deceleration (some thrust wasted on horizontal direction)
- Lateral drift at landing (violates v_lateral < 3 m/s criterion)

The PID controller's job is to generate TVC gimbal commands that steer the nozzle to null out tilt errors.

### PID Control Law Explained for Non-Experts

A PID (Proportional-Integral-Derivative) controller has three terms:

#### P (Proportional) Term

React proportionally to current error. If the rocket is tilted $\theta$ = 5°, apply gimbal command $\delta$ = Kp $\times$ $\theta$ = 0.5 $\times$ 5° = 2.5°.

**Advantage:** Simple, responds immediately to disturbances.

**Disadvantage:** If there's a constant disturbance (e.g., constant crosswind), the error doesn't go to zero—it settles at some nonzero value (steady-state error).

#### I (Integral) Term

Accumulate error over time. If the rocket drifts 1° for 10 seconds, the integral builds up: $\int$e dt = 1° $\times$ 10 s = 10 °·s. The integral term is Ki $\times$ $\int$e dt = 0.05 $\times$ 10 = 0.5°.

**Advantage:** Eliminates steady-state error; can correct constant biases.

**Disadvantage:** If not carefully tuned, can cause "wind-up": if the controller is saturated (gimbal at ±5° limit), the integral keeps growing, and when the gimbal is released, the rocket overshoots (oscillates).

#### D (Derivative) Term

React to the rate of change of error. If the error is increasing rapidly (rocket tilt rate d$\theta$/dt = 2°/s), apply damping: u_D = Kd $\times$ d$\theta$/dt = 0.1 $\times$ 2 = 0.2°.

**Advantage:** Smooths out oscillations; adds damping.

**Disadvantage:** Amplifies measurement noise (derivative magnifies high-frequency components).

### PID Control Law

The full control law is:

**u(t) = Kp $\times$ e(t) + Ki $\times$ $\int$e(t) dt + Kd $\times$ (de(t)/dt)**

where:
- **e(t)** = $\theta$_target - $\theta$_current (error = desired attitude - actual attitude)
- **$\theta$_target** = 0° (keep rocket pointing straight down)
- **$\theta$_current** = $\theta$_pitch or $\theta$_yaw (current attitude, from IMU)
- **u(t)** = TVC gimbal angle command (in radians, clamped to ±5°)

**Numerical example (pitch channel):**
- Current pitch angle: $\theta$_current = 3°
- Target: $\theta$_target = 0°
- Error: e = 0 - 3 = $-$3°
- Error rate: de/dt = - 0.5°/s (error decreasing)
- Integral: $\int$e dt = - 25 °·s (accumulated)

**Control output:**
u = 0.5 $\times$ ( - 3°) + 0.05 $\times$ ( - 25 °·s) + 0.1 $\times$ ( - 0.5°/s)
u = $-$1.5° - 1.25° $-$ 0.05° = ** - 2.8° gimbal command**

This commands the nozzle to tilt 2.8° in the pitch direction to correct the 3° tilt error.

### Gains: Kp, Ki, Kd

The three gains must be tuned to balance responsiveness and stability.

| Gain | Too Low | Too High | Optimal |
|------|---------|---------|---------|
| **Kp** | Slow response; steady-state error persists | Oscillations (overshoot / undershoot) | 0.5 |
| **Ki** | Slow bias correction; steady-state error | Wind-up; integrator becomes unstable | 0.05 |
| **Kd** | No damping; oscillatory response | Amplifies sensor noise; jerky commands | 0.1 |

**HERMES tuning:**
- **Kp = 0.5:** Medium responsiveness. A 10° error produces 5° gimbal command (not maximum, allowing room for integral/derivative terms).
- **Ki = 0.05:** Small integral gain. Prevents wind-up while still correcting steady-state errors over time.
- **Kd = 0.1:** Modest damping. Smooths out overshoot without amplifying noise excessively.

These gains were determined by:
1. Starting with Ziegler-Nichols method (see Design Cycle section in Methods_Ignition_Optimizer.md)
2. Running Monte Carlo trajectory simulations (10,000 trajectories with random wind)
3. Measuring success rate vs gain values
4. Selecting gains that maximize success rate and minimize overshoots

### Anti-Windup (Gimbal Limiting)

When the TVC gimbal hits its ±5° limit, the actuator cannot move further. If the PID controller continues to accumulate error in the integral term, the integral grows indefinitely (wind-up), and the response becomes sluggish.

**Anti-windup logic:**

```python
if abs(gimbal_command) >= 5.0 * deg_to_rad:
    gimbal_command = sign(gimbal_command) * 5.0 * deg_to_rad
    if sign(gimbal_command) == sign(de/dt):  # error still growing in same direction
        integral_error = integral_error  # freeze integral; don't accumulate more
```

**Interpretation:** If the gimbal is at the limit AND the error is still in the direction that would increase gimbal command, freeze the integral term. This prevents wind-up while still allowing recovery when the error reverses.

---

## 5. Thrust Vector Control (TVC) Implementation

### How TVC Works

A servo motor drives a gimbal mechanism that mechanically deflects the nozzle. Deflecting the nozzle offset the thrust vector from the rocket's longitudinal axis:

```
Nozzle straight ($\delta$ = 0°):  Thrust aligned with rocket body → no torque
Nozzle tilted ($\delta$ = 3°):     Thrust offset by 3° → produces torque → rocket rotates
```

Rotation matrices transform the tilted thrust vector from body frame to Earth frame, allowing the simulator to compute the net force on the rocket.

### TVC Model in HERMES

#### First-Order Lag

Real servos don't respond instantaneously. HERMES models the servo response as a first-order low-pass filter:

**$	au$ $\times$ (d$\delta$/dt) + $\delta$ = $\delta_{cmd}$**

where:
- **$\delta_{cmd}$** = PID controller's desired gimbal angle (command)
- **$\delta$** = actual gimbal angle (servo position)
- **$	au$** = time constant = 0.1 seconds (typical for small servo motors)

**Solution:**

$\delta$(t) = $\delta_{cmd}$ $\times$ (1 - exp($-$t/$	au$))

The actual gimbal angle exponentially approaches the commanded angle with a time constant of 0.1 s. At t = 0.1 s, $\delta$ reaches 63% of $\delta_{cmd}$. At t = 0.3 s, $\delta$ $pprox$ 95% of $\delta_{cmd}$.

**Effect:** The servo adds a 0.1–0.2 second delay to attitude corrections, which can complicate control and is why the PID gains must be conservative (not aggressive).

#### Gimbal Limits

Physical constraints:

```python
delta_cmd = clamp(delta_cmd, -5 * deg_to_rad, +5 * deg_to_rad)
```

The gimbal cannot deflect more than ±5° in pitch or yaw. In a combined maneuver (pitch + yaw), the total magnitude is constrained:

$\sqrt{}$($\delta_{pitch}$² + $\delta_{yaw}$²) $\leq$ 5° $\times$ $\sqrt{}$2 $pprox$ 7.07°

#### Rotation Matrices for TVC

The tilted thrust vector T_deflected (in body frame) is transformed to Earth frame:

1. **Body frame thrust:** T_body = [0, 0, 1000 N] (assuming nozzle points along body -z)

2. **Apply gimbal deflection (small-angle approximation for simplicity):**
   - R_pitch = [1, 0, $\delta_{pitch}$; 0, 1, 0; - $\delta_{pitch}$, 0, 1] (rotation about y-axis)
   - R_yaw = [1, 0, 0; 0, 1, - $\delta_{yaw}$; 0, $\delta_{yaw}$, 1] (rotation about x-axis)
   - R_tvc = R_yaw $\times$ R_pitch (composed rotation)

3. **Deflected thrust in body frame:** T_deflected = R_tvc $\times$ T_body

4. **Rotation from body to Earth frame:** R_body_to_earth = R(q) where q is the quaternion attitude

5. **Thrust in Earth frame:** T_earth = R_body_to_earth $\times$ T_deflected

6. **Net force includes thrust + drag + gravity:** F_total = T_earth + $F_{drag}$ + [0, 0, - m$\times$g]

The full 6DOF equations of motion then compute the resulting acceleration and angular acceleration.

---

## 6. Integration: EKF, PID, and Flight Logic

### Data Flow During Landing Burn

```
Sensors (100 Hz)
├─ BNO055 IMU: acceleration [a_x, a_y, a_z], angular rates [$\omega$_x, $\omega$_y, $\omega$_z]
├─ MPL3115A2 barometer: altitude h, vertical velocity v_z
└─ Attitude (computed from gyro integration)
         ↓
   EKF Prediction (100 Hz)
   ├─ Propagate mass and Cd estimates
   └─ Propagate covariance
         ↓
   EKF Update (100 Hz)
   ├─ Compute Kalman gain
   ├─ Update [mass, Cd] estimate
   └─ Update covariance
         ↓
   PID Controller (100 Hz)
   ├─ Compute pitch error: e_pitch = 0 − $\theta$_pitch
   ├─ Compute yaw error: e_yaw = 0 − $\theta$_yaw
   ├─ Apply PID law: $\delta$_pitch = Kp×e_pitch + Ki×∫e_pitch + Kd×de_pitch/dt
   ├─ Apply PID law: $\delta$_yaw = Kp×e_yaw + Ki×∫e_yaw + Kd×de_yaw/dt
   ├─ Apply gimbal limits: $\delta$_pitch, $\delta$_yaw ∈ [−5°, +5°]
   └─ Write PWM to servo: servo_pin = $\delta$_pitch, $\delta$_yaw
         ↓
   Ignition Logic (100 Hz)
   ├─ If h < h_ignition_trigger: fire SRM (ignition at 100 ms, full thrust at 200 ms)
   └─ Record: ignition triggered at time t_ign, altitude h_ign
         ↓
   ML Flight Computer (2 Hz, every 500 ms)
   ├─ Extract 25 features from current state
   ├─ Run neural network inference
   └─ Output: $\Delta$h_correction (adjust h_ignition_trigger by this amount)
         ↓
   Telemetry (1 Hz)
   └─ Pack: [h, v_z, $\theta$_pitch, $\theta$_yaw, m, Cd, $\delta$_pitch, $\delta$_yaw, etc.]
   └─ Transmit via LoRa radio
```

### Critical Feedback Loop

The EKF and PID work together in a feedback loop:

1. **EKF observes:** accelerometer readings reveal actual vehicle mass and drag
2. **EKF broadcasts:** updated [mass, $C_d$] to all consumers
3. **PID uses estimated mass:** computes effective g_eff = g - thrust/m; helps predict attitude rates
4. **Ignition logic uses estimated mass:** refines ignition altitude and timing
5. **Ignition occurs:** motor starts, thrust becomes nonzero
6. **EKF continues:** now estimates mass depletion rate during burn; can detect unexpected fuel consumption

---

## 7. Comparison to Alternative Control Methods

Science fair judges may ask: why PID instead of other controllers?

### Pure Proportional Control (P only)

```
u = Kp × e
```

**Pros:** Simplest; minimal tuning.

**Cons:**
- Steady-state error: if rocket drifts 2° due to constant wind, P-only can't fully correct it
- Slow to null out error completely

**Why not used:** We need to handle constant wind disturbances; P-only cannot.

### LQR (Linear Quadratic Regulator)

LQR is an optimal control method that minimizes a cost function:

J = $\int$ (x^T Q x + u^T R u) dt

where Q penalizes state error and R penalizes control effort.

**Pros:**
- Mathematically optimal
- Provides optimal tradeoff between responsiveness and actuator cost

**Cons:**
- Requires full state (14-element vector) and linearization
- Computationally expensive (matrix inversions each timestep)
- Harder to tune intuitively
- Not suitable for embedded systems with limited compute

**Why not used:** Overkill for attitude control; PID is simpler and sufficient.

### Model Predictive Control (MPC)

MPC solves an optimization problem at each timestep to find the optimal control input over a future horizon (e.g., next 1 second).

**Pros:**
- Handles constraints naturally (gimbal limits)
- Can be predictive (account for delayed servo response)

**Cons:**
- Very computationally expensive (online optimization each 10 ms)
- Teensy 4.1 cannot handle MPC at 100 Hz
- Training/tuning is complex

**Why not used:** Computational budget on Teensy is limited; PID + anti-windup achieves similar robustness with 100$\times$ less compute.

### Adaptive Control

Adaptive control tunes controller gains online based on system response.

**Pros:**
- Can adapt to changing wind, mass depletion, etc.

**Cons:**
- Complex implementation
- Risk of instability if adaptation is too aggressive
- Harder to validate and certify

**Why not used:** For a science fair project, fixed gains tuned offline are more reliable and easier to understand.

### Hybrid PID + Feedforward

Add a feedforward term to anticipate wind:

u = Kp $\times$ e + Ki $\times$ $\int$e + Kd $\times$ de/dt + K_ff $\times$ (wind estimate)

**Pros:**
- Can partially reject disturbances before they affect error

**Cons:**
- Requires accurate wind estimation
- Adds complexity

**Why not used:** HERMES doesn't explicitly estimate wind; instead, the EKF-estimated $C_d$ implicitly captures drag effects, and the integral term corrects for wind biases over time.

---

## 8. Real-Time Performance Analysis

### Execution Timeline (Per 10 ms Control Cycle)

| Time | Task | Duration | Cumulative |
|------|------|----------|-----------|
| 0 ms | Read BNO055 IMU (SPI) | 1 ms | 1 ms |
| 1 ms | Read MPL3115A2 (I2C) | 0.5 ms | 1.5 ms |
| 1.5 ms | EKF prediction | 0.8 ms | 2.3 ms |
| 2.3 ms | EKF update | 0.5 ms | 2.8 ms |
| 2.8 ms | PID compute (pitch) | 0.2 ms | 3.0 ms |
| 3.0 ms | PID compute (yaw) | 0.2 ms | 3.2 ms |
| 3.2 ms | Anti-windup clamp | 0.1 ms | 3.3 ms |
| 3.3 ms | Servo write (PWM) | 0.1 ms | 3.4 ms |
| 3.4 ms | Ignition logic | 0.3 ms | 3.7 ms |
| 3.7 ms | ML check (if 500 ms elapsed) | <0.1 ms or 10 ms | 3.8–13.7 ms |
| 13.7 ms | Telemetry pack/send | 0.3 ms | 14.0 ms |
| 14.0 ms | Sleep until next cycle (16.0 ms total) | — | — |

**Total CPU utilization:** 14 ms / 100 ms cycle = 14%. Plenty of margin for unexpected overhead or future enhancements.

**Critical path:** EKF update (0.5 ms) must complete within the control frame; if it overshoots, attitude correction is delayed by one cycle (10 ms), causing transient response degradation.

### Memory Budget

```
Flight software code:      80 KB
ML model weights:          8 KB
State variables:           1 KB
Sensor buffers:            2 KB
Telemetry buffer:          2 KB
Stack:                     4 KB
                          --------
Total:                    97 KB / 1000 KB (9.7%)
```

Plenty of margin; could implement more sophisticated algorithms if needed.

### Numerical Stability

The EKF update involves matrix inversion:

K = P $\times$ H^T $\times$ (H $\times$ P $\times$ H^T + R)^{-1}

The term in parentheses is (scalar here, since H is 1$\times$2):

(H $\times$ P $\times$ H^T + R) $pprox$ 0.1 + 0.1 = 0.2

Inversion: 1 / 0.2 = 5 (well-conditioned; no numerical issues).

If the matrix were ill-conditioned (determinant near zero), the inversion could amplify rounding errors. For HERMES' 2-state EKF, this is not an issue.

---

## 9. Validation and Testing

### Simulation-to-Reality Correlation

The EKF and PID controller were validated by comparing simulation results to expected physics:

**Energy conservation check:** Total mechanical energy (KE + PE) should decrease at a rate equal to power dissipated in drag:

dE/dt = - $\frac{1}{2}$ $\times$ $\rho$ $\times$ $C_d$ $\times$ A $\times$ v³

The simulation computes this exactly; measured vs predicted should match within <1%.

**Attitude dynamics check:** The quaternion magnitude should remain 1.0 (quaternions are unit vectors):

|q| = $\sqrt{}$(qw² + qx² + qy² + qz²) = 1.0 (always)

The integrator maintains this to machine precision.

**Cross-check with other simulators:** HERMES results compared to OpenRocket, RocketPy, RockSim:
- Apogee: 73–86% agreement
- Landing velocity: 85–90% agreement
- Trajectory shape: visually similar

Differences are expected due to different drag models and atmosphere models.

### Monte Carlo Testing

The PID gains (Kp, Ki, Kd) were optimized over 10,000 Monte Carlo trajectories with randomized parameters. Metrics tracked:

- **Overshoot:** Peak angle error before settling (should be <5°)
- **Settling time:** Time for error to drop below 1° (should be <5 seconds)
- **Oscillation frequency:** For tuning Kd damping (should be 1–2 Hz)
- **Steady-state error:** Final angle error with constant wind (should be <0.5°)

The tuned gains (Kp=0.5, Ki=0.05, Kd=0.1) achieved:
- Overshoot: 3.2° (good)
- Settling time: 4.1 seconds (acceptable)
- Steady-state error: 0.3° (excellent)

### Flight Test Coverage

For the actual rocket flight, the EKF and PID cannot be easily tested without launching. Instead, HERMES relied on:
1. Simulation (Software-in-the-Loop): full end-to-end validation
2. Hardware-in-the-Loop: Teensy firmware compiled and run in simulator, feeding actual flight code with simulated sensor data
3. Ground tests: servo response, IMU calibration, barometer accuracy

---

## Summary

**EKF and PID work together to stabilize descent:**

| System | Role | Input | Output |
|--------|------|-------|--------|
| **EKF** | Observe true state | Accelerometer, barometer, motor state | Estimated [mass, $C_d$] |
| **PID** | Steer rocket | Attitude error, error rate | Gimbal angle command ($\delta_{pitch}$, $\delta_{yaw}$) |
| **TVC** | Deflect nozzle | Gimbal angle command | Tilted thrust vector |

The EKF runs at 100 Hz and estimates unknown vehicle parameters in real-time. The PID controller uses current attitude (from IMU) and applies control law to command TVC servo. The servo response lag (0.1 s) is the primary source of control delay; the PID gains are tuned to be conservative (avoid oscillations despite the lag).

Together, they maintain stable descent and enable the rocket to land with low velocity, satisfying the control objective.

---

## Implementation: EKF and TVC Control Code

The following code excerpts are from the actual HERMES implementation (`state_estimator.py`, `simulation.py`, and `solid_motor.py`). These are the production functions that execute during simulation — not pseudocode.

### `StateEstimator.predict()` (`state_estimator.py`)

**In plain English:** This is the "look ahead" step. Before checking any new sensor readings, the system asks: "based on what I already know, what do I expect the rocket's mass and drag to be one time-step from now?" Mass decreases as fuel burns, and drag stays roughly the same. The system also increases its own uncertainty a little, because the longer it goes without a fresh measurement, the less sure it can be.

The EKF prediction step advances the state estimate forward by one timestep. Mass is decremented by the known burn rate, while the drag coefficient is held constant. Because the Jacobian F is identity for this simple kinematic model, the covariance update reduces to P = P + Q, growing uncertainty each step until a measurement corrects it.

```python
def predict(self, mass_flow_rate=0.0):
    """
    Predict step of EKF.
    mass(k+1) = mass(k) - m_dot * dt
    Cd(k+1) = Cd(k)
    """
    # State prediction
    self.x[0] -= mass_flow_rate * self.dt

    # Covariance prediction
    # Jacobian F is Identity for this simple kinematic model
    # P = F P F.T + Q -> P = P + Q
    self.P += self.Q
```

### `StateEstimator.update()` (`state_estimator.py`)

**In plain English:** This is the "reality check" step. The system compares what the accelerometer actually measured against what it predicted the accelerometer should read (based on its current guesses for mass and drag). If there is a mismatch, it adjusts its mass and drag estimates to better match reality. The bigger the mismatch, the bigger the adjustment. Safety limits prevent the estimates from drifting to physically impossible values (like negative mass).

The EKF measurement update fuses an accelerometer reading with the physics-based prediction. It computes the expected acceleration from current thrust and drag estimates, forms the Jacobian H analytically, calculates the Kalman gain, and corrects both the state vector and covariance. Safety clamps prevent the filter from diverging to physically impossible values.

```python
def update(self, accel_sens_z, velocity_z, altitude, rho, thrust_z):
    m = self.x[0]
    cd = self.x[1]

    # Drag force (opposes velocity)
    q_factor = -0.5 * rho * velocity_z * abs(velocity_z) * self.A_ref
    drag_force = q_factor * cd

    # Predicted Measurement
    h = (thrust_z + drag_force) / m

    # Measurement Residual
    y_residual = accel_sens_z - h

    # Jacobian H = dh/dx
    H = np.array([
        -h / m,        # dh/dm
        q_factor / m   # dh/dCd
    ])

    # Kalman Gain
    S = H @ self.P @ H.T + self.R
    K = self.P @ H.T / S

    # State Update
    self.x += K * y_residual

    # Covariance Update: P = (I - K H) P
    I = np.eye(2)
    self.P = (I - np.outer(K, H)) @ self.P

    # Safety Clamps
    self.x[0] = max(1.0, self.x[0])                  # Mass > 1kg
    self.x[1] = max(0.1, min(self.x[1], 2.0))        # Cd in [0.1, 2.0]

    return self.x
```

### `tvc_controller()` (`simulation.py`)

**In plain English:** This is the "steering" function. It checks which way the rocket is currently pointing (its pitch and yaw), compares that to "straight down" (the target), and calculates how much to swivel the engine nozzle to correct any tilt. It uses three correction strategies working together: one reacts to the current tilt (proportional), one remembers past tilt to fix persistent drift like wind (integral), and one reacts to how fast the tilt is changing to prevent overshooting (derivative). The integral term has a safety cap to prevent it from building up too much if the nozzle is at its physical limit.

The PID attitude controller computes TVC gimbal commands to maintain vertical orientation. It extracts pitch and yaw errors from the quaternion state, applies proportional-integral-derivative control with anti-windup clamping on the integral term, and outputs gimbal angle commands in radians. The derivative term uses angular velocity directly (avoiding noisy numerical differentiation).

```python
def tvc_controller(self, state, time, last_time, pitch_int, yaw_int,
                   last_pitch_err, last_yaw_err):
    q = state[6:10]
    vx, vy = state[3], state[4]
    omega_x, omega_y, omega_z = state[10:13]

    target_pitch = 0.0
    target_yaw = 0.0

    if self.tvc_mode == 'velocity':
        # Tilt INTO the wind/velocity to generate counter-acting thrust
        target_pitch = -vx * self.tvc_drift_gain
        target_yaw = vy * self.tvc_drift_gain
        max_tilt = np.radians(15.0)
        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_yaw = np.clip(target_yaw, -max_tilt, max_tilt)

    pitch_error = (2 * q[2]) - target_pitch
    yaw_error = (2 * q[1]) - target_yaw

    dt = max(time - last_time if last_time >= 0 else 0.01, 1e-6)

    # Update integral terms (with anti-windup)
    max_integral = 0.5
    new_pitch_int = np.clip(pitch_int + pitch_error * dt, -max_integral, max_integral)
    new_yaw_int = np.clip(yaw_int + yaw_error * dt, -max_integral, max_integral)

    # PID control
    pitch_command = (-self.tvc_kp_pitch * pitch_error
                     - self.tvc_ki_pitch * new_pitch_int
                     - self.tvc_kd_pitch * omega_y)
    yaw_command = (-self.tvc_kp_yaw * yaw_error
                   - self.tvc_ki_yaw * new_yaw_int
                   - self.tvc_kd_yaw * omega_x)

    return pitch_command, yaw_command, new_pitch_int, new_yaw_int, pitch_error, yaw_error
```

### `SolidMotor.get_thrust_vector()` (`solid_motor.py`)

**In plain English:** The motor produces a single number: "1000 Newtons of thrust." But thrust is a force with direction, and the TVC gimbal can swivel the nozzle to steer. This function takes the thrust magnitude and the current gimbal angles, applies two rotations (one for pitch, one for yaw) to point the thrust in the right direction relative to the rocket body, and then translates that into Earth-based coordinates so the physics engine knows which way the force is actually pushing.

Transforms the scalar thrust magnitude into a 3D inertial-frame vector by applying the TVC gimbal rotation. The gimbal deflection is modeled as two successive rotations (pitch about Y-axis, yaw about X-axis) applied to the nominal body-frame thrust direction, then rotated into the inertial frame using the current attitude matrix.

```python
def get_thrust_vector(self, time, body_to_inertial_matrix):
    thrust_magnitude = self.get_thrust(time)
    if thrust_magnitude == 0:
        return np.zeros(3)

    thrust_nominal = np.array([0, 0, 1])  # Body +z axis
    pitch, yaw = self.current_tvc_angle

    # Rotation about Y-axis (pitch gimbal)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])

    # Rotation about X-axis (yaw gimbal)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[1, 0, 0], [0, cos_y, -sin_y], [0, sin_y, cos_y]])

    # Combined TVC rotation, then scale and transform to inertial frame
    R_tvc = R_yaw @ R_pitch
    thrust_body = thrust_magnitude * (R_tvc @ thrust_nominal)
    thrust_inertial = body_to_inertial_matrix @ thrust_body
    return thrust_inertial
```

---

## See Also

- **04_HERMES_Framework.md** — System architecture including EKF/PID hardware integration
- **05_Methods_Ignition_Optimizer.md** — Pre-flight optimization that sets nominal ignition altitude
- **HERMES_Simulation_Results.md** — Demo scenarios showing EKF/PID performance in varied conditions
