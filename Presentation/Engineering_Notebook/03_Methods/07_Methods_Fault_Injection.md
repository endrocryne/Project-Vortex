# 07. Methods: Fault Injection and Robustness Testing

## 1. Why Fault Injection?

A rocket simulator that only works perfectly is worthless in the real world. Every physical system has uncertainties: manufacturing tolerances, sensor drift, environmental variations, and structural failures are not exceptions — they are the norm. A robust landing system must not merely succeed under nominal conditions; it must succeed *despite* imperfect conditions.

Fault injection is the practice of deliberately introducing errors into a simulation to systematically test how the control system responds to known failure modes. Real rockets experience:

- **Manufacturing variation**: SRM thrust curves vary motor-to-motor by ±5%; structural mass varies by ±2–3%
- **Environmental changes**: air density shifts with temperature and altitude; wind gusts can exceed 15 m/s
- **Sensor degradation**: accelerometers drift, barometers saturate, gyroscopes accumulate bias
- **Structural failures**: unplanned mass loss from impact; aerodynamic damage increasing drag

The goal of fault injection is not to prevent these faults — that's impossible — but to find the worst-case scenarios, understand system limits, and ensure graceful degradation rather than catastrophic failure.

## 2. What Is Fault Injection?

Fault injection is a controlled testing methodology borrowed from software engineering (stress testing) and applied to physics simulation. The process:

1. **Run nominal simulation** — establish baseline performance
2. **Inject a fault** — deliberately perturb a system parameter (mass, thrust, drag, wind)
3. **Observe response** — how does the control system react? Does it recover?
4. **Measure outcome** — does the system still land safely, or does it fail?
5. **Repeat with combinations** — test single faults, pairs of faults, worst-case scenarios

Analogies:
- In software: fuzzing (random input testing) reveals crashes; fault injection is structured fuzzing
- In aviation: stall testing or failure-mode studies determine safe operating envelope
- In engineering: stress testing physical prototypes to destruction establishes safety margins

For HERMES, fault injection combined with Monte Carlo optimization means: *find the ignition altitude that is robust to faults, not just nominal conditions*.

## 3. HERMES Fault Taxonomy

### 3.1 Four Fault Types

| Type | What It Simulates | Implementation | Example |
|------|------------------|----------------|---------|
| **MASS_LOSS** | Unplanned structural mass change | Adds delta $\Delta$m to simulated total mass | Propellant leak, structural failure, part ejection |
| **THRUST_VAR** | Motor thrust higher or lower than nominal | Multiplies thrust by factor (1 ± magnitude) | Motor-to-motor batch variation, defective motor |
| **DRAG_CHANGE** | Different aerodynamic drag than expected | Multiplies $C_d$ by factor (1 ± magnitude) | Unexpected angle of attack, damaged surface |
| **WIND_GUST** | Sudden wind disturbance | Adds delta $\Delta v_{wind}$ to wind velocity vector | Atmospheric turbulence, local shear layer |

Each fault type independently modifies the dynamics via the FaultInjectionManager:
- `mass_delta`: scalar added to m in state vector
- `thrust_multiplier`: scalar multiplying $F_{thrust}$
- `drag_multiplier`: scalar multiplying $C_d$ in $F_{drag}$ = $\frac{1}{2}$$\rho$$C_d$Av²
- `wind_delta`: vector added to wind_velocity

### 3.2 Four Trigger Modes

| Mode | When Fault Activates | Use Case |
|------|---------------------|----------|
| **ABSOLUTE_TIME** | At T + X seconds from launch | Motor timing variation; payload separation delay |
| **TIME_SINCE_APOGEE** | X seconds after apogee is reached | Descent-phase sensor failure; parachute deployment issues |
| **ALTITUDE_THRESHOLD** | When altitude drops below X meters | Near-ground pressure sensor failure; low-altitude wind shear |
| **MANUAL** | When externally commanded during simulation | Controlled experiments; human-in-loop testing |

Example trigger scenarios:
- MASS_LOSS triggered at ABSOLUTE_TIME 5.5 s: simulate payload separation after ascent motor burnout
- WIND_GUST triggered at ALTITUDE_THRESHOLD 500 m: simulate surface layer wind disturbance during final descent
- DRAG_CHANGE triggered at TIME_SINCE_APOGEE 2.0 s: simulate aerodynamic stall during powered descent

## 4. Fault Configuration

Each fault is described by a `FaultConfig` dataclass with the following fields:

| Field | Type | Purpose |
|-------|------|---------|
| `fault_type` | FAULT_TYPE enum | Which type of fault (MASS_LOSS, THRUST_VAR, etc.) |
| `trigger_mode` | TRIGGER_MODE enum | How fault is activated (ABSOLUTE_TIME, etc.) |
| `trigger_value` | float | Trigger parameter (time in seconds, altitude in meters) |
| `magnitude` | float | Fault scale: for THRUST_VAR, multiplier (1.0 = nominal); for MASS_LOSS, delta in kg |
| `duration` | float | How long fault persists: 0 = permanent, >0 = seconds until fault clears |
| `target` | TARGET enum | SIMULATED_STATE or SENSOR_ONLY |
| `randomize_magnitude` | bool | If true, magnitude is randomized within magnitude_range |
| `magnitude_range` | tuple(float, float) | Min/max bounds if randomized (e.g., -10%, +10%) |
| `probability` | float | Probability [0, 1] that fault occurs in any given trial |
| `active` | bool | Whether this fault is enabled in the current fault configuration |

### 4.1 Two Target Types: Dynamics vs. Sensing

**SIMULATED_STATE**: Fault affects actual physics. The rocket *truly* changes — mass is lost, thrust drops, drag increases. This is the "real" fault affecting the true trajectory.

**SENSOR_ONLY**: Fault corrupts sensor *readings* without changing actual state. The rocket physics are unaffected; only the EKF state estimate is corrupted. This tests **estimator robustness** — how well the control system recovers from sensor noise and bias without being fooled into wrong control decisions.

Example:
- SIMULATED_STATE mass loss: rocket actually becomes lighter; EKF detects this via lower acceleration; control adapts
- SENSOR_ONLY accelerometer bias: EKF estimates higher acceleration than reality; control over-throttles; eventually true dynamics reveal the mistake and EKF corrects

### 4.2 Fault Groups and Concurrency

Multiple faults can be organized into a `FaultGroup`:

```
FaultGroup {
  faults: [fault_1, fault_2, fault_3]
  concurrent: true/false
}
```

- `concurrent=true`: all faults in group activate simultaneously (realistic worst-case scenario)
- `concurrent=false`: faults activate sequentially (fault 1 at T+2s, fault 2 at T+5s, etc.)

Example worst-case scenario (Demo 6): concurrent WIND_GUST (10.3 m/s) + DRAG_CHANGE (+20% $C_d$) + MASS_LOSS (+5% mass) all active during the descent burn.

## 5. Environmental Conditions — 12 Scenarios

Beyond fault injection, the environment itself is varied. These 12 environmental conditions test how atmosphere, wind, and location affect landing precision.

| # | Condition | Description | Effect on Landing |
|---|-----------|-------------|------------------|
| 1 | Standard atmosphere | $\rho$ = 1.225 exp( - h/8500), T = 15°C | Baseline performance |
| 2 | High temperature | T = +35°C, $\rho$ $pprox$ 1.15 kg/m³ | Lower density → less drag → softer landing harder to achieve |
| 3 | Low temperature | T = - 5°C, $\rho$ $pprox$ 1.32 kg/m³ | Higher density → more drag → easier to slow |
| 4 | High altitude site | Launch from 1500 m elevation | Baseline $\rho$ is lower throughout; less terminal velocity control |
| 5 | Light crosswind | Steady 5 m/s perpendicular to trajectory | Moderate lateral drift; TVC within limits |
| 6 | Moderate crosswind | Steady 10 m/s crosswind | Significant lateral offset; TVC near limits; acceptable if controlled |
| 7 | Strong crosswind | Steady 15+ m/s | TVC gimbal limits reached; difficult landing; error-prone |
| 8 | Wind shear | Wind speed changes with altitude | Unpredictable during descent; difficult to estimate in real-time |
| 9 | Headwind | Wind opposing descent direction | Increases effective descent speed; less drag to work with |
| 10 | Tailwind | Wind assisting descent direction | Decreases effective descent speed; easier to slow down |
| 11 | Gusty wind | Random bursts (Gaussian noise on wind vector) | Sudden perturbations; tests EKF filtering and control responsiveness |
| 12 | Combined atmospheric | High temperature + strong shear + high altitude | Worst-case realistic scenario; tests all subsystems |

## 6. Fault Injection Factors — 12 Factors

These 12 factors are the specific fault magnitudes and timings tested in the Monte Carlo study:

| # | Factor | Fault Type | Magnitude | Trigger | Notes |
|---|--------|-----------|-----------|---------|-------|
| 1 | Motor thrust nominal | THRUST_VAR | 1.0 | MANUAL | Baseline (no fault) |
| 2 | Motor thrust +5% | THRUST_VAR | +5% | ABSOLUTE_TIME 0.5s | Higher-than-nominal production lot |
| 3 | Motor thrust - 5% | THRUST_VAR | $-$5% | ABSOLUTE_TIME 0.5s | Lower-than-nominal production lot |
| 4 | Payload separation | MASS_LOSS | - 2.0 kg | ABSOLUTE_TIME 5.5s | Payload separation occurs at nominal time |
| 5 | Propellant leak | MASS_LOSS | $-$0.5 kg | TIME_SINCE_APOGEE 1.0s | Leak during descent phase |
| 6 | Drag nominal | DRAG_CHANGE | 1.0 | MANUAL | Baseline |
| 7 | Drag +10% | DRAG_CHANGE | +10% | ABSOLUTE_TIME 0.5s | Slight surface roughness or off-angle |
| 8 | Drag +20% | DRAG_CHANGE | +20% | ABSOLUTE_TIME 0.5s | Significant aerodynamic issue |
| 9 | Light crosswind | WIND_GUST | 5 m/s | MANUAL | Constant throughout flight |
| 10 | Moderate crosswind | WIND_GUST | 10.3 m/s | MANUAL | Typical day with wind |
| 11 | Strong wind + shear | WIND_GUST | 15 m/s + varying | MANUAL | Challenging launch day |
| 12 | Combined scenario | Multiple | Mixed | Mixed | Wind + Drag + Mass simultaneously |

## 7. Rocket Configuration Options — 21 Variations

Beyond faults and environment, the rocket itself has design parameters that are varied to test robustness:

| # | Parameter | Nominal | Variation | Impact |
|---|-----------|---------|-----------|--------|
| 1–3 | Dry mass | 50 kg | - 10%, $-$5%, nominal, +5%, +10% | Higher mass → worse TWR → harder landing |
| 4–5 | Propellant mass | 10 kg | ±5% | Less fuel → shorter burn → higher landing velocity |
| 6–7 | Center of gravity | @ 2.4 m | ±5 cm | Aft CG → more stable; forward CG → less stable |
| 8–9 | Coefficient of drag | $C_d$ = 0.5 | ±10% | Affects terminal velocity and descent rate |
| 10–11 | Max TVC gimbal angle | ±10° | ±2° variation | Tighter limits → less control authority |
| 12–14 | PID gains (P, I, D) | Tuned optimal | ±20% | Aggressive gains → overshoot; conservative → sluggish |
| 15–17 | Sensor noise | 1% accel, 0.1 m baro | Low/nominal/high | High noise → EKF diverges → worse control |
| 18–20 | Actuator response time | 20 ms | 10/20/30 ms | Slower actuation → lag → tracking error |
| 21 | Mass flow rate uncertainty | ±2% | ±2% throughout burn | Affects thrust-to-weight trajectory |

Total combinations: 21 variations $\times$ 12 environmental conditions $\times$ 12 fault factors = 3,024 distinct configurations tested across Monte Carlo optimization (200 altitude candidates $\times$ 100 trials per candidate = 20,000 simulations per run, covering these variations probabilistically).

## 8. The Nine Demo Scenarios

Nine scenarios progress from benign to severe, demonstrating HERMES capability across the difficulty spectrum. Each scenario is a specific combination of faults and environmental conditions, designed to represent realistic failure modes.

### 8.1 Comprehensive Scenario Table

| # | Scenario Name | Faults Applied | Control Method | Launch Env. | Result | Landing Vel. (m/s) | ML Improvement (m/s) |
|---|---|---|---|---|---|---|---|
| 1 | **Baseline** | None | Optimizer | Standard atm. | ✓ SUCCESS | 0.52 | — |
| 2 | **Wind Alone** | 10.3 m/s crosswind | Optimizer | Standard atm. | ✗ FAIL | 3.90 | — |
| 3 | **Wind Alone** | 10.3 m/s crosswind | ML Controller | Standard atm. | ✓ SUCCESS | 0.78 | - 3.12 |
| 4 | **Drag + Mass** | $C_d$ +20%, mass +5% | Optimizer | Standard atm. | ✗ FAIL | 8.60 | — |
| 5 | **Drag + Mass** | $C_d$ +20%, mass +5% | ML Controller | Standard atm. | ✓ SUCCESS | 0.84 | - 7.76 |
| 6 | **Severe Combined** | Wind +10.3 m/s, $C_d$ +20%, mass +5% | Optimizer | Standard atm. | ✗ CRASH | 12.80 | — |
| 7 | **Severe Combined** | Wind +10.3 m/s, $C_d$ +20%, mass +5% | ML Controller | Standard atm. | ✓ SUCCESS | 1.02 | - 11.78 |
| 8 | **Extreme** | All faults extreme* | Optimizer | Combined atm. | ✗ FAIL | 4.10 | — |
| 9 | **Extreme** | All faults extreme* | ML Controller | Combined atm. | ✓ DEGRADED | 4.90 | $-$0.80 (mitigation) |

*Extreme scenario parameters: 15 m/s wind with shear, $C_d$ +20%, mass +10%, thrust ±10%, high-altitude cold environment. Demos 8 and 9 use different fault intensities; the extreme ML case is designed to be slightly more feasible than extreme optimizer (still difficult).

### 8.2 Scenario Progression and Analysis

**Demo 1 (Baseline)**: No faults. Optimizer chooses ideal ignition altitude; rocket lands cleanly. Establishes baseline landing velocity of 0.52 m/s — well within the 2 m/s success threshold. Shows deterministic optimization works perfectly under nominal conditions.

**Demos 2–3 (Wind Alone)**: Single fault — 10.3 m/s crosswind (moderate to strong). Optimizer-controlled rocket cannot maintain trajectory; lateral TVC limits saturate; rocket lands with dangerous 3.9 m/s velocity (nearly 7$\times$ nominal). ML controller adapts in real-time to compensate; lands within spec (0.78 m/s). **Finding**: single-axis disturbances are within optimizer capability for well-tuned systems; wind is an exception due to TVC gimbal limits.

**Demos 4–5 (Drag + Mass)**: Two simultaneous faults. Increased drag increases required thrust to slow down; increased mass decreases available thrust per unit weight. Optimizer fails (8.6 m/s landing velocity). ML adapts by increasing thrust saturation during descent and reducing PID aggressiveness. **Finding**: combined faults exceed optimizer tuning range; ML's learned adaptation becomes critical.

**Demos 6–7 (Severe Combined)**: Worst-case realistic scenario. Simultaneous wind, drag, and mass faults represent a day with bad conditions + unexpected manufacturing tolerance stack-up. Optimizer fails catastrophically (12.8 m/s → crash). ML succeeds (1.02 m/s). **Finding**: ML eliminates crashes; the control system doesn't blindly follow a pre-computed trajectory but adapts in real-time.

**Demos 8–9 (Extreme)**: Out-of-distribution faults with worst-case atmosphere. Both systems struggle. Optimizer fails completely (4.1 m/s). ML degrades gracefully (4.9 m/s) — still above 2 m/s spec but better than crash. This represents the limit of current ML training data; the faults are more severe than any training scenario. **Finding**: even learned systems have limits; extreme conditions require even better training or hardware redundancy.

### 8.3 Cross-Reference

- See **Figure 5**: `fig_05_demo_scenario_comparison.png` — Bar chart of all 9 scenarios showing landing velocity per method
- See **Figure 8**: `fig_08_fault_injection_diagram.png` — Schematic showing fault types, triggers, and combinations

## 9. Statistical Robustness Testing — Integration with Monte Carlo

Fault injection does not exist in isolation; it is integrated into the Monte Carlo optimization loop:

### 9.1 Optimization with Faults

Standard Monte Carlo:
1. For each ignition altitude candidate (h = 1000 m to 1500 m, in 5 m increments = 200 candidates)
2. For each trial (1 to 100 trials per altitude)
3. Run simulation with:
   - Thrust ±5%, Drag ±10%, air density ±5%, TVC ±10%, sensors ±1%, mass flow ±2% (random per trial)
   - This is called "per-trial randomization" and represents realistic variability
4. Count successes (landing velocity < 2 m/s) and failures
5. Plot success rate curve; optimal altitude is where success rate = 100% (or highest percentage)

With fault injection enabled:
1. Same structure as above, but **in addition** to per-trial randomization, probabilistically inject one or more faults
2. For instance, at each trial, 50% probability of WIND_GUST; 30% probability of DRAG_CHANGE; etc.
3. If fault triggers, it activates at its configured trigger time/altitude
4. Simulation continues with fault active
5. Final success rate now reflects robustness to both random variability AND known fault modes

### 9.2 Success Rate Interpretation

If Monte Carlo optimization reports "100% success rate at 1250 m ignition altitude with per-trial randomization AND faults enabled", this means:

- Across 100 independent trials with different randomization and fault injection, 100 of them landed successfully
- The rocket lands within 2 m/s in nearly all realistic scenarios
- The control system is robust

If it reports "78% success rate", 22% of trials ended in crash or unsafe landing. This is not acceptable for a critical mission; the altitude must be adjusted.

### 9.3 Fault Probability Tuning

The `probability` field in FaultConfig allows the engineer to fine-tune how often faults are injected:
- probability = 0.0: fault never activates (useful for disabling a fault without removing it from config)
- probability = 0.5: fault activates in ~50% of trials
- probability = 1.0: fault always activates

Tuning probabilities allows realistic scenarios. For instance:
- WIND_GUST at a typical launch site: probability = 0.3 (wind is occasional, not constant)
- THRUST_VAR: probability = 1.0 (every motor has some tolerance; assume it always exists)
- MASS_LOSS payload separation: probability = 0.95 (should happen, but occasionally fails)

## 10. Key Findings from Fault Injection Study

### 10.1 Single vs. Multiple Faults

**Single faults** (wind alone, drag alone): Traditional optimized control systems handle these reasonably well. The optimizer tuned the system assuming a nominal system; single deviations are small enough that the control loop can correct them within the gimbal limits.

**Combined severe faults** (Demo 6: wind + drag + mass): Optimizer fails. The system was not designed for simultaneous multi-axis failures. Each fault pushes the control authority 20–30%; combined, they exceed available authority by 50%+. ML correction is essential.

**Extreme out-of-distribution faults** (Demo 8–9): Both systems struggle. Even ML, trained on fault scenarios similar to Demos 1–7, has not learned strategies for faults 50%+ worse. This suggests:
- Better training data is needed (more extreme scenarios in training set)
- Hardware improvements (larger TVC gimbal, higher thrust margin) would help
- Hybrid classical + ML approaches might be needed

### 10.2 Dominant Failure Modes

From systematic testing:
1. **Wind is the dominant failure mode** — TVC gimbal limits are hit first; wind pushes the rocket sideways faster than thrust vector can correct
2. **Drag+Mass combination is second** — once thrust margin is consumed, drag prevents deceleration; adding mass makes it worse
3. **Single temperature variation is negligible** — air density changes ±5%; this is easily corrected by EKF real-time estimation

### 10.3 Control System Limits

- **PID alone**: handles single faults, fails on combinations
- **EKF + PID**: improves performance by real-time state estimation; still limited by control authority (gimbal angle limits)
- **ML correction**: learns to sacrifice precision in other axes to preserve altitude; trades lateral accuracy for descent rate control; succeeds where classical fails

## 11. Implications for System Design

Fault injection testing directly informed design decisions:

1. **Wind Tolerance**: Expanded TVC gimbal from ±8° (initial design) to ±10° (final design) to increase control authority against wind

2. **Thrust Margin**: Increased propellant from 8 kg to 10 kg, increasing TWR from 1.6 to 2.04, to provide 27% additional margin against drag and mass loss faults

3. **ML Integration**: Realized classical PID cannot handle combined faults; integrated ML controller for adaptive correction (demos 6–7 show this works)

4. **Sensor Redundancy**: SENSOR_ONLY fault testing (corrupting barometer, accelerometer independently) revealed that EKF with redundancy is more robust than single-sensor; upgraded from single barometer to dual sensor fusion

5. **Fault Tolerance Specifications**: Documented acceptable failure modes (single faults OK, severe combinations require ML) to inform flight operations and launch weather constraints

## 12. Summary

Fault injection transformed HERMES from a "works in ideal conditions" simulator to a "handles real-world variations" system. By systematically testing 12 environmental conditions, 12 fault factors, 21 rocket configurations, and their combinations across 20,000+ Monte Carlo trials, the team identified system limits, validated the control architecture, and proved that ML correction is not merely an enhancement — it is essential for safe landing in realistic conditions.

## 13. Fault Intensity Quantification — Mathematical Framework

**Why do we need this?** Imagine you are a doctor comparing injuries. A paper cut and a broken arm are both injuries, but they are obviously not equally serious. You need a consistent "severity score" to compare them. Rockets face a similar problem: a 5% drop in thrust is bad, but is it worse than a 10 m/s wind gust? Without a common yardstick, there is no principled way to compare, rank, or visualize different faults. This section develops that yardstick -- a single number between 0 (no impact) and 1 (worst possible) for every fault, computed purely from its configuration (no simulation needed).

### 13.1 Overview

Sections 3–12 describe *what* faults are injected and *when* they fire. A complementary question arises: **how severe is a given fault?** Answering this quantitatively is essential for three reasons:

1. **Visualization** — Our dashboard color-codes active faults on trajectory plots (red for severe, green for mild). A severity score gives us an objective way to choose those colors instead of guessing.
2. **Prioritized analysis** — When testing thousands of fault configurations, engineers need a quick way to sort them from "most dangerous" to "least dangerous" and focus on what matters.
3. **Combined-fault reasoning** — When multiple faults happen at once (for example, wind AND extra drag AND mass loss), we need a fair way to combine their individual severities into a single overall danger rating.

We therefore define a normalised fault intensity $I \in [0,\,1]$ for every `FaultConfig` instance. The intensity is computed analytically — no simulation is required — from the fault's type, magnitude, trigger timing, duration, and probability. The derivation below corresponds exactly to `calculate_fault_intensity()` and its helper functions in `faults.py`.

### 13.2 The Master Formula

**In plain English:** The overall severity of a fault depends on three things multiplied together: (1) how likely it is to happen, (2) how bad it is in terms of magnitude, and (3) how much the timing and duration make it worse. If any one of these factors is zero, the overall severity is zero -- a fault that never happens (probability = 0) is not dangerous, no matter how large it would be.

$$I = P \cdot B \cdot C$$

where

| Symbol | Name | Domain | Meaning |
|--------|------|--------|---------|
| $P$ | Probability | $[0,\,1]$ | Probability that the fault occurs in a given trial (`FaultConfig.probability`) |
| $B$ | Base intensity | $[0,\,1]$ | Severity attributable to fault *magnitude* alone, independent of timing or duration |
| $C$ | Context modifier | $[C_0,\,1]$ | Amplification factor capturing *when* the fault fires and *how long* it persists |

Because each factor is bounded in $[0,\,1]$ (or $[C_0,\,1]$ with $C_0 > 0$), the product $I$ is guaranteed to lie in $[0,\,1]$ without any clamping or saturation step.

### 13.3 Base Intensity $B$ — Hill / Sigmoid Functions

**Why not just use a straight line?** You might think that if 10% thrust loss is bad, then 20% must be exactly twice as bad. But that is not how real systems work. Think of it like pain: a paper cut hurts, but a slightly deeper cut hurts a lot more -- the relationship is not proportional. And at the extreme end, once an injury is already very severe, making it slightly worse does not change your experience much (you are already in a lot of pain). The Hill function captures this S-shaped relationship: small faults barely register because the control system absorbs them, medium faults are where danger ramps up steeply, and very large faults plateau near maximum severity because the system is already overwhelmed.

This is the same functional form used in biology to describe how drugs affect the body (the Hill-Langmuir equation). It turns out that rocket faults and biological responses follow similar patterns: there is a threshold below which little happens, a steep transition zone, and a saturation region where the effect levels off.

#### 13.3.1 Functional Form

The base intensity for every fault type is computed via a **Hill function** (also known as the Hill–Langmuir equation):

$$B(x;\;K,\,n) \;=\; \frac{x^{n}}{x^{n} + K^{n}}$$

This is the same functional form used in enzyme kinetics (Michaelis-Menten at $n=1$) and cooperative binding models (Hill equation at general $n$). Its properties make it well suited for measuring fault severity:

- $B(0) = 0$ — a zero-magnitude fault produces zero intensity. (No fault means no danger.)
- $B(K) = 0.5$ — the parameter $K$ is the point where the effect reaches half its maximum. Think of it as the "tipping point" where the fault goes from "manageable" to "getting serious."
- $B(x) \to 1$ as $x \to \infty$ — the function levels off and never exceeds 1, even for an enormous fault. This makes physical sense: even the worst possible thrust failure cannot make the situation infinitely bad.
- For $n > 1$ the curve is **S-shaped** (sigmoidal): small faults produce barely any severity, then severity rises steeply near the tipping point, then levels off. This matches reality -- the control system can absorb small problems, but once you exceed its margins, things get bad fast.
- For $n < 1$ the curve rises steeply at first and then flattens: even a small deviation is immediately noticeable, but making it worse adds less and less danger. This shape is used for faults like excess thrust, where even a small surplus matters but a huge surplus is not proportionally worse.
- For $n = 1$ the curve reduces to the standard Michaelis-Menten hyperbola (a smooth, gradual rise).

#### 13.3.2 Per-Fault-Type Parameters

**In plain English:** Each type of fault uses different settings for the Hill function because they affect the rocket in fundamentally different ways. Losing thrust is much more dangerous than gaining thrust (because you can always burn shorter with extra thrust, but you cannot create thrust you do not have). Similarly, losing drag is more dangerous than gaining drag (less drag means the rocket falls faster with no way to slow down aerodynamically). The table below shows the specific "tipping point" ($K$) and "steepness" ($n$) chosen for each fault type, along with the reasoning.

Each fault type maps its physical magnitude to a normalised input $x$ and uses a tuned $(K,\,n)$ pair. The table below summarises the choices implemented in `faults.py`:

| Fault Type | Input $x$ | $K$ (half-saturation) | $n$ (cooperativity) | Cap | Physical Reasoning |
|---|---|---|---|---|---|
| **Mass Loss** | $\lvert\Delta m\rvert / m_{ref}$ | 0.15 | 1.4 | — | S-curve onset: small losses ($<5\%$) are absorbed by the control system's thrust margin; losses exceeding ~15% rapidly eat into that margin and become dangerous |
| **Thrust Reduction** ($T_{mult} < 1$) | $1 - T_{mult}$ | 0.20 | 2.0 | — | Steeply rising: a 10% thrust loss yields $B \approx 0.20$; a 20% loss jumps to $B \approx 0.50$; a hoverslam vehicle has almost no thrust to spare |
| **Thrust Increase** ($T_{mult} > 1$) | $T_{mult} - 1$ | 0.25 | 0.7 | 0.4 | Gently rising: excess thrust wastes fuel and stresses the airframe but is far less dangerous than thrust loss because the vehicle can simply burn shorter; capped at 0.4 to reflect this asymmetry |
| **Drag Decrease** ($D_{mult} < 1$) | $1 - D_{mult}$ | 0.25 | 1.8 | — | The vehicle cannot slow itself down through air resistance alone; in a powered-landing design this forces the motor to compensate, burning extra fuel and shrinking the recovery window |
| **Drag Increase** ($D_{mult} > 1$) | $D_{mult} - 1$ | 0.30 | 0.8 | 0.6 | Extra air resistance slows the vehicle, which is less dangerous -- the vehicle arrives slower, not faster; gently rising Hill curve reflects diminishing severity; capped at 0.6 |
| **Wind Gust** | $\lvert v_{wind}\rvert / 15$ | 0.40 | 0.75 | 1.0 | Gently rising: the first few m/s of crosswind impose the steepest difficulty increase because the TVC gimbal must start deflecting; additional wind adds proportionally less new difficulty once the gimbal is already near its limit. Tipping point at $0.4 \times 15 = 6$ m/s |

Notes on the reference values:

- $m_{ref} = 60$ kg (default `reference_mass` parameter) — total vehicle mass at descent initiation.
- The 15 m/s wind reference corresponds to the maximum gust tested in the fault injection study (Section 6, Factor 11).
- Thrust and drag multipliers are dimensionless; $x$ is simply the fractional deviation from nominal (1.0).

### 13.4 Timing Criticality $T_n$ — Exponential Urgency Integral

**In plain English:** When a fault happens matters enormously. Think of braking a car: if you notice a stop sign 500 meters ahead, you have plenty of time to slow down gently. But if you notice it 10 meters ahead, you must slam the brakes -- and you might not stop in time. The same principle applies to rocket faults. A problem that appears early in the descent gives the control system plenty of time to adapt. A problem that appears just seconds before landing is far more dangerous because there is almost no time to react.

This timing factor captures that "urgency curve." It starts near zero for early faults (plenty of recovery time) and climbs exponentially toward 1 as the fault occurs closer and closer to touchdown. The exponential shape means the urgency accelerates: the last 20% of the descent is far more critical than the first 20%.

Another way to think about it: imagine trying to catch a ball. If someone tells you "the ball is coming" when it is 100 feet away, you can easily position yourself. If they tell you when it is 5 feet away, good luck.

#### 13.4.1 Derivation

The control system's ability to fix a fault depends on how much time remains after the fault appears. The later the fault fires, the less corrective thrust can be applied before landing:

$$J_{remaining} = \int_{t_{trigger}}^{T_{descent}} F_{max}\,dt$$

As the trigger time approaches touchdown, this remaining correction capacity shrinks, and each additional moment of delay becomes more dangerous. This exponential growth in urgency, after normalizing to a 0-to-1 scale, gives us:

$$T_n(\tau) = \frac{e^{\alpha\tau} - 1}{e^{\alpha} - 1}, \qquad \tau = \frac{t_{trigger}}{T_{descent}}, \qquad \alpha = 3.0$$

#### 13.4.2 Properties

- $T_n(0) = 0$ — a fault at the very start of descent has minimal timing penalty (maximum recovery time available).
- $T_n(1) = 1$ — a fault at touchdown is maximally critical.
- The curve is convex for $\alpha > 0$: criticality grows slowly at first and accelerates toward the end of descent. With $\alpha = 3.0$ a fault at 80% descent ($\tau = 0.8$) produces $T_n \approx 0.55$, while a fault at 40% descent ($\tau = 0.4$) produces $T_n \approx 0.17$ — a ratio of approximately 3:1, calibrated against Monte Carlo landing-failure statistics.

#### 13.4.3 Trigger-Mode Mapping

The raw timing fraction $\tau$ is derived differently for each trigger mode:

| Trigger Mode | $\tau$ computation | Default |
|---|---|---|
| `ABSOLUTE_TIME` | $\min(1,\; t_{trigger} / T_{descent})$ | — |
| `TIME_SINCE_APOGEE` | $\min(1,\; t_{trigger} / T_{descent})$ | — |
| `ALTITUDE_THRESHOLD` | $1 - \min(1,\; h_{trigger} / h_{typical})$ | — |
| `MANUAL` | 0.75 (conservative mid-to-late assumption) | 0.75 |

For `MANUAL` triggers where no timing information is available, a default of $\tau = 0.75$ is used. This is deliberately conservative — it assumes the fault occurs in the latter half of descent, which biases the intensity estimate upward rather than underestimating risk.

### 13.5 Duration Severity $D_n$ — Exponential Saturation

**In plain English:** How long a fault lasts also matters. A brief glitch that fixes itself is far less dangerous than a permanent failure. Think of driving with a flat tire: if you get a slow leak, you lose a little air and can still drive to a gas station. But if the tire blows out completely and stays flat, you are in real trouble. This duration factor captures that idea. The first few seconds of a fault cause the most additional danger. After that, the trajectory is already messed up, and each additional second of fault adds less new harm -- the damage is already done.

#### 13.5.1 Derivation

A fault of constant magnitude $\Delta F$ persisting for duration $d$ inflicts a cumulative velocity error:

$$\Delta v = \frac{\Delta F \cdot d}{m}$$

However, the *additional* danger from each extra second of fault diminishes once the trajectory is already badly off course — the vehicle has already departed from its planned path, and further deviation adds comparatively less new risk. This motivates an exponential saturation model (a curve that rises quickly at first, then levels off) rather than a straight line:

$$D_n(\tau) = 1 - e^{-\lambda\tau}, \qquad \tau = \frac{dur}{T_{descent}}, \qquad \lambda = 3.0$$

#### 13.5.2 Properties

- $D_n(0) = 0$ — an instantaneous fault has zero duration severity. (In practice, instantaneous faults still register through $B$ and the context floor $C_0$.)
- $D_n(\tau) \to 1$ as $\tau \to \infty$ — a permanent fault achieves maximum duration severity.
- **Permanent faults** (`duration = 0` in the `FaultConfig` convention) are mapped to $D_n = 1$ exactly, by treating $\tau \to \infty$.
- The time constant is $1/\lambda \approx 0.33$ of the descent duration: a fault lasting one-third of the descent already achieves $D_n = 1 - e^{-1} \approx 0.632$, capturing the intuition that a fault persisting for a substantial fraction of the flight is "almost as bad as permanent."

### 13.6 Context Modifier $C$ — Bilinear Interaction

**In plain English:** The context modifier combines the "when" (timing) and "how long" (duration) into a single adjustment factor. The key insight is that a fault that is both late AND long-lasting is much worse than you would expect from adding those two effects together. Imagine a basketball player who trips: tripping early in the game (lots of time to recover) is not too bad. Tripping for just a split second (even late) is recoverable. But tripping in the final seconds of a tied game AND staying down? That is catastrophically worse than the sum of "late" plus "long." The math here captures that same idea -- certain combinations of timing and duration are more dangerous than either factor alone.

#### 13.6.1 Formula

$$C = C_0 + (1 - C_0)\,\bigl(w_t \cdot T_n + w_d \cdot D_n + w_c \cdot T_n \cdot D_n\bigr)$$

with constants:

| Constant | Value | Role |
|----------|-------|------|
| $C_0$ | 0.25 | Context floor: even the best-timed, briefest fault registers at 25% of maximum context weight |
| $w_t$ | 0.35 | Weight of timing criticality (linear term) |
| $w_d$ | 0.35 | Weight of duration severity (linear term) |
| $w_c$ | 0.30 | Weight of bilinear interaction term $T_n \cdot D_n$ |

The weights satisfy $w_t + w_d + w_c = 1.0$; the interaction term "borrows" equally from each linear contribution.

#### 13.6.2 The Bilinear Interaction Term

The cross-product $w_c \cdot T_n \cdot D_n$ is the key modeling choice. Without it, the context modifier would simply add the timing and duration effects -- implying that a fault which is late *or* persistent is just as dangerous as one that is late *and* persistent. In reality, the combination is worse than the sum of its parts:

- A **late, brief** fault fires close to touchdown but clears quickly — the vehicle may still recover in the remaining seconds.
- An **early, persistent** fault fires with plenty of recovery time — the control system has the entire descent to adapt.
- A **late, persistent** fault fires close to touchdown *and* never clears — there is neither time nor opportunity for recovery. This scenario is multiplicatively more dangerous than the sum of its parts.

The $T_n \cdot D_n$ term captures precisely this "worse than the sum of its parts" interaction.

#### 13.6.3 Boundary Behaviour

- **Minimum**: $T_n = D_n = 0 \;\Longrightarrow\; C = C_0 = 0.25$. Even a fault with no timing or duration information contributes 25% of the maximum context amplification, ensuring that $I$ is never trivially zero when $B > 0$ and $P > 0$.
- **Maximum**: $T_n = D_n = 1 \;\Longrightarrow\; C = 0.25 + 0.75 \cdot (0.35 + 0.35 + 0.30) = 0.25 + 0.75 \cdot 1.0 = 1.0$. The context modifier reaches unity for a permanent fault firing at touchdown — as expected.

### 13.7 Combined Fault Intensity — Probabilistic-OR Combination

**In plain English:** What happens when multiple things go wrong at the same time? We cannot just add their severities (that could produce a number greater than 1, which makes no sense on our 0-to-1 scale). Instead, we use a formula borrowed from probability: "what is the chance that at least one of these faults causes a problem?" If one fault has severity 0.3 (30%) and another has severity 0.3, the combined severity is not 0.6 but rather 0.51 -- because there is some overlap (both faults might cause the same kind of damage). This keeps the combined score realistic and always between 0 and 1.

When multiple faults are active simultaneously, their individual intensities $I_1, I_2, \ldots, I_k$ are combined into a single aggregate intensity using the **inclusion-exclusion** (probabilistic-OR) formula:

$$I_{combined} = 1 - \prod_{i=1}^{k}(1 - I_i)$$

This is equivalent to treating each $I_i$ as an independent probability of mission degradation and computing the probability that *at least one* fault degrades the mission. The formula is implemented in `calculate_combined_fault_intensity()`.

**Properties:**

- **Always between 0 and 1**: since each $I_i \in [0,\,1]$, the combined result is also in $[0,\,1]$.
- **Order does not matter**: the result is the same regardless of the order faults are listed.
- **Adding a fault always increases severity**: adding a fault with $I_i > 0$ always makes $I_{combined}$ larger.
- **No double-counting**: $I_{combined} \leq \sum I_i$ for all non-trivial cases, capturing the physical overlap of fault effects -- two 30% faults do not produce a 60% combined intensity but rather $1 - (0.7)^2 = 0.51$.
- **Zero faults contribute nothing**: a fault with $I_i = 0$ has no effect on the combined intensity.
- **Maximum severity dominates**: a fault with $I_i = 1$ drives $I_{combined} = 1$ regardless of other faults.

### 13.8 Boundary Properties of the Complete Model

**In plain English:** Before trusting a formula, we need to make sure it behaves sensibly in extreme cases. Here we verify five common-sense checks: (1) the severity score is always between 0 and 1, (2) a fault with zero magnitude has zero impact, (3) a fault that never happens (probability = 0) contributes nothing, (4) the worst possible score approaches but never quite reaches 1.0 (because even a catastrophic fault does not guarantee failure with 100% certainty), and (5) making any aspect of a fault worse (bigger, later, longer, more likely) never makes the score go down. All five checks pass by mathematical construction -- no special-case "if" statements needed in the code.

The following properties hold **by construction** — they follow automatically from the math and do not require any special-case handling in the code:

1. **$I \in [0,\,1]$** — guaranteed because $P \in [0,\,1]$, $B \in [0,\,1]$, and $C \in [0.25,\,1]$, so their product $I = P \cdot B \cdot C$ can never exceed 1.

2. **Zero magnitude means zero impact** — if the fault magnitude is zero, $B(0;\,K,\,n) = 0$, so $I = P \cdot 0 \cdot C = 0$. No fault, no danger.

3. **Zero probability means zero contribution** — if $P = 0$, then $I = 0$, and the fault contributes nothing to the combined intensity. A fault that never happens cannot cause harm.

4. **Maximum severity is approached but never quite reached** — the highest possible $I$ is 1, but it requires infinitely large magnitude, probability of exactly 1, and the worst possible timing and duration. In practice, even a catastrophic fault does not guarantee failure with 100% certainty because the control system always has some chance of partial recovery.

5. **Worse inputs always produce a worse score** — $I$ never decreases when you increase the probability, magnitude, lateness, or duration of a fault. This matches common sense: a bigger, later, longer, more likely fault is never less dangerous.

---

---

## Implementation: Fault System Code

The following code excerpts are from the actual HERMES implementation (`faults.py`). These are the production functions that execute during simulation — not pseudocode.

### `calculate_fault_intensity()` (`faults.py`)

Computes a normalized intensity I in [0, 1] for a single fault, representing its expected impact on landing safety. The formula is I = P * B * C, where B is the base intensity from a Hill function on fault magnitude, and C is a context modifier combining timing criticality and duration severity with a bilinear interaction term. This function drives the fault severity visualization in the dashboard.

```python
def calculate_fault_intensity(fault, typical_descent_time=10.0,
                              typical_altitude=1000.0, reference_mass=60.0):
    C_FLOOR, W_T, W_D, W_C = 0.25, 0.35, 0.35, 0.30
    ALPHA, LAMBDA = 3.0, 3.0

    # STEP 1 -- Base intensity B using Hill functions
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

    # STEP 2 -- Timing criticality T_n
    timing_frac = 0.6  # default
    if fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
        timing_frac = 1.0 - min(1.0, fault.trigger_value / typical_altitude)
    # ... (other trigger modes compute timing_frac similarly)
    T_n = _exp_timing(timing_frac, ALPHA)

    # STEP 3 -- Duration severity D_n
    D_n = 1.0 if fault.duration == 0.0 else _exp_saturation(fault.duration / typical_descent_time, LAMBDA)

    # STEP 4 -- Context modifier C with bilinear interaction
    context = C_FLOOR + (1.0 - C_FLOOR) * (W_T * T_n + W_D * D_n + W_C * T_n * D_n)

    # STEP 5 -- Final intensity I = P * B * C
    return fault.probability * base_intensity * context
```

### `_hill()`, `_exp_timing()`, `_exp_saturation()` (`faults.py`)

These three helper functions implement the mathematical building blocks of the intensity model. The Hill function provides sigmoidal base intensity with tunable half-saturation and cooperativity. The exponential timing function models the shrinking recovery margin as the vehicle approaches touchdown. The exponential saturation models diminishing marginal severity for long-duration faults.

```python
def _hill(x, K, n):
    """Hill function: f(x) = x^n / (x^n + K^n).  f(0)=0, f(K)=0.5, f(inf)->1"""
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

### `FaultInjectionManager.check_triggers()` (`faults.py`)

Called each simulation timestep, this method iterates over all configured faults and evaluates whether each fault's trigger condition has been met (absolute time, time since apogee, altitude threshold, or manual). Faults that trigger are activated with probability sampling and magnitude randomization. This is the runtime entry point for the fault injection system.

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
        elif fault.trigger_mode == TriggerMode.MANUAL:
            triggered = False

        if triggered:
            if np.random.random() <= fault.probability:
                self._activate_fault(fault, current_time)
                newly_triggered.append(fault)

    return newly_triggered
```

### `FaultInjectionManager.apply_faults_to_state()` (`faults.py`)

Applies all currently active faults to the simulation state vector. For mass loss faults targeting the simulated state, it directly modifies the mass element of the 14-element state vector (with a floor of 1 kg to prevent division-by-zero in the physics engine). Other fault types (thrust, drag, wind) are applied via separate multiplier/delta accessors queried by the simulation loop.

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

**See also:**
- [08_Results_Apogee_Two_Stage.md](../05_Results/08_Results_Apogee_Two_Stage.md) — Results from two-stage testing
- [09_Results_Accuracy_Comparison.md](../05_Results/09_Results_Accuracy_Comparison.md) — Accuracy validation
- Figure 5: `../figures/fig_05_demo_scenario_comparison.png`
- Figure 8: `../figures/fig_08_fault_injection_diagram.png`
