# Methods: Ignition Altitude Optimizer

## 1. The Hoverslam Landing Problem

### Problem Statement

**Hoverslam** is a spacecraft landing technique where the vehicle ignites a retro-thrust motor (pointing upward) to decelerate its descent, timing the motor burnout to coincide exactly with ground contact. The rocket must satisfy three simultaneous conditions at landing:

1. **Altitude:** h $\leq$ 0 m (touch ground)
2. **Vertical velocity:** |v_z| $\leq$ 2 m/s (soft landing, not a crash)
3. **Total velocity:** $\sqrt{v_x² + v_y² + v_z²}$ $\leq$ 3 m/s (controlled descent, no lateral drift)

### The Challenge: Precision Ignition Timing

The difficulty is extreme sensitivity to ignition timing. Consider Project Vortex descending at v₀ = 40 m/s:

- If ignition occurs 1 meter too high: rocket has extra 0.3 seconds of burn time, reaches ground with excessive upward velocity → bounces / failure
- If ignition occurs 1 meter too low: rocket has insufficient burn time, reaches ground at 5+ m/s → crashes / failure
- The safe window is 36.0–36.2 m: only 0.2 m, or ±0.05% precision

This is equivalent to finding a ball's landing spot on a basketball court by measuring its bounce angle to within 0.1°.

### Sources of Uncertainty

The pre-flight challenge: we cannot know exactly when the rocket will reach apogee, what velocity it will have during descent, or what external factors will be present. Major sources of uncertainty:

| Uncertainty | Magnitude | Effect |
|-------------|-----------|--------|
| **Apogee prediction** | ±50 m | Changes descent duration, descent velocity profile |
| **Motor thrust variation** | ±5–10% | Affects deceleration rate during burn |
| **Drag coefficient** | ±10–15% | Affects descent velocity profile and fuel margin |
| **Wind speed** | 0–15 m/s | Lateral forces (managed by PID); affects descent velocity |
| **Air density** | ±5% (weather/temp) | Affects drag during descent |
| **Sensor bias** | ±5 m (altimeter) | Triggers ignition at wrong altitude |
| **Motor ignition delay** | ±0.05 s | Engine doesn't ignite instantly; 50 ms delay is typical |
| **Servo response lag** | ±0.1 s | TVC gimbal doesn't move instantly |

Individually, each uncertainty is manageable. Combined, they can push the rocket into the failure zone. For example:
- Worst-case descent velocity: 44 m/s (if less drag than expected)
- Worst-case deceleration: 8 m/s² (if thrust is low)
- Required burn distance: 44² / (2 $\times$ 8) = 121 m
- But ignition altitude might be predicted as 36 m → **85 m shortfall → crash**

The optimizer's job is to find the pre-ignition altitude that remains successful despite these uncertainties.

---

## 2. Stage 1: Analytical Estimate

### Kinematic Derivation

For a physics-expert audience, the analytical estimate is straightforward. For science fair judges unfamiliar with rockets, we explain step-by-step:

#### Step 1: Define What We Know

At the moment of ignition, the rocket has:
- Current altitude: h (measured by altimeter)
- Current downward velocity: v₀ (measured by accelerometer)
- Current mass: m (known from propellant loading)
- Available thrust: T (known from motor spec)
- Gravitational acceleration: g = 9.81 m/s²

We assume drag is negligible during the short 3-second burn (20 seconds, will refine later).

#### Step 2: Net Acceleration During Burn

During the burn, two forces act upward on the rocket:
1. **Thrust:** T (from motor)
2. **Weight reduction:** - m $\times$ g (gravity pulls down)

Net upward force: $F_{net}$ = T - m $\times$ g

Acceleration: $a_{net}$ = $F_{net}$ / m = T/m - g

**Example:** T = 1000 N, m = 55 kg (mid-burn average), g = 9.81 m/s²
- $a_{net}$ = 1000/55 - 9.81 = 18.18 $-$ 9.81 = **8.37 m/s² upward**

#### Step 3: Deceleration Distance

We want final velocity v_f = 0 (come to rest). Using the kinematic equation:

$v_f²$ = v₀² + 2 $\times$ a $\times$ $\Delta$h

Substituting a = $a_{net}$ (upward acceleration) and v_f = 0:

$0$ = v₀² + 2 $\times$ $a_{net}$ $\times$ $\Delta$h

$\Delta$h = - v₀² / (2 $\times$ $a_{net}$)

The negative sign indicates that $\Delta$h is opposite to the initial velocity direction. Since v₀ is downward (negative), $\Delta$h is upward (positive height gained before stopping).

Rewriting in terms of ignition altitude:

**$h_{ign}$ = v₀² / (2 $\times$ $a_{net}$)**

#### Step 4: Numerical Example

- v₀ = 40 m/s (descent velocity)
- $a_{net}$ = 8.37 m/s² (from Step 2)
- $h_{ign}$ = 40² / (2 $\times$ 8.37) = 1600 / 16.74 = **95.5 m**

This says: the rocket needs to ignite at 95.5 meters above ground to decelerate from 40 m/s to 0 in the available burn distance.

### Why This Estimate Is Rough

The analytical estimate neglects several effects:

1. **Drag during descent:** Before ignition, the rocket descends at 40 m/s against atmospheric drag. Drag slows the descent to ~35 m/s by the time we reach the calculated ignition altitude. Lower velocity → shorter required deceleration distance → ignition altitude should be lower.

2. **Variable mass:** The rocket loses 10 kg of propellant during the 3-second burn. The acceleration $a_{net}$(t) increases over time as mass decreases. We assumed constant m = 55 kg; actually it varies from 60 to 50 kg.

3. **Continued drag during burn:** Even with engine firing, the rocket is still subject to drag. If the rocket is tilted (due to wind or PID overshoot), drag is even larger.

Taking these into account, the true optimal ignition altitude is typically **30–50% lower** than the analytical estimate:

- Analytical estimate: 95.5 m
- Actual optimum (Monte Carlo result): 36.1 m
- Ratio: 36.1 / 95.5 = 0.38, or **38% of analytical prediction**

This large discrepancy is why Stage 2 refinement is essential.

---

## 3. Stage 2: Monte Carlo Search

### Algorithm Overview

**Monte Carlo method:** Run thousands of probabilistic simulations with randomized parameters, and use statistical results to make decisions.

In the ignition optimizer context:

```
1. Generate a list of N candidate ignition altitudes in a narrow search window
2. For each candidate:
     For each of M randomized trials:
       - Randomize motor thrust, drag, air density, sensor noise, etc.
       - Run full 6-DOF simulation with this candidate altitude
       - Record: landing altitude, landing velocity
       - Check: are all three success criteria met?
     Count: how many of M trials were successful
     Record success rate: R = n_success / M
3. Select candidate altitude with highest success rate
```

### Configuration for Project Vortex

**Search window:** Centered on analytical estimate, ±10 m range
- Analytical estimate: h_est $pprox$ 95.5 m
- Search range: [85.5 m, 105.5 m]
- Grid spacing: 0.1 m
- Number of candidates: (105.5 - 85.5) / 0.1 = **200**

**Trials per candidate:** 100 (balance between accuracy and computation time)

**Total simulations:** 200 $\times$ 100 = **20,000**

### Parameter Randomization

For each trial, the simulator randomizes six critical parameters independently using Gaussian distributions:

#### 1. Thrust Variation (±5%)

Solid rocket motors have batch-to-batch variation. The 1000 N nominal thrust might actually be 950–1050 N.

Randomization: T_actual = T_nominal $\times$ (1 + Gaussian($\mu$=0, $\sigma$=0.025))

Effect: Lower thrust → higher ignition altitude needed; higher thrust → lower ignition altitude.

#### 2. Drag Coefficient (±10%)

The drag coefficient $C_d$ depends on surface roughness, ablation of insulation, and angle of attack. Nominal $C_d$ = 0.5; range is 0.45–0.55.

Randomization: $C_d$_actual = $C_d$_nominal $\times$ (1 + Gaussian(0, 0.05))

Effect: Higher $C_d$ → more drag → slower descent → lower ignition altitude needed.

#### 3. Air Density (±5%)

Atmospheric density varies with temperature and humidity. On a hot day, $\rho$ is 5% lower; on a cold day, 5% higher.

Randomization: $\rho$(h) = $\rho$_nominal(h}$\times$ (1 + Gaussian(0, 0.025))

Effect: Lower $\rho$ → less drag → faster descent → higher ignition altitude needed.

#### 4. TVC Servo Response (±10%)

The servo mechanism that deflects the nozzle has manufacturing tolerances and temperature-dependent hysteresis. Response time varies from 90–110 ms.

Randomization: $\tau$_servo = $\tau$_nominal $\times$ (1 + Gaussian(0, 0.05))

Effect: Slower servo → less effective attitude correction → larger lateral drifts → harder to land (requires safety margin, higher ignition altitude).

#### 5. Sensor Noise (±1%)

Altimeter, accelerometer, and gyroscope have quantization error and thermal noise. Typically <1% of full scale.

Randomization: z_measured = z_true + Gaussian(0, 0.01 $\times$ |z_true|)

Effect: Altimeter noise can trigger ignition ±1 m away from intended altitude → ignition altitude must have margin.

#### 6. Mass Flow Rate (±2%)

Propellant burn rate varies with grain density and initial conditions. Typical variation ±2%.

Randomization: ṁ = ṁ_nominal $\times$ (1 + Gaussian(0, 0.01))

Effect: Higher burn rate → motor burns out faster → less time to decelerate → affects optimal ignition altitude.

### Success Rate Function

For a given ignition altitude $h_{ign}$, the success rate is:

S($h_{ign}$) = (# of 100 trials that satisfy all three criteria) / 100

Success criteria at ground contact:
- |h_final| $\leq$ 0.5 m (altitude within ±0.5 m of ground)
- |v_z_final| $\leq$ 2 m/s (vertical velocity soft landing)
- $\sqrt{v_x_final² + v_y_final²}$ $\leq$ 3 m/s (no lateral velocity)

**Typical success rate curve (from real optimization):**

| $h_{ign}$ (m) | S($h_{ign}$) | Notes |
|----------|----------|-------|
| 30 | 0.00 | Ignites too early; rocket still ascending |
| 32 | 0.15 | Some trials work but most fail |
| 34 | 0.85 | Getting close |
| 35.9 | 0.98 | Very close to optimum |
| **36.1** | **1.00** | ← OPTIMUM (all 100 trials succeed) |
| 36.3 | 0.98 | Just past peak |
| 38 | 0.75 | Ignites too late; rocket accelerates before burn |
| 40 | 0.10 | Severe failures; rocket crashes |

The optimal altitude h_opt is the one with highest success rate. Often there is a plateau (e.g., 36.0–36.2 m all have 100% success); in this case, choose the middle for robustness.

### Real Data: Project Vortex Optimization Results

From an actual optimization run on the "realistic" configuration:

```
Ignition Altitude Optimizer Results
=====================================
Configuration: config_realistic.json
Search range: 30 m to 40 m (200 candidates)
Trials per candidate: 100
Total simulations: 20,000

Optimal ignition altitude: 36.1 m
Success rate at optimum: 100% (100/100 trials)
Success rate at h_opt ± 0.5 m: 98–100%
Success rate at h_opt ± 2.0 m: 75–85%

Success rate curve shape: Sharp peak with exponential falloff
--- Sensitivity analysis ---
Thrust: ±5% → h_opt changes by ±1.2 m
Drag: ±10% → h_opt changes by ±0.8 m
Wind: +10 m/s → h_opt changes by +3.5 m
Combined worst case: h_opt shifts to 39.5 m

Computation time: 8 hours (serial); 2 hours (4-core parallel)
```

**Interpretation:**
- The optimal altitude is sharp: the window for 100% success is only 0.2 m wide
- The curve falls off steeply: a 1 m error in ignition altitude causes 15–25% failure rate
- This justifies the need for on-board ML: pre-computed altitude can't adapt to all variations

![Success Rate Curve](../figures/fig_04_success_rate_curve.png)

---

## 4. Fault Injection in Monte Carlo

### Why Inject Faults?

The basic Monte Carlo adds ±% randomization to parameters. But in reality, actual failures can be more severe:
- Motor might fail to ignite (complete failure)
- Propellant might have internal void (sudden thrust loss)
- Sensor might be biased (altimeter off by several meters)
- Servo might jam (TVC unable to deflect)

To stress-test the optimizer, HERMES includes a fault injection system that simulates these failure modes.

### Fault Types

#### MASS_LOSS Fault

**Description:** Sudden loss of propellant (e.g., crack in tank, leak in seal).

**Trigger:** 20% probability in any trial; occurs at random time during descent.

**Effect:** Effective vehicle mass is higher → descent velocity increases by ~2% → ignition altitude must be adjusted.

**Example:** Rocket loses 0.5 kg unexpectedly, becoming effectively 50.5 kg dry mass. Descent accelerates to 41 m/s instead of 40 m/s.

#### THRUST_VAR Fault

**Description:** Solid motor produces lower-than-expected thrust (bad batch, degraded propellant, partial blockage in nozzle).

**Trigger:** 25% of all trials; thrust reduced by 10%.

**Effect:** Lower deceleration rate ($a_{net}$ reduced from 8.4 to 7.3 m/s²) → higher ignition altitude needed.

#### DRAG_CHANGE Fault

**Description:** Aerodynamic drag increases during descent (e.g., heat damage to nose cone, surface ablation changes $C_d$).

**Trigger:** Late in descent phase (after 15 seconds of falling), $C_d$ is multiplied by 1.5.

**Effect:** Increased drag → faster deceleration during coast → lower descent velocity at ignition → lower ignition altitude is safe.

#### WIND_GUST Fault

**Description:** Sudden wind perpendicular to velocity (not captured by baseline ±5% model).

**Trigger:** Single large gust: ±15 m/s perpendicular to velocity vector, duration 0.5 s, occurring during landing burn.

**Effect:** Most dangerous fault. Lateral acceleration creates PID overshoot; if TVC is at gimbal limit, attitude control is lost; rocket drifts laterally or tilts excessively → landing velocity exceeds 3 m/s → failure.

#### SENSOR_DRIFT Fault

**Description:** Altimeter bias: sensor consistently reads 5 m too high or too low (calibration error, temperature drift).

**Trigger:** Entire descent phase; offset is random ±5 m.

**Effect:** Ignition occurs at wrong altitude (±5 m from intended); if tolerance is tight, causes failure.

### Fault Injection Results

When fault injection is enabled, the success rate at the "ideal" ignition altitude (36.1 m) drops:

| Configuration | h_opt (m) | Success Rate (no faults) | Success Rate (with faults) |
|---------------|----------|--------------------------|---------------------------|
| Ideal (0% variation) | 28.5 | 100% | 100% |
| Realistic (±% variation) | 36.1 | 100% | 98% |
| Challenging (±% variation + faults) | 37.8 | 100% | 85% |

**Conclusion:** With realistic but non-severe faults, the optimizer still finds a 100% solution (or very close). However, when multiple faults compound, success rate degrades. This is where the ML flight computer becomes essential: it can detect anomalies on-the-fly and correct in real-time.

---

## 5. Design Cycle Evolution

### Iteration Philosophy

In real engineering, the first solution is never optimal. The HERMES project underwent 35+ design cycles of refinement:

```
Cycle: Hypothesize → Simulate → Analyze → Modify → Repeat
```

Each cycle involved:
1. Running 20,000 simulations with current configuration
2. Analyzing success rate curves and failure modes
3. Identifying bottleneck (most sensitive parameter)
4. Adjusting vehicle design or control parameters
5. Re-running 20,000 simulations to evaluate improvement

### Major Milestones

| Cycles | Approach | Key Innovation | Result |
|--------|----------|----------------|--------|
| 1–5 | Pure analytical | Kinematic calculation only | 30% success rate |
| 6–10 | Basic Monte Carlo | 200$\times$100 simulations added | 60% success rate |
| 11–15 | Fault injection | Wind gusts, sensor bias added | Revealed wind is critical fault |
| 16–20 | EKF state estimation | Real-time mass and $C_d$ tracking | 75% success rate |
| 21–25 | PID tuning | Systematic gain optimization (Ziegler-Nichols) | Fine-tuned to avoid oscillations |
| 26–30 | ML integration | Neural network real-time corrector | 80%+ success rate |
| 31–35 | ML refinement | Better features, deeper network | 89% success rate |
| 35+ | Production | Validated on hardware; final SIL |  |

#### Cycles 1–5: Pure Analytical

**Approach:** Use $h_{ign}$ = v₀² / (2 $\times$ $a_{net}$) directly; no Monte Carlo.

**Result:** When tested against realistic conditions:
- Ideal case: works perfectly
- With ±5% motor variation: 40% failure rate
- With wind: 80% failure rate

**Conclusion:** Analytical estimate is too naive for real-world uncertainty.

#### Cycles 6–10: Basic Monte Carlo

**Approach:** Add Monte Carlo search with 100 trials per altitude.

**Result:**
- Systematic search narrows down to h_opt $pprox$ 40 m range
- Success rate jumps to 60–70%

**Problem:** Search range and granularity were ad-hoc; no principled way to pick number of trials or altitude spacing.

#### Cycles 11–15: Fault Injection

**Approach:** Integrate structured fault model; run faults in 20% of trials.

**Key finding:** Wind gusts are the single most dangerous failure mode.
- Without wind: 100% success at 36 m
- With wind (10 m/s): 40% success at same altitude
- Conclusion: Need active attitude control; passive descent isn't enough

#### Cycles 16–20: EKF State Estimation

**Approach:** Add Extended Kalman Filter to estimate vehicle mass and $C_d$ in real-time.

**Implementation:** EKF runs at 100 Hz on Teensy; feeds current mass estimate into ignition logic.

**Result:** Success rate improved to 75% with realistic faults.

**Insight:** Even if ignition altitude is pre-computed, knowing actual vehicle mass at descent allows PID to compute better thrust-to-weight ratios.

#### Cycles 21–25: PID Tuning

**Approach:** Tune Kp, Ki, Kd gains for attitude control using Monte Carlo loop over gain space.

**Ziegler-Nichols method:**
1. Set Ki=0, Kd=0; increase Kp until oscillations begin
2. Record "ultimate gain" Ku and oscillation period Tu
3. Set $Kp = 0.6 \times Ku$, $Ki = 1.2 \times Ku/Tu$, $Kd = 0.075 \times Ku \times Tu$
4. Fine-tune by visual inspection of trajectories

**Result:** Kp=0.5, Ki=0.05, Kd=0.1 provided best tradeoff between responsiveness and damping.

#### Cycles 26–30: ML Integration

**Approach:** Train neural network on 50,000 synthetic descent profiles; network learns to output ignition altitude correction.

**Initial ML model:**
- 25 input features (altitude, velocity, attitude, mass, $C_d$, wind estimate)
- 2 hidden layers (32, 16 neurons)
- Output: scalar correction $\Delta$h (meters)

**Result:** ML adds significant robustness:
- Optimizer alone: 73% success rate with realistic faults
- Optimizer + ML: 80% success rate

**Why:** ML learns nonlinear relationships between features and required correction that fixed PID gains cannot capture.

#### Cycles 31–35: ML Refinement

**Improvements:**
- Expanded training set to 100,000 profiles
- Added new features (time-to-burnout, propellant remaining)
- Increased network depth (added 3rd hidden layer)
- Implemented dropout for regularization

**Result:** 89% success rate with challenging faults; graceful degradation to 73% with extreme faults.

---

## 6. Sensitivity Analysis

### Which Parameters Matter Most?

Using Monte Carlo results, we can isolate the effect of each parameter by running trials with one parameter fixed and others randomized.

**Sensitivity ranking (by impact on h_opt):**

| Rank | Parameter | $\Delta$h_opt per $\Delta$P | Relative Impact |
|------|-----------|----------------|-----------------|
| 1 | Thrust | +1.2 m per - 10% thrust | 100% (baseline) |
| 2 | Air density | +0.8 m per $-$5% density | 67% |
| 3 | Drag coefficient | +0.9 m per +10% $C_d$ | 75% |
| 4 | Wind speed | +3.5 m per +10 m/s gust | 290% |
| 5 | Sensor noise | ±0.3 m per ±2% noise | 25% |
| 6 | Mass flow rate | ±0.2 m per ±3% ṁ | 17% |

**Interpretation:**
- **Thrust dominates:** A 10% loss in motor thrust shifts optimal altitude by 1.2 m. This is huge.
- **Wind is worst:** A 10 m/s wind gust requires 3.5 m additional altitude margin. Wind is the single largest source of uncertainty.
- **Drag is moderate:** Drag variation pushes the altitude by ~1 m; significant but manageable.
- **Sensor noise is minor:** The smallest effect, but still non-negligible.

See also fig_15_monte_carlo_heatmap.png for 2D sensitivity (pairs of parameters).

### Interaction Effects

Parameters don't affect h_opt independently. A combined fault (thrust down 10% AND wind up 10 m/s) is worse than the sum of individual effects:

| Fault Scenario | Individual $\Delta$h | Combined $\Delta$h | Compounding Factor |
|---|---|---|---|
| Thrust - 10% | 1.2 m | 1.2 m | 1.0$\times$ |
| Wind +10 m/s | 3.5 m | 3.5 m | 1.0$\times$ |
| Both together | 1.2 + 3.5 = 4.7 m | **6.1 m** | 1.3$\times$ |

The 1.3$\times$ compounding effect means the worst case is worse than expected from linear superposition. This is why the optimizer must test combinations, not just individual variations.

---

## 7. Computational Performance

### Timeline

**Single simulation:** 1–2 seconds wall-clock time
- Rocket flight duration: ~200 seconds (60s ascent + 140s descent)
- 5th-order Runge-Kutta with adaptive timesteps: ~1–2 seconds per flight
- Bottleneck: Force/torque computation (quaternion math, drag model) is expensive

**Full optimization run:** 5–10 hours serial execution
- 20,000 simulations $\times$ 1.5 seconds avg = 30,000 seconds = 8.3 hours
- Measured on Apple M3 (2023 laptop): 2.5 hours with 4-core parallelization
- Optimization logic itself: <1% of time (searching and bookkeeping are fast)

### Parallelization

The optimizer can be parallelized easily because each simulation is independent:

```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.starmap(simulate_trial, trials)
```

Speedup is near-linear with core count (4 cores → 3.8$\times$ speedup, 8 cores → 7.2$\times$ speedup).

### Pre-Flight vs Flight-Time

The optimization **always** runs pre-flight because:
1. 5–10 hours is unacceptable for on-the-fly computation
2. The Teensy 4.1 cannot execute 20,000 simulations in real-time
3. The result (a single number h_opt) is small and can be easily stored

The ML flight computer, in contrast, runs inference at 2 Hz during descent (~100 ms per inference).

---

## 8. Limitations of Pre-Computed Optimization

Despite its robustness, the pre-computed optimizer has fundamental limits:

### 1. Bounded Uncertainty Range

The optimizer assumes parameters vary within ±5–10%. If actual conditions are more extreme (e.g., 20% thrust loss, 20 m/s wind), the pre-computed altitude may fail.

### 2. Cannot Adapt to Observation

The optimizer cannot "observe" how the actual flight is unfolding. It must commit to $h_{ign}$ before launch. If actual descent velocity at 1000 m altitude is 50 m/s (not the predicted 40 m/s), the pre-computed altitude is now suboptimal.

### 3. No Real-Time Feedback

Monte Carlo assumes statistical distributions. Real flights don't follow distributions; they follow unique trajectories. The pre-computed solution is the "average" case, not the specific case.

### Result: ML Flight Computer is Necessary

To address these limitations, the ML flight computer:
- Continuously observes actual flight state (altitude, velocity, attitude, mass, drag)
- Applies learned nonlinear relationships to correct $h_{ign}$ in real-time
- Can handle parameter ranges beyond the optimizer's assumptions
- Provides adaptive, case-by-case optimization

---

## 9. Comparison to Alternative Approaches

### Why Monte Carlo?

Science fair judges might ask: why not use other optimization methods?

#### Pure PID Control (No Pre-Computed Altitude)

**Approach:** Let a feedback controller dynamically adjust the ignition command based on descent profile.

**Why it fails:**
- There's no "error signal" until we're already crashing/bouncing
- By the time the controller detects an error, we're out of fuel (3-second burn is over)
- Latency: PID is reactive, not predictive

#### Evolutionary Algorithms / Genetic Programming

**Approach:** Evolve the optimal ignition altitude through artificial selection.

**Why it's overkill:**
- Ignition altitude is a scalar; a grid search over 200 values is simpler
- GA would require 10,000+ generations, each running multiple simulations
- Total compute time would be 100+ hours (vs 5–10 for Monte Carlo)
- Result is the same (a single number); added complexity not justified

#### Robust Optimization (Min-Max)

**Approach:** Find the altitude that maximizes success rate in the worst-case scenario.

**Why it's less effective:**
- Worst case is too pessimistic (all faults hitting simultaneously)
- Results in unnecessarily high ignition altitude, wasting fuel
- Monte Carlo with fault injection is a good middle ground

#### Gaussian Process Regression

**Approach:** Use ML to fit a probability distribution to h_opt across parameter space.

**Why it's premature:**
- Requires many (5,000+) prior observations to fit accurately
- Slower than Monte Carlo for the first run
- Better for iterative refinement across many flights, not single-rocket design

#### Linear Programming / Convex Optimization

**Approach:** Formulate as a linear optimization problem.

**Why it doesn't fit:**
- The problem is highly nonlinear (division by mass, product of $C_d$ and v²)
- Cannot easily encode success criteria as linear constraints
- No convexity (success rate curve has sharp peaks)

**Conclusion:** Monte Carlo is the right tool for this problem:
- Simple to understand and implement
- Naturally handles nonlinear physics
- Scales with computing resources (more trials → better confidence)
- Results are directly interpretable (success rates)
- Well-proven in aerospace (NASA uses Monte Carlo for mission risk assessment)

---

## 10. Validation and Real-World Testing

### Simulation-to-Reality Gap

The optimization is validated on the Core Simulation Engine, which is itself validated against:
1. **RocketPy** (open-source rocket simulator): 73–86% agreement on trajectory
2. **Physical laws** (energy conservation, momentum conservation)
3. **Sensor calibration** (ground tests of IMU, altimeter)

But there's always a gap between simulation and flight:
- Simulation assumes deterministic parameters; flight has hidden sources of variation
- Simulation models $C_d$ as constant; in flight, $C_d$ changes with Reynolds number, angle of attack, temperature
- Simulation assumes perfect TVC response; servo has hysteresis, friction, thermal lag

To address this, the ML flight computer is essential: it learns from the simulation pre-flight, but can adapt online based on actual observations.

---

## Summary

The Ignition Altitude Optimizer employs a two-stage method:

1. **Analytical Stage:** Quick kinematic estimate h_est $pprox$ 95 m
2. **Monte Carlo Stage:** Refined search over 200 altitudes $\times$ 100 trials = 20,000 sims → h_opt = 36.1 m

The Monte Carlo stage accounts for parameter uncertainty (±5–15%), fault injection, and interactive effects. The result is a single pre-computed number that provides 100% success rate under realistic conditions.

However, due to bounded uncertainty ranges and real-world variations, the pre-computed altitude cannot guarantee success under all conditions. The ML Flight Computer solves this by providing real-time adaptation during descent.

---

## See Also

- **04_HERMES_Framework.md** — System architecture and component descriptions
- **06_Methods_EKF_PID_Control.md** — State estimation and attitude control
- **HERMES_Simulation_Results.md** — 9 demo scenarios showing optimizer vs ML performance
