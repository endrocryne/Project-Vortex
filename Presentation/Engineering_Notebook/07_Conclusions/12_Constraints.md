# Constraints: Regulatory, Budget, Technical, and Infrastructure

## 1. Constraint Overview

Project HERMES operates within a complex web of constraints spanning four domains: regulatory (federal and amateur rocketry), financial (science fair budget), technical (physics and control limitations), and logistical (launch site requirements). Understanding these constraints is essential to appreciating why certain design choices were made.

**Table 1: Constraint Summary**

| Domain | Constraint | Impact | Mitigation Strategy |
|--------|-----------|--------|---------------------|
| **Regulatory** | NAR L2 certification required | Cannot conduct physical test without certification | Simulation-first; build for future certified launch |
| **Regulatory** | FAA waiver for >400 ft altitude | Launch coordination needed | Within established model rocket waiver corridors at NAR sites |
| **Budget** | ~$3,700 total project cost | Limits motor class, airframe material, sensor quality | SRM-based (cheaper than liquid), proven components |
| **Technical** | SRM cannot throttle | No abort; must predict ignition perfectly | Optimizer + ML hybrid provides robust prediction |
| **Technical** | TVC gimbal limited to ±5° | Cannot recover from high tilt angles | Design for low attitudes during descent; PID stability critical |
| **Infrastructure** | Launch site <1/year availability | Slow iteration cycle | Simulation-based validation (35+ iterations pre-test) |
| **Environmental** | Wind sensitivity (>10 m/s degrades performance) | Operational launch window limited | ML designed for 0–15 m/s wind robustness |

---

## 2. Regulatory Constraints

### 2.1 NAR and TRA Certifications

**National Association of Rocketry (NAR)** and **Tripoli Rocketry Association (TRA)** are volunteer organizations that oversee model and high-power rocketry in the United States. They set safety codes and maintain a registry of certified launch sites.

#### Motor Classification System

Solid rocket motors are classified by **total impulse** (thrust integrated over burn time). Each letter represents a doubling of impulse:

| Class | Impulse Range (N·s) | Example | Application |
|-------|-------------------|---------|-------------|
| A | 1.25–2.5 | 2 s at 1 N | Toy model rockets (no cert required) |
| B | 2.5–5 | 2 s at 2.5 N | Model rockets (NAR section) |
| C | 5–10 | 3 s at 3 N | Model rockets |
| D | 10–20 | 3 s at 5 N | Model rockets |
| E | 20–40 | 3 s at 12 N | High-power (HP); **NAR L1 required** |
| F | 40–80 | 3 s at 25 N | High-power (HP) |
| G | 80–160 | 3 s at 50 N | High-power (HP) |
| H | 160–320 | 3 s at 100 N | High-power (HP) |
| **I** | **320–640** | **3 s at 200 N** | **High-power (HP)** |
| **J** | **640–1,280** | **3 s at 400 N** | **High-power (HP); L2 required** |
| **K** | **1,280–2,560** | **3 s at 800 N** | **High-power (HP); L2 required** |
| L | 2,560–5,120 | — | Experimental |
| M+ | >5,120 | — | Experimental |

**HERMES uses K-class motors** (ascent + descent SRMs total ~3,000 N·s). Classification: **high-power**.

#### Certification Requirements

**NAR L1 Certification**:
- Prerequisite: None
- Requirements:
  - Attend a certified NAR launch
  - Successfully fly a single-engine rocket with H-class or I-class motor
  - Recovery must be safe (parachute deployed, no damage)
- Validity: Lifetime (honorary)
- Cost: ~$50 membership + motor cost (~$50–100)

**NAR L2 Certification**:
- Prerequisite: L1 certification
- Requirements:
  - Pass written exam (50 questions covering rocket physics, safety, regulations)
  - Successfully fly a single-engine rocket with J-class, K-class, or L-class motor
  - Recovery must be safe
  - Rocket must be your own design (or heavily modified kit)
- Validity: Lifetime (honorary)
- Cost: ~$50 exam + motor cost (~$150–300)

**Timeline for HERMES**:
- Year 1 (2026): Obtain L1 (attend NAR launch, fly H-class, pass flight)
- Year 2 (2027): Obtain L2 (study, pass exam, fly K-class)
- Year 3+ (2028+): Eligible for high-power launch with K-class motors

### 2.2 Federal Regulations: FAA

**Title 14 CFR Part 101** covers amateur rockets.

#### Altitude Limits

- **<400 ft AGL**: No FAA coordination needed; launches at NAR-approved sites allowed
- **400–50,000 ft AGL**: Requires FAA waiver and NOTAM (Notice to Airmen)
- **>50,000 ft**: Requires FAA waiver + coordination with Air Force

**HERMES target apogee**: 2,500 m (8,200 ft) → requires FAA waiver

#### Safety Corridors

NAR certified launch sites maintain established safety corridors (typically 5–10 km radius) where waivers are **pre-approved** by the FAA. Launches within these corridors proceed without per-flight coordination.

**HERMES approach**: Fly at an NAR site with established waiver. No additional FAA paperwork required.

#### Tracking and Recovery

- Rockets >1 kg must have recovery system (parachute, streamer)
- Tumble recovery (no parachute) allowed only for low-altitude flights (<100 m AGL landing)
- HERMES uses tumble recovery on landing descent (rocket falls slowly during main coast; lands under control via hoverslam)

### 2.3 Launch Site Coordination

Flying a K-class rocket with an *experimental propulsive landing system* requires **special approval** from the launch site safety officer (SO).

**Typical launch site restrictions**:
- Propulsive burns are allowed (part of multi-stage flights)
- Exotic recovery (hoverslam) requires demonstration of control and safety
- SO may request simulation data, trajectory predictions, and contingency plans

**HERMES mitigation**:
- Provide SO with comprehensive simulation report (this engineering notebook)
- Demonstrate via Monte Carlo analysis that landing is possible and safe
- Pre-flight briefing with SO explaining control algorithms
- Telemetry downlink for real-time mission monitoring

### 2.4 Launch Site Requirements

Licensed NAR/TRA launch sites must provide:
- **Minimum 2 km radius clear zone** (no buildings, vehicles, or spectators outside designated area)
- **Level or gently sloping terrain** (for recovery)
- **Established waiver** (coordinate with FAA)
- **Range control officer** (trained to manage flight operations)

Most established launch sites in the US (Arizona, California, Texas, etc.) meet these requirements.

---

## 3. Budget Constraints

### 3.1 Total Project Budget

**Available**: ~$3,700 (combination of personal savings, science fair grants if available)

**Allocation**:

| Category | Amount | Notes |
|----------|--------|-------|
| Avionics (flight computer stack) | ~$100 | Teensy, IMU, altimeter, radio, servo, battery |
| Airframe structure | ~$1,500 | Phenolic/fiberglass tubes, centering rings, couplers, shock cord |
| Motors (ascent + landing) | ~$1,200 | K-class ascent motor (~$300) + backup (~$300) + K-class landing motor (~$300) + backup (~$300) |
| Recovery system | ~$400 | Main parachute (1.5 m), droguechute (0.5 m), harness, electronics |
| Miscellaneous | ~$500 | Test equipment (altimeter, scale), shipping, tools, adhesives, sundries |
| **TOTAL** | **~$3,700** | One-time hardware investment |

**Per-launch variable cost**: ~$465 (motors + consumables)

### 3.2 How Budget Constraints Shaped Design

#### SRM vs Liquid Propellant

**Solid Rocket Motors (SRM)**:
- Cost per unit: $200–300
- Advantages: Pre-manufactured, turn-key, no handling infrastructure
- Disadvantages: Cannot throttle, high peak pressure

**Liquid Propellant** (e.g., LOX/ethanol):
- Cost per launch: $200,000–300,000 (includes tank, pumps, launch equipment, facility rental)
- Advantages: Throttleable, reusable infrastructure, controllable thrust curve
- Disadvantages: Requires cryogenic handling, hazmat certification, facility access

**Choice**: SRM (required for $3,700 budget; liquid out of reach)

#### Motor Class Selection

K-class was chosen as the optimal balance:
- **Cost**: K-class motors ~$200–300 each; within reach
- **Performance**: 1,280–2,560 N·s sufficient for 2,500 m apogee
- **Availability**: Off-the-shelf from commercial vendors (CTI, Loki, AT)
- **Certification**: Requires L2 (achievable in 2-3 years)

Larger motors (L-class, M-class) would cost $400–600 each and require specialized facilities.

#### Airframe Material

**Phenolic (paper composite)**:
- Cost: $200–300 for complete tube set
- Pros: Low cost, easy to machine
- Cons: Prone to surface erosion under high vibration

**Fiberglass**:
- Cost: $500–800 for complete tube set
- Pros: Superior strength, lower drag, smoother surface
- Cons: Expensive; requires composite layup skills

**Choice**: Phenolic for initial build (budget); fiberglass for future upgrade (after proving concept)

#### Sensor Selection

**BNO055 IMU** ($8) vs **ICM-20689** ($20) vs **Custom IMU stack** ($50+):
- **BNO055**: Proven in hobby aerospace, affordable, adequate noise specs
- Trade-off: Simpler design vs slightly lower performance
- **Choice**: BNO055 (budget-optimal; sufficient for 0.1 m landing accuracy requirement)

#### Avionics Redundancy

**Single-string vs Triple-redundancy**:
- Single string: All sensors/flight computer integrated; single point of failure
- Redundant: 3 independent flight computers; expensive and heavy
- **Choice**: Single string (justified by budget constraint; simulation validates robustness)

---

## 4. Technical Constraints — Solid Rocket Motors

### 4.1 Unthrottleable Burn

**The fundamental constraint**: Once a solid rocket motor is ignited, its thrust follows the **thrust curve** defined by the motor design. The rocket cannot throttle, reduce thrust, or abort the burn.

**Physical reason**: The propellant (composite or black powder) burns at a fixed regression rate determined by:
- Propellant composition and density
- Grain geometry (core diameter, length)
- Chamber pressure and temperature
- These are fixed at manufacturing time

**Implication for landing**:
- Ignition must occur at the *precisely correct altitude*
- If ignition is too early: rocket overshoots target altitude by 50+ m
- If ignition is too late: insufficient burn time; rocket hits ground at high velocity
- No "mid-burn throttle" to compensate

**Typical K-class thrust curve**:
```
Thrust
  ↑
  │     ╱╲
800│    ╱  ╲         Burn duration: 3.0 s
  │   ╱    ╲        Peak thrust: ~800 N
600│  ╱      ╲       Avg thrust: ~600 N
  │ ╱        ╲      Impulse: ~1,800 N·s
400│╱          ╲
  │            ╲
200│             ╲___
  │                  ╲___
  └────────────────────────→ Time (s)
  0    1    2    3    4
```

Time-averaged thrust during 3-second burn: 600 N.

Vehicle mass during burn: 50 kg.

**Time-averaged acceleration**: a = 600/50 = 12 m/s² (upward).

This must decelerate the rocket from, say, 30 m/s downward to 1 m/s downward. The margin is slim.

### 4.2 Motor-to-Motor Variation

Solid rocket motor manufacturers specify a **±10% thrust tolerance**:

- **Nominal K-class motor**: 1,800 N·s impulse
- **Low end**: 1,620 N·s (10% lower)
- **High end**: 1,980 N·s (10% higher)

**Effect on landing**:
- Low thrust motor: Average acceleration 10% lower → rocket falls faster during burn
- High thrust motor: Average acceleration 10% higher → rocket decelerates more
- Ignition altitude target must be chosen to work for both extremes

This is why **Monte Carlo simulation** is critical: by testing the rocket against 10,000 simulated motors with random thrust variations, we find an ignition altitude that's robust to ±10% variation.

### 4.3 Burn-Once, Expendable

Each landing attempt requires a fresh motor. You cannot recover and re-ignite a spent SRM.

**Cost implication**:
- Each landing test costs ~$200 (motor) + ~$50 (consumables) = $250–300
- Over 10 landings: $2,500–3,000 for motors alone
- This is why simulation validation is so important: validates design with near-zero per-iteration cost before expensive physical tests

---

## 5. Technical Constraints — Control Authority

### 5.1 Thrust Vector Control Gimbal Limits

The TVC system moves the rocket nozzle (and thus thrust vector) by rotating it about a pivot point. Standard TVC gimbals provide **±5°** rotation.

**Effect on control authority**:
- At altitude 100 m, falling at 30 m/s downward
- Vehicle is tilted 20° nose-down (pointing toward ground)
- TVC rotates thrust vector by +5° toward vertical
- Net thrust vector is still 15° off vertical
- Effective upward thrust component: F cos(15°) $\approx$ 0.97 F (only 3% reduction from vertical case)

**Control limitation**: If the rocket is tilted more than ~15° at ignition, TVC cannot restore it to vertical within the 3-second burn window.

**HERMES mitigation**:
- Design for low descent velocity and minimal tilt at ignition
- EKF monitors attitude in real-time
- PID adjusts gimbal to null attitude error during burn
- ML predicts when ignition altitude should shift to achieve lower tilt

### 5.2 Aerodynamic Control Effectiveness at Low Speed

Rocket fins provide aerodynamic stability, but their effectiveness depends on **dynamic pressure** $q = \frac{1}{2}\rho V^2$.

| Descent velocity | q (Pa) | Fin effectiveness | Notes |
|-----------------|--------|-------------------|-------|
| 100 m/s | 6,000 | Excellent | High-altitude descent |
| 50 m/s | 1,500 | Good | Nominal descent |
| 20 m/s | 240 | Fair | Slow descent (low q) |
| 5 m/s | 15 | Poor | Hoverslam phase |
| 2 m/s | 2 | Negligible | Final descent |

During hoverslam (final 2–3 seconds of descent when velocity <5 m/s), fins are aerodynamically ineffective. **Only the TVC motor can control attitude.**

This is why ignition altitude is critical: must ignite high enough that TVC has time to establish stable attitude before velocity becomes too low.

### 5.3 Short Burn Window

The landing motor burns for only 3 seconds. During this time, all attitude corrections must occur.

**Control problem**: Rocket enters hoverslam phase with arbitrary attitude (roll, pitch, yaw). The PID controller has 3 seconds to:
1. Measure current attitude (from IMU)
2. Compute error (deviation from vertical)
3. Command TVC servo
4. Wait for servo to move (mechanical lag ~50 ms)
5. Wait for attitude to change (rotational inertia ~100 ms)
6. Repeat until attitude is near-vertical

At 100 Hz control update rate, there are 300 control steps available. For a rocket with inertia $I_{yy}$ $\approx$ 50 kg·m², angular acceleration from TVC is roughly:

$	au$_TVC = F_offset $\times$ L $\approx$ 100 N (offset thrust) $\times$ 1 m (distance from CG) $\approx$ 100 N·m

$\alpha$ = $\tau$ / I $\approx$ 100 / 50 = 2 rad/s² (30 deg/s² angular acceleration)

**To null a 10° tilt error**: $\Delta$$\omega$ = $\alpha$ $\Delta$t → 0.17 rad/s = 2 rad/s² $\times$ 0.087 s → needs ~87 ms control time. Feasible.

**To null a 30° tilt error**: Would need ~260 ms (still feasible within 3 s burn).

**Margin**: Adequate, but tight. If rocket enters hoverslam tilted >45°, TVC may not be able to recover.

---

## 6. Infrastructure Constraints

### 6.1 Launch Site Availability

NAR/TRA certified launch sites typically operate **once per month** (weekends), often with advance scheduling.

**Implication**:
- Physical test flights are infrequent (12 flights/year maximum)
- If a flight fails, next attempt is 1 month away
- This is why **simulation is essential**: allows 35+ design iterations in the time it would take to perform 3 physical test flights

### 6.2 Environmental Requirements

#### Wind

- **Operational limit**: 0–15 m/s sustained wind
- **Beyond 15 m/s**: Launch cancelled (safety; rocket too difficult to control)
- **HERMES performance**:
  - 0–5 m/s: Optimizer alone sufficient
  - 5–10 m/s: Optimizer marginal; ML beneficial
  - 10–15 m/s: Optimizer fails; ML enables success (as shown in Scenario 3)

#### Weather

- Clear weather for visibility
- No precipitation (wet motors don't light reliably; risk of electrical hazard)
- Temperature 0–40°C (SRM performance varies outside range)

#### Launch Window

Typical launch window at NAR site: 9 AM – 4 PM on a single weekend day. Limited to 2–3 flights per event (flight prep, recovery, turnaround time).

### 6.3 Recovery Area Requirements

HERMES lands under control, so landing dispersion is small (typically <1 m lateral). However, contingency planning requires:

- **Clear land area**: 2 km radius minimum (for tumble recovery or parachute drift in case of motor failure)
- **Water avoidance**: No large lakes or water hazards where rocket might drift
- **Spectator safety**: All non-essential personnel outside 2 km radius during burn phase

### 6.4 FAA Waiver Coordination

Launches >400 ft AGL require FAA waiver. Process:

1. **Identify launch site** with established waiver corridor
2. **Submit flight notification** 48 hours before launch (via NOTAM system or site coordinator)
3. **Verify airspace** (check for TFRs, military operations, airshows)
4. **Launch approval** (range control officer confirms green light)

**Timeline**: At established sites with pre-approved waivers, process takes <24 hours.

---

## 7. Environmental Constraints

### 7.1 Wind Sensitivity

Wind is the single largest environmental hazard for HERMES landing.

**Mechanism**:
- Constant wind at 10 m/s → lateral drift of 50 m during a 5-second descent
- At ignition time (e.g., 100 m AGL), rocket has 5 m/s lateral velocity
- During 3-second burn, lateral acceleration from TVC is small (~1–2 m/s²)
- Cannot arrest 5 m/s lateral velocity in 3 seconds
- Result: Landing 50 m downwind of target

**HERMES mitigation**:
- Altimeter measures barometric altitude; not affected by wind
- IMU measures attitude and lateral velocity; detects wind
- ML observes lateral velocity (features 6–7) and corrects ignition altitude downward
- By igniting lower (less time to drift), lateral displacement reduces from 50 m → 1 m

**Operational limit**: Wind >15 m/s makes landing challenging even with ML; launch would be cancelled.

### 7.2 Air Density Variation

Air density decreases with altitude and temperature:

$\rho = P / (R T)$

Where P is pressure, R is gas constant, T is temperature.

| Altitude | Density | Pressure |
|----------|---------|----------|
| Sea level, 15°C | 1.225 kg/m³ | 101.3 kPa |
| 1,000 m, 10°C | 1.111 kg/m³ | 89.9 kPa |
| 2,500 m, 5°C | 0.949 kg/m³ | 75.3 kPa |

**Effect on drag force**:
$F_{\text{drag}} = \frac{1}{2} \rho C_d A V^2$

At 2,500 m altitude, air density is 22% lower → drag force is 22% lower → rocket falls faster.

**HERMES mitigation**:
- EKF estimates air density from barometric altitude and temperature
- ML includes air_density as feature #19
- Network learns: "low density → ignite lower"

### 7.3 Motor Thrust Sensitivity to Temperature

Solid rocket motor thrust varies with propellant temperature:

| Propellant Temperature | Thrust Variation |
|------------------------|------------------|
| 0°C (cold) | -15% |
| 20°C (nominal) | 0% |
| 40°C (hot) | +15% |

**Cause**: Propellant burn rate increases with temperature; hotter propellant burns faster → higher regression rate → higher thrust.

**HERMES mitigation**:
- Store motor in shade before launch
- Measure motor housing temperature; adjust ignition altitude expectation
- ML feature #20 (ambient_temperature) captures this

---

## 8. Timeline Implications

### 8.1 Why Simulation-First?

Physical test flights are expensive and slow:
- $200–300 per motor
- 1 month between launch opportunities
- Uncertain outcomes (rocket might crash, destroying hardware)

**Monte Carlo simulation**:
- $0 per iteration (computational cost negligible)
- Can run 35+ design iterations in a month
- Validate robustness before spending money on hardware

### 8.2 Path to Physical Validation

| Year | Milestone | Work |
|------|-----------|------|
| 2026 | Obtain L1 certification | Design and build single-engine rocket with H-class motor; fly at NAR event |
| 2027 | Obtain L2 certification | Study rocket physics; pass NAR exam; fly dual-stage rocket with K-class ascent motor |
| 2028 | Build HERMES avionics | Construct flight computer stack; bench test EKF, PID, ML inference |
| 2028 | Test hoverslam concept | Fly HERMES with landing motor (non-propulsive recovery first to validate control)
| 2029+ | Attempt propulsive landing | Coordinate with NAR launch site safety officer; fly hoverslam sequence |

**This timeline is realistic** given the regulatory and logistical constraints.

---

## 9. Constraint Interactions

Some constraints reinforce each other:

### 9.1 Budget $\times$ Regulatory

Budget ($3,700) is insufficient for liquid propellant infrastructure. This forces SRM choice. SRMs cannot throttle. This motivates optimizer + ML hybrid. The hybrid requires Monte Carlo simulation. Simulation is cheap, allowing many iterations within budget.

**Result**: Design choices are mutually reinforcing.

### 9.2 Technical $\times$ Infrastructure

Control authority is limited (±5° TVC, 3 s burn). This is acceptable because launch site location is guaranteed to be low-wind, level terrain. High-wind locations would make landing impossible even with optimization.

### 9.3 Regulatory $\times$ Timeline

NAR L2 certification takes 2–3 years. This delays physical testing but allows ample time for simulation validation. By the time hardware is built, control algorithms are well-tested.

---

## 10. Risk Mitigation Strategies

| Risk | Constraint | Mitigation |
|------|-----------|-----------|
| Motor fails to ignite | Technical (SRM reliability) | Use redundant ignition (two igniters in parallel) |
| Parachute fails; tumble recovery inadequate | Infrastructure (launch site) | Design rocket to survive tumble from 2,500 m; absorb landing shock in foam |
| ML model fails on real flight data | Technical (out-of-distribution) | Optimizer provides fallback; always have pre-computed ignition altitude as backup |
| Wind exceeds 15 m/s | Environmental | Check weather forecast; cancel launch if necessary |
| FAA denies waiver | Regulatory | Fly at established site with pre-approved waiver corridor |
| Budget overrun | Budget | Design for minimum viable avionics; defer fiberglass airframe to future |

---

## Cross-References

- **ML architecture**: See [10_Results_ML_Landing.md](10_Results_ML_Landing.md) for how constraints motivated the design
- **Hardware design**: See [11_Final_Build_Avionics.md](11_Final_Build_Avionics.md) for component selection driven by budget and technical constraints
- **Validation**: See [13_Validation_Criteria.md](13_Validation_Criteria.md) for testing protocols that account for infrastructure limits
- **Future work**: See [14_Conclusions_Future_Work.md](14_Conclusions_Future_Work.md) for plans to overcome constraints

---

**See also:**
- [10_Results_ML_Landing.md](10_Results_ML_Landing.md) — ML design driven by SRM limitations
- [11_Final_Build_Avionics.md](11_Final_Build_Avionics.md) — Component selection under budget constraint
- [13_Validation_Criteria.md](13_Validation_Criteria.md) — Testing despite infrastructure limits
- [14_Conclusions_Future_Work.md](14_Conclusions_Future_Work.md) — Future work to overcome constraints
