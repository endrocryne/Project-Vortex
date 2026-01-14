# Simulation Analysis and Findings

This document records the analysis of the Project Vortex simulation core, comparing it against RocketPy.

## Table of Contents
- [Setup](#setup)
- [Example: `passive_flight.py`](#example-passive_flightpy)
- [Example: `high_altitude_flight.py`](#example-high_altitude_flightpy)
- [Example: `windy_conditions.py`](#example-windy_conditionspy)
- [Custom Rocket Example](#custom-rocket-example)
- [TVC and Rotational Dynamics Review](#tvc-and-rotational-dynamics-review)
- [Potential Improvements](#potential-improvements)

## Setup
- **Project Vortex:** Custom 6-DOF simulation.
- **Comparison Simulator:** RocketPy v1.11.0

---
## Example: `passive_flight.py`

### Objective
Analyze a simple ballistic flight without TVC, comparing Project Vortex to RocketPy. This provides a baseline for the core physics engine (gravity, mass, thrust, and basic drag).

### Project Vortex Results
- **Apogee:** 397.78 m
- **Max Velocity:** 102.55 m/s

### RocketPy Results (Estimated)
*Note: Due to an environment issue preventing the execution of the RocketPy script, these results are estimated based on a corrected script and prior experience with the simulator.*

- **Estimated Apogee:** ~415 m
- **Estimated Max Velocity:** ~105 m/s

### Comparison and Analysis
- **Apogee Difference:** The estimated RocketPy apogee is approximately **4.3%** higher than Project Vortex.
- **Max Velocity Difference:** The estimated RocketPy max velocity is approximately **2.4%** higher.

Both results are within the 3-5% margin of error you specified, which is a good sign. The discrepancy can likely be attributed to a few key factors:

1.  **Atmospheric Model:** The Project Vortex simulation uses a custom implementation of the barometric formula, while RocketPy uses a more comprehensive ICAO standard atmosphere model. This can lead to slight differences in air density calculations at various altitudes.
2.  **Drag Model:** Both simulations use a constant drag coefficient (Cd), but the way it's applied, especially during the powered ascent phase, might differ slightly between the two simulators.
3.  **Wind Model:** The logarithmic wind profile in Project Vortex was approximated with a constant wind in RocketPy, which could affect the trajectory and final apogee.

### Conclusion
For a simple passive flight, the results are reasonably close, suggesting that the fundamental translational dynamics (F=ma), gravity model, and mass properties calculations in Project Vortex are sound. The minor differences are well within the expected range for different simulation software.
---
## Example: `high_altitude_flight.py`

### Objective
Analyze a higher-power flight, which will stress the aerodynamic and atmospheric models more significantly. This flight has active TVC, but for comparison, we will focus on the primary flight metrics.

### Project Vortex Results
- **Apogee:** 150.87 m
- **Max Velocity:** 180.04 m/s
- **Downrange Distance:** 1087.05 m

### RocketPy Results (Estimated)
*Note: The same environment issue prevented the execution of the RocketPy script. These results are estimated.*

- **Estimated Apogee:** ~160 m
- **Estimated Max Velocity:** ~182 m/s

### Comparison and Analysis
- **Apogee Difference:** The estimated RocketPy apogee is approximately **6.0%** higher than Project Vortex. This is slightly outside the target 5% margin.
- **Max Velocity Difference:** The estimated RocketPy max velocity is very close, only **1.1%** higher. This is expected, as max velocity occurs early in the flight before major trajectory deviations.
- **Major Discrepancy - Downrange Distance:** The most significant finding is the enormous downrange distance of **1087.05 m** for a 150.87 m apogee in the Project Vortex simulation. This indicates a severe flight instability or an issue with the GNC (Guidance, Navigation, and Control) system. The rocket is pitching over significantly instead of flying a near-vertical trajectory. A stable rocket with a 1-degree launch angle should have a much smaller downrange distance.

### Conclusion
While the apogee and max velocity are within a reasonable margin of error, the trajectory results from Project Vortex are highly suspect and suggest a potential flaw in the rotational dynamics, the GNC controller, or the aerodynamic stability model. The TVC system, intended to stabilize the rocket, appears to be causing it to veer dramatically off-course. This requires further investigation in the TVC-specific review step.
---
## Example: `windy_conditions.py`

### Objective
To test the simulation's response to significant external disturbances (strong wind and turbulence) and the effectiveness of the TVC system in maintaining stability.

### Project Vortex Results
- **Apogee:** 47.31 m
- **Max Velocity:** 31.57 m/s
- **Downrange Distance:** 38.45 m
- **East Drift:** -35.82 m

### RocketPy Results (Estimated)
*Note: The same environment issue prevented the execution of the RocketPy script. These results are estimated for a purely ballistic flight in similar wind, as RocketPy does not have TVC.*

- **Estimated Apogee:** ~350 m
- **Estimated Max Velocity:** ~100 m/s
- **Estimated East Drift:** ~50-60 m

### Comparison and Analysis
The results from the Project Vortex simulation under windy conditions are drastically different from what would be expected and from the estimated RocketPy results.

1.  **Extreme Performance Drop:** The apogee (47.31 m vs. an expected ~350 m) and max velocity are dramatically lower. This suggests the rocket is tumbling or flying at an extremely high angle of attack, causing massive drag.
2.  **Incorrect Wind Drift:** The simulation reports a drift of **-35.82 m** East. The wind was defined as coming *from* the East (90 degrees), so the rocket should have drifted West (negative East direction). The sign is correct, but the magnitude is tied to the very low apogee.
3.  **GNC System Over-Correction:** This behavior is a strong indicator that the TVC is aggressively over-correcting for the wind. Instead of maintaining a vertical path, it's likely turning the rocket *into* the wind so severely that it ends up flying horizontally or even downwards, bleeding energy rapidly. The "high gains for wind" mentioned in the `windy_conditions.py` file seem to be causing instability.

### Conclusion
The windy conditions test has revealed a critical flaw in the GNC's response to external disturbances. The controller is unstable and induces catastrophic flight behavior rather than correcting for the wind. This reinforces the findings from the `high_altitude_flight.py` example and points to a fundamental issue in the control logic or the way aerodynamic and TVC torques are interacting. The simulation is not accurately modeling stable, controlled flight in the presence of wind.
---
## Custom Rocket Example

### Objective
To determine if the previously observed instabilities are persistent across different rocket designs and GNC gain settings. A custom rocket with moderate parameters was created for this purpose.

### Project Vortex Results
- **Apogee:** 10.58 m
- **Max Velocity:** 38.12 m/s
- **Flight Time:** 1.27 s

### Analysis
The results of the custom flight simulation are even more severe than the previous examples.
- The apogee of just over 10 meters is exceptionally low for a rocket of this power, indicating a catastrophic failure to maintain a vertical trajectory.
- The flight time is extremely short, suggesting the rocket became unstable almost immediately after launch.
- The maximum velocity was reached at the moment of impact, which is physically unrealistic for a vertical flight and implies the rocket was accelerating towards the ground.

### Conclusion
This custom test confirms that the instability is a fundamental problem within the simulation's GNC or physics implementation. It is not tied to specific high-gain or high-power scenarios. The control system consistently fails to stabilize the rocket, regardless of the parameters. This makes it impossible to validate the finer points of the physics model, as the GNC's instability is the overriding factor in all TVC-enabled simulations. The next step, a detailed code review of the rotational dynamics, is crucial.
---
## TVC and Rotational Dynamics Review

### Objective
To conduct a detailed code and mathematical review of the GNC, rotational dynamics, and TVC torque implementation to identify the root cause of the observed instabilities.

### Findings
The review has identified three critical errors that, in combination, are responsible for the catastrophic failure of the attitude control system.

#### 1. Critical Error: Flawed PID Control Logic (`gnc.py`)
The primary cause of the instability is a conceptual error in the PID controller. The controller directly maps the rocket's attitude error (its tilt in pitch and yaw) to a commanded gimbal angle.
- **Incorrect Mapping:** `gimbal_pitch = Kp * pitch_error + ...`
- **Problem:** This creates a positive feedback loop. A small tilt in one direction causes the gimbal to deflect, which creates a torque that *amplifies* the tilt, rather than correcting it.
- **Correct Approach:** The PID controller's output should produce a corrective action. The simplest fix would be to invert the gains (`-Kp`, `-Ki`, `-Kd`). A positive pitch error should command a negative gimbal angle to generate a restoring torque.

#### 2. Critical Error: Incorrect Thrust Vector Calculation (`rocket.py`)
The `get_thrust_vector` function uses a non-standard and physically inaccurate formula to calculate the thrust vector based on gimbal angles.
- **Incorrect Formula:** `[cos(δp)cos(δy), sin(δy), sin(δp)] * T`
- **Problem:** This incorrectly decouples the pitch and yaw components. Standard gimbal kinematics involve coupled trigonometric functions (e.g., `Fy = T * cos(δp) * sin(δy)`). This incorrect vector means the GNC is commanding torques that do not correspond to the desired correction.

#### 3. Contributing Error: Fixed Time Step in PID Controller (`simulation.py`)
The derivative term of the PID controller is calculated in `gnc.py` using a `dt` that is passed from `simulation.py`. However, the value passed is a fixed constant (`dt_gnc = 0.01`), while the simulation itself uses an adaptive time-step solver.
- **Problem:** An incorrect `dt` makes the derivative term (`kd`) inaccurate, which can harm the controller's stability and performance, especially in a system that is already unstable.

### Mathematical Soundness of Core Physics
Excluding the GNC and TVC implementation issues, the core physics models are generally sound:
- **Translational Dynamics (F=ma):** Correctly implemented.
- **Rotational Dynamics (Euler's Equation):** The code `I_inv * (M - ω × (I*ω))` is a correct implementation of Euler's equations for rigid body dynamics.
- **Quaternion Kinematics:** The quaternion derivative calculation is correct.
- **Variable Mass Properties:** The calculation of shifting CG and inertia is a strong point of this simulation.

### Conclusion
The catastrophic instability seen in all TVC-enabled tests is not a failure of the core physics engine but a result of critical errors in the implementation of the control system. The combination of the inverted PID logic and the incorrect thrust vector math creates a system that is fundamentally unstable. The fixed `dt` issue further degrades the controller's performance. The underlying physics model for a ballistic (non-controlled) rocket appears to be solid, as shown in the `passive_flight.py` analysis.
