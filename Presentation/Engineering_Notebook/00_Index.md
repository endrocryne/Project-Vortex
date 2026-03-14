# Project HERMES: Engineering Notebook

**Project Title:** HERMES — Hoverslam Evaluation for Rocket Motor Employment via Simulation
**Subtitle:** 6DOF Propulsive Landing Simulation and Control System for Solid-Fuel Rockets
**Author:** Agastya
**Year:** 2026
**Category:** Aerospace Engineering / Computer Science / Systems Engineering

---

## Executive Summary

Project HERMES is a comprehensive simulation and control system designed to prove the feasibility of landing a solid-fuel sounding rocket using propulsive "hoverslam" (suicide burn) landing. Unlike expensive commercial liquid-engine systems (costing \$200k–\$300k+ per launch), HERMES demonstrates how affordable solid rocket motors can enable reusable suborbital access.

The core innovation combines three technologies: (1) an accurate 6DOF physics simulation for trajectory prediction, (2) Monte Carlo optimization to find the precise ignition altitude, and (3) a machine-learning adaptive controller that corrects for real-world faults. Key results: **42.7% apogee improvement** using two-stage architecture, **73–86% better accuracy** than industry tools (RocketPy, OpenRocket, RockSim), and **machine learning succeeds in 5/9 challenging fault scenarios** where rule-based optimization fails.

---

## How to Use This Notebook

This notebook is organized into **7 topic folders with 16 technical documents** designed for deep-dive exploration. Each document is self-contained and cross-referenced.

**For Judges During Presentation:**
- Use the **Quick Reference Map** (below) to find documents matching poster sections
- Each document is 600–1000+ lines with equations, figures, and validation data
- All figures are in `../figures/` and referenced throughout

**For Technical Deep-Dives:**
- Start with your area of interest (framework, methods, results, hardware)
- Follow cross-references to supporting physics, control, or validation material

**For Complete Context:**
- Read the Introduction folder first (context and motivation)
- Then follow the Methods → Results → Hardware → Conclusions path

---

## Folder Structure Overview

| Folder | Contents | Purpose |
|--------|----------|---------|
| **01_Introduction** | Project intro, background, engineering objectives | Context: why propulsive landing matters, how HERMES fits into aerospace |
| **02_Framework** | HERMES 4-component system architecture | System overview: simulator, optimizer, EKF, PID controller |
| **03_Methods** | Simulation engine, ignition optimizer, EKF, PID, fault injection | Technical methods: physics models, estimation, control algorithms, testing |
| **04_ML_Model** | Machine learning architecture and adaptation strategy | ML controller: 25-feature NN, real-time ignition correction, 400-scenario training |
| **05_Results** | Apogee comparison, accuracy validation, ML landing performance | Quantitative results: 9 demo scenarios, ML vs. optimizer, landing precision |
| **06_Hardware** | Avionics schematic, component selection, real-time integration | Physical system: sensors (BNO055, MPL3115A2), Teensy 4.1, RFM95W radio |
| **07_Conclusions** | Constraints, validation criteria, conclusions, future work | Summary: requirements achieved, roadmap to physical build, next steps |

---

## Complete Table of Contents with Document Links

### 01_Introduction/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 01 | [01_Introduction.md](01_Introduction/01_Introduction.md) | **Introduction** | What is suborbital flight? Why propulsive landing matters. SRM vs. liquid engines. The hoverslam challenge. |
| 02 | [02_Background.md](01_Introduction/02_Background.md) | **Background** | History of propulsive landing (SpaceX Starship, Blue Origin New Shepard, BPS.Space). Existing sim tools (RocketPy, OpenRocket, RockSim) and why they fall short. |
| 03 | [03_Engineering_Objectives.md](01_Introduction/03_Engineering_Objectives.md) | **Engineering Objectives** | Quantitative requirements, success criteria, validation strategy, out-of-scope items. |

### 02_Framework/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 04 | [04_HERMES_Framework.md](02_Framework/04_HERMES_Framework.md) | **HERMES Framework Overview** | System architecture: 4 major subsystems (simulator, optimizer, EKF, PID). Data flow, integration strategy. |

### 03_Methods/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 05 | [Core_Simulation_Engine.md](03_Methods/Core_Simulation_Engine.md) | **Core Simulation Engine** | 6DOF physics: equations of motion, quaternion attitude, atmospheric model, aerodynamics, RK45 integration. |
| 06 | [05_Methods_Ignition_Optimizer.md](03_Methods/05_Methods_Ignition_Optimizer.md) | **Ignition Altitude Optimizer** | Analytical ignition estimation and Monte Carlo search (20,000 trials). Success rate curve. |
| 07 | [06_Methods_EKF_PID_Control.md](03_Methods/06_Methods_EKF_PID_Control.md) | **EKF + PID Control** | Kalman filter state estimation (mass, drag) + PID feedback via thrust-vector control. Gimbal authority. |
| 08 | [07_Methods_Fault_Injection.md](03_Methods/07_Methods_Fault_Injection.md) | **Fault Injection & Robustness** | 12 environmental conditions, 12 fault scenarios (wind, mass error, $C_d$ error, sensor noise). Test harness, 192 total test cases. Fault intensity quantification (Hill functions, exponential urgency, bilinear interaction). |

### 04_ML_Model/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 09 | [ML_Model_Architecture.md](04_ML_Model/ML_Model_Architecture.md) | **ML Model Architecture** | 25-feature neural network. Real-time ignition correction. Training on 400 scenarios. Performance across fault types. |

### 05_Results/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 10 | [08_Results_Apogee_Two_Stage.md](05_Results/08_Results_Apogee_Two_Stage.md) | **Results: Two-Stage Apogee** | Single-stage vs. two-stage architecture. 42.7% apogee improvement. Burn profile analysis. |
| 11 | [09_Results_Accuracy_Comparison.md](05_Results/09_Results_Accuracy_Comparison.md) | **Results: Accuracy Validation** | HERMES vs. RocketPy / OpenRocket / RockSim. 73–86% better accuracy. Why HERMES is more faithful. |
| 12 | [10_Results_ML_Landing.md](05_Results/10_Results_ML_Landing.md) | **Results: ML + Landing** | 9 demo scenarios, ML vs. optimizer performance. 5/9 ML successes vs. 3/9 optimizer. Landing velocity scatter plot. RMSE analysis (0.387 m overall). |

### 06_Hardware/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 13 | [11_Final_Build_Avionics.md](06_Hardware/11_Final_Build_Avionics.md) | **Final Build & Avionics** | Hardware BOM, sensor selection (BNO055 IMU, MPL3115A2 baro, Teensy 4.1, RFM95W radio). Data rates, integration, cost breakdown. |

### 07_Conclusions/
| Doc | Filename | Title | Summary |
|-----|----------|-------|---------|
| 14 | [12_Constraints.md](07_Conclusions/12_Constraints.md) | **Constraints & Assumptions** | Simulation-only scope, SRM availability, gimbal limits, atmospheric model bounds, sensor noise assumptions. |
| 15 | [13_Validation_Criteria.md](07_Conclusions/13_Validation_Criteria.md) | **Validation Criteria** | 12 environmental conditions, 12 fault scenarios, 144 total tests. Success definitions (velocity, altitude, position). Validation achievement summary. |
| 16 | [14_Conclusions_Future_Work.md](07_Conclusions/14_Conclusions_Future_Work.md) | **Conclusions & Future Work** | Key findings, cost-effectiveness (500–650× reduction vs. commercial), roadmap to physical build, CFD validation, custom SRM design. |

---

## Quick Reference: Poster Section → Document Map

Use this table to jump directly from poster sections to relevant documents during live presentation:

| Poster Section | Primary Document(s) | Secondary Reference(s) |
|---|---|---|
| **Introduction** | [01_Introduction.md](01_Introduction/01_Introduction.md) | Background (02), Objectives (03) |
| **Background** | [02_Background.md](01_Introduction/02_Background.md) | Introduction (01) |
| **Engineering Objectives** | [03_Engineering_Objectives.md](01_Introduction/03_Engineering_Objectives.md) | Validation Criteria (15) |
| **HERMES Framework Overview** | [04_HERMES_Framework.md](02_Framework/04_HERMES_Framework.md) | Core Simulation (05), EKF + PID (07), ML (09) |
| **Core Simulation** | [Core_Simulation_Engine.md](03_Methods/Core_Simulation_Engine.md) | Framework (04), Accuracy (11) |
| **Ignition Altitude Optimizer** | [05_Methods_Ignition_Optimizer.md](03_Methods/05_Methods_Ignition_Optimizer.md) | Framework (04), ML (09) |
| **EKF + PID Control** | [06_Methods_EKF_PID_Control.md](03_Methods/06_Methods_EKF_PID_Control.md) | Fault Injection (08), ML (09) |
| **Fault Injection** | [07_Methods_Fault_Injection.md](03_Methods/07_Methods_Fault_Injection.md) | EKF + PID (07), Results ML (12) |
| **ML Model** | [ML_Model_Architecture.md](04_ML_Model/ML_Model_Architecture.md) | Framework (04), Results ML (12) |
| **Results: Apogee (42.7%)** | [08_Results_Apogee_Two_Stage.md](05_Results/08_Results_Apogee_Two_Stage.md) | Framework (04), Conclusions (16) |
| **Results: Accuracy (73–86%)** | [09_Results_Accuracy_Comparison.md](05_Results/09_Results_Accuracy_Comparison.md) | Core Simulation (05), Background (02) |
| **Results: ML + Landing** | [10_Results_ML_Landing.md](05_Results/10_Results_ML_Landing.md) | ML (09), Fault Injection (08) |
| **Final Build / Avionics** | [11_Final_Build_Avionics.md](06_Hardware/11_Final_Build_Avionics.md) | Framework (04), Conclusions (16) |
| **Constraints** | [12_Constraints.md](07_Conclusions/12_Constraints.md) | Objectives (03), Validation (15) |
| **Validation Criteria** | [13_Validation_Criteria.md](07_Conclusions/13_Validation_Criteria.md) | Fault Injection (08), Results (10–12) |
| **Conclusions & Future Work** | [14_Conclusions_Future_Work.md](07_Conclusions/14_Conclusions_Future_Work.md) | All results (10–12), Hardware (13) |

---

## Key Performance Metrics at a Glance

| Metric | Value | Context |
|--------|-------|---------|
| **Apogee Improvement (2-stage)** | +42.7% | vs. single-stage equivalent |
| **Accuracy vs. Industry Tools** | 73–86% better | vs. RocketPy, OpenRocket, RockSim |
| **ML Success Rate** | 5/9 scenarios | vs. 3/9 for optimizer alone |
| **Worst-Case ML Landing** | 1.02 m/s | Severe fault (vs. 12.8 m/s crash without ML) |
| **Landing Altitude Precision** | ±0.5 m | Consumer GPS-grade accuracy |
| **Vertical Landing Velocity** | <2 m/s | Survivable by foam/spring gear |
| **Total Landing Velocity** | <3 m/s | Structural integrity threshold |
| **Cost Reduction Factor** | ~500–650× | vs. commercial operators (\$200k–\$300k+) |

---

## Figure Index

All figures are stored in `../figures/` and referenced throughout the notebook:

| # | Filename | Title | Primary Docs |
|---|----------|-------|---|
| 1 | fig_01_system_architecture.png | HERMES 4-Component Architecture Block Diagram | 04 |
| 2 | fig_02_flight_profile.png | 5-Phase Mission Profile (Ascent → Separation → Coast → Descent → Suicide Burn) | 01, 02, 08 |
| 3 | fig_03_trajectory_baseline.png | Baseline Trajectory: Altitude & Velocity vs. Time (demo_01) | 05 |
| 4 | fig_04_success_rate_curve.png | Monte Carlo Success Rate vs. Ignition Altitude (20,000 trials) | 06 |
| 5 | fig_05_demo_scenario_comparison.png | 9 Demo Scenarios: Landing Velocity Bar Chart | 12 |
| 6 | fig_06_ekf_block_diagram.png | EKF State Estimation Block Diagram with Equations | 07 |
| 7 | fig_07_pid_block_diagram.png | PID Control Loop Diagram with Anti-Windup | 07 |
| 8 | fig_08_fault_injection_diagram.png | Fault Types and Trigger Modes Summary | 08 |
| 9 | fig_09_two_stage_comparison.png | Single-Stage vs. Two-Stage Apogee Comparison (42.7% gain) | 10 |
| 10 | fig_10_accuracy_comparison.png | HERMES vs. RocketPy vs. OpenRocket vs. RockSim Accuracy | 05, 11 |
| 11 | fig_11_ml_vs_optimizer.png | ML vs. Optimizer Landing Velocity Across 9 Paired Scenarios | 12 |
| 12 | fig_12_landing_scatter.png | Landing Position Accuracy Scatter Plot (XY coordinates) | 12 |
| 13 | fig_13_avionics_schematic.png | Avionics Hardware Block Diagram (BNO055 → Teensy → RFM95W) | 13 |
| 14 | fig_14_budget_breakdown.png | Cost Breakdown Pie Chart + Per-Launch Consumables | 13, 16 |
| 15 | fig_15_monte_carlo_heatmap.png | Monte Carlo Parameter Sensitivity Heatmap | 11, 16 |
| 16 | fig_16_validation_summary.png | Validation Criteria Achievement Summary Chart | 03, 15 |
| 17 | fig_17_sensor_drift_example.png | Sensor Drift with ML Correction Example | 09, 12 |
| 18 | fig_18_secondary_body.png | Two-Stage Separation: Primary Vehicle vs. Secondary Payload | 05, 10 |
| 19 | fig_19_ml_rockets_tested.png | ML vs. Optimizer: Landing Velocity Across Rocket Configurations | 12 |

---

## Rocket Specifications (Reference)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Dry Mass** | 50 kg | Airframe + avionics + TVC system |
| **Propellant Mass** | 10 kg | Landing motor only (ignition burn) |
| **Total Length** | 5 m | Typical suborbital sounding rocket |
| **Diameter** | 0.3 m | 12-inch diameter |
| **Landing Motor Thrust** | 1000 N (fixed) | Solid motor (fixed throttle) |
| **Landing Burn Duration** | ~3.0 s | Typical for soft landing |
| **TVC Gimbal Limit** | ±5° | Thrust-vector control authority |
| **Thrust-to-Weight Ratio** | 2.04 | Supports vertical hover + control margins |

---

## How to Cite This Work

**Simulation system:** "HERMES 6DOF simulator with RK45 integration and quaternion attitude tracking"
**Optimizer:** "Monte Carlo ignition altitude optimization (20,000 trials)"
**Control:** "Dual-mode control: EKF state estimation + PID attitude feedback + ML adaptation"
**Validation:** "Tested against 9 demo scenarios and 144 fault-injection conditions"

**Suggested citation:**
> HERMES: A 6DOF Simulation and Control System for Propulsive Landing of Solid-Fuel Suborbital Rockets. Engineering Notebook. Project Vortex, Agastya. March 2026.

---

**End of Document 00: Index**

*Next: [01_Introduction.md](01_Introduction/01_Introduction.md)*
