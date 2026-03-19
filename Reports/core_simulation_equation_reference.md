# Core Simulation Equation Reference (6DOF)

This document records the governing equations used to validate the core simulation (`physics_engine.py`, `simulation.py`, `solid_motor.py`, `state_estimator.py`) before code-level review.

## 1) Rigid-body translational dynamics

In inertial frame:

\[
m\dot{\mathbf{v}} = \mathbf{F}_g + \mathbf{F}_d + \mathbf{F}_t
\]

\[
\dot{\mathbf{r}} = \mathbf{v}
\]

Where:
- \(\mathbf{F}_g = [0, 0, -mg]\)
- \(\mathbf{F}_d = -\frac{1}{2}\rho C_d A \|\mathbf{v}_{rel}\|\mathbf{v}_{rel}\)
- \(\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}\)
- \(\mathbf{F}_t\) is thrust vector transformed from body frame to inertial frame.

## 2) Atmospheric density (engineering approximation)

Exponential model (for \(h \ge 0\)):

\[
\rho(h) = \rho_0 e^{-h/H}
\]

with scale height \(H \approx 8500\text{ m}\).

## 3) Wind models

- Constant horizontal wind
- Altitude power-law wind profile \(v(h)=v_{ref}(h/h_{ref})^\alpha\)
- Gust model (deterministic sinusoidal perturbations)

These are pragmatic simulation models and not full turbulence/LES models.

## 4) Rotational dynamics (6DOF)

Euler rigid-body equation in body frame:

\[
\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega}) = \mathbf{M}
\]

\[
\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1}\left(\mathbf{M} - \boldsymbol{\omega}\times(\mathbf{I}\boldsymbol{\omega})\right)
\]

Primary moment terms in current model:
- Thrust-vector-control moment from nozzle offset/cg offset
- Simplified angular damping term.

## 5) Quaternion kinematics

Quaternion state \(q=[q_w,q_x,q_y,q_z]\), angular-rate quaternion \(\omega_q=[0,\omega_x,\omega_y,\omega_z]\):

\[
\dot{q} = \frac{1}{2}q\otimes \omega_q
\]

Quaternion normalization should be applied to avoid numerical drift.

## 6) Solid motor and mass depletion

- Thrust is interpolated from thrust-time curve.
- Mass flow is modeled as:

\[
\dot{m} = -\dot{m}_{prop}
\]

with \(\dot{m}_{prop}\) assumed piecewise-constant over burn for simplicity.

## 7) TVC actuation

First-order actuator lag:

\[
\dot{\theta} = \frac{\theta_{cmd}-\theta}{\tau}
\]

with angle saturation at \(\pm \theta_{max}\).

## 8) EKF mass/Cd estimator

State:
\[
\mathbf{x}=[m,\ C_d]^T
\]

Predict:
\[
m_{k+1}=m_k-\dot{m}\Delta t,\quad C_{d,k+1}=C_{d,k}
\]

Measurement model (specific-force style in vertical axis):
\[
a_{sens,z}\approx \frac{T_z + D_z}{m}
\]

with drag sign consistent with velocity direction.

## 9) Implementation quality criteria used in audit

1. Physics consistency: no stochastic forcing inside repeated derivative evaluations at same state/time.
2. Numerical consistency: deterministic ODE RHS for identical \((t, x)\) within a simulation run.
3. Frame/sign consistency: drag opposes relative velocity, gravity negative \(z\), thrust rotation correctly applied.
4. Bounds/safety: non-negative altitude for atmosphere query, stable quaternion normalization behavior.
