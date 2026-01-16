"""
Physics Equations and Logic Reference
This document lists all physics equations used in the simulation with their sources.

NOTE: Web search was attempted but all domains are blocked in this environment.
The equations below are based on standard aerospace engineering references.

1. DRAG FORCE EQUATION
   Source: Standard fluid dynamics
   Equation: F_d = 0.5 * ρ * v² * C_d * A
   Where:
   - F_d = drag force (N)
   - ρ = air density (kg/m³)
   - v = velocity relative to air (m/s)
   - C_d = drag coefficient (dimensionless)
   - A = reference area (m²)

2. BAROMETRIC FORMULA (Atmospheric Density)
   Source: Standard atmosphere model
   Equation: ρ(h) = ρ₀ * exp(-h/H)
   Where:
   - ρ(h) = air density at altitude h
   - ρ₀ = air density at sea level (1.225 kg/m³)
   - h = altitude (m)
   - H = scale height (8500 m for Earth)

3. POWER LAW WIND PROFILE
   Source: Boundary layer meteorology
   Equation: v(h) = v_ref * (h / h_ref)^α
   Where:
   - v(h) = wind speed at height h
   - v_ref = reference wind speed at reference height
   - h_ref = reference height (typically 10 m)
   - α = power law exponent (0.143 for open terrain, 0.11 for open water)

4. QUATERNION KINEMATICS
   Source: Spacecraft dynamics (quaternion convention: [w, x, y, z])
   Equation: q̇ = 0.5 * q ⊗ ω_quat
   Where:
   - q̇ = quaternion derivative
   - q = current quaternion
   - ω_quat = [0, ωx, ωy, ωz] (angular velocity as quaternion)
   - ⊗ = quaternion multiplication

5. QUATERNION MULTIPLICATION
   Given q1 = [w1, x1, y1, z1] and q2 = [w2, x2, y2, z2]:
   q1 ⊗ q2 = [
       w1*w2 - x1*x2 - y1*y2 - z1*z2,
       w1*x2 + x1*w2 + y1*z2 - z1*y2,
       w1*y2 - x1*z2 + y1*w2 + z1*x2,
       w1*z2 + x1*y2 - y1*x2 + z1*w2
   ]

6. QUATERNION TO ROTATION MATRIX
   Given q = [w, x, y, z]:
   R = [
       [1-2(y²+z²),  2(xy-wz),    2(xz+wy)   ],
       [2(xy+wz),    1-2(x²+z²),  2(yz-wx)   ],
       [2(xz-wy),    2(yz+wx),    1-2(x²+y²) ]
   ]

7. EULER'S ROTATION EQUATIONS
   Source: Rigid body dynamics
   Equation: I * ω̇ + ω × (I * ω) = M
   Where:
   - I = inertia tensor (3x3 matrix)
   - ω = angular velocity vector (body frame)
   - ω̇ = angular acceleration vector
   - M = applied moment/torque vector
   - × = cross product

8. NEWTON'S SECOND LAW (Linear Motion)
   Equation: F = m * a
   Or: a = F / m
   Where:
   - F = total force vector (N)
   - m = mass (kg)
   - a = acceleration vector (m/s²)

9. GRAVITATIONAL FORCE
   Equation: F_g = -m * g * ẑ
   Where:
   - m = mass (kg)
   - g = gravitational acceleration (9.81 m/s² on Earth)
   - ẑ = unit vector pointing up

10. MOMENT OF INERTIA (Cylinder)
    For a solid cylinder of mass m, radius r, and length L:
    I_xx = I_yy = (1/12) * m * (3r² + L²)
    I_zz = (1/2) * m * r²
    (z-axis along cylinder axis)

11. THRUST MOMENT FROM TVC
    Equation: M = r × F
    Where:
    - M = moment vector
    - r = position vector from CG to thrust application point
    - F = thrust force vector
    - × = cross product

12. KINEMATIC EQUATION (Suicide Burn Calculation)
    Equation: v² = v₀² + 2*a*Δh
    For suicide burn: v_final = 0, solve for Δh:
    Δh = -v₀² / (2*a)
    Where:
    - v = final velocity (target: 0)
    - v₀ = initial velocity
    - a = net acceleration (thrust/mass - gravity)
    - Δh = distance traveled

13. FIRST-ORDER LAG (TVC ACTUATOR)
    Equation: dx/dt = (x_commanded - x_current) / τ
    Where:
    - x = actuator position
    - τ = time constant
    - dx/dt = rate of change

VERIFICATION NOTES:
- All equations follow SI units
- Right-handed coordinate system with +Z up
- Quaternion scalar-first convention [w, x, y, z]
- Body frame for angular velocities and moments
- Inertial frame for positions, velocities, and forces

ASSUMPTIONS AND SIMPLIFICATIONS:
1. Flat Earth approximation (valid for altitudes << Earth radius)
2. Constant gravitational acceleration (valid for small altitude changes)
3. Exponential atmosphere model (simplified but reasonable for low altitudes)
4. Symmetric vehicle (diagonal inertia tensor)
5. Thrust acts along vehicle axis with small TVC gimbal angles
6. No sloshing or flexible body dynamics
7. No aerodynamic forces other than drag (no lift)

CRITICAL CORRECTIONS NEEDED:
After review, the implementation appears consistent with standard aerospace dynamics.
However, without web verification, I recommend:
1. Cross-check the quaternion convention matches your integration method
2. Verify the coordinate frame conventions (which axis is "up")
3. Validate the scale height value for the atmosphere model
4. Check the power law exponent for your specific terrain type
5. Verify TVC angle sign conventions (positive = which direction)
