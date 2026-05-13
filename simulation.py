"""
Suicide Burn Simulation
6DOF flight dynamics with solid motor and TVC control
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time as pytime
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

from physics_engine import PhysicsEngine
from solid_motor import SolidMotor


class SuicideBurnSimulation:
    """Complete 6DOF suicide burn simulation"""
    
    def __init__(self, rocket_config, environment_config, simulation_config):
        """
        Initialize simulation
        
        Args:
            rocket_config: dict with rocket parameters
            environment_config: dict with environment parameters
            simulation_config: dict with simulation parameters
        """
        self.rocket_config = rocket_config
        self.environment_config = environment_config
        self.simulation_config = simulation_config
        
        # Physics engine
        self.physics = PhysicsEngine(environment_config)
        
        # Solid motor
        self.motor = SolidMotor(rocket_config)
        
        # Rocket parameters
        self.dry_mass = rocket_config.get('dry_mass', 50.0)
        self.propellant_mass = rocket_config.get('propellant_mass', 10.0)
        self.initial_mass = self.dry_mass + self.propellant_mass
        
        self.length = rocket_config.get('length', 5.0)
        self.diameter = rocket_config.get('diameter', 0.3)
        
        # Dynamic CG/inertia option
        self.use_dynamic_inertia = rocket_config.get('use_dynamic_inertia', False)
        
        # Moments of inertia (simplified cylinder)
        # Initial inertia tensor
        r = self.diameter / 2
        self.I_xx_initial = (1/12) * self.initial_mass * (3*r**2 + self.length**2)
        self.I_yy_initial = self.I_xx_initial
        self.I_zz_initial = (1/2) * self.initial_mass * r**2
        self.inertia_tensor = np.diag([self.I_xx_initial, self.I_yy_initial, self.I_zz_initial])
        
        # CG location parameters (for dynamic calculation)
        # Assume fuel tank at bottom, dry mass CG at center
        self.fuel_tank_bottom = -self.length / 2  # Bottom of rocket
        self.fuel_tank_top = 0.0  # Middle of rocket
        self.dry_mass_cg = 0.0  # Dry mass CG at geometric center
        
        # Control parameters - PID gains for pitch (y-axis) and yaw (x-axis)
        self.tvc_kp_pitch = rocket_config.get('tvc_kp_pitch', 0.5)
        self.tvc_ki_pitch = rocket_config.get('tvc_ki_pitch', 0.05)
        self.tvc_kd_pitch = rocket_config.get('tvc_kd_pitch', 0.1)
        
        self.tvc_kp_yaw = rocket_config.get('tvc_kp_yaw', 0.5)
        self.tvc_ki_yaw = rocket_config.get('tvc_ki_yaw', 0.05)
        self.tvc_kd_yaw = rocket_config.get('tvc_kd_yaw', 0.1)
        
        # New TVC Mode and Drift Correction Gain
        self.tvc_mode = rocket_config.get('tvc_mode', simulation_config.get('tvc_mode', 'orientation'))
        self.tvc_drift_gain = rocket_config.get('tvc_drift_gain', simulation_config.get('tvc_drift_gain', 0.1))
        
        # Integral error accumulators
        self.pitch_integral_error = 0.0
        self.yaw_integral_error = 0.0
        self.last_time = 0.0
        
        # Sensor accuracy (Monte Carlo)
        self.altimeter_error = simulation_config.get('altimeter_error', 0.0)
        self.velocity_sensor_error = simulation_config.get('velocity_sensor_error', 0.0)
        
        # Ignition offsets
        self.ignition_percent_offset = simulation_config.get('ignition_percent_offset', 0.0)
        self.ignition_hard_offset = simulation_config.get('ignition_hard_offset', 0.0)
        # Minimum time between motor burnout and ground contact (s)
        self.ignition_burnout_buffer = simulation_config.get('ignition_burnout_buffer', 0.5)

        # Simulation mode and config
        self.simulate_ascent = simulation_config.get('simulate_ascent', False)
        self.ascent_motor_casing_mass = rocket_config.get('ascent_motor_casing_mass', 2.0)
        
        # Ascent parameters
        self.ascent_initial_pitch = simulation_config.get('ascent_initial_pitch', 0.0) # degrees
        self.ascent_initial_yaw = simulation_config.get('ascent_initial_yaw', 0.0) # degrees
        self.ascent_initial_roll = simulation_config.get('ascent_initial_roll', 0.0) # degrees
        
        # Descent override parameters
        self.descent_initial_pitch = simulation_config.get('descent_initial_pitch', 0.0) # degrees
        self.descent_initial_yaw = simulation_config.get('descent_initial_yaw', 0.0) # degrees
        self.descent_initial_roll = simulation_config.get('descent_initial_roll', 0.0) # degrees
        
        # RCS parameters
        self.rcs_thrust = rocket_config.get('rcs_thrust', 1000.0)
        self.rcs_arm = rocket_config.get('rcs_arm', self.length * 0.4) # Near nose
        self.rcs_enabled = False
        self.rcs_commands = np.zeros(4) # [pitch_pos, pitch_neg, yaw_pos, yaw_neg]

        # Results storage
        self.history = None

    def run_ascent_phase(self, initial_state):
        """
        Simulate the ascent phase:
        1. Burn ascent motor (full fuel)
        2. Coast to apogee (vz <= 0)
        
        Args:
            initial_state: State vector at launch pad
            
        Returns:
            apogee_state: State vector at apogee
            ascent_history: History dict for ascent phase
        """
        # Create a FRESH motor for ascent
        self.motor = SolidMotor(self.rocket_config)
        
        # Modify initial mass to include Ascent Motor Casing + Ascent Propellant
        # The 'initial_state' passed in likely has the descent mass. We need to add ascent components.
        # However, to avoid confusion, let's reconstruct the mass.
        # Ascent Start Mass = Dry(Landing) + Prop(Landing) + Casing(Ascent) + Prop(Ascent)
        base_descent_mass = self.dry_mass + self.propellant_mass
        ascent_total_mass = base_descent_mass + self.ascent_motor_casing_mass + self.motor.propellant_mass
        
        # Modify the mass in state vector
        current_state = initial_state.copy()
        current_state[13] = ascent_total_mass
        
        # Setup events
        def apogee_event(t, y):
             # Ignore apogee check during first 1.0s to allow for thrust ramp-up
             # (prevents immediate trigger if T < W at t=0)
             if t < 1.0: 
                 return 100.0
             return y[5] # vz
        apogee_event.terminal = True
        apogee_event.direction = -1
        
        # 1. Powered Ascent (Burn Time)
        burn_time = self.motor.burn_time
        
        # PID reset
        self.pitch_integral_error = 0.0
        self.yaw_integral_error = 0.0
        self.last_time = 0.0

        # Reset PID for ascent
        self.last_time = 0.0
        self.pitch_integral_error = 0.0
        self.yaw_integral_error = 0.0
        self.last_pitch_error = 0.0
        self.last_yaw_error = 0.0
        
        def powered_derivative(t, state):
            # TVC Logic (Vertical hold for ascent)
            p_cmd, y_cmd, p_int, y_int, p_err, y_err = self.tvc_controller(
                state, t, self.last_time, 
                self.pitch_integral_error, self.yaw_integral_error,
                self.last_pitch_error, self.last_yaw_error
            )
            
            if t > self.last_time:
                self.motor.set_tvc_command(p_cmd, y_cmd)
                self.pitch_integral_error = p_int
                self.yaw_integral_error = y_int
                self.last_pitch_error = p_err
                self.last_yaw_error = y_err
                self.last_time = t
            
            # Update mass and other physics
            return self.state_derivative(t, state)

        # Ignite!
        self.motor.ignite(0.0)
        
        sol_powered = solve_ivp(
            powered_derivative,
            [0, burn_time],
            current_state,
            events=[apogee_event], # Just in case it hits apogee during burn
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        # 2. Coast to Apogee
        state_after_burn = sol_powered.y[:, -1]
        time_after_burn = sol_powered.t[-1]
        
        if len(sol_powered.t_events[0]) > 0:
            # Reached apogee during burn? highly unlikely but possible
            t_coast = np.array([])
            y_coast = np.empty((14, 0))
            apogee_state = state_after_burn
        else:
            # Coast physics (motor is spent, but we still have empty casing mass attached until apogee)
            # Actually, the motor class handles burnout behavior (thrust=0), so we can just use state_derivative
            # but we need to ensure the mass derivative is 0.
            # state_derivative calls get_mass_flow_rate, which returns 0 after burnout. Correct.
            
            sol_coast = solve_ivp(
                self.state_derivative,
                [time_after_burn, time_after_burn + 100.0], # 100s timeout
                state_after_burn,
                events=[apogee_event],
                method='RK45',
                rtol=1e-6
            )
            t_coast = sol_coast.t
            y_coast = sol_coast.y
            apogee_state = sol_coast.y[:, -1]

        # Combine history
        t_combined = np.concatenate([sol_powered.t, t_coast])
        y_combined = np.concatenate([sol_powered.y, y_coast], axis=1)
        
        return apogee_state, t_combined, y_combined

    def calculate_dynamic_cg(self, current_mass):
        """
        Calculate center of gravity location as fuel burns
        
        Args:
            current_mass: current vehicle mass (kg)
            
        Returns:
            cg_location: CG location in body frame (m, along z-axis)
        """
        # Calculate remaining fuel mass
        fuel_remaining = current_mass - self.dry_mass
        fuel_remaining = max(0.0, min(fuel_remaining, self.propellant_mass))
        
        # Fuel CG (assuming uniform distribution in tank)
        fuel_cg_z = (self.fuel_tank_bottom + self.fuel_tank_top) / 2
        
        # Combined CG using parallel axis theorem
        if current_mass > 0:
            cg_z = (self.dry_mass * self.dry_mass_cg + fuel_remaining * fuel_cg_z) / current_mass
        else:
            cg_z = self.dry_mass_cg
        
        return cg_z
    
    def calculate_dynamic_inertia(self, current_mass, cg_location):
        """
        Calculate inertia tensor accounting for fuel depletion and CG shift
        
        Args:
            current_mass: current vehicle mass (kg)
            cg_location: current CG location in body frame (m)
            
        Returns:
            inertia_tensor: 3x3 inertia tensor in body frame
        """
        if not self.use_dynamic_inertia:
            return self.inertia_tensor
        
        # Calculate remaining fuel mass
        fuel_remaining = current_mass - self.dry_mass
        fuel_remaining = max(0.0, min(fuel_remaining, self.propellant_mass))
        
        r = self.diameter / 2
        
        # Dry mass inertia about its own CG
        I_xx_dry = (1/12) * self.dry_mass * (3*r**2 + self.length**2)
        I_yy_dry = I_xx_dry
        I_zz_dry = (1/2) * self.dry_mass * r**2
        
        # Fuel inertia (model as cylinder in tank)
        fuel_length = (self.fuel_tank_top - self.fuel_tank_bottom) * (fuel_remaining / self.propellant_mass if self.propellant_mass > 0 else 0)
        I_xx_fuel = (1/12) * fuel_remaining * (3*r**2 + fuel_length**2) if fuel_remaining > 0 else 0
        I_yy_fuel = I_xx_fuel
        I_zz_fuel = (1/2) * fuel_remaining * r**2 if fuel_remaining > 0 else 0
        
        # Fuel CG
        fuel_cg_z = (self.fuel_tank_bottom + self.fuel_tank_top) / 2 if fuel_remaining > 0 else 0
        
        # Use parallel axis theorem to move to combined CG
        # I_total = I_dry + m_dry * d_dry^2 + I_fuel + m_fuel * d_fuel^2
        d_dry = self.dry_mass_cg - cg_location
        d_fuel = fuel_cg_z - cg_location
        
        I_xx = I_xx_dry + self.dry_mass * d_dry**2 + I_xx_fuel + fuel_remaining * d_fuel**2
        I_yy = I_yy_dry + self.dry_mass * d_dry**2 + I_yy_fuel + fuel_remaining * d_fuel**2
        I_zz = I_zz_dry + I_zz_fuel  # No parallel axis for rotation about z
        
        return np.diag([I_xx, I_yy, I_zz])

    def check_feasibility(self, initial_velocity, initial_altitude):
        """
        Analyze if a safe landing is physically possible with current configuration.
        
        Returns:
            is_possible (bool): True if landing is theoretically possible
            report (dict): Detailed metrics (dv_capacity, dv_required, margin)
        """
        # Motor constants
        v_e = self.motor.total_impulse / self.motor.propellant_mass
        t_burn = self.motor.burn_time
        g = self.physics.g
        
        m0 = self.initial_mass
        mf = m0 - self.motor.propellant_mass
        
        # 1. Delta-V Capacity (Tsiolkovsky)
        # We also subtract gravity losses because we are fighting g the whole time
        # DeltaV_effective = ve * ln(m0/mf) - g*t_burn
        dv_gross = v_e * np.log(m0/mf)
        dv_gravity_loss = g * t_burn
        dv_capacity = dv_gross - dv_gravity_loss
        
        # 2. Required Delta-V
        # Energy at impact: 0.5*m*v^2
        # Impact velocity if we just fell: sqrt(v0^2 + 2gh)
        # We need to shed *at least* this much velocity
        v_impact_unpowered = np.sqrt(initial_velocity**2 + 2 * g * initial_altitude)
        
        # 3. Thrust-to-Weight Ratio checks
        # If T/W < 1 at burnout, we can never stop falling
        # Approximate max thrust from constant curve or average
        max_thrust = self.motor.total_impulse / t_burn 
        
        final_weight = mf * g
        max_twr = max_thrust / final_weight
        
        # Margin: dv_capacity - v_impact
        margin = dv_capacity - v_impact_unpowered
        
        is_possible = (margin > 0) and (max_twr > 1.05) # 5% TWR margin
        
        return is_possible, {
            "v_impact_unpowered": v_impact_unpowered,
            "dv_capacity": dv_capacity,
            "dv_gross": dv_gross,
            "dv_gravity_loss": dv_gravity_loss,
            "margin": margin,
            "max_twr": max_twr
        }
        
    def calculate_ignition_altitude(self, initial_velocity, initial_altitude_cg):
        """
        Refined analytical estimate for ignition altitude.
        Accounts for drag using terminal velocity approximation and uses 
        iterative refinement for mass consumption.
        """
        # Motor constants
        v_e = self.motor.total_impulse / self.motor.propellant_mass
        t_burn = self.motor.burn_time
        g = self.physics.g
        m0 = self.initial_mass
        mf = self.initial_mass - self.motor.propellant_mass
        
        # Calculate current nozzle-to-cg offset
        cg_z_body = self.calculate_dynamic_cg(m0)
        nozzle_offset = cg_z_body - self.fuel_tank_bottom
        initial_altitude_nozzle = initial_altitude_cg - nozzle_offset

        # 1. Estimate terminal velocity
        # F_drag = 0.5 * rho * v^2 * Cd * A = mg
        # v_term = sqrt(2mg / (rho * Cd * A))
        rho = self.physics.get_air_density(initial_altitude_cg / 2)
        v_term = np.sqrt((2 * m0 * g) / (rho * self.physics.Cd * self.physics.A_ref))
        
        # 2. Maximum Delta-V capacity
        dv_max = (v_e * np.log(m0/mf)) - (g * t_burn)
        
        # 3. Estimate actual impact speed with drag
        # Using analytical solution for falling with quadratic drag: v^2 = v_term^2 * (1 - exp(-2gh/v_term^2))
        h_fall = initial_altitude_nozzle
        v_impact_sq = v_term**2 * (1 - np.exp(-2 * g * h_fall / v_term**2)) + initial_velocity**2
        v_impact = np.sqrt(max(0, v_impact_sq))
        
        # 4. Decision: Is a safe landing even possible?
        if v_impact > dv_max:
            # IMPOSSIBLE CASE: Rocket will crash. 
            h_ign = (v_impact * t_burn) - (0.5 * (dv_max/t_burn) * t_burn**2)
            return max(0.1, min(initial_altitude_nozzle, h_ign))

        # 5. POSSIBLE CASE: Iterative search for h_ign
        m_avg = (m0 + mf) / 2
        a_thrust_avg = (self.motor.total_impulse / t_burn) / m_avg
        
        # Initial guess (Work-Energy)
        # h_ign = v_impact_at_h_ign^2 / (2 * (a_thrust - g))
        # v_impact_at_h_ign^2 is approx (h_fall - h_ign) / h_fall * v_impact_sq
        h_ign = (v_impact_sq / (2 * a_thrust_avg)) # Crude starting point
        
        # Refine guess
        for _ in range(3):
            # Fall distance to this h_ign
            h_fall_dist = max(0.1, initial_altitude_nozzle - h_ign)
            v_ign_sq = v_term**2 * (1 - np.exp(-2 * g * h_fall_dist / v_term**2)) + initial_velocity**2
            v_ign = np.sqrt(max(0, v_ign_sq))
            
            # Re-estimate required burn duration
            t_req = v_ign / (max(0.1, a_thrust_avg - g))
            m_burn_final = m0 - (self.motor.mass_flow_rate * min(t_burn, t_req))
            m_avg_new = (m0 + m_burn_final) / 2
            a_thrust_new = (self.motor.total_impulse / t_burn) / m_avg_new
            
            # Re-solve: h_ign = v_ign^2 / (2 * (a_thrust_new - g))
            # Avoid dividing by a very small or negative net acceleration.
            denom = a_thrust_new - g
            if denom <= 1e-3:
                # Net acceleration insufficient to arrest descent; be conservative and
                # suggest igniting as early as possible (nozzle altitude)
                return max(0.1, min(initial_altitude_nozzle, initial_altitude_nozzle))
            h_ign = v_ign_sq / (2 * denom)

        # The previous conservative check (ensuring freefall time > burn time) was incorrect for suicide burns.
        # It caused the estimated ignition altitude to be much higher than necessary because it didn't account
        # for the fact that the motor slows the rocket down, extending the flight time for a given distance.
        # We rely on the iterative energy/kinematics calculation above.

        return max(0.1, min(initial_altitude_nozzle, h_ign))

    def tvc_controller(self, state, time, last_time, pitch_int, yaw_int, last_pitch_err, last_yaw_err):
        """
        Stateless TVC PID controller.
        Calculates commanded gimbal angles based on current orientation.
        
        Args:
            state: current state vector
            time: current time
            last_time: time of last update
            pitch_int, yaw_int: current integral errors
            last_pitch_err, last_yaw_err: previous errors
            
        Returns:
            pitch_cmd, yaw_cmd: gimbal commands
            new_pitch_int, new_yaw_int: updated integrals
            pitch_err, yaw_err: current errors
        """
        # Target: stay vertical (identity quaternion)
        # In body frame, Z is up. 
        # We want to minimize the x and y components of the Z-axis in the inertial frame?
        # No, easier: get current orientation and align with [0,0,1]
        
        q = state[6:10]
        vx, vy = state[3], state[4]
        omega_x, omega_y, omega_z = state[10:13]

        # Target setpoints
        target_pitch = 0.0
        target_yaw = 0.0
        
        if self.tvc_mode == 'velocity':
            # Velocity Hold (Drift Correction):
            # Tilt INTO the wind/velocity to generate a counter-acting thrust component.
            # VX+ (East) needs TargetPitch < 0 to get -X thrust component.
            # VY+ (North) needs TargetYaw > 0 to get -Y thrust component.
            target_pitch = -vx * self.tvc_drift_gain
            target_yaw = vy * self.tvc_drift_gain
            
            # Limit target tilt to 15 degrees to prevent loss of control
            max_tilt = np.radians(15.0)
            target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
            target_yaw = np.clip(target_yaw, -max_tilt, max_tilt)

        # Simple attitude control: we want to minimize error relative to targets
        # Pitch tilt is related to qy (2*qy approx pitch in rad)
        # Yaw tilt is related to qx (2*qx approx yaw in rad)
        
        pitch_error = (2 * q[2]) - target_pitch
        yaw_error = (2 * q[1]) - target_yaw
        
        dt = time - last_time if last_time >= 0 else 0.01
        dt = max(dt, 1e-6)
        
        # Update integral terms (with anti-windup)
        max_integral = 0.5  # Limit integral term to prevent windup
        new_pitch_int = pitch_int + pitch_error * dt
        new_pitch_int = np.clip(new_pitch_int, -max_integral, max_integral)
        
        new_yaw_int = yaw_int + yaw_error * dt
        new_yaw_int = np.clip(new_yaw_int, -max_integral, max_integral)
        
        # PID control for pitch (y-axis motor)
        pitch_command = (
            -self.tvc_kp_pitch * pitch_error 
            - self.tvc_ki_pitch * new_pitch_int
            - self.tvc_kd_pitch * omega_y 
        )
        
        # PID control for yaw (x-axis motor)
        yaw_command = (
            -self.tvc_kp_yaw * yaw_error 
            - self.tvc_ki_yaw * new_yaw_int
            - self.tvc_kd_yaw * omega_x 
        )
        
        return pitch_command, yaw_command, new_pitch_int, new_yaw_int, pitch_error, yaw_error
    
    def rcs_controller(self, state, target_quat):
        """Simple PD controller for RCS attitude control"""
        q_curr = state[6:10]
        omega = state[10:13]

        # inv(q) = [w, -x, -y, -z]
        q_inv = np.array([q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]])

        # q_err = q_target * q_inv
        # q_target = [1, 0, 0, 0]
        q_err = q_inv

        # Proportional term
        kp = 200.0 # Increased for faster flip
        kd = 100.0

        # Torque commands (body frame)
        # We want to minimize error. If q_err[2] (qy) is positive, we need torque to reduce it.
        torque_y = kp * q_err[2] - kd * omega[1]
        torque_x = kp * q_err[1] - kd * omega[0]

        # Map torque to thruster commands [pitch_pos, pitch_neg, yaw_pos, yaw_neg]
        commands = np.zeros(4)
        if torque_y > 0.1: commands[0] = 1.0
        elif torque_y < -0.1: commands[1] = 1.0

        if torque_x > 0.1: commands[2] = 1.0
        elif torque_x < -0.1: commands[3] = 1.0

        return commands

    def get_rcs_torque(self, cg_location):
        """
        Calculate torque from RCS thrusters in body frame

        Args:
            cg_location: current CG z-coordinate in body frame

        Returns:
            torque: 3D torque vector in body frame
        """
        if not self.rcs_enabled:
            return np.zeros(3)

        # rcs_commands: [pitch_pos, pitch_neg, yaw_pos, yaw_neg]
        # Leverage arm from CG to RCS (RCS near nose)
        arm_z = self.rcs_arm - cg_location

        # Pitch torque (about Y): force in X
        # Pitch_pos (Pitch UP) -> Force in +X -> Positive Y torque (arm_z * f_x)
        f_x = (self.rcs_commands[0] - self.rcs_commands[1]) * self.rcs_thrust

        # Yaw torque (about X): force in Y
        # Yaw_pos -> Torque_x > 0 -> Force in -Y (Torque = r x F = [0,0,z] x [0, -f, 0] = [z*f, 0, 0])
        f_y = (self.rcs_commands[3] - self.rcs_commands[2]) * self.rcs_thrust

        return np.array([-arm_z * f_y, arm_z * f_x, 0.0])

    def state_derivative(self, t, state):
        """
        Calculate state derivative for integration
        
        State vector:
        [x, y, z, vx, vy, vz, qw, qx, qy, qz, omega_x, omega_y, omega_z, mass]
        
        Args:
            t: time
            state: state vector
            
        Returns:
            derivative: state derivative
        """
        # Extract state
        position = state[0:3]
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        mass = state[13]
        
        # Normalize quaternion
        quaternion = self.physics.normalize_quaternion(quaternion)
        
        # Rotation matrix (body to inertial)
        R_body_to_inertial = self.physics.quaternion_to_rotation_matrix(quaternion)
        
        # Calculate dynamic CG and inertia if enabled
        cg_location = self.calculate_dynamic_cg(mass)
        current_inertia_tensor = self.calculate_dynamic_inertia(mass, cg_location)
        
        # CG offset from thrust point (accounting for dynamic CG)
        # Thrust point is at bottom of rocket
        cg_offset_from_thrust = np.array([0, 0, cg_location - self.fuel_tank_bottom])
        
        # Forces in inertial frame
        # Gravity
        F_gravity = np.array([0, 0, -mass * self.physics.g])
        
        # Drag
        F_drag = self.physics.get_drag_force(velocity, position, t)
        
        # Thrust
        F_thrust = self.motor.get_thrust_vector(t, R_body_to_inertial)
        
        # Total force
        F_total = F_gravity + F_drag + F_thrust
        
        # Linear acceleration
        acceleration = F_total / mass if mass > 0 else np.zeros(3)
        
        # Moments in body frame
        # Thrust moment from TVC (using dynamic CG offset)
        M_thrust = self.motor.get_thrust_moment(t, cg_offset_from_thrust)
        
        # RCS Torque
        M_rcs = self.get_rcs_torque(cg_location)

        # Aerodynamic moment (Stability from CP + Damping)
        M_aero_stability = self.physics.get_aero_torque(velocity, position, quaternion, t, cg_location)

        # Aerodynamic damping
        M_aero_damping = -0.5 * angular_velocity

        M_aero = M_aero_stability + M_aero_damping
        
        # Total moment
        M_total = M_thrust + M_rcs + M_aero
        
        # Angular acceleration (Euler's equation: I*ω̇ + ω × (I*ω) = M)
        I_omega = current_inertia_tensor @ angular_velocity
        omega_cross_I_omega = np.cross(angular_velocity, I_omega)
        angular_acceleration = np.linalg.solve(current_inertia_tensor, M_total - omega_cross_I_omega)
        
        # Quaternion derivative
        # q̇ = 0.5 * q ⊗ [0, ω]
        omega_quat = np.array([0, angular_velocity[0], angular_velocity[1], angular_velocity[2]])
        q_dot_quat = self.physics.quaternion_multiply(quaternion, omega_quat)
        q_dot = 0.5 * q_dot_quat
        
        # Mass derivative
        mass_dot = -self.motor.get_mass_flow_rate(t)
        
        # Assemble derivative
        derivative = np.concatenate([
            velocity,
            acceleration,
            q_dot,
            angular_acceleration,
            [mass_dot]
        ])
        
        return derivative
    
    def run_simulation(self, initial_state, ignition_altitude=None, max_time=60.0):
        """
        Run a single simulation
        """
        # Ensure fresh motor for every run
        from solid_motor import SolidMotor
        self.motor = SolidMotor(self.rocket_config)
        
        # Results containers
        t_ascent = np.array([])
        y_ascent = np.empty((14, 0))
        
        # Prepare running state
        current_sim_state = initial_state.copy()
        current_time_offset = 0.0
        
        # Determine Body frame offsets
        # fuel_tank_bottom is -length/2
        initial_mass = initial_state[13] if not self.simulate_ascent else (self.dry_mass + self.propellant_mass + self.ascent_motor_casing_mass + self.motor.propellant_mass)
        initial_cg_body = self.calculate_dynamic_cg(initial_mass)
        # z_cg_local = cg_z_body - fuel_tank_bottom (height above nozzle)
        z_cg_offset_local = initial_cg_body - self.fuel_tank_bottom
        
        # PHASE 1: ASCENT (Optional)
        if self.simulate_ascent:
            # Use Ascent Configuration for Orientation
            q_start = self.physics.euler_to_quaternion(
                np.radians(self.ascent_initial_roll),
                np.radians(self.ascent_initial_pitch),
                np.radians(self.ascent_initial_yaw)
            )
            # Update initial state orientation
            current_sim_state[6:10] = q_start
            # Offset position such that Z in initial_state refers to NOZZLE
            # pos_cg = pos_nozzle + R @ [0,0,offset]
            R_start = self.physics.quaternion_to_rotation_matrix(q_start)
            pos_nozzle = initial_state[0:3]
            current_sim_state[0:3] = pos_nozzle + R_start @ np.array([0, 0, z_cg_offset_local])
            
            # Run Ascent
            apogee_state, t_asc, y_asc = self.run_ascent_phase(current_sim_state)
            
            # Transition to Descent
            # 1. Eject Ascent Motor Casing -> Mass drops to Descent Mass
            descent_start_mass = self.dry_mass + self.propellant_mass
            
            # State Handover
            state_for_descent = apogee_state.copy()
            state_for_descent[13] = descent_start_mass # Reset mass
            
            # Store ascent data
            t_ascent = t_asc
            y_ascent = y_asc
            
            # Reset Motor for Descent
            self.motor = SolidMotor(self.rocket_config)
            
            # Update running variables
            current_sim_state = state_for_descent
            current_time_offset = t_asc[-1]
            
        else:
            # Direct Descent Simulation
            q_start = self.physics.euler_to_quaternion(
                np.radians(self.descent_initial_roll),
                np.radians(self.descent_initial_pitch),
                np.radians(self.descent_initial_yaw)
            )
            current_sim_state[6:10] = q_start
            # Offset position
            R_start = self.physics.quaternion_to_rotation_matrix(q_start)
            pos_nozzle = initial_state[0:3]
            current_sim_state[0:3] = pos_nozzle + R_start @ np.array([0, 0, z_cg_offset_local])
            current_time_offset = 0.0

        # Reset RCS state
        self.rcs_enabled = False
        self.rcs_commands = np.zeros(4)

        # PHASE 2: DESCENT / FLIP / SUICIDE BURN
        
        # Calculate ignition parameters based on CURRENT state (at apogee or start)
        current_alt = current_sim_state[2]
        current_vel = current_sim_state[5]
        
        if ignition_altitude is None:
            raw_ignition_alt = self.calculate_ignition_altitude(current_vel, current_alt)
            # Apply offsets: h_new = h_old * (1 + %) + hard
            ignition_altitude = raw_ignition_alt * (1.0 + self.ignition_percent_offset) + self.ignition_hard_offset
        
        # Add sensor noise to ignition altitude CHECK
        ignition_altitude_sensed = ignition_altitude * (1.0 + np.random.uniform(
            -self.altimeter_error, self.altimeter_error))
        
        # Event: motor ignition
        class Event:
            def __init__(self, func, terminal: bool = False, direction: int = 0):
                self._func = func
                self.terminal = terminal
                self.direction = direction
            def __call__(self, t, y):
                return self._func(t, y)
        
        # Flip Maneuver Altitude (e.g. 2x ignition altitude or 250m above it)
        flip_altitude = ignition_altitude + 250.0

        def _flip_event_func(t, state):
            vz = state[5]
            if vz > -0.1: return 1.0
            R_mat = self.physics.quaternion_to_rotation_matrix(state[6:10])
            off_local = self.calculate_dynamic_cg(state[13]) - self.fuel_tank_bottom
            pos_n = state[0:3] - R_mat @ np.array([0, 0, off_local])
            alt_n = pos_n[2]
            return alt_n - flip_altitude

        flip_event = Event(_flip_event_func, terminal=True, direction=-1)

        def _ignition_event_func(t, state):
            # Only trigger if descending (vz < -0.1)
            # Using a small negative threshold to ensure we are clearly falling
            vz = state[5]
            if vz > -0.1:
                return 1.0  # Return positive value while rising/peak to avoid crossing zero
            
            # Compare nozzle altitude (not CG altitude) to the sensed ignition altitude
            R_mat = self.physics.quaternion_to_rotation_matrix(state[6:10])
            off_local = self.calculate_dynamic_cg(state[13]) - self.fuel_tank_bottom
            pos_n = state[0:3] - R_mat @ np.array([0, 0, off_local])
            return pos_n[2] - ignition_altitude_sensed
        
        ignition_event = Event(_ignition_event_func, terminal=True, direction=-1)
         
        # Event: ground contact (using Nozzle position)
        def _ground_event_func(t, state):
            R_mat = self.physics.quaternion_to_rotation_matrix(state[6:10])
            off_local = self.calculate_dynamic_cg(state[13]) - self.fuel_tank_bottom
            pos_n = state[0:3] - R_mat @ np.array([0, 0, off_local])
            return pos_n[2]  # Nozzle altitude
        
        ground_event = Event(_ground_event_func, terminal=True, direction=-1)
        
        # stop_climb_event removed: avoid terminating the entire integration on small positive vz
        
        # Integrate until ignition or ground
        
        # Wrapper to handle time offset for 'state_derivative' if it depended on absolute time (it currently doesn't, but good practice)
        # But wait, solve_ivp works with relative time chunks usually, but we want continuous history.
        # We will pass absolute time to solve_ivp, starting from current_time_offset
        
        # 2a. Freefall until flip altitude
        def freefall_derivative(t, state):
            self.rcs_enabled = False
            self.rcs_commands = np.zeros(4)
            return self.state_derivative(t, state)

        sol_preflip = solve_ivp(
            freefall_derivative,
            [current_time_offset, current_time_offset + max_time],
            current_sim_state,
            events=[flip_event, ignition_event, ground_event],
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=0.1
        )

        t_parts = [sol_preflip.t]
        y_parts = [sol_preflip.y]
        rcs_active_parts = [np.zeros((4, len(sol_preflip.t)))]

        current_sim_state = sol_preflip.y[:, -1]
        current_time_offset = sol_preflip.t[-1]

        # Check if we should start flip
        if len(sol_preflip.t_events[0]) > 0:
            # Reached flip altitude
            self.rcs_enabled = True

            def flip_derivative(t, state):
                self.rcs_commands = self.rcs_controller(state, np.array([1, 0, 0, 0]))
                return self.state_derivative(t, state)

            sol_flip = solve_ivp(
                flip_derivative,
                [current_time_offset, current_time_offset + max_time],
                current_sim_state,
                events=[ignition_event, ground_event],
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                max_step=0.1
            )

            # Store RCS commands used during flip
            rcs_history = np.zeros((4, len(sol_flip.t)))
            for i in range(len(sol_flip.t)):
                rcs_history[:, i] = self.rcs_controller(sol_flip.y[:, i], np.array([1, 0, 0, 0]))

            t_parts.append(sol_flip.t)
            y_parts.append(sol_flip.y)
            rcs_active_parts.append(rcs_history)

            current_sim_state = sol_flip.y[:, -1]
            current_time_offset = sol_flip.t[-1]

            # Check if motor ignited
            if len(sol_flip.t_events[0]) > 0:
                ignited = True
                ignition_time = sol_flip.t_events[0][0]
                state_at_ignition = sol_flip.y_events[0][0]
            else:
                ignited = False
        else:
            # Check if motor ignited or hit ground before flip altitude
            if len(sol_preflip.t_events[1]) > 0:
                ignited = True
                ignition_time = sol_preflip.t_events[1][0]
                state_at_ignition = sol_preflip.y_events[1][0]
            else:
                ignited = False

        # PHASE 3: POWERED DESCENT
        if ignited:
            # Ignite motor and reset PID integral terms
            self.motor.ignite(ignition_time)

            # Keep RCS enabled for stability during ignition transition?
            # Or turn it off? User said "turns itself back around... before the main motor ignites".
            # Let's turn it off for the main burn to avoid interference with TVC,
            # but maybe keep it for a split second. Actually, let's just disable it.
            self.rcs_enabled = False
            self.rcs_commands = np.zeros(4)
            
            # Ignite motor and reset PID integral terms
            self.motor.ignite(ignition_time)
            
            # We use a stateful wrapper that only updates state if time progresses.
            # This is a compromise given the 14-element state vector constraint.
            self.last_time = ignition_time
            self.pitch_integral_error = 0.0
            self.yaw_integral_error = 0.0
            self.last_pitch_error = 0.0
            self.last_yaw_error = 0.0
            
            def burning_state_derivative(t, state):
                # Update TVC controller if moving forward
                # Note: solve_ivp may query t slightly behind last_time due to RK stages
                # We only update the "real" internal state of the PID if it's a new time step.
                if t > self.last_time:
                    p_cmd, y_cmd, p_int, y_int, p_err, y_err = self.tvc_controller(
                        state, t, self.last_time, 
                        self.pitch_integral_error, self.yaw_integral_error,
                        self.last_pitch_error, self.last_yaw_error
                    )
                    # For the derivative calculation, we use these updated values
                    self.motor.set_tvc_command(p_cmd, y_cmd)
                    
                    # Store for next step
                    self.pitch_integral_error = p_int
                    self.yaw_integral_error = y_int
                    self.last_pitch_error = p_err
                    self.last_yaw_error = y_err
                    
                    # Update TVC actuator
                    dt = t - self.last_time
                    self.motor.update_tvc(dt)
                    self.last_time = t
                else:
                    # Keep same command for intermediate/backward steps
                    pass
                
                return self.state_derivative(t, state)
            
            sol_powered = solve_ivp(
                burning_state_derivative,
                [ignition_time, ignition_time + max_time],
                state_at_ignition,
                events=[ground_event],
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                max_step=0.1
            )
            
            t_parts.append(sol_powered.t)
            y_parts.append(sol_powered.y)
            rcs_active_parts.append(np.zeros((4, len(sol_powered.t))))

            t_descent = np.concatenate(t_parts)
            y_descent = np.concatenate(y_parts, axis=1)
            rcs_history_full = np.concatenate(rcs_active_parts, axis=1)

            # If the powered phase ended because it hit the configured time limit (no ground_event),
            # try extending the integration until ground is reached.
            try:
                ground_triggered = len(sol_powered.t_events[0]) > 0
            except Exception:
                ground_triggered = False

            t_end_expected = current_time_offset + max_time
            # If the powered phase completed without hitting the ground, always attempt
            # to continue integrating until `ground_event` is triggered. Previously this
            # only happened when the powered phase hit the overall time limit which
            # could leave the simulation terminating in mid-air after a burnout.
            if not ground_triggered:
                extend_sol = solve_ivp(
                    self.state_derivative,
                    [sol_powered.t[-1], sol_powered.t[-1] + max_time],
                    sol_powered.y[:, -1],
                    events=[ground_event],
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-9,
                    max_step=0.1
                )
                if extend_sol is not None and extend_sol.t.size > 1:
                    t_descent = np.concatenate([t_descent, extend_sol.t[1:]])
                    y_descent = np.concatenate([y_descent, extend_sol.y[:, 1:]], axis=1)
        else:
            # No ignition (hit ground before ignition altitude)
            t_descent = np.concatenate(t_parts)
            y_descent = np.concatenate(y_parts, axis=1)
            rcs_history_full = np.concatenate(rcs_active_parts, axis=1)

            # If freefall ended due to time limit without reaching ground, extend to try to reach ground
            try:
                ground_triggered_free = len(sol_freefall.t_events[1]) > 0
            except Exception:
                ground_triggered_free = False

            t_end_expected_free = current_time_offset + max_time
            # If freefall completed without reaching ground, always attempt to extend
            # the integration until ground is reached rather than only when the time
            # limit was hit. This prevents the simulation from stopping in mid-air
            # (e.g., right after burnout).
            if not ground_triggered_free:
                extend_sol = solve_ivp(
                    self.state_derivative,
                    [sol_freefall.t[-1], sol_freefall.t[-1] + max_time],
                    sol_freefall.y[:, -1],
                    events=[ground_event],
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-9,
                    max_step=0.1
                )
                if extend_sol is not None and extend_sol.t.size > 1:
                    t_descent = np.concatenate([t_descent, extend_sol.t[1:]])
                    y_descent = np.concatenate([y_descent, extend_sol.y[:, 1:]], axis=1)
        
        # STITCH ASCENT AND DESCENT HISTORY
        if len(t_ascent) > 0:
            # Avoid duplicate time points
            t_combined = np.concatenate([t_ascent, t_descent[1:]])
            y_combined = np.concatenate([y_ascent, y_descent[:, 1:]], axis=1)
        else:
            t_combined = t_descent
            y_combined = y_descent
            
        # CONVERT TO NOZZLE POSITION FOR VISUALIZATION
        x_nozzle = np.zeros_like(t_combined)
        y_nozzle = np.zeros_like(t_combined)
        z_nozzle = np.zeros_like(t_combined)
        
        for i in range(len(t_combined)):
            R_mat = self.physics.quaternion_to_rotation_matrix(y_combined[6:10, i])
            off_local = self.calculate_dynamic_cg(y_combined[13, i]) - self.fuel_tank_bottom
            pos_n = y_combined[0:3, i] - R_mat @ np.array([0, 0, off_local])
            x_nozzle[i], y_nozzle[i], z_nozzle[i] = pos_n
            
        # Extract final state
        final_state = y_combined[:, -1]
        final_altitude = z_nozzle[-1] 
        final_velocity = y_combined[3:6, -1]
        final_speed = np.linalg.norm(final_velocity)
        
        # Check success criteria
        altitude_ok = abs(final_altitude) < 1.0
        velocity_ok = abs(final_velocity[2]) < 2.0
        total_velocity_ok = final_speed < 3.0
        
        success = altitude_ok and velocity_ok and total_velocity_ok
        
        # Build history
        history = {
            't': t_combined,
            'x': x_nozzle,
            'y': y_nozzle,
            'z': z_nozzle,
            'vx': y_combined[3, :],
            'vy': y_combined[4, :],
            'vz': y_combined[5, :],
            'qw': y_combined[6, :],
            'qx': y_combined[7, :],
            'qy': y_combined[8, :],
            'qz': y_combined[9, :],
            'omega_x': y_combined[10, :],
            'omega_y': y_combined[11, :],
            'wx': y_combined[10, :],
            'wy': y_combined[11, :],
            'wz': y_combined[12, :],
            'omega_x': y_combined[10, :], # Backward compatibility
            'omega_y': y_combined[11, :],
            'omega_z': y_combined[12, :],
            'mass': y_combined[13, :],
            'thrust': np.array([self.motor.get_thrust(t) for t in t_combined]),
            'rcs_p_pos': rcs_history_full[0, :],
            'rcs_p_neg': rcs_history_full[1, :],
            'rcs_y_pos': rcs_history_full[2, :],
            'rcs_y_neg': rcs_history_full[3, :],
            'success': success,
            'final_altitude': final_altitude,
            'final_velocity': final_speed,
            'ignition_altitude': ignition_altitude
        }
        
        self.history = history
        
        return success, final_state, history
    
    def optimize_ignition_altitude(self, initial_state, num_monte_carlo=100, 
                                   altitude_search_range=10.0, altitude_step=0.1,
                                   progress_callback=None,
                                   save_each_trial=False, results_folder=None, save_plots_per_trial=False):
        """
        Optimize ignition altitude using Monte Carlo simulation
        
        Args:
            initial_state: initial state vector
            num_monte_carlo: number of Monte Carlo runs per altitude
            altitude_search_range: range to search around analytical estimate (m)
            altitude_step: step size for altitude search (m)
            progress_callback: optional callable(progress_completed, total)
            save_each_trial: if True, save raw data (CSV) for every individual trial
            results_folder: path to a folder where per-run files will be saved (required if save_each_trial=True)
            save_plots_per_trial: if True and save_each_trial=True, generate per-trial PNG plots
            
        Returns:
            optimal_altitude: best ignition altitude (m)
            success_rates: dict with altitude -> success rate mapping
            best_history: history from best run
        """
        # Calculate analytical estimate
        if self.simulate_ascent:
            # Run a nominal ascent (no noise) to find expected apogee
            # Setup initial rotation
            q_start = self.physics.euler_to_quaternion(
                np.radians(self.ascent_initial_roll),
                np.radians(self.ascent_initial_pitch),
                np.radians(self.ascent_initial_yaw)
            )
            temp_state = initial_state.copy()
            temp_state[6:10] = q_start
            
            # Record current motor mass so we don't mess up state (run_ascent_phase creates its own motor)
            apogee_state, _, _ = self.run_ascent_phase(temp_state)
            v_for_est = apogee_state[5]
            h_for_est = apogee_state[2]
            print(f"Nominal apogee: {h_for_est:.2f} m, velocity: {v_for_est:.2f} m/s")
        else:
            v_for_est = initial_state[5]  # vz
            h_for_est = initial_state[2]  # z
        
        estimate = self.calculate_ignition_altitude(v_for_est, h_for_est)
        
        print(f"Analytical ignition altitude estimate: {estimate:.2f} m")
        
        # Search around estimate
        altitudes = np.arange(
            max(0, estimate - altitude_search_range),
            estimate + altitude_search_range + altitude_step,
            altitude_step
        )
        
        success_rates = {}
        best_altitude = estimate
        best_success_rate = 0.0
        best_history = None
        
        total_runs = len(altitudes) * int(num_monte_carlo)
        runs_completed = 0
        
        trials_dir = None
        if save_each_trial:
            if not results_folder:
                raise ValueError("results_folder must be provided when save_each_trial=True")
            trials_dir = os.path.join(results_folder, 'trials')
            os.makedirs(trials_dir, exist_ok=True)

        for altitude in altitudes:
            successes = 0
            histories = []
            
            for i in range(int(num_monte_carlo)):
                # Run simulation
                success, final_state, history = self.run_simulation(
                    initial_state.copy(), altitude
                )
                
                if success:
                    successes += 1
                    histories.append(history)

                # Save per-trial raw data if requested
                if save_each_trial and trials_dir is not None:
                    # Safe altitude string for filenames
                    alt_str = f"{altitude:.2f}".replace('.', 'p')
                    idx_str = f"{i+1:04d}"
                    status = 'success' if success else 'fail'
                    csv_name = os.path.join(trials_dir, f"trial_alt{alt_str}_idx{idx_str}_{status}.csv")
                    try:
                        with open(csv_name, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass', 'AltitudeTest'])
                            for k in range(len(history['t'])):
                                writer.writerow([
                                    history['t'][k],
                                    history['x'][k],
                                    history['y'][k],
                                    history['z'][k],
                                    history['vx'][k],
                                    history['vy'][k],
                                    history['vz'][k],
                                    history['qw'][k],
                                    history['qx'][k],
                                    history['qy'][k],
                                    history['qz'][k],
                                    history['mass'][k],
                                    altitude
                                ])
                    except Exception as e:
                        print(f"Could not save trial CSV {csv_name}: {e}")

                    # Optionally save a small plot (Altitude vs Time)
                    if save_plots_per_trial:
                        png_name = os.path.join(trials_dir, f"trial_alt{alt_str}_idx{idx_str}_{status}.png")
                        try:
                            plt.figure(figsize=(6, 3))
                            plt.plot(history['t'], history['z'], 'b-')
                            plt.xlabel('Time (s)')
                            plt.ylabel('Altitude (m)')
                            plt.title(f'Trial {idx_str} (alt {altitude:.2f})')
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(png_name, dpi=100, bbox_inches='tight')
                            plt.close()
                        except Exception as e:
                            print(f"Could not save trial PNG {png_name}: {e}")

                # Update progress
                runs_completed += 1
                if progress_callback is None:
                    # Console progress
                    print(f"\rRunning optimization: {runs_completed}/{total_runs} simulations", end='', flush=True)
                else:
                    try:
                        progress_callback(runs_completed, total_runs)
                    except Exception:
                        print("MishraPy Runtime: An error occured at line 529 in simulation.py during [AI] progress callback. [AI]")
            
            success_rate = successes / float(num_monte_carlo)
            success_rates[altitude] = success_rate
            
            # Ensure console has a newline after printing inline progress
            if progress_callback is None:
                print('')

            print(f"Altitude {altitude:.1f} m: {success_rate*100:.1f}% success rate")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_altitude = altitude
                if len(histories) > 0:
                    best_history = histories[0]        
        print(f"\nOptimal ignition altitude: {best_altitude:.2f} m "
              f"({best_success_rate*100:.1f}% success rate)")
        
        return best_altitude, success_rates, best_history

    def optimize_ignition_altitude_adaptive(self, initial_state, num_monte_carlo=10,
                                            altitude_search_range=10.0,
                                            max_iterations=3, samples_per_step=10,
                                            target_step=0.01,
                                            progress_callback=None,
                                            save_each_trial=False, results_folder=None, 
                                            save_plots_per_trial=False):
        """
        Adaptive optimization for ignition altitude.
        Iteratively narrows the search range based on the trend of final velocity.
        
        Args:
            initial_state: initial state vector
            num_monte_carlo: runs per altitude in the final refinement stage
            altitude_search_range: initial range to search around analytical estimate
            max_iterations: maximum number of refinement steps
            samples_per_step: number of altitudes to test in each zoom-in step
            target_step: stop refining if step size is smaller than this
            progress_callback: optional callback(completed, total)
            save_each_trial: save CSVs for every run?
            results_folder: folder for saving trials
            
        Returns:
            optimal_altitude, convergence_history
        """
        # Calculate analytical estimate
        if self.simulate_ascent:
            q_start = self.physics.euler_to_quaternion(
                np.radians(self.ascent_initial_roll),
                np.radians(self.ascent_initial_pitch),
                np.radians(self.ascent_initial_yaw)
            )
            temp_state = initial_state.copy()
            temp_state[6:10] = q_start
            apogee_state, _, _ = self.run_ascent_phase(temp_state)
            v_for_est = apogee_state[5]
            h_for_est = apogee_state[2]
        else:
            v_for_est = initial_state[5]
            h_for_est = initial_state[2]
        
        center_alt = self.calculate_ignition_altitude(v_for_est, h_for_est)
        current_range = altitude_search_range
        
        convergence_history = []
        best_overall_altitude = center_alt
        best_overall_velocity = float('inf')
        best_overall_history = None
        
        # Total runs calculation: (max_iterations-1) * samples * 5 + 1 * samples * num_monte_carlo
        total_estimated_runs = (max_iterations - 1) * samples_per_step * 5 + samples_per_step * int(num_monte_carlo)
        runs_completed = 0

        trials_dir = None
        if save_each_trial:
            if not results_folder:
                raise ValueError("results_folder must be provided when save_each_trial=True")
            trials_dir = os.path.join(results_folder, 'trials')
            os.makedirs(trials_dir, exist_ok=True)

        for iteration in range(max_iterations):
            step_size = (current_range * 2) / (samples_per_step - 1)
            if step_size < target_step / 2:
                break
                
            alt_to_test = np.linspace(
                max(0.1, center_alt - current_range),
                center_alt + current_range,
                samples_per_step
            )
            
            print(f"\nAdaptive Iteration {iteration+1}/{max_iterations}: Range [{alt_to_test[0]:.2f}, {alt_to_test[-1]:.2f}], Step {step_size:.3f}")
            
            iteration_results = []
            for alt in alt_to_test:
                test_runs = 5 if iteration < max_iterations - 1 else num_monte_carlo
                
                final_velocities = []
                for i in range(int(test_runs)):
                    success, final_state, history = self.run_simulation(initial_state.copy(), alt)
                    # Metric is final vertical velocity absolute
                    final_velocities.append(abs(final_state[5]))
                    
                    # Save per-trial raw data if requested
                    if save_each_trial and trials_dir is not None:
                        alt_str = f"{alt:.2f}".replace('.', 'p')
                        idx_str = f"iter{iteration+1}_{i+1:03d}"
                        status = 'success' if success else 'fail'
                        csv_name = os.path.join(trials_dir, f"trial_alt{alt_str}_{idx_str}_{status}.csv")
                        try:
                            with open(csv_name, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'QW', 'QX', 'QY', 'QZ', 'Mass', 'AltitudeTest'])
                                for k in range(len(history['t'])):
                                    writer.writerow([
                                        history['t'][k], history['x'][k], history['y'][k], history['z'][k],
                                        history['vx'][k], history['vy'][k], history['vz'][k],
                                        history['qw'][k], history['qx'][k], history['qy'][k], history['qz'][k],
                                        history['mass'][k], alt
                                    ])
                        except Exception as e:
                            print(f"Could not save adaptive trial CSV {csv_name}: {e}")

                        if save_plots_per_trial:
                            png_name = os.path.join(trials_dir, f"trial_alt{alt_str}_{idx_str}_{status}.png")
                            try:
                                plt.figure(figsize=(6, 3))
                                plt.plot(history['t'], history['z'], 'b-')
                                plt.xlabel('Time (s)')
                                plt.ylabel('Altitude (m)')
                                plt.title(f'Iter {iteration+1} alt {alt:.2f}')
                                plt.grid(True)
                                plt.tight_layout()
                                plt.savefig(png_name, dpi=100, bbox_inches='tight')
                                plt.close()
                            except Exception:
                                pass

                    runs_completed += 1
                    if progress_callback:
                        progress_callback(runs_completed, total_estimated_runs)
                    else:
                        print(f"\rProgress: {runs_completed}/{total_estimated_runs} simulations", end='')

                avg_vel = np.mean(final_velocities)
                iteration_results.append((alt, avg_vel))
                print(f"\n  Alt {alt:.2f} m -> Avg Final Velocity: {avg_vel:.2f} m/s")
                if avg_vel < best_overall_velocity:
                    best_overall_velocity = avg_vel
                    best_overall_altitude = alt
                    # We store the latest history of the best altitude as an example
                    best_overall_history = history
            
            # Sort and find best in this iteration
            iteration_results.sort(key=lambda x: x[1])
            best_iter_alt, best_iter_vel = iteration_results[0]
            
            convergence_history.append({
                'iteration': iteration,
                'range': (alt_to_test[0], alt_to_test[-1]),
                'best_alt': best_iter_alt,
                'best_vel': best_iter_vel
            })
            
            # Zoom in: new center is the best alt, new range is reduced
            center_alt = best_iter_alt
            current_range = current_range / (samples_per_step / 2.0) # Zoom factor
            
        print(f"\nAdaptive optimization converged to {best_overall_altitude:.3f} m")
        return best_overall_altitude, convergence_history, best_overall_history
