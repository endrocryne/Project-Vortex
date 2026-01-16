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
        
        # Integral error accumulators
        self.pitch_integral_error = 0.0
        self.yaw_integral_error = 0.0
        self.last_time = 0.0
        
        # Sensor accuracy (Monte Carlo)
        self.altimeter_error = simulation_config.get('altimeter_error', 0.0)
        self.velocity_sensor_error = simulation_config.get('velocity_sensor_error', 0.0)
        
        # Results storage
        self.history = None
    
    def calculate_dynamic_cg(self, current_mass):
        """
        Calculate center of gravity location as fuel burns
        
        Args:
            current_mass: current vehicle mass (kg)
            
        Returns:
            cg_location: CG location in body frame (m, along z-axis)
        """
        if not self.use_dynamic_inertia:
            return 0.0
        
        # Calculate remaining fuel mass
        fuel_remaining = current_mass - self.dry_mass
        fuel_remaining = max(0.0, min(fuel_remaining, self.propellant_mass))
        
        # Fuel CG (assuming uniform distribution in tank)
        fuel_fraction = fuel_remaining / self.propellant_mass if self.propellant_mass > 0 else 0.0
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
        
    def calculate_ignition_altitude(self, initial_velocity, initial_altitude):
        """
        Calculate analytical estimate for ignition altitude
        
        Uses kinematic equation: v^2 = v0^2 + 2*a*(h - h0)
        Solve for h when v = 0 at burnout
        
        Args:
            initial_velocity: initial vertical velocity (m/s, negative = falling)
            initial_altitude: initial altitude (m)
            
        Returns:
            ignition_altitude: estimated altitude to ignite motor (m)
        """
        # Average thrust during burn
        avg_thrust = self.motor.total_impulse / self.motor.burn_time
        
        # Average mass during burn
        avg_mass = self.initial_mass - self.motor.propellant_mass / 2
        
        # Net acceleration (thrust - weight - drag)
        # Simplified: ignore drag for initial estimate
        a_net = avg_thrust / avg_mass - self.physics.g
        
        # Distance traveled during burn to reach v=0
        # v^2 = v0^2 + 2*a*d  =>  d = -v0^2 / (2*a)
        if abs(a_net) < 0.1:
            # Not enough thrust to decelerate
            return 0.0
        
        v0 = abs(initial_velocity)
        distance_to_stop = v0**2 / (2 * a_net)
        
        # Ignition altitude = current altitude - distance to stop
        ignition_altitude = max(0.0, initial_altitude - distance_to_stop)
        
        return ignition_altitude
    
    def tvc_controller(self, state, time):
        """
        TVC PID controller for attitude stabilization
        Separate gains for pitch (y-axis) and yaw (x-axis)
        
        Args:
            state: current state vector
            time: current time
            
        Returns:
            pitch_command, yaw_command: TVC angles (radians)
        """
        # Extract state
        qw, qx, qy, qz = state[6:10]
        omega_x, omega_y, omega_z = state[10:13]
        
        # Calculate time step
        dt = time - self.last_time if self.last_time > 0 else 0.01
        dt = max(dt, 1e-6)  # Prevent division by zero
        self.last_time = time
        
        # Target: vertical orientation (pointing up)
        # Target quaternion: [1, 0, 0, 0]
        
        # Error quaternion (simplified - just use rotation components)
        # For small angles: pitch ≈ 2*qy, yaw ≈ 2*qx
        pitch_error = 2 * qy
        yaw_error = 2 * qx
        
        # Update integral terms (with anti-windup)
        max_integral = 0.5  # Limit integral term to prevent windup
        self.pitch_integral_error += pitch_error * dt
        self.pitch_integral_error = np.clip(self.pitch_integral_error, -max_integral, max_integral)
        
        self.yaw_integral_error += yaw_error * dt
        self.yaw_integral_error = np.clip(self.yaw_integral_error, -max_integral, max_integral)
        
        # PID control for pitch (y-axis motor)
        pitch_command = (
            -self.tvc_kp_pitch * pitch_error 
            - self.tvc_ki_pitch * self.pitch_integral_error
            - self.tvc_kd_pitch * omega_y
        )
        
        # PID control for yaw (x-axis motor)
        yaw_command = (
            -self.tvc_kp_yaw * yaw_error 
            - self.tvc_ki_yaw * self.yaw_integral_error
            - self.tvc_kd_yaw * omega_x
        )
        
        return pitch_command, yaw_command
    
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
        
        # Aerodynamic moment (simplified - stabilizing)
        omega_body = angular_velocity
        M_aero = -0.1 * omega_body
        
        # Total moment
        M_total = M_thrust + M_aero
        
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
    
    def run_simulation(self, initial_state, ignition_altitude, max_time=30.0):
        """
        Run a single simulation
        
        Args:
            initial_state: initial state vector
            ignition_altitude: altitude at which to ignite motor (m)
            max_time: maximum simulation time (s)
            
        Returns:
            success: True if landing was successful
            final_state: final state at touchdown or timeout
            history: dict with time history of all variables
        """
        # Add sensor noise to ignition altitude
        ignition_altitude_sensed = ignition_altitude * (1.0 + np.random.uniform(
            -self.altimeter_error, self.altimeter_error))
        
        # Event: motor ignition
        def ignition_event(t, state):
            altitude = state[2]
            return altitude - ignition_altitude_sensed
        ignition_event.terminal = False
        ignition_event.direction = -1  # Trigger when decreasing
        
        # Event: ground contact
        def ground_event(t, state):
            return state[2]  # altitude
        ground_event.terminal = True
        ground_event.direction = -1
        
        # Integrate until ignition or ground
        sol_freefall = solve_ivp(
            self.state_derivative,
            [0, max_time],
            initial_state,
            events=[ignition_event, ground_event],
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            max_step=0.01
        )
        
        # Check if motor should ignite
        if len(sol_freefall.t_events[0]) > 0:
            # Motor ignited
            ignition_time = sol_freefall.t_events[0][0]
            state_at_ignition = sol_freefall.y_events[0][0]
            
            # Ignite motor and reset PID integral terms
            self.motor.ignite(ignition_time)
            self.pitch_integral_error = 0.0
            self.yaw_integral_error = 0.0
            self.last_time = ignition_time
            
            # Continue simulation with motor burning
            def burning_state_derivative(t, state):
                # Update TVC controller
                pitch_cmd, yaw_cmd = self.tvc_controller(state, t)
                self.motor.set_tvc_command(pitch_cmd, yaw_cmd)
                
                # Update TVC actuator
                if t > ignition_time:
                    dt = 0.01
                    self.motor.update_tvc(dt)
                
                return self.state_derivative(t, state)
            
            sol_powered = solve_ivp(
                burning_state_derivative,
                [ignition_time, max_time],
                state_at_ignition,
                events=[ground_event],
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                max_step=0.01
            )
            
            # Combine solutions
            t_combined = np.concatenate([sol_freefall.t, sol_powered.t])
            y_combined = np.concatenate([sol_freefall.y, sol_powered.y], axis=1)
            
        else:
            # No ignition (hit ground before ignition altitude)
            t_combined = sol_freefall.t
            y_combined = sol_freefall.y
        
        # Extract final state
        final_state = y_combined[:, -1]
        final_altitude = final_state[2]
        final_velocity = final_state[3:6]
        final_speed = np.linalg.norm(final_velocity)
        
        # Check success criteria
        # Success: final altitude ≈ 0, final vertical speed < 2 m/s
        altitude_ok = abs(final_altitude) < 0.5
        velocity_ok = abs(final_velocity[2]) < 2.0
        total_velocity_ok = final_speed < 3.0
        
        success = altitude_ok and velocity_ok and total_velocity_ok
        
        # Build history
        history = {
            't': t_combined,
            'x': y_combined[0, :],
            'y': y_combined[1, :],
            'z': y_combined[2, :],
            'vx': y_combined[3, :],
            'vy': y_combined[4, :],
            'vz': y_combined[5, :],
            'qw': y_combined[6, :],
            'qx': y_combined[7, :],
            'qy': y_combined[8, :],
            'qz': y_combined[9, :],
            'omega_x': y_combined[10, :],
            'omega_y': y_combined[11, :],
            'omega_z': y_combined[12, :],
            'mass': y_combined[13, :],
            'success': success,
            'final_altitude': final_altitude,
            'final_velocity': final_speed,
            'ignition_altitude': ignition_altitude
        }
        
        self.history = history
        
        return success, final_state, history
    
    def optimize_ignition_altitude(self, initial_state, num_monte_carlo=100, 
                                   altitude_search_range=10.0, altitude_step=0.1):
        """
        Optimize ignition altitude using Monte Carlo simulation
        
        Args:
            initial_state: initial state vector
            num_monte_carlo: number of Monte Carlo runs per altitude
            altitude_search_range: range to search around analytical estimate (m)
            altitude_step: step size for altitude search (m)
            
        Returns:
            optimal_altitude: best ignition altitude (m)
            success_rates: dict with altitude -> success rate mapping
            best_history: history from best run
        """
        # Calculate analytical estimate
        initial_velocity = initial_state[5]  # vz
        initial_altitude = initial_state[2]  # z
        
        estimate = self.calculate_ignition_altitude(initial_velocity, initial_altitude)
        
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
        
        for altitude in altitudes:
            successes = 0
            histories = []
            
            for i in range(num_monte_carlo):
                # Reset motor
                self.motor = SolidMotor(self.rocket_config)
                
                # Run simulation
                success, final_state, history = self.run_simulation(
                    initial_state.copy(), altitude
                )
                
                if success:
                    successes += 1
                    histories.append(history)
            
            success_rate = successes / num_monte_carlo
            success_rates[altitude] = success_rate
            
            print(f"Altitude {altitude:.1f} m: {success_rate*100:.1f}% success rate")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_altitude = altitude
                if len(histories) > 0:
                    best_history = histories[0]
        
        print(f"\nOptimal ignition altitude: {best_altitude:.2f} m "
              f"({best_success_rate*100:.1f}% success rate)")
        
        return best_altitude, success_rates, best_history
