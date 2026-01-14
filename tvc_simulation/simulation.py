"""
Main simulation class for 6-DOF TVC rocket dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .utils import (
    quaternion_derivative, quaternion_normalize, 
    quaternion_to_rotation_matrix, rotate_vector_by_quaternion
)


class Simulation:
    """
    6-DOF simulation of TVC rocket.
    
    State vector (13 elements):
    X = [Px, Py, Pz, Vx, Vy, Vz, q0, q1, q2, q3, ωx, ωy, ωz]
    
    - Position (P): Inertial frame (NED)
    - Velocity (V): Inertial frame (NED)
    - Quaternion (q): [w, x, y, z] mapping Inertial to Body
    - Angular velocity (ω): Body frame
    """
    
    def __init__(self, rocket, environment, gnc):
        """
        Initialize simulation.
        
        Args:
            rocket: Rocket instance
            environment: Environment instance
            gnc: GNC instance
        """
        self.rocket = rocket
        self.environment = environment
        self.gnc = gnc
        
        # Initial state
        self.state = None
        self.time = 0.0
        
        # Track if rocket has lifted off (for ground impact detection)
        self.has_lifted_off = False
        self.liftoff_altitude_threshold = 1.0  # meters
        
        # History for logging
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'quaternion': [],
            'angular_velocity': [],
            'euler_angles': [],
            'gimbal_angles': [],
            'thrust': [],
            'mass': [],
            'cg_position': [],
        }
    
    def set_initial_state(self, position, velocity, quaternion, angular_velocity):
        """
        Set initial state of the simulation.
        
        Args:
            position: Initial position [x, y, z] in inertial frame (m)
            velocity: Initial velocity [vx, vy, vz] in inertial frame (m/s)
            quaternion: Initial quaternion [w, x, y, z]
            angular_velocity: Initial angular velocity [ωx, ωy, ωz] in body frame (rad/s)
        """
        # Normalize quaternion
        q = quaternion_normalize(np.array(quaternion))
        
        self.state = np.concatenate([
            np.array(position),
            np.array(velocity),
            q,
            np.array(angular_velocity)
        ])
        
        self.time = 0.0
    
    def _state_derivative(self, t, state):
        """
        Compute state derivative for integration.
        
        This is the core physics function implementing:
        - Translational dynamics: F_total = m * dV/dt
        - Rotational dynamics (Euler's equations): M_total = I*dω/dt + ω × (I*ω)
        - Quaternion kinematics: dq/dt = 0.5 * q ⊗ ω
        
        Args:
            t: Current time (s)
            state: Current state vector [13 elements]
        
        Returns:
            State derivative dstate/dt [13 elements]
        """
        # Unpack state
        position = state[0:3]
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        
        # Normalize quaternion to prevent drift
        quaternion = quaternion_normalize(quaternion)
        
        # Get current rocket properties
        mass = self.rocket.get_total_mass()
        inertia = self.rocket.get_inertia()
        
        # Altitude (negative z in NED convention)
        altitude = -position[2]
        
        # ========== CONTROL ==========
        # Compute gimbal angles from GNC
        dt_gnc = 0.01  # Small dt for control derivative
        gimbal_pitch, gimbal_yaw = self.gnc.compute_control(
            quaternion, angular_velocity, dt_gnc, t
        )
        
        # ========== THRUST ==========
        # Get thrust magnitude
        thrust_magnitude = self.rocket.get_thrust(t)
        
        # Get thrust vector in body frame
        thrust_body = self.rocket.get_thrust_vector(gimbal_pitch, gimbal_yaw, thrust_magnitude)
        
        # Transform thrust to inertial frame
        # R^T (transpose) rotates from Body to Inertial
        R = quaternion_to_rotation_matrix(quaternion)
        thrust_inertial = R.T @ thrust_body
        
        # ========== AERODYNAMICS ==========
        # Get wind vector
        wind_vector = self.environment.get_wind(altitude)
        
        # Relative velocity (rocket velocity relative to wind, in inertial frame)
        velocity_rel_inertial = velocity - wind_vector
        
        # Transform relative velocity to body frame
        velocity_rel_body = R @ velocity_rel_inertial
        
        # Dynamic pressure and drag
        rho = self.environment.get_density(altitude)
        v_rel_magnitude = np.linalg.norm(velocity_rel_inertial)
        
        if v_rel_magnitude > 0.1:
            # Drag force in inertial frame (opposes relative velocity)
            drag_force_inertial = -0.5 * rho * v_rel_magnitude**2 * self.rocket.cd * self.rocket.area_ref * (velocity_rel_inertial / v_rel_magnitude)
        else:
            drag_force_inertial = np.zeros(3)
        
        # ========== GRAVITY ==========
        gravity_force_inertial = mass * self.environment.get_gravity_vector()
        
        # ========== TRANSLATIONAL DYNAMICS ==========
        # F_total = Thrust + Drag + Gravity
        force_total_inertial = thrust_inertial + drag_force_inertial + gravity_force_inertial
        
        # Newton's second law: F = m * a
        if mass > 0:
            acceleration_inertial = force_total_inertial / mass
        else:
            acceleration_inertial = np.zeros(3)
        
        # ========== AERODYNAMIC TORQUE ==========
        # Simplified model: aerodynamic forces create restoring torque if CP is behind CG
        # This provides static stability
        cp_position = self.rocket.cp_position
        cg_position = self.rocket.cg_position
        
        # Vector from CG to CP in body frame (along x-axis)
        r_cp_cg = np.array([cp_position - cg_position, 0.0, 0.0])
        
        # Aerodynamic force in body frame (simplified: perpendicular to velocity)
        # Use cross-flow drag as a simple model
        if v_rel_magnitude > 0.1:
            # Angle of attack components
            v_x = velocity_rel_body[0]
            v_perp = np.sqrt(velocity_rel_body[1]**2 + velocity_rel_body[2]**2)
            
            if v_perp > 0.1:
                # Normal force coefficient (simplified)
                CN = 2.0  # Normal force coefficient
                normal_force_magnitude = 0.5 * rho * v_perp**2 * CN * self.rocket.area_ref
                
                # Direction of normal force (opposes perpendicular velocity)
                normal_direction = np.array([0, velocity_rel_body[1], velocity_rel_body[2]]) / v_perp
                aero_force_body = -normal_force_magnitude * normal_direction
            else:
                aero_force_body = np.zeros(3)
        else:
            aero_force_body = np.zeros(3)
        
        # Aerodynamic torque: M_aero = r_cp_cg × F_aero
        aero_torque_body = np.cross(r_cp_cg, aero_force_body)
        
        # ========== TVC TORQUE ==========
        tvc_torque_body = self.rocket.get_tvc_torque(thrust_body)
        
        # ========== ROTATIONAL DYNAMICS (EULER'S EQUATIONS) ==========
        # M_total = M_TVC + M_aero
        torque_total_body = tvc_torque_body + aero_torque_body
        
        # Euler's equation: M = I*dω/dt + ω × (I*ω)
        # Solve for dω/dt: dω/dt = I^(-1) * (M - ω × (I*ω))
        I_omega = inertia @ angular_velocity
        omega_cross_I_omega = np.cross(angular_velocity, I_omega)
        
        try:
            inertia_inv = np.linalg.inv(inertia)
            angular_acceleration = inertia_inv @ (torque_total_body - omega_cross_I_omega)
        except np.linalg.LinAlgError:
            # If inertia is singular, no angular acceleration
            angular_acceleration = np.zeros(3)
        
        # ========== QUATERNION KINEMATICS ==========
        # dq/dt = 0.5 * q ⊗ ω
        quaternion_dot = quaternion_derivative(quaternion, angular_velocity)
        
        # ========== ASSEMBLE STATE DERIVATIVE ==========
        state_dot = np.concatenate([
            velocity,                # dP/dt = V
            acceleration_inertial,   # dV/dt = a
            quaternion_dot,          # dq/dt
            angular_acceleration     # dω/dt
        ])
        
        return state_dot
    
    def _event_ground_impact(self, t, state):
        """
        Event function to detect ground impact.
        
        In NED convention:
        - z = 0 is ground level
        - z < 0 means above ground (in the air)
        - z > 0 would be underground (not physical)
        
        We want to stop when the rocket returns to ground after liftoff.
        """
        z_position = state[2]
        altitude = -z_position  # Altitude above ground
        
        # Check if rocket has lifted off
        if not self.has_lifted_off and altitude > self.liftoff_altitude_threshold:
            self.has_lifted_off = True
        
        # Only detect impact after liftoff, when rocket returns to ground
        if self.has_lifted_off and z_position >= -0.01:
            return z_position  # Will cross 0 when hitting ground
        else:
            return -1.0  # Keep event inactive
    
    _event_ground_impact.terminal = True
    _event_ground_impact.direction = 1  # Detect crossing from negative to positive
    
    def run(self, t_span, max_step=0.01, method='RK45'):
        """
        Run the simulation.
        
        Args:
            t_span: Tuple (t_start, t_end) in seconds
            max_step: Maximum integration step size
            method: Integration method ('RK45', 'DOP853', etc.)
        
        Returns:
            Dictionary containing simulation results
        """
        if self.state is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")
        
        # Reset GNC
        self.gnc.reset()
        
        # Reset liftoff flag
        self.has_lifted_off = False
        
        # Solve ODE with event detection for ground impact
        solution = solve_ivp(
            fun=self._state_derivative,
            t_span=t_span,
            y0=self.state,
            method=method,
            max_step=max_step,
            events=self._event_ground_impact,
            dense_output=False,
            vectorized=False
        )
        
        # Process results
        results = {
            'time': solution.t,
            'state': solution.y,
            'success': solution.success,
            'message': solution.message
        }
        
        # Store in history and add derived quantities
        self._process_results(results)
        
        return results
    
    def _process_results(self, results):
        """
        Process simulation results and compute derived quantities.
        
        Args:
            results: Dictionary from solve_ivp
        """
        from .utils import quaternion_to_euler
        
        times = results['time']
        states = results['state']
        
        for i, t in enumerate(times):
            state = states[:, i]
            
            # Unpack state
            position = state[0:3]
            velocity = state[3:6]
            quaternion = quaternion_normalize(state[6:10])
            angular_velocity = state[10:13]
            
            # Update rocket mass properties at this time point
            thrust = self.rocket.get_thrust(t)
            if i > 0:
                dt = times[i] - times[i-1]
                self.rocket.update_mass_properties(dt, thrust)
            
            # Compute Euler angles
            euler_angles = quaternion_to_euler(quaternion)
            
            # Compute gimbal angles (recompute control)
            dt_gnc = 0.01
            gimbal_pitch, gimbal_yaw = self.gnc.compute_control(
                quaternion, angular_velocity, dt_gnc, t
            )
            
            # Store in history
            self.history['time'].append(t)
            self.history['position'].append(position.copy())
            self.history['velocity'].append(velocity.copy())
            self.history['quaternion'].append(quaternion.copy())
            self.history['angular_velocity'].append(angular_velocity.copy())
            self.history['euler_angles'].append(euler_angles.copy())
            self.history['gimbal_angles'].append([gimbal_pitch, gimbal_yaw])
            self.history['thrust'].append(thrust)
            self.history['mass'].append(self.rocket.get_total_mass())
            self.history['cg_position'].append(self.rocket.get_cg_position())
    
    def get_history(self):
        """
        Get simulation history as numpy arrays.
        
        Returns:
            Dictionary with history arrays
        """
        history_arrays = {}
        for key, value in self.history.items():
            if key == 'time':
                history_arrays[key] = np.array(value)
            else:
                history_arrays[key] = np.array(value)
        
        return history_arrays
