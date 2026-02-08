"""
Solid Motor Model with Thrust Vector Control
Non-throttleable motor with fixed thrust curve
"""

import numpy as np
from scipy.interpolate import interp1d


class SolidMotor:
    """Solid fuel rocket motor model"""
    
    def __init__(self, config):
        """
        Initialize solid motor
        
        config should contain:
        - thrust_curve: list of [time, thrust] pairs
        - total_impulse: total impulse (N·s)
        - propellant_mass: total propellant mass (kg)
        - burn_time: total burn time (s)
        - tvc_max_angle: max TVC gimbal angle (degrees)
        - tvc_response_time: TVC response time constant (s)
        - tvc_response_variation: Monte Carlo variation in TVC response
        """
        self.config = config
        
        # Thrust curve
        thrust_curve = config.get('thrust_curve', [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]])
        times = [t for t, _ in thrust_curve]
        thrusts = [T for _, T in thrust_curve]
        
        self.thrust_interp = interp1d(times, thrusts, kind='linear', 
                                      bounds_error=False, fill_value=0.0)
        
        self.burn_time = config.get('burn_time', max(times))
        self.propellant_mass = config.get('propellant_mass', 10.0)
        # Calculate total impulse using trapezoidal rule
        from scipy.integrate import trapezoid
        self.total_impulse = config.get('total_impulse', trapezoid(thrusts, times))
        
        # Calculate mass flow rate (assume constant for simplicity)
        # Safety check to prevent division by zero
        if self.burn_time > 0:
            self.mass_flow_rate = self.propellant_mass / self.burn_time
        else:
            # Instantaneous burn (unrealistic but safe fallback)
            self.mass_flow_rate = 0.0
        
        # TVC parameters
        self.tvc_max_angle = np.radians(config.get('tvc_max_angle', 5.0))
        self.tvc_response_time = config.get('tvc_response_time', 0.1)
        self.tvc_response_variation = config.get('tvc_response_variation', 0.0)
        
        # Monte Carlo variation factors (sampled once per motor instance)
        self.thrust_variation = config.get('thrust_variation', 0.0)
        self.thrust_variation_factor = 1.0 + np.random.uniform(-self.thrust_variation, self.thrust_variation)
        
        mass_variation = config.get('mass_variation', 0.0)
        self.mass_flow_variation_factor = 1.0 + np.random.uniform(-mass_variation, mass_variation)
        
        # Motor state
        self.ignited = False
        self.ignition_time = None
        self.current_tvc_angle = np.array([0.0, 0.0])  # pitch, yaw
        self.commanded_tvc_angle = np.array([0.0, 0.0])
        
    def ignite(self, time):
        """Ignite the motor"""
        if not self.ignited:
            self.ignited = True
            self.ignition_time = time
            
    def is_burning(self, time):
        """Check if motor is currently burning"""
        if not self.ignited:
            return False
        
        time_since_ignition = time - self.ignition_time
        return 0 <= time_since_ignition <= self.burn_time
    
    def get_thrust(self, time):
        """Get thrust magnitude at given time"""
        if not self.is_burning(time):
            return 0.0
        
        time_since_ignition = time - self.ignition_time
        nominal_thrust = float(self.thrust_interp(time_since_ignition))
        
        return nominal_thrust * self.thrust_variation_factor
    
    def get_mass_flow_rate(self, time):
        """Get mass flow rate at given time"""
        if not self.is_burning(time):
            return 0.0
        
        return self.mass_flow_rate * self.mass_flow_variation_factor
    
    def set_tvc_command(self, pitch_angle, yaw_angle):
        """Set commanded TVC angle (radians)"""
        # Clamp to max angle
        pitch_angle = np.clip(pitch_angle, -self.tvc_max_angle, self.tvc_max_angle)
        yaw_angle = np.clip(yaw_angle, -self.tvc_max_angle, self.tvc_max_angle)
        
        self.commanded_tvc_angle = np.array([pitch_angle, yaw_angle])
    
    def update_tvc(self, dt):
        """Update TVC actuator state (first-order lag)"""
        # Apply TVC response variation
        response_time = self.tvc_response_time * (1.0 + np.random.uniform(
            -self.tvc_response_variation, self.tvc_response_variation))
        
        # First-order lag: dx/dt = (commanded - current) / tau
        tau = max(response_time, 0.001)
        self.current_tvc_angle += (self.commanded_tvc_angle - self.current_tvc_angle) * dt / tau
    
    def get_thrust_vector(self, time, body_to_inertial_matrix):
        """
        Get thrust force vector in inertial frame
        
        Args:
            time: current time
            body_to_inertial_matrix: rotation matrix from body to inertial frame
            
        Returns:
            thrust_vector: 3D thrust vector in inertial frame
        """
        thrust_magnitude = self.get_thrust(time)
        
        if thrust_magnitude == 0:
            return np.zeros(3)
        
        # Nominal thrust direction in body frame (along +z axis before gimbal)
        thrust_nominal = np.array([0, 0, 1])
        
        # TVC angles: pitch (rotation about y), yaw (rotation about x)
        pitch, yaw = self.current_tvc_angle
        
        # Build rotation matrices for TVC gimbal
        # Rotation about Y-axis (pitch)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Rotation about X-axis (yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_yaw = np.array([
            [1, 0, 0],
            [0, cos_yaw, -sin_yaw],
            [0, sin_yaw, cos_yaw]
        ])
        
        # Combined TVC rotation: first pitch, then yaw
        R_tvc = R_yaw @ R_pitch
        
        # Apply TVC rotation to nominal thrust direction
        thrust_direction_body = R_tvc @ thrust_nominal
        
        # Scale by thrust magnitude
        thrust_body = thrust_magnitude * thrust_direction_body
        
        # Transform to inertial frame
        thrust_inertial = body_to_inertial_matrix @ thrust_body
        
        return thrust_inertial
    
    def get_thrust_moment(self, time, cg_offset):
        """
        Get moment generated by thrust about center of mass
        
        Args:
            time: current time
            cg_offset: offset from thrust application point to CG in body frame
            
        Returns:
            moment: 3D moment vector in body frame
        """
        thrust_magnitude = self.get_thrust(time)
        
        if thrust_magnitude == 0:
            return np.zeros(3)
        
        # Nominal thrust direction in body frame (along +z axis before gimbal)
        thrust_nominal = np.array([0, 0, 1])
        
        # TVC angles: pitch (rotation about y), yaw (rotation about x)
        pitch, yaw = self.current_tvc_angle
        
        # Build rotation matrices for TVC gimbal
        # Rotation about Y-axis (pitch)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        R_pitch = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        # Rotation about X-axis (yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R_yaw = np.array([
            [1, 0, 0],
            [0, cos_yaw, -sin_yaw],
            [0, sin_yaw, cos_yaw]
        ])
        
        # Combined TVC rotation: first pitch, then yaw
        R_tvc = R_yaw @ R_pitch
        
        # Apply TVC rotation to nominal thrust direction
        thrust_direction_body = R_tvc @ thrust_nominal
        
        # Scale by thrust magnitude
        thrust_body = thrust_magnitude * thrust_direction_body
        
        # Moment = r × F
        moment = np.cross(cg_offset, thrust_body)
        
        return moment
