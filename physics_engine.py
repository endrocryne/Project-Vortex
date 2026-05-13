"""
Physics Engine for 6DOF Flight Dynamics Simulation
Handles gravity, drag, wind, and mass variation
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class PhysicsEngine:
    """6 Degrees of Freedom physics simulation"""
    
    def __init__(self, config):
        """
        Initialize physics engine with configuration
        
        config should contain:
        - gravity: gravitational acceleration (m/s^2)
        - air_density: air density at sea level (kg/m^3)
        - temperature: temperature (K)
        - drag_coefficient: drag coefficient
        - reference_area: reference area for drag (m^2)
        - wind_model: 'constant', 'altitude_varying', 'gusts'
        - wind_speed: base wind speed (m/s)
        - wind_direction: wind direction (degrees)
        """
        self.config = config
        self.g = config.get('gravity', 9.81)
        self.rho_0 = config.get('air_density', 1.225)
        self.T_0 = config.get('temperature', 288.15)
        self.Cd = config.get('drag_coefficient', 0.5)
        self.A_ref = config.get('reference_area', 0.1)
        self.wind_model = config.get('wind_model', 'constant')
        self.wind_speed = config.get('wind_speed', 0.0)
        self.wind_direction = np.radians(config.get('wind_direction', 0.0))
        
        # Monte Carlo variation parameters
        self.drag_variation = config.get('drag_variation', 0.0)
        self.air_density_variation = config.get('air_density_variation', 0.0)

        # Aerodynamic center of pressure (z-coordinate in body frame)
        # Nose is at +length/2, Tail at -length/2.
        # For stability, CP should be behind CG.
        self.cp_z = config.get('cp_z', -0.5) # Default slightly behind center
        
    def get_air_density(self, altitude):
        """Calculate air density at given altitude using barometric formula"""
        # Scale height for exponential atmosphere model
        H = 8500  # meters
        rho_variation = 1.0 + np.random.uniform(-self.air_density_variation, self.air_density_variation)
        return self.rho_0 * np.exp(-altitude / H) * rho_variation
    
    def get_wind_velocity(self, position, time):
        """Get wind velocity at given position and time"""
        altitude = position[2]
        
        if self.wind_model == 'constant':
            # Constant wind
            wind_x = self.wind_speed * np.cos(self.wind_direction)
            wind_y = self.wind_speed * np.sin(self.wind_direction)
            wind_z = 0.0
            
        elif self.wind_model == 'altitude_varying':
            # Power law wind profile: v(h) = v_ref * (h / h_ref)^alpha
            h_ref = 10.0  # reference height (m)
            alpha = 0.143  # power law exponent for open terrain
            wind_factor = (max(altitude, 1.0) / h_ref) ** alpha
            wind_x = self.wind_speed * wind_factor * np.cos(self.wind_direction)
            wind_y = self.wind_speed * wind_factor * np.sin(self.wind_direction)
            wind_z = 0.0
            
        elif self.wind_model == 'gusts':
            # Wind with random gusts
            gust_freq = 0.1  # Hz
            gust_amplitude = self.wind_speed * 0.5
            gust_x = gust_amplitude * np.sin(2 * np.pi * gust_freq * time)
            gust_y = gust_amplitude * np.cos(2 * np.pi * gust_freq * time * 1.3)
            
            wind_x = self.wind_speed * np.cos(self.wind_direction) + gust_x
            wind_y = self.wind_speed * np.sin(self.wind_direction) + gust_y
            wind_z = 0.0
        else:
            wind_x = wind_y = wind_z = 0.0
            
        return np.array([wind_x, wind_y, wind_z])
    
    def get_drag_force(self, velocity, position, time):
        """Calculate drag force in inertial frame"""
        altitude = position[2]
        rho = self.get_air_density(altitude)
        
        # Wind velocity
        wind = self.get_wind_velocity(position, time)
        
        # Relative velocity (velocity relative to air)
        v_rel = velocity - wind
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag < 0.01:
            return np.zeros(3)
        
        # Drag force: F_d = 0.5 * rho * v^2 * Cd * A
        Cd_actual = self.Cd * (1.0 + np.random.uniform(-self.drag_variation, self.drag_variation))
        drag_magnitude = 0.5 * rho * v_rel_mag**2 * Cd_actual * self.A_ref
        
        # Drag force opposes relative velocity
        drag_force = -drag_magnitude * (v_rel / v_rel_mag)
        
        return drag_force

    def get_aero_torque(self, velocity, position, quaternion, time, cg_location):
        """
        Calculate aerodynamic torque (stability + damping)

        Args:
            velocity: inertial velocity
            position: inertial position
            quaternion: attitude [w, x, y, z]
            time: current time
            cg_location: CG z-coordinate in body frame

        Returns:
            torque: 3D torque vector in body frame
        """
        # 1. Stability Torque (Restoring Moment)
        # Relative velocity in inertial frame
        wind = self.get_wind_velocity(position, time)
        v_rel_inertial = velocity - wind
        v_rel_mag = np.linalg.norm(v_rel_inertial)

        if v_rel_mag < 0.1:
            return np.zeros(3)

        # Rotate relative velocity to body frame
        # (Transposing rotation matrix R_body_to_inertial gives R_inertial_to_body)
        R_b2i = self.quaternion_to_rotation_matrix(quaternion)
        R_i2b = R_b2i.T
        v_rel_body = R_i2b @ v_rel_inertial

        # Aerodynamic force magnitude (using drag equation)
        altitude = position[2]
        rho = self.get_air_density(altitude)
        drag_mag = 0.5 * rho * v_rel_mag**2 * self.Cd * self.A_ref

        # Force vector in body frame (simplified: force at CP opposing v_rel_body)
        f_aero_body = -drag_mag * (v_rel_body / v_rel_mag)

        # Leverage arm: vector from CG to CP in body frame
        # cp_z is usually constant, cg_location varies with fuel
        r_cp = np.array([0, 0, self.cp_z - cg_location])

        # Stability Torque = r x F
        # Use a higher multiplier to ensure flip
        torque_stability = np.cross(r_cp, f_aero_body) * 5.0

        return torque_stability
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def normalize_quaternion(self, q):
        """Normalize quaternion"""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
        
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles (in radians) to quaternion [w, x, y, z]
        Order: Z (yaw) -> Y (pitch) -> X (roll) intrinsic rotations
        """
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return self.normalize_quaternion(np.array([w, x, y, z]))
    
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw) in radians
        
        Args:
            q: quaternion [w, x, y, z]
            
        Returns:
            np.array([roll, pitch, yaw]) in radians
        """
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Clamp to avoid numerical issues with arcsin
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])