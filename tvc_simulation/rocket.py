"""
Rocket class for TVC model rocket with variable mass properties.
"""

import numpy as np
from scipy.interpolate import interp1d


class Rocket:
    """
    Represents a TVC-capable model rocket with variable mass properties.
    
    Attributes:
        mass_dry: Dry mass (without fuel) in kg
        mass_fuel_initial: Initial fuel mass in kg
        mass_fuel: Current fuel mass in kg
        length: Total rocket length in m
        diameter: Rocket body diameter in m
        cd: Drag coefficient (dimensionless)
        area_ref: Reference area for drag in m²
        
        Motor properties:
        motor_position: Position of motor mount from nose in m
        motor_length: Motor length in m
        grain_outer_radius: Outer radius of fuel grain in m
        grain_inner_radius_initial: Initial inner radius of fuel grain in m
        grain_inner_radius: Current inner radius of fuel grain in m
        grain_height: Height of fuel grain in m
        
        Thrust curve:
        thrust_curve: Interpolated thrust vs time function
        isp: Specific impulse in seconds
        
        TVC:
        gimbal_position: Distance of gimbal pivot from nose in m
        max_gimbal_angle: Maximum gimbal deflection in radians
        
        Aerodynamics:
        cp_position: Center of pressure distance from nose in m
    """
    
    def __init__(self, config):
        """
        Initialize rocket with configuration dictionary.
        
        Args:
            config: Dictionary containing rocket parameters
        """
        # Mass properties
        self.mass_dry = config['mass_dry']
        self.mass_fuel_initial = config['mass_fuel']
        self.mass_fuel = config['mass_fuel']
        
        # Geometry
        self.length = config['length']
        self.diameter = config['diameter']
        self.area_ref = np.pi * (self.diameter / 2)**2
        
        # Aerodynamics
        self.cd = config['cd']
        self.cp_position = config['cp_position']
        
        # Motor properties
        self.motor_position = config['motor_position']
        self.motor_length = config['motor_length']
        
        # Fuel grain geometry (hollow cylinder)
        self.grain_outer_radius = config['grain_outer_radius']
        self.grain_inner_radius_initial = config['grain_inner_radius_initial']
        self.grain_inner_radius = config['grain_inner_radius_initial']
        self.grain_height = config['grain_height']
        
        # Thrust curve (time vs thrust)
        thrust_times = np.array(config['thrust_curve_time'])
        thrust_values = np.array(config['thrust_curve_thrust'])
        self.thrust_curve = interp1d(
            thrust_times, thrust_values,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        self.burn_time = thrust_times[-1]
        
        # Specific impulse
        self.isp = config['isp']
        self.g0 = 9.80665  # Standard gravity
        
        # TVC
        self.gimbal_position = config['gimbal_position']
        self.max_gimbal_angle = np.radians(config['max_gimbal_angle_deg'])
        
        # Inertia tensor (will be computed)
        self.inertia = np.eye(3)
        self.cg_position = 0.0  # Distance from nose
        
        # Compute initial properties
        self._update_mass_properties()
    
    def _update_mass_properties(self):
        """
        Update center of gravity and inertia tensor based on current fuel mass.
        Models fuel grain as hollow cylinder burning from inside out.
        """
        # Total mass
        m_total = self.mass_dry + self.mass_fuel
        
        # Estimate dry rocket CG (assuming uniform distribution)
        cg_dry = self.length / 2
        
        # Fuel grain CG (at motor location)
        cg_fuel = self.motor_position + self.motor_length / 2
        
        # Combined CG using weighted average
        if m_total > 0:
            self.cg_position = (self.mass_dry * cg_dry + self.mass_fuel * cg_fuel) / m_total
        else:
            self.cg_position = cg_dry
        
        # Inertia tensor computation
        # Dry rocket approximated as thin cylinder
        m_dry = self.mass_dry
        r_body = self.diameter / 2
        l_body = self.length
        
        # Longitudinal MOI (about x-axis, along rocket body)
        I_dry_xx = 0.5 * m_dry * r_body**2
        
        # Transverse MOI (about y and z axes)
        I_dry_yy = (1/12) * m_dry * (3*r_body**2 + l_body**2)
        I_dry_zz = I_dry_yy
        
        # Fuel grain inertia (hollow cylinder)
        if self.mass_fuel > 0:
            r_out = self.grain_outer_radius
            r_in = self.grain_inner_radius
            h_grain = self.grain_height
            m_fuel = self.mass_fuel
            
            # Longitudinal MOI (hollow cylinder formula)
            I_fuel_xx = 0.5 * m_fuel * (r_out**2 + r_in**2)
            
            # Transverse MOI
            I_fuel_yy = (1/12) * m_fuel * (3*(r_out**2 + r_in**2) + h_grain**2)
            I_fuel_zz = I_fuel_yy
            
            # Parallel axis theorem: shift fuel inertia to rocket CG
            d_fuel = abs(cg_fuel - self.cg_position)
            I_fuel_yy += m_fuel * d_fuel**2
            I_fuel_zz += m_fuel * d_fuel**2
        else:
            I_fuel_xx = 0
            I_fuel_yy = 0
            I_fuel_zz = 0
        
        # Parallel axis theorem: shift dry inertia to rocket CG
        d_dry = abs(cg_dry - self.cg_position)
        I_dry_yy += m_dry * d_dry**2
        I_dry_zz += m_dry * d_dry**2
        
        # Total inertia tensor (diagonal, assuming symmetry)
        self.inertia = np.diag([
            I_dry_xx + I_fuel_xx,
            I_dry_yy + I_fuel_yy,
            I_dry_zz + I_fuel_zz
        ])
    
    def update_mass_properties(self, dt, thrust):
        """
        Update mass and inertia based on fuel consumption.
        
        Args:
            dt: Time step in seconds
            thrust: Current thrust in Newtons
        """
        if thrust > 0 and self.mass_fuel > 0:
            # Mass flow rate: dm/dt = -Thrust / (Isp * g0)
            mass_flow_rate = thrust / (self.isp * self.g0)
            dm = mass_flow_rate * dt
            
            # Update fuel mass
            self.mass_fuel = max(0.0, self.mass_fuel - dm)
            
            # Update grain inner radius (burning from inside out)
            if self.mass_fuel > 0:
                # Volume of hollow cylinder: V = π * h * (r_out² - r_in²)
                # Mass = density * volume
                # Assume constant density: ρ = m_fuel_initial / V_initial
                
                V_initial = np.pi * self.grain_height * (
                    self.grain_outer_radius**2 - self.grain_inner_radius_initial**2
                )
                density = self.mass_fuel_initial / V_initial
                
                # Current volume
                V_current = self.mass_fuel / density
                
                # Solve for new inner radius
                # V = π * h * (r_out² - r_in²)
                r_in_squared = self.grain_outer_radius**2 - V_current / (np.pi * self.grain_height)
                self.grain_inner_radius = np.sqrt(max(0, r_in_squared))
            else:
                self.grain_inner_radius = self.grain_outer_radius
            
            # Recompute mass properties
            self._update_mass_properties()
    
    def get_thrust(self, time):
        """
        Get thrust at given time from thrust curve.
        
        Args:
            time: Current simulation time in seconds
        
        Returns:
            Thrust in Newtons
        """
        return float(self.thrust_curve(time))
    
    def get_thrust_vector(self, gimbal_pitch, gimbal_yaw, thrust):
        """
        Compute thrust vector in body frame given gimbal angles.
        
        TVC actuation: motor mount pivots by angles δp (pitch) and δy (yaw).
        Thrust vector in body frame:
        F_thrust^(B) = [cos(δp)cos(δy), sin(δy), sin(δp)] * T(t)
        
        Args:
            gimbal_pitch: Gimbal pitch angle in radians (δp)
            gimbal_yaw: Gimbal yaw angle in radians (δy)
            thrust: Scalar thrust magnitude in Newtons
        
        Returns:
            3D thrust vector in body frame [Fx, Fy, Fz]
        """
        # Clamp gimbal angles to mechanical limits
        gimbal_pitch = np.clip(gimbal_pitch, -self.max_gimbal_angle, self.max_gimbal_angle)
        gimbal_yaw = np.clip(gimbal_yaw, -self.max_gimbal_angle, self.max_gimbal_angle)
        
        # Compute thrust vector in body frame using a standard yaw-then-pitch rotation.
        # Body frame: X points out nose, Y is to starboard, Z is down.
        # A positive pitch gimbal (δp) should create a negative Z force.
        # A positive yaw gimbal (δy) should create a positive Y force.
        cp = np.cos(gimbal_pitch)
        sp = np.sin(gimbal_pitch)
        cy = np.cos(gimbal_yaw)
        sy = np.sin(gimbal_yaw)

        thrust_vector = thrust * np.array([
            cp * cy,
            cp * sy,
            -sp
        ])
        
        return thrust_vector
    
    def get_tvc_torque(self, thrust_vector_body):
        """
        Compute torque generated by TVC.
        
        M_TVC = r_pivot × F_thrust
        where r_pivot is vector from CG to gimbal pivot point.
        
        Args:
            thrust_vector_body: Thrust vector in body frame [Fx, Fy, Fz]
        
        Returns:
            Torque vector in body frame [Mx, My, Mz]
        """
        # Vector from CG to gimbal pivot in body frame
        # CG and gimbal positions are measured from nose
        # In body frame, positive x points forward (nose direction)
        # Distance from CG to gimbal (negative if gimbal is behind CG)
        r_pivot_x = self.gimbal_position - self.cg_position
        r_pivot = np.array([r_pivot_x, 0.0, 0.0])
        
        # Torque = r × F
        torque = np.cross(r_pivot, thrust_vector_body)
        
        return torque
    
    def get_total_mass(self):
        """Get current total mass."""
        return self.mass_dry + self.mass_fuel
    
    def get_inertia(self):
        """Get current inertia tensor."""
        return self.inertia.copy()
    
    def get_cg_position(self):
        """Get current CG position from nose."""
        return self.cg_position
