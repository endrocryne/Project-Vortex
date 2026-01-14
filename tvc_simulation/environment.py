"""
Environment class for atmospheric and wind modeling.
"""

import numpy as np


class Environment:
    """
    Models atmospheric conditions and wind effects.
    
    Implements:
    - Standard atmosphere model for density vs altitude
    - Logarithmic wind shear model
    - Turbulence/gust capability
    """
    
    def __init__(self, config):
        """
        Initialize environment with configuration.
        
        Args:
            config: Dictionary containing environment parameters
        """
        # Gravity
        self.gravity = config.get('gravity', 9.80665)  # m/s²
        
        # Standard atmosphere at sea level
        self.rho_0 = config.get('rho_0', 1.225)  # kg/m³
        self.T_0 = config.get('T_0', 288.15)  # K
        self.pressure_0 = config.get('pressure_0', 101325)  # Pa
        
        # Atmospheric model parameters
        self.temperature_lapse_rate = config.get('temperature_lapse_rate', 0.0065)  # K/m
        self.R = 287.05  # Specific gas constant for air, J/(kg·K)
        self.gamma = 1.4  # Specific heat ratio for air
        
        # Wind model parameters
        self.wind_reference_speed = config.get('wind_reference_speed', 5.0)  # m/s
        self.wind_reference_altitude = config.get('wind_reference_altitude', 10.0)  # m
        self.wind_roughness_length = config.get('wind_roughness_length', 0.1)  # m
        self.wind_direction = config.get('wind_direction', 0.0)  # radians from North
        
        # Turbulence
        self.turbulence_intensity = config.get('turbulence_intensity', 0.1)  # fraction
        self.enable_turbulence = config.get('enable_turbulence', False)
        
        # Random seed for turbulence
        self.rng = np.random.RandomState(config.get('random_seed', 42))
    
    def get_density(self, altitude):
        """
        Calculate air density at given altitude using standard atmosphere model.
        
        Uses barometric formula:
        ρ(h) = ρ₀ * (T(h) / T₀)^((g/(R*L)) - 1)
        
        Args:
            altitude: Altitude above ground level in meters (negative z in NED)
        
        Returns:
            Air density in kg/m³
        """
        # Ensure altitude is non-negative
        h = max(0.0, altitude)
        
        # Temperature at altitude
        T = self.T_0 - self.temperature_lapse_rate * h
        
        # Ensure temperature doesn't go negative
        if T <= 0:
            T = self.T_0 * 0.1
        
        # Density using barometric formula
        exponent = (self.gravity / (self.R * self.temperature_lapse_rate)) - 1.0
        density = self.rho_0 * (T / self.T_0) ** exponent
        
        return density
    
    def get_wind(self, altitude):
        """
        Calculate wind velocity vector at given altitude.
        
        Implements logarithmic wind shear:
        V_wind(z) = V_ref * ln(z / z_0) / ln(z_ref / z_0)
        
        Args:
            altitude: Altitude above ground level in meters
        
        Returns:
            Wind velocity vector in inertial frame [Vx, Vy, Vz] in m/s
        """
        # Ensure altitude is above roughness length
        h = max(self.wind_roughness_length + 0.01, altitude)
        
        # Logarithmic wind shear
        if h > self.wind_roughness_length and self.wind_reference_altitude > self.wind_roughness_length:
            wind_speed = self.wind_reference_speed * (
                np.log(h / self.wind_roughness_length) /
                np.log(self.wind_reference_altitude / self.wind_roughness_length)
            )
        else:
            wind_speed = 0.0
        
        # Add turbulence if enabled
        if self.enable_turbulence:
            # Simple Dryden-like turbulence (random fluctuations)
            turbulence = self.rng.randn(3) * self.turbulence_intensity * wind_speed
        else:
            turbulence = np.zeros(3)
        
        # Wind vector in inertial frame (NED convention)
        # Wind direction: 0 = North, π/2 = East
        wind_north = wind_speed * np.cos(self.wind_direction) + turbulence[0]
        wind_east = wind_speed * np.sin(self.wind_direction) + turbulence[1]
        wind_down = turbulence[2]  # Vertical turbulence only
        
        wind_vector = np.array([wind_north, wind_east, wind_down])
        
        return wind_vector
    
    def get_gravity_vector(self):
        """
        Get gravity vector in inertial frame (NED convention).
        
        Returns:
            Gravity vector [0, 0, g] where g is positive (down is positive Z)
        """
        return np.array([0.0, 0.0, self.gravity])
