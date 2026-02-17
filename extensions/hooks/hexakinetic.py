"""
HexaKinetic Extension Hooks
=============================

Hook types for HexaKinetic (simulation GUI) extensions.
Extensions targeting HexaKinetic should subclass HexaKineticExtension.
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from extensions.base import VortexExtension, ExtensionType


class FaultModelDefinition:
    """Defines a custom fault model that can be injected during simulation."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "⚡", category: str = "Custom Faults",
                 parameters: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            key: Unique fault model identifier
            label: Display name for the fault model
            description: What this fault simulates
            icon: Emoji or short prefix
            category: Grouping category
            parameters: List of parameter definitions, each a dict with:
                        - name: str (parameter name)
                        - label: str (display label)
                        - type: str ('float', 'int', 'bool', 'choice')
                        - default: Any
                        - min/max: float (for numeric types)
                        - options: list (for 'choice' type)
        """
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.category = category
        self.parameters = parameters or []


class MotorProfileDefinition:
    """Defines a custom motor thrust profile."""

    def __init__(self, key: str, label: str, description: str = "",
                 manufacturer: str = "",
                 total_impulse: float = 0.0,
                 burn_time: float = 0.0):
        self.key = key
        self.label = label
        self.description = description
        self.manufacturer = manufacturer
        self.total_impulse = total_impulse
        self.burn_time = burn_time


class ControllerDefinition:
    """Defines a custom control algorithm."""

    def __init__(self, key: str, label: str, description: str = "",
                 icon: str = "🎮", parameters: Optional[List[Dict[str, Any]]] = None):
        self.key = key
        self.label = label
        self.description = description
        self.icon = icon
        self.parameters = parameters or []


class HexaKineticExtension(VortexExtension):
    """
    Base class for HexaKinetic extensions.

    Provides hooks for:
        - Custom fault injection models
        - Custom motor thrust curves / profiles
        - Custom control algorithms
        - Custom atmosphere models
        - Pre/post simulation hooks
        - Custom analysis passes on simulation results
    """

    @property
    def extension_type(self) -> ExtensionType:
        return ExtensionType.HEXAKINETIC

    # --- Fault model hooks ---

    def register_fault_models(self) -> List[FaultModelDefinition]:
        """Register custom fault injection models."""
        return []

    def apply_fault(self, fault_key: str, state: np.ndarray, t: float,
                    params: Dict[str, Any]) -> np.ndarray:
        """
        Apply a custom fault to the simulation state.

        Args:
            fault_key: The fault model key from register_fault_models()
            state: Current 14-element state vector [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, mass]
            t: Current simulation time (seconds)
            params: User-configured fault parameters

        Returns:
            Modified state vector
        """
        return state

    # --- Motor profile hooks ---

    def register_motor_profiles(self) -> List[MotorProfileDefinition]:
        """Register custom motor thrust profiles."""
        return []

    def get_thrust_at_time(self, motor_key: str, t: float,
                           params: Dict[str, Any]) -> float:
        """
        Get thrust magnitude at time t for a custom motor profile.

        Args:
            motor_key: The motor profile key
            t: Time since motor ignition (seconds)
            params: Motor configuration parameters

        Returns:
            Thrust in Newtons
        """
        return 0.0

    # --- Controller hooks ---

    def register_controllers(self) -> List[ControllerDefinition]:
        """Register custom control algorithms."""
        return []

    def compute_control(self, controller_key: str, state: np.ndarray,
                        target: np.ndarray, t: float,
                        params: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute TVC gimbal angles using a custom controller.

        Args:
            controller_key: The controller key
            state: Current 14-element state vector
            target: Target state (typically [0, 0, 0] for landing)
            t: Current simulation time
            params: Controller parameters

        Returns:
            Tuple of (gimbal_y, gimbal_z) angles in radians
        """
        return (0.0, 0.0)

    # --- Atmosphere hooks ---

    def get_atmosphere(self, altitude: float, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Provide custom atmospheric conditions at a given altitude.

        Args:
            altitude: Height above ground (meters)
            params: Atmosphere model parameters

        Returns:
            Dict with keys: 'density' (kg/m³), 'pressure' (Pa),
                           'temperature' (K), 'wind_x' (m/s),
                           'wind_y' (m/s), 'wind_z' (m/s)
        """
        return {}

    # --- Simulation lifecycle hooks ---

    def pre_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before simulation starts. Can modify the config.

        Args:
            config: Simulation configuration dict

        Returns:
            Possibly modified configuration dict
        """
        return config

    def post_simulation(self, config: Dict[str, Any],
                        results: Dict[str, Any]) -> None:
        """
        Called after simulation completes.

        Args:
            config: Simulation configuration that was used
            results: Simulation results dict
        """
        pass

    def on_timestep(self, state: np.ndarray, t: float, dt: float) -> None:
        """
        Called on each simulation timestep (for logging/monitoring).

        Args:
            state: Current state vector
            t: Current time
            dt: Timestep size
        """
        pass
