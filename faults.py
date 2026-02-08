"""
Fault Injection System for Suicide Burn Simulation
Supports dynamic fault injection during simulation runtime
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy


class FaultType(Enum):
    """Enumeration of supported fault types"""
    MASS_LOSS = "mass_loss"
    THRUST_VAR = "thrust_var"
    DRAG_CHANGE = "drag_change"
    WIND_GUST = "wind_gust"


class TriggerMode(Enum):
    """Fault trigger modes"""
    ABSOLUTE_TIME = "absolute_time"
    TIME_SINCE_APOGEE = "time_since_apogee"
    ALTITUDE_THRESHOLD = "altitude_threshold"
    MANUAL = "manual"


class FaultTarget(Enum):
    """Where the fault affects"""
    SIMULATED_STATE = "simulated_state"  # Affects actual dynamics
    SENSOR_ONLY = "sensor_only"  # Only affects sensor readings


@dataclass
class FaultConfig:
    """Configuration for a single fault"""
    fault_type: FaultType
    trigger_mode: TriggerMode
    trigger_value: float  # time (s) or altitude (m) depending on mode
    magnitude: float  # Multiplier or absolute value depending on fault type
    duration: float  # Duration in seconds (0 = permanent)
    target: FaultTarget = FaultTarget.SIMULATED_STATE
    randomize_magnitude: bool = False
    magnitude_range: tuple = (0.8, 1.2)  # Min/max for randomization
    probability: float = 1.0  # Probability of occurring (0-1)
    active: bool = False  # Runtime flag
    start_time: float = 0.0  # When fault was activated
    applied_magnitude: float = 0.0  # Actual magnitude after randomization
    fault_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'fault_type': self.fault_type.value,
            'trigger_mode': self.trigger_mode.value,
            'trigger_value': self.trigger_value,
            'magnitude': self.magnitude,
            'duration': self.duration,
            'target': self.target.value,
            'randomize_magnitude': self.randomize_magnitude,
            'magnitude_range': list(self.magnitude_range),
            'probability': self.probability,
            'fault_id': self.fault_id
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'FaultConfig':
        """Create from dictionary"""
        return FaultConfig(
            fault_type=FaultType(d['fault_type']),
            trigger_mode=TriggerMode(d['trigger_mode']),
            trigger_value=d['trigger_value'],
            magnitude=d['magnitude'],
            duration=d['duration'],
            target=FaultTarget(d.get('target', 'simulated_state')),
            randomize_magnitude=d.get('randomize_magnitude', False),
            magnitude_range=tuple(d.get('magnitude_range', [0.8, 1.2])),
            probability=d.get('probability', 1.0),
            fault_id=d.get('fault_id', 0)
        )


@dataclass
class FaultGroup:
    """Group of faults that execute concurrently or sequentially"""
    faults: List[FaultConfig] = field(default_factory=list)
    concurrent: bool = True  # If True, all faults trigger together
    group_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'faults': [f.to_dict() for f in self.faults],
            'concurrent': self.concurrent,
            'group_id': self.group_id
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'FaultGroup':
        """Create from dictionary"""
        return FaultGroup(
            faults=[FaultConfig.from_dict(f) for f in d['faults']],
            concurrent=d.get('concurrent', True),
            group_id=d.get('group_id', 0)
        )


class FaultInjectionManager:
    """
    Manages fault injection during simulation
    Handles triggering, application, and deactivation of faults
    """
    
    def __init__(self, fault_groups: Optional[List[FaultGroup]] = None):
        """
        Initialize fault manager
        
        Args:
            fault_groups: List of fault groups to manage
        """
        self.fault_groups = fault_groups or []
        self.all_faults: List[FaultConfig] = []
        self.active_faults: List[FaultConfig] = []
        self.fault_history: List[Dict[str, Any]] = []
        self.apogee_time: Optional[float] = None
        self.apogee_detected: bool = False
        self.has_ascended: bool = False
        
        # Flatten all faults for easy access
        self._flatten_faults()
        
        # State modifications (accumulated)
        self.mass_delta = 0.0
        self.thrust_multiplier = 1.0
        self.drag_multiplier = 1.0
        self.wind_speed_delta = 0.0
        self.wind_direction_delta = 0.0
        
    def _flatten_faults(self):
        """Flatten all faults from groups into single list"""
        self.all_faults = []
        for group in self.fault_groups:
            self.all_faults.extend(group.faults)
    
    def add_fault_group(self, group: FaultGroup):
        """Add a fault group"""
        self.fault_groups.append(group)
        self._flatten_faults()
    
    def clear_faults(self):
        """Clear all faults and reset state"""
        self.fault_groups = []
        self.all_faults = []
        self.active_faults = []
        self.fault_history = []
        self.reset()
        
    def reset(self):
        """Reset apogee detection state"""
        self.apogee_time = None
        self.apogee_detected = False
        self.has_ascended = False
        self.mass_delta = 0.0
        self.thrust_multiplier = 1.0
        self.drag_multiplier = 1.0
        self.wind_speed_delta = 0.0
        self.wind_direction_delta = 0.0
        # Re-flatten to ensure any group modifications are picked up
        self._flatten_faults()
    
    def detect_apogee(self, current_time: float, vz: float, altitude: float):
        """
        Detect apogee based on vertical velocity and flight history
        
        Args:
            current_time: Current simulation time
            vz: Vertical velocity
            altitude: Current altitude
        """
        if not self.apogee_detected:
            # Track if we have started ascending
            # Require both velocity and a bit of altitude gain to be sure
            if vz > 2.0 and altitude > 10.0:
                self.has_ascended = True
            
            # Detect apogee if:
            # 1. We were ascending and now we are not (vz <= 0)
            # OR
            # 2. We are starting high up (initial condition) and not ascending
            if vz <= 0:
                # Extra check: avoid triggering at start if on pad
                if self.has_ascended:
                    self.apogee_time = current_time
                    self.apogee_detected = True
                elif current_time < 0.1 and altitude > 10.0:
                    # Case: starting already at altitude (descent-only sim)
                    self.apogee_time = current_time
                    self.apogee_detected = True
    
    def check_triggers(self, current_time: float, altitude: float, vz: float) -> List[FaultConfig]:
        """
        Check if any faults should be triggered
        
        Args:
            current_time: Current simulation time (s)
            altitude: Current altitude (m)
            vz: Vertical velocity (for apogee detection)
            
        Returns:
            List of newly triggered faults
        """
        # Detect apogee
        self.detect_apogee(current_time, vz, altitude)
        
        newly_triggered = []
        
        for fault in self.all_faults:
            if fault.active:
                continue  # Already active
            
            triggered = False
            
            # Check trigger condition
            if fault.trigger_mode == TriggerMode.ABSOLUTE_TIME:
                triggered = current_time >= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.TIME_SINCE_APOGEE:
                if self.apogee_detected and self.apogee_time is not None:
                    time_since_apogee = current_time - self.apogee_time
                    triggered = time_since_apogee >= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
                triggered = altitude <= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.MANUAL:
                # Manual faults don't auto-trigger
                triggered = False
            
            if triggered:
                # Check probability
                if np.random.random() <= fault.probability:
                    self._activate_fault(fault, current_time)
                    newly_triggered.append(fault)
        
        return newly_triggered
    
    def _activate_fault(self, fault: FaultConfig, current_time: float):
        """Activate a fault"""
        fault.active = True
        fault.start_time = current_time
        
        # Randomize magnitude if requested
        if fault.randomize_magnitude:
            fault.applied_magnitude = np.random.uniform(
                fault.magnitude_range[0], 
                fault.magnitude_range[1]
            )
        else:
            fault.applied_magnitude = fault.magnitude
        
        self.active_faults.append(fault)
        
        # Log activation
        self.fault_history.append({
            'time': current_time,
            'action': 'activate',
            'fault_type': fault.fault_type.value,
            'magnitude': fault.applied_magnitude,
            'duration': fault.duration,
            'fault_id': fault.fault_id
        })
    
    def manually_trigger_fault(self, fault_id: int, current_time: float):
        """Manually trigger a specific fault by ID"""
        for fault in self.all_faults:
            if fault.fault_id == fault_id and not fault.active:
                self._activate_fault(fault, current_time)
                break
    
    def check_deactivations(self, current_time: float) -> List[FaultConfig]:
        """
        Check if any active faults should be deactivated
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of deactivated faults
        """
        deactivated = []
        
        for fault in self.active_faults[:]:  # Copy list to allow removal
            if fault.duration > 0:  # 0 means permanent
                elapsed = current_time - fault.start_time
                if elapsed >= fault.duration:
                    self._deactivate_fault(fault, current_time)
                    deactivated.append(fault)
        
        return deactivated
    
    def _deactivate_fault(self, fault: FaultConfig, current_time: float):
        """Deactivate a fault"""
        fault.active = False
        self.active_faults.remove(fault)
        
        # Log deactivation
        self.fault_history.append({
            'time': current_time,
            'action': 'deactivate',
            'fault_type': fault.fault_type.value,
            'fault_id': fault.fault_id
        })
    
    def apply_faults_to_state(self, state: np.ndarray, current_time: float) -> np.ndarray:
        """
        Apply active faults to simulation state
        
        Args:
            state: Current state vector [x,y,z,vx,vy,vz,qw,qx,qy,qz,ωx,ωy,ωz,mass]
            current_time: Current simulation time
            
        Returns:
            Modified state vector
        """
        modified_state = state.copy()
        
        # Reset accumulated modifications
        self.mass_delta = 0.0
        
        for fault in self.active_faults:
            if fault.target != FaultTarget.SIMULATED_STATE:
                continue
            
            if fault.fault_type == FaultType.MASS_LOSS:
                # Apply mass loss (negative magnitude means loss)
                self.mass_delta += fault.applied_magnitude
                modified_state[13] = max(1.0, modified_state[13] + fault.applied_magnitude)
        
        return modified_state
    
    def get_thrust_multiplier(self) -> float:
        """Get current thrust multiplier from active faults"""
        multiplier = 1.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.THRUST_VAR:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    multiplier *= fault.applied_magnitude
        return multiplier
    
    def get_drag_multiplier(self) -> float:
        """Get current drag multiplier from active faults"""
        multiplier = 1.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.DRAG_CHANGE:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    multiplier *= fault.applied_magnitude
        return multiplier
    
    def get_wind_modification(self) -> tuple:
        """
        Get wind modifications from active faults
        
        Returns:
            (speed_delta, direction_delta) in m/s and degrees
        """
        speed_delta = 0.0
        direction_delta = 0.0
        
        for fault in self.active_faults:
            if fault.fault_type == FaultType.WIND_GUST:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    # Magnitude represents wind speed change
                    speed_delta += fault.applied_magnitude
        
        return speed_delta, direction_delta
    
    def apply_sensor_noise(self, sensor_readings: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sensor-only faults to sensor readings
        
        Args:
            sensor_readings: Dictionary of sensor values
            
        Returns:
            Modified sensor readings
        """
        modified = sensor_readings.copy()
        
        for fault in self.active_faults:
            if fault.target != FaultTarget.SENSOR_ONLY:
                continue
            
            # Apply sensor-specific modifications
            if fault.fault_type == FaultType.MASS_LOSS:
                # Would affect inferred mass from accelerometer
                if 'inferred_mass' in modified:
                    modified['inferred_mass'] *= fault.applied_magnitude
            
            # Add more sensor-specific logic as needed
        
        return modified
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current fault injection status"""
        return {
            'total_faults': len(self.all_faults),
            'active_faults': len(self.active_faults),
            'apogee_detected': self.apogee_detected,
            'apogee_time': self.apogee_time,
            'active_fault_types': [f.fault_type.value for f in self.active_faults],
            'fault_history_count': len(self.fault_history)
        }
    
    def get_fault_log(self) -> List[Dict[str, Any]]:
        """Get complete fault history log"""
        return self.fault_history.copy()
    
    def to_config_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'fault_groups': [g.to_dict() for g in self.fault_groups]
        }
    
    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> 'FaultInjectionManager':
        """Create manager from configuration dictionary"""
        groups = [FaultGroup.from_dict(g) for g in d.get('fault_groups', [])]
        return FaultInjectionManager(groups)


def create_default_fault_configs() -> List[FaultGroup]:
    """Create default fault configurations for testing"""
    # Example: Mass loss at 50% through descent
    mass_loss = FaultConfig(
        fault_type=FaultType.MASS_LOSS,
        trigger_mode=TriggerMode.TIME_SINCE_APOGEE,
        trigger_value=2.0,
        magnitude=-5.0,  # Lose 5 kg
        duration=0.0,  # Permanent
        probability=0.0,  # Disabled by default
        fault_id=1
    )
    
    # Example: Thrust variation
    thrust_var = FaultConfig(
        fault_type=FaultType.THRUST_VAR,
        trigger_mode=TriggerMode.ALTITUDE_THRESHOLD,
        trigger_value=500.0,
        magnitude=0.8,  # 80% thrust
        duration=1.0,  # 1 second
        probability=0.0,
        fault_id=2
    )
    
    # Example: Drag increase
    drag_change = FaultConfig(
        fault_type=FaultType.DRAG_CHANGE,
        trigger_mode=TriggerMode.ABSOLUTE_TIME,
        trigger_value=10.0,
        magnitude=1.5,  # 150% drag
        duration=2.0,
        probability=0.0,
        fault_id=3
    )
    
    # Example: Wind gust
    wind_gust = FaultConfig(
        fault_type=FaultType.WIND_GUST,
        trigger_mode=TriggerMode.ALTITUDE_THRESHOLD,
        trigger_value=200.0,
        magnitude=10.0,  # +10 m/s wind
        duration=3.0,
        probability=0.0,
        fault_id=4
    )
    
    # Create groups (all individual for now)
    groups = [
        FaultGroup(faults=[mass_loss], concurrent=False, group_id=1),
        FaultGroup(faults=[thrust_var], concurrent=False, group_id=2),
        FaultGroup(faults=[drag_change], concurrent=False, group_id=3),
        FaultGroup(faults=[wind_gust], concurrent=False, group_id=4),
    ]
    
    return groups
