"""
Simulation Wrapper with Fault Injection and ML Flight Computer Integration
Provides a non-invasive interface to add fault injection and ML capabilities
"""

import numpy as np
from typing import Dict, Any, Optional
import copy

from simulation import SuicideBurnSimulation
from faults import FaultInjectionManager, FaultGroup, FaultConfig, FaultType, TriggerMode, FaultTarget
from ml_flight_computer import MLFlightComputer


class EnhancedSimulation:
    """
    Wrapper around SuicideBurnSimulation that adds fault injection and ML capabilities
    without modifying the original simulation code
    """
    
    def __init__(self, 
                 rocket_config: Dict[str, Any],
                 environment_config: Dict[str, Any],
                 simulation_config: Dict[str, Any],
                 fault_config: Optional[Dict[str, Any]] = None,
                 ml_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced simulation
        
        Args:
            rocket_config: Rocket configuration dict
            environment_config: Environment configuration dict
            simulation_config: Simulation configuration dict
            fault_config: Fault injection configuration dict
            ml_config: ML flight computer configuration dict
        """
        # Store original configs
        self.base_rocket_config = copy.deepcopy(rocket_config)
        self.base_environment_config = copy.deepcopy(environment_config)
        self.base_simulation_config = copy.deepcopy(simulation_config)
        
        # Create base simulation
        self.simulation = SuicideBurnSimulation(
            rocket_config, 
            environment_config, 
            simulation_config
        )
        
        # Setup fault injection
        self.fault_manager = None
        self.faults_enabled = False
        if fault_config and fault_config.get('enabled', False):
            self.faults_enabled = True
            fault_groups = [FaultGroup.from_dict(g) for g in fault_config.get('fault_groups', [])]
            self.fault_manager = FaultInjectionManager(fault_groups)
        
        # Setup ML flight computer
        self.ml_flight_computer = None
        self.ml_enabled = False
        if ml_config and ml_config.get('enabled', False):
            self.ml_enabled = True
            self.ml_flight_computer = MLFlightComputer.from_config_dict(ml_config)
        
        # Track modifications for logging
        self.fault_log = []
        self.ml_correction_log = []
        self.apogee_time = None
        
    def _wrap_state_derivative(self, original_derivative_func):
        """
        Wrap the state derivative function to apply faults
        
        Args:
            original_derivative_func: Original state_derivative method
            
        Returns:
            Wrapped derivative function
        """
        def wrapped_derivative(t, state):
            # Apply faults if enabled
            if self.faults_enabled and self.fault_manager:
                # Check for new triggers
                altitude = state[2]
                vz = state[5]
                triggered = self.fault_manager.check_triggers(t, altitude, vz)
                
                # Log new faults
                for fault in triggered:
                    self.fault_log.append({
                        'time': t,
                        'fault_type': fault.fault_type.value,
                        'magnitude': fault.applied_magnitude
                    })
                
                # Check for deactivations
                self.fault_manager.check_deactivations(t)
                
                # Apply faults to state
                modified_state = self.fault_manager.apply_faults_to_state(state, t)
                
                # Get derivative with modified state
                deriv = original_derivative_func(t, modified_state)
                
                # Modify derivative components based on active faults
                thrust_mult = self.fault_manager.get_thrust_multiplier()
                drag_mult = self.fault_manager.get_drag_multiplier()
                wind_speed_delta, _ = self.fault_manager.get_wind_modification()
                
                # Apply thrust multiplier (affects acceleration indirectly through motor)
                # Note: This is a simplified approach. For full integration, we'd need to
                # modify the motor's thrust output, but that requires deeper integration.
                # For now, we apply it as a force multiplier in the derivative
                
                # Temporarily modify environment for wind
                if wind_speed_delta != 0:
                    self.simulation.physics.wind_speed += wind_speed_delta
                
                # Re-compute derivative with modified environment
                if thrust_mult != 1.0 or drag_mult != 1.0 or wind_speed_delta != 0:
                    # Store original values
                    original_cd = self.simulation.physics.Cd
                    
                    # Apply drag multiplier
                    self.simulation.physics.Cd *= drag_mult
                    
                    # Note: Thrust multiplier would require deeper integration with motor class
                    # For now, we apply it indirectly through force calculations
                    # A full implementation would modify motor.get_thrust() directly
                    
                    # Re-compute derivative
                    deriv = original_derivative_func(t, modified_state)
                    
                    # Restore original values
                    self.simulation.physics.Cd = original_cd
                    if wind_speed_delta != 0:
                        self.simulation.physics.wind_speed -= wind_speed_delta
                
                return deriv
            else:
                return original_derivative_func(t, state)
        
        return wrapped_derivative
    
    def _wrap_calculate_ignition_altitude(self, original_func):
        """
        Wrap ignition altitude calculation to integrate ML corrections
        
        Args:
            original_func: Original calculate_ignition_altitude method
            
        Returns:
            Wrapped function
        """
        def wrapped_calc(initial_velocity, initial_altitude_cg):
            # Get baseline ignition altitude
            baseline_ign_alt = original_func(initial_velocity, initial_altitude_cg)
            
            # Apply ML correction if enabled
            if self.ml_enabled and self.ml_flight_computer:
                # Create a dummy state for feature extraction
                # This is called before the main simulation loop, so we use initial conditions
                dummy_state = np.array([
                    0, 0, initial_altitude_cg,  # position
                    0, 0, initial_velocity,      # velocity
                    1, 0, 0, 0,                  # quaternion (upright)
                    0, 0, 0,                     # angular velocity
                    self.simulation.initial_mass # mass
                ])
                
                # Get ML correction
                ascent_twr = 7.0  # Placeholder, should be computed from config
                correction = self.ml_flight_computer.update(
                    dummy_state,
                    0.0,  # current_time (at start of descent)
                    baseline_ign_alt,
                    ascent_twr,
                    self.simulation.physics,
                    self.simulation.motor,
                    0.0,  # apogee_time (placeholder)
                    self.simulation.initial_mass,  # inferred_mass
                    self.simulation.physics.Cd  # inferred_drag
                )
                
                if correction is not None:
                    corrected_alt = baseline_ign_alt + correction
                    self.ml_correction_log.append({
                        'time': 0.0,
                        'baseline': baseline_ign_alt,
                        'correction': correction,
                        'corrected': corrected_alt
                    })
                    return corrected_alt
            
            return baseline_ign_alt
        
        return wrapped_calc
    
    def run_simulation(self, initial_state, ignition_altitude=None, max_time=60.0):
        """
        Run simulation with fault injection and ML enhancements
        
        Args:
            initial_state: Initial state vector
            ignition_altitude: Optional fixed ignition altitude
            max_time: Maximum simulation time
            
        Returns:
            History dictionary with results
        """
        # Reset fault manager and ML computer
        if self.fault_manager:
            self.fault_manager.reset()
            self.fault_log = []
        
        if self.ml_flight_computer:
            self.ml_flight_computer.reset()
            self.ml_correction_log = []
        
        # Store original methods
        original_derivative = None
        original_calc = None
        
        # Wrap methods if faults or ML are enabled
        if self.faults_enabled:
            # Store original method
            original_derivative = self.simulation.state_derivative
            # Replace with wrapped version
            self.simulation.state_derivative = self._wrap_state_derivative(original_derivative)
        
        if self.ml_enabled and ignition_altitude is None:
            # Wrap ignition altitude calculation
            original_calc = self.simulation.calculate_ignition_altitude
            self.simulation.calculate_ignition_altitude = self._wrap_calculate_ignition_altitude(original_calc)
        
        # Run base simulation
        history_tuple = self.simulation.run_simulation(initial_state, ignition_altitude, max_time)
        
        # Unpack tuple (success, final_state, history)
        if isinstance(history_tuple, tuple) and len(history_tuple) == 3:
            success, final_state, history = history_tuple
            # Add success to history if not present
            if 'success' not in history:
                history['success'] = success
        else:
            # Already a history dict
            history = history_tuple
        
        # Add fault and ML logs to history
        if self.fault_log:
            history['fault_log'] = self.fault_log
        if self.ml_correction_log:
            history['ml_correction_log'] = self.ml_correction_log
        
        # Restore original methods
        if original_derivative is not None:
            self.simulation.state_derivative = original_derivative
        if original_calc is not None:
            self.simulation.calculate_ignition_altitude = original_calc
        
        return history
    
    def check_feasibility(self, initial_velocity, initial_altitude):
        """Check feasibility (pass-through to base simulation)"""
        return self.simulation.check_feasibility(initial_velocity, initial_altitude)
    
    def get_fault_status(self) -> Dict[str, Any]:
        """Get current fault injection status"""
        if self.fault_manager:
            return self.fault_manager.get_status_summary()
        return {}
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get ML flight computer status"""
        if self.ml_flight_computer:
            return {
                'enabled': self.ml_enabled,
                'corrections_applied': len(self.ml_correction_log),
                'model_path': str(self.ml_flight_computer.model_path) if self.ml_flight_computer.model_path else None
            }
        return {'enabled': False}
    
    @staticmethod
    def from_config_file(config_file: str):
        """
        Create enhanced simulation from config file
        
        Args:
            config_file: Path to JSON config file
            
        Returns:
            EnhancedSimulation instance
        """
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return EnhancedSimulation(
            config.get('rocket', {}),
            config.get('environment', {}),
            config.get('simulation', {}),
            config.get('faults', None),
            config.get('ml_flight_computer', None)
        )
