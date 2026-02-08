"""
ML-Enhanced Flight Computer Wrapper
Integrates trained ML correction models with fault injection system
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import warnings


class MLFlightComputer:
    """
    Simulated flight computer with ML-based ignition altitude correction
    Runs periodically during descent to adjust ignition altitude based on current conditions
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 update_interval: float = 0.5,
                 enabled: bool = False):
        """
        Initialize ML flight computer
        
        Args:
            model_path: Path to .keras, .h5, or .tflite model file
            scaler_path: Path to .pkl scaler file
            update_interval: How often to run ML inference (seconds)
            enabled: Whether ML guidance is active
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.update_interval = update_interval
        self.enabled = enabled
        
        self.model = None
        self.scaler = None
        self.model_type = None  # 'keras' or 'tflite'
        self.last_update_time = -999.0
        self.correction_history = []
        
        # Feature names (must match training data)
        self.feature_names = [
            'baseline_ignition_altitude',
            'ascent_twr',
            'descent_velocity',
            'current_altitude',
            'vertical_acceleration',
            'lateral_velocity_x',
            'lateral_velocity_y',
            'lateral_position_x',
            'lateral_position_y',
            'pitch_angle',
            'roll_angle',
            'yaw_angle',
            'omega_x',
            'omega_y',
            'omega_z',
            'inferred_mass',
            'inferred_drag_coeff',
            'estimated_drag_area',
            'air_density',
            'ambient_temp',
            'wind_speed',
            'dynamic_pressure',
            'time_since_apogee',
            'thrust_available',
            'burn_time_remaining',
            'predicted_ignition_altitude'
        ]
        
        # Load model and scaler if provided
        if self.enabled and model_path:
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Load ML model and scaler
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler pickle file
        """
        from pathlib import Path
        model_path_obj = Path(model_path)
        
        if not model_path_obj.exists():
            warnings.warn(f"Model file not found: {model_path}")
            self.enabled = False
            return
        
        # Detect model type by extension
        if model_path_obj.suffix in ['.keras', '.h5']:
            self.model_type = 'keras'
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(model_path_obj))
            except Exception as e:
                warnings.warn(f"Failed to load Keras model: {e}")
                self.enabled = False
                return
        
        elif model_path_obj.suffix == '.tflite':
            self.model_type = 'tflite'
            try:
                import tensorflow as tf
                self.model = tf.lite.Interpreter(model_path=str(model_path_obj))
                self.model.allocate_tensors()
                self.input_details = self.model.get_input_details()
                self.output_details = self.model.get_output_details()
            except Exception as e:
                warnings.warn(f"Failed to load TFLite model: {e}")
                self.enabled = False
                return
        else:
            warnings.warn(f"Unsupported model format: {model_path_obj.suffix}")
            self.enabled = False
            return
        
        # Load scaler
        if scaler_path:
            scaler_path_obj = Path(scaler_path)
            if scaler_path_obj.exists():
                try:
                    with open(scaler_path_obj, 'rb') as f:
                        self.scaler = pickle.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load scaler: {e}")
                    self.scaler = None
            else:
                warnings.warn(f"Scaler file not found: {scaler_path}")
        
        self.enabled = True
    
    def should_update(self, current_time: float) -> bool:
        """Check if it's time to run ML inference"""
        if not self.enabled:
            return False
        return (current_time - self.last_update_time) >= self.update_interval
    
    def extract_features(self,
                        state: np.ndarray,
                        current_time: float,
                        baseline_ignition_alt: float,
                        ascent_twr: float,
                        physics_engine,
                        motor,
                        apogee_time: float,
                        inferred_mass: Optional[float] = None,
                        inferred_drag: Optional[float] = None) -> np.ndarray:
        """
        Extract feature vector from current simulation state
        
        Args:
            state: Current state vector [x,y,z,vx,vy,vz,qw,qx,qy,qz,ωx,ωy,ωz,mass]
            current_time: Current simulation time
            baseline_ignition_alt: Baseline ignition altitude (m)
            ascent_twr: Ascent thrust-to-weight ratio
            physics_engine: Physics engine instance
            motor: Motor instance
            apogee_time: Time of apogee
            inferred_mass: Inferred mass from state estimator
            inferred_drag: Inferred drag coefficient from state estimator
            
        Returns:
            Feature vector (26 features)
        """
        # Extract state components
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        qw, qx, qy, qz = state[6:10]
        wx, wy, wz = state[10:13]
        mass = state[13]
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = physics_engine.quaternion_to_euler(state[6:10])
        
        # Calculate derived quantities
        lateral_velocity = np.sqrt(vx**2 + vy**2)
        lateral_position = np.sqrt(x**2 + y**2)
        descent_velocity = vz
        vertical_accel = -physics_engine.g  # Approximate
        
        # Environmental conditions
        air_density = physics_engine.get_air_density(z)
        ambient_temp = physics_engine.get_temperature(z)
        wind_speed = physics_engine.wind_speed
        
        # Dynamic pressure
        velocity_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        dynamic_pressure = 0.5 * air_density * velocity_mag**2
        
        # Time since apogee
        time_since_apogee = max(0.0, current_time - apogee_time)
        
        # Motor parameters
        thrust_available = motor.get_thrust(0.0) if hasattr(motor, 'get_thrust') else 0.0
        burn_time_remaining = motor.burn_time if hasattr(motor, 'burn_time') else 0.0
        
        # State estimator values (use actual if not provided)
        if inferred_mass is None:
            inferred_mass = mass
        if inferred_drag is None:
            inferred_drag = physics_engine.Cd
        
        estimated_drag_area = inferred_drag * physics_engine.A_ref
        
        # Predicted ignition altitude (using same baseline for now)
        predicted_ignition_altitude = baseline_ignition_alt
        
        # Assemble feature vector
        features = np.array([
            baseline_ignition_alt,
            ascent_twr,
            descent_velocity,
            z,  # current_altitude
            vertical_accel,
            vx,  # lateral_velocity_x
            vy,  # lateral_velocity_y
            x,   # lateral_position_x
            y,   # lateral_position_y
            pitch,
            roll,
            yaw,
            wx,  # omega_x
            wy,  # omega_y
            wz,  # omega_z
            inferred_mass,
            inferred_drag,
            estimated_drag_area,
            air_density,
            ambient_temp,
            wind_speed,
            dynamic_pressure,
            time_since_apogee,
            thrust_available,
            burn_time_remaining,
            predicted_ignition_altitude
        ])
        
        return features
    
    def predict(self, features: np.ndarray) -> float:
        """
        Run ML inference to get ignition altitude correction
        
        Args:
            features: Feature vector (26 features)
            
        Returns:
            Predicted correction to ignition altitude (meters)
        """
        if not self.enabled or self.model is None:
            return 0.0
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            try:
                # Handle dict-based scaler (with 'scaler_X' key)
                if isinstance(self.scaler, dict):
                    scaler_X = self.scaler.get('scaler_X')
                    if scaler_X is not None:
                        features = scaler_X.transform(features)
                else:
                    # Direct scaler object
                    features = self.scaler.transform(features)
            except Exception as e:
                warnings.warn(f"Scaler transformation failed: {e}")
        
        # Run inference
        try:
            if self.model_type == 'keras':
                prediction = self.model.predict(features, verbose=0)
                correction = float(prediction[0, 0])
            
            elif self.model_type == 'tflite':
                self.model.set_tensor(self.input_details[0]['index'], features.astype(np.float32))
                self.model.invoke()
                prediction = self.model.get_tensor(self.output_details[0]['index'])
                correction = float(prediction[0, 0])
            
            else:
                correction = 0.0
            
            # Inverse transform correction if scaler available
            if self.scaler is not None and isinstance(self.scaler, dict):
                scaler_y = self.scaler.get('scaler_y')
                if scaler_y is not None:
                    try:
                        correction_scaled = np.array([[correction]])
                        correction = float(scaler_y.inverse_transform(correction_scaled)[0, 0])
                    except:
                        pass  # Use raw prediction
            
            return correction
        
        except Exception as e:
            warnings.warn(f"ML prediction failed: {e}")
            return 0.0
    
    def update(self, 
               state: np.ndarray,
               current_time: float,
               baseline_ignition_alt: float,
               ascent_twr: float,
               physics_engine,
               motor,
               apogee_time: float,
               inferred_mass: Optional[float] = None,
               inferred_drag: Optional[float] = None) -> Optional[float]:
        """
        Update ML flight computer and get ignition altitude correction
        
        Args:
            (Same as extract_features)
            
        Returns:
            Correction to ignition altitude (meters) or None if not time to update
        """
        if not self.should_update(current_time):
            return None
        
        # Extract features
        features = self.extract_features(
            state, current_time, baseline_ignition_alt, ascent_twr,
            physics_engine, motor, apogee_time, inferred_mass, inferred_drag
        )
        
        # Get prediction
        correction = self.predict(features)
        
        # Update state
        self.last_update_time = current_time
        
        # Log correction
        self.correction_history.append({
            'time': current_time,
            'altitude': state[2],
            'correction': correction,
            'baseline': baseline_ignition_alt
        })
        
        return correction
    
    def get_correction_history(self) -> list:
        """Get history of ML corrections"""
        return self.correction_history.copy()
    
    def reset(self):
        """Reset flight computer state"""
        self.last_update_time = -999.0
        self.correction_history = []
    
    def to_config_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'enabled': self.enabled,
            'model_path': str(self.model_path) if self.model_path else None,
            'scaler_path': str(self.scaler_path) if self.scaler_path else None,
            'update_interval': self.update_interval
        }
    
    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> 'MLFlightComputer':
        """Create from configuration dictionary"""
        return MLFlightComputer(
            model_path=d.get('model_path'),
            scaler_path=d.get('scaler_path'),
            update_interval=d.get('update_interval', 0.5),
            enabled=d.get('enabled', False)
        )
