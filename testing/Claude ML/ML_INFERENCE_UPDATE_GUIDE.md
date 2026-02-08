# Inference Code Update Guide

## Overview
This guide shows how to update `simulation_inject.py` to use the new 26-feature ML model.

---

## Required Changes

### 1. Update Feature Extraction

**Location:** `simulation_inject.py`, method `predict_ignition_altitude_ml`

**Current code extracts 8 features:**
```python
def predict_ignition_altitude_ml(self, state, est_mass, est_cd):
    if self.ml_model is None or self.ml_scaler is None:
        return None
    
    # Extract features [OLD - 8 features]
    features = np.array([[
        self.ascent_twr,                     # 1. ascent_twr
        state[5],                            # 2. descent_velocity (vz)
        state[2],                            # 3. current_altitude (z)
        est_mass,                            # 4. inferred_mass
        est_cd,                              # 5. inferred_drag_coeff
        288.15,                              # 6. ambient_temp
        self.environment_config.get('wind_speed', 0.0),  # 7. wind_speed
        self.calculate_ignition_altitude(state[5], state[2])  # 8. predicted
    ]])
```

**New code extracts 26 features:**
```python
def predict_ignition_altitude_ml(self, state, est_mass, est_cd):
    if self.ml_model is None or self.ml_scaler is None:
        return None
    
    # Extract quaternion and convert to Euler angles
    q = state[6:10]
    euler = self.physics.quaternion_to_euler(q)
    roll, pitch, yaw = euler
    
    # Calculate derived quantities
    horizontal_vel = np.sqrt(state[3]**2 + state[4]**2)
    prop_frac = max(0, (state[13] - self.rocket_config['dry_mass']) / 
                        self.rocket_config['propellant_mass'])
    air_density_at_alt = self.physics.get_air_density(state[2])
    
    # Extract thrust curve parameters
    thrust_curve = self.rocket_config.get('thrust_curve', [[0, 0], [0.1, 1000], [3.0, 1000], [3.1, 0]])
    peak_thrust = max([thrust for _, thrust in thrust_curve])
    burn_time = max([time for time, _ in thrust_curve])
    
    # NEW - 26 features in exact order
    features = np.array([[
        # Rocket characteristics (8)
        self.ascent_twr,                                          # 1. ascent_twr
        self.rocket_config.get('length', 5.0),                   # 2. rocket_length
        self.rocket_config.get('diameter', 0.3),                 # 3. rocket_diameter
        burn_time,                                               # 4. burn_time
        peak_thrust,                                             # 5. peak_thrust
        self.rocket_config.get('tvc_max_angle', 5.0),           # 6. tvc_max_angle
        self.rocket_config.get('tvc_response_time', 0.1),       # 7. tvc_response_time
        
        # 3D state (5)
        state[2],                                                # 8. current_altitude (z)
        state[5],                                                # 9. descent_velocity (vz)
        horizontal_vel,                                          # 10. horizontal_velocity
        state[3],                                                # 11. vx
        state[4],                                                # 12. vy
        
        # Attitude (6)
        pitch,                                                   # 13. pitch_angle
        yaw,                                                     # 14. yaw_angle
        roll,                                                    # 15. roll_angle
        state[10],                                               # 16. omega_x
        state[11],                                               # 17. omega_y
        state[12],                                               # 18. omega_z
        
        # Estimated state (3)
        est_mass,                                                # 19. inferred_mass
        est_cd,                                                  # 20. inferred_drag_coeff
        prop_frac,                                               # 21. propellant_fraction
        
        # Environment (4)
        self.environment_config.get('wind_speed', 0.0),         # 22. wind_speed
        self.environment_config.get('wind_direction', 0.0),     # 23. wind_direction
        air_density_at_alt,                                      # 24. air_density_at_altitude
        self.environment_config.get('drag_coefficient', 0.5),   # 25. base_drag_coefficient
        
        # Baseline prediction (1)
        self.calculate_ignition_altitude(state[5], state[2])    # 26. predicted_ignition_altitude
    ]])
    
    # Scale and predict
    features_scaled = self.ml_scaler.transform(features)
    prediction = self.ml_model.predict(features_scaled, verbose=0)[0][0]
    
    return float(prediction)
```

---

### 2. Add Euler Conversion Helper (if not already present)

**Location:** Add to `PhysicsEngine` class in `physics_engine.py`

```python
def quaternion_to_euler(self, q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        q: quaternion [w, x, y, z]
        
    Returns:
        np.array([roll, pitch, yaw]) in radians
    """
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    
    # Pitch (y-axis rotation)
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    
    # Yaw (z-axis rotation)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    
    return np.array([roll, pitch, yaw])
```

---

### 3. Update Model Loading

**Location:** `simulation_inject.py`, method `load_ml_model`

**Change model filename:**
```python
def load_ml_model(self, model_path='ignition_model_improved.keras', 
                        scaler_path='scaler_improved.pkl'):
    """Load the trained ML model and scaler if they exist."""
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            import tensorflow as tf
            import pickle
            
            self.ml_model = tf.keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.ml_scaler = pickle.load(f)
            
            print(f"ML model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            print(f"Model expects {self.ml_model.input_shape[1]} features")  # Should print 26
            
            return True
        else:
            print(f"ML model or scaler not found. Using analytical method only.")
            return False
    except Exception as e:
        print(f"Error loading ML model: {e}")
        self.ml_model = None
        self.ml_scaler = None
        return False
```

---

### 4. Ensure Config Files Have Required Parameters

**Add to rocket config** (if not already present):
```json
{
  "rocket": {
    "length": 5.0,
    "diameter": 0.3,
    "tvc_max_angle": 5.0,
    "tvc_response_time": 0.1,
    ...
  },
  "environment": {
    "wind_direction": 0.0,  // Add if missing
    ...
  }
}
```

---

## Testing the Updated Code

### Step 1: Backup Original
```bash
cp simulation_inject.py simulation_inject.py.backup
```

### Step 2: Make Changes
Apply the code updates above.

### Step 3: Test Feature Extraction
Add a test at the end of `simulation_inject.py`:

```python
if __name__ == "__main__":
    # Test feature extraction
    import json
    
    # Load a config
    with open('config_realistic.json', 'r') as f:
        config = json.load(f)
    
    sim = SuicideBurnSimulation(
        config['rocket'],
        config['environment'],
        config['simulation']
    )
    
    # Load ML model
    sim.load_ml_model('ignition_model_improved.keras', 'scaler_improved.pkl')
    
    # Create a test state
    test_state = np.array([
        0, 0, 300,        # x, y, z
        2, -1, -40,       # vx, vy, vz
        1, 0, 0, 0,       # qw, qx, qy, qz
        0.1, -0.05, 0,    # omega_x, omega_y, omega_z
        55.0              # mass
    ])
    
    # Test prediction
    est_mass = 54.5
    est_cd = 0.52
    
    prediction = sim.predict_ignition_altitude_ml(test_state, est_mass, est_cd)
    
    if prediction is not None:
        print(f"\n✅ Feature extraction successful!")
        print(f"   Predicted ignition altitude: {prediction:.2f} m")
    else:
        print("\n❌ Feature extraction failed!")
```

### Step 4: Run Test
```bash
python simulation_inject.py
```

Expected output:
```
ML model loaded from ignition_model_improved.keras
Scaler loaded from scaler_improved.pkl
Model expects 26 features

✅ Feature extraction successful!
   Predicted ignition altitude: 42.35 m
```

---

## Common Issues & Fixes

### Issue 1: "Input shape mismatch"
**Error:** `ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 26), found shape=(None, 8)`

**Fix:** You're still using the old 8-feature extraction code. Update to 26 features as shown above.

---

### Issue 2: Missing config parameters
**Error:** `KeyError: 'length'` or similar

**Fix:** Add default values with `.get()`:
```python
self.rocket_config.get('length', 5.0)  # Default to 5.0 if not in config
```

---

### Issue 3: Euler conversion not found
**Error:** `AttributeError: 'PhysicsEngine' object has no attribute 'quaternion_to_euler'`

**Fix:** Add the `quaternion_to_euler` method to `PhysicsEngine` class (see section 2 above).

---

## Verification Checklist

- [ ] `quaternion_to_euler` method added to `PhysicsEngine`
- [ ] `predict_ignition_altitude_ml` updated to extract 26 features
- [ ] Feature order matches training notebook exactly
- [ ] Model and scaler filenames updated to `_improved` versions
- [ ] Config files have `length`, `diameter`, `wind_direction` parameters
- [ ] Test script runs without errors
- [ ] Predictions seem reasonable (not NaN, not wildly off)

---

## Performance Comparison

After updating, you can compare predictions:

```python
# Analytical method (simple physics)
analytical = sim.calculate_ignition_altitude(vz=-40, z=300)

# ML method (learned from oracle)  
ml_prediction = sim.predict_ignition_altitude_ml(test_state, est_mass, est_cd)

print(f"Analytical: {analytical:.2f} m")
print(f"ML (improved): {ml_prediction:.2f} m")
print(f"Difference: {abs(ml_prediction - analytical):.2f} m")
```

The ML prediction should be more accurate (closer to oracle-optimized value) than the analytical estimate.

---

## Rollback Instructions

If something goes wrong:

```bash
# Restore backup
cp simulation_inject.py.backup simulation_inject.py

# Use old model files
# The old load_ml_model will look for:
# - ignition_model.keras
# - scaler.pkl
```

---

## Next Steps After Update

1. **Generate new training data** using `train_model_improved.ipynb`
2. **Train the new model**
3. **Copy files to simulation directory:**
   ```bash
   cp ignition_model_improved.keras /path/to/simulation/
   cp scaler_improved.pkl /path/to/simulation/
   ```
4. **Run full simulation tests** with new model
5. **Compare performance** against analytical and old ML methods

---

## Questions?

If you encounter issues not covered here, check:
1. Feature ordering matches training notebook exactly
2. All 26 features are present and in correct order
3. Model and scaler files are from the _improved_ training run
4. Config files have all required parameters

The most common mistake is feature ordering - the order in inference code MUST match the order in training exactly!
