
import json
import numpy as np
from simulation import SuicideBurnSimulation

# Config from results/single_run_20260128_154553/config.json
config = {
  "rocket": {
    "dry_mass": 10.0,
    "propellant_mass": 1.0,
    "length": 5.0,
    "diameter": 0.3,
    "use_dynamic_inertia": False,
    "thrust_curve": [
      [0.0, 0.0],
      [0.1, 500.0],
      [3.0, 500.0],
      [3.1, 0.0]
    ],
    "burn_time": 3.0,
    "tvc_max_angle": 5.0,
    "tvc_response_time": 0.1,
    "tvc_kp_pitch": 0.5,
    "tvc_ki_pitch": 0.05,
    "tvc_kd_pitch": 0.1,
    "tvc_kp_yaw": 0.5,
    "tvc_ki_yaw": 0.05,
    "tvc_kd_yaw": 0.1,
    "thrust_variation": 0.05,
    "tvc_response_variation": 0.1,
    "mass_variation": 0.02,
    "ascent_motor_casing_mass": 1.0
  },
  "environment": {
    "gravity": 9.81,
    "air_density": 1.225,
    "temperature": 288.15,
    "drag_coefficient": 0.5,
    "reference_area": 0.07068583470577035,
    "wind_model": "constant",
    "wind_speed": 5.0,
    "wind_direction": 0.0,
    "drag_variation": 0.1,
    "air_density_variation": 0.05,
    "initial_altitude": 0.0,
    "initial_velocity": 0.0
  },
  "simulation": {
    "num_monte_carlo": 100,
    "altitude_search_range": 10.0,
    "altitude_step": 0.1,
    "altimeter_error": 0.01,
    "velocity_sensor_error": 0.01,
    "ignition_percent_offset": 0.5,
    "ignition_hard_offset": 0.0,
    "show_plots": True,
    "simulate_ascent": True,
    "ascent_initial_pitch": 0.0,
    "ascent_initial_yaw": 0.0,
    "ascent_initial_roll": 0.0
  }
}

def run():
    print("Initializing simulation...")
    sim = SuicideBurnSimulation(config['rocket'], config['environment'], config['simulation'])
    
    # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, mass]
    initial_mass = config['rocket']['dry_mass'] + config['rocket']['propellant_mass']
    initial_state = np.array([0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 
                              1.0, 0.0, 0.0, 0.0, 
                              0.0, 0.0, 0.0, 
                              initial_mass])
    
    print("Running simulation...")
    success, final_state, history = sim.run_simulation(initial_state, max_time=60.0)
    
    print(f"Simulation finished. Success: {success}")
    print(f"Final Altitude: {history['z'][-1]}")
    print(f"Final VZ: {history['vz'][-1]}")
    print(f"History length: {len(history['t'])}")
    
    if len(history['t']) > 0:
        print(f"Last time: {history['t'][-1]}")

if __name__ == "__main__":
    run()
