import numpy as np
import matplotlib.pyplot as plt
from rocketpy import Environment, SolidMotor, Rocket, Flight

# --- Parameters from the original script ---

# Rocket Properties
DRY_MASS_MEAN = 0.060  # 60 grams
ROCKET_DIAMETER = 0.034  # 34mm diameter

# Motor Properties
PROPELLANT_MASS_MEAN = 0.0125  # 12.5g
TIME_POINTS = np.array([
    0.0, 0.01248, 0.014, 0.026, 0.067, 0.099, 0.15, 0.183, 0.207, 0.219,
    0.262, 0.333, 0.349, 0.392, 0.475, 0.653, 0.913, 1.366, 1.607, 1.745,
    1.978, 2.023, 2.024
])
THRUST_POINTS_BASE = np.array([
    0.0, 0.0227, 0.633, 1.533, 2.726, 5.136, 9.103, 11.465, 11.635, 11.391,
    6.377, 5.014, 5.209, 4.722, 4.771, 4.746, 4.673, 4.625, 4.625, 4.868,
    4.795, 0.828, 0.0
])
thrust_curve = np.array([TIME_POINTS, THRUST_POINTS_BASE]).T

# Environment
GROUND_WIND_SPEED_MEAN = 4.0  # m/s

# --- 1. Environment Setup ---
print("Setting up the environment...")
env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
env.set_atmospheric_model(
    type="custom_atmosphere",
    pressure=None,
    temperature=None,
    wind_u=lambda h: GROUND_WIND_SPEED_MEAN * (max(h, 0) / 10.0)**0.15,
    wind_v=0
)

# --- 2. Motor Definition ---
print("\nDefining the solid motor...")
motor = SolidMotor(
    thrust_source=thrust_curve,
    dry_mass=0.015,
    dry_inertia=(0.005, 0.005, 0.0001),
    nozzle_radius=0.005,
    burn_time=TIME_POINTS[-1],
    grain_number=1,
    grain_density=1800,
    grain_outer_radius=0.015,
    grain_initial_inner_radius=0.005,
    grain_initial_height=0.05,
    nozzle_position=0,
    interpolation_method="linear",
    grain_separation=0.005,
    grains_center_of_mass_position=-0.035,
    center_of_dry_mass_position=-0.035
)

# --- 3. Rocket Definition ---
print("\nDefining the rocket structure...")
rocket = Rocket(
    radius=ROCKET_DIAMETER / 2,
    mass=DRY_MASS_MEAN,
    inertia=(.007, .007, .0001),
    power_off_drag=0.6,
    power_on_drag=0.6,
    # --- ARGUMENT ADDED TO FIX THE ERROR ---
    # This specifies the center of mass of the rocket *before* the motor is added.
    # The value is an estimate in meters from the rocket's tail (position 0).
    center_of_mass_without_motor=0.15
)
rocket.add_motor(motor, position=0)
nose_cone = rocket.add_nose(length=0.10, kind="ogive", position=0.35)
fins = rocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.05,
    tip_chord=0.02,
    span=0.04,
    position=0.0
)

# --- 4. Parachute System ---
print("\nAdding parachute system...")
main_chute = rocket.add_parachute(
    "Main",
    cd_s=1.0,
    trigger="apogee",
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

# --- 5. Flight Simulation ---
print("\nSimulating flight...")
test_flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=1.0,
    inclination=85,
    heading=0,
    max_time=30.0
)

# --- 6. Display Results ---
print("\n--- Flight Results ---")
test_flight.prints.all()

# --- 7. Generate Plots ---
print("\nGenerating standard plots...")
test_flight.plots.trajectory_3d()
test_flight.plots.linear_kinematics()

# --- 8. Custom Position vs. Time Plot ---
print("\nGenerating custom Position vs. Time plot...")
plt.figure(figsize=(10, 6))
plt.plot(test_flight.t, test_flight.x, label='X Position (East)')
plt.plot(test_flight.t, test_flight.y, label='Y Position (North)')
plt.plot(test_flight.t, test_flight.z, label='Z Position (Altitude)', linewidth=2.5)
plt.title("Position Components vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)
plt.show()

print("\nSimulation complete. Close plot windows to exit.")