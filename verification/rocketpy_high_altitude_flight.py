
import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight

# Parameters from high_altitude_flight.py
thrust_time = np.array([0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.6])
thrust_values = np.array([0.0, 120.0, 110.0, 90.0, 80.0, 75.0, 60.0, 30.0, 0.0])
mass_dry = 0.400  # kg
mass_fuel = 0.060  # kg
rocket_length = 0.8  # m
rocket_diameter = 0.05  # m
cd = 0.45
cp_position = 0.65 # m from nose
isp = 140

# 1. Environment Setup
Env = Environment(
    gravity=9.80665,
    date=(2024, 1, 1, 12),
    latitude=32.990254,
    longitude=-106.974998,
)
Env.set_atmospheric_model(type='StandardAtmosphere', pressure=101325, temperature=288.15)
Env.set_wind(2.0, 0) # 2 m/s wind from the North

# 2. Motor Setup
ProMotor = SolidMotor(
    thrust_source=(thrust_time, thrust_values),
    burn_time=thrust_time[-1],
    propellant_initial_mass=mass_fuel,
    lato_mass=mass_fuel,
    nozzle_radius=0.01,
    throat_radius=0.005,
    grain_number=1,
    grain_density=1800,
    grain_outer_radius=0.018,
    grain_initial_inner_radius=0.006,
    grain_initial_height=0.10,
    interpolation_method='linear',
    nozzle_position=0.70,
    coordinate_system_orientation='nozzle_to_combustion_chamber'
)
ProMotor.isp = isp

# 3. Rocket Setup
VortexRocket = Rocket(
    radius=rocket_diameter / 2,
    mass=mass_dry,
    inertia=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    power_off_drag=cd,
    power_on_drag=cd,
    center_of_mass_without_propellant=rocket_length / 2,
    coordinate_system_orientation='tail_to_nose'
)
VortexRocket.add_motor(ProMotor, position=rocket_length - 0.10)

Fins = VortexRocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.05,
    tip_chord=0.03,
    span=0.05,
    position=0.67,
    cant_angle=0,
    radius=None,
    airfoil=None,
)
VortexRocket.cp_pos = cp_position

# 4. Flight Setup
# Launch with 1 degree tilt from vertical
TestFlight = Flight(
    rocket=VortexRocket,
    environment=Env,
    inclination=89, # 90 - 1
    heading=0,
    rail_length=0,
    terminate_on_apogee=True
)

# Run simulation and print results
TestFlight.all_info()
