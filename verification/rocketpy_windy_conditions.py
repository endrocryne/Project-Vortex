
import numpy as np
from rocketpy import Environment, SolidMotor, Rocket, Flight

# Parameters from windy_conditions.py
thrust_time = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1])
thrust_values = np.array([0.0, 25.0, 22.0, 18.0, 15.0, 14.0, 13.5, 13.0, 12.5, 12.0, 10.0, 7.0, 3.0, 0.0])
mass_dry = 0.150  # kg
mass_fuel = 0.025  # kg
rocket_length = 0.5  # m
rocket_diameter = 0.04  # m
cd = 0.5
cp_position = 0.4 # m from nose
isp = 120

# 1. Environment Setup
Env = Environment(
    gravity=9.80665,
    date=(2024, 1, 1, 12),
    latitude=32.990254,
    longitude=-106.974998,
)
Env.set_atmospheric_model(type='StandardAtmosphere', pressure=101325, temperature=288.15)
# Strong wind: 10 m/s from East (90 degrees)
Env.set_wind(10.0, 90)

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
    grain_outer_radius=0.013,
    grain_initial_inner_radius=0.004,
    grain_initial_height=0.07,
    interpolation_method='linear',
    nozzle_position=0.45,
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
VortexRocket.add_motor(ProMotor, position=rocket_length - 0.07)

Fins = VortexRocket.add_trapezoidal_fins(
    n=4,
    root_chord=0.04,
    tip_chord=0.02,
    span=0.04,
    position=0.42,
    cant_angle=0,
    radius=None,
    airfoil=None,
)
VortexRocket.cp_pos = cp_position

# 4. Flight Setup
# Launch vertically
TestFlight = Flight(
    rocket=VortexRocket,
    environment=Env,
    inclination=90,
    heading=0,
    rail_length=0,
    terminate_on_apogee=True
)

# Run simulation and print results
TestFlight.all_info()
