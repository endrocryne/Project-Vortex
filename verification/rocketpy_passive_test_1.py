#!/usr/bin/env python3
from rocketpy import Environment, SolidMotor, Rocket, Flight
import numpy as np

# Test 1: Heavy Rocket, Low Power
thrust_time = np.array([0.0, 0.1, 0.5, 0.8, 0.9])
thrust_values = np.array([0.0, 5.0, 4.5, 2.0, 0.0])

Env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
Env.set_atmospheric_model(type='StandardAtmosphere')

motor = SolidMotor(
    thrust_source=(thrust_time, thrust_values),
    burn_time=0.9,
    propellant_initial_mass=0.010,
    lato_mass=0.010,
    nozzle_radius=0.01,
    throat_radius=0.005,
    interpolation_method='linear'
)

rocket = Rocket(
    radius=0.06/2,
    mass=0.500,
    inertia=(0, 0, 0, 0, 0, 0),
    power_off_drag=0.6,
    power_on_drag=0.6,
    center_of_mass_without_propellant=0.7/2,
    coordinate_system_orientation='tail_to_nose'
)
rocket.add_motor(motor, position=0.7 - 0.05)
rocket.cp_pos = 0.5

flight = Flight(rocket=rocket, environment=Env, inclination=88, heading=0, rail_length=0, terminate_on_apogee=True)
flight.all_info()
