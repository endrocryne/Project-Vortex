import json
from pathlib import Path
from simulation import SuicideBurnSimulation

cfg_path = Path('results/single_run_20260128_145428/config.json')
cfg = json.loads(cfg_path.read_text())
rocket, env, simc = cfg['rocket'], cfg['environment'], cfg['simulation']

sim = SuicideBurnSimulation(rocket, env, simc)

initial_mass = rocket['dry_mass'] + rocket['propellant_mass']
initial_state = [0,0,0, 0,0,0, 1,0,0,0, 0,0,0, initial_mass]

success, final_state, history = sim.run_simulation(initial_state, max_time=60.0)
print('success', success)
print('final altitude', history['final_altitude'])
print('final vz', history['vz'][-1])
print('final speed', history['final_velocity'])
print('last time', history['t'][-1])
