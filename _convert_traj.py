import csv, json, os

base = 'results/demo_runs'
demos = [
    'demo_01_baseline', 'demo_02_wind_opt', 'demo_03_wind_ml',
    'demo_04_dragmass_opt', 'demo_05_dragmass_ml',
    'demo_06_severe_opt', 'demo_07_severe_ml',
    'demo_08_extreme_opt', 'demo_09_extreme_ml'
]

# Load manifest
with open(os.path.join(base, 'demo_manifest.json')) as f:
    manifest = json.load(f)

# Downsample every 5th row for compact embedding
all_data = {}
for d in demos:
    path = os.path.join(base, d, 'trajectory.csv')
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        all_rows = list(reader)
        for i in range(0, len(all_rows), 5):
            vals = [round(float(v), 4) for v in all_rows[i]]
            rows.append(vals)
        if (len(all_rows) - 1) % 5 != 0:
            vals = [round(float(v), 4) for v in all_rows[-1]]
            rows.append(vals)
    all_data[d] = rows
    print(f'{d}: {len(rows)} sampled rows')

# Generate JS data block
traj_js = json.dumps(all_data, separators=(',', ':'))
manifest_js = json.dumps(manifest, separators=(',', ':'))

with open('_traj_data_js.txt', 'w') as f:
    f.write(f'const TRAJECTORY_DATA = {traj_js};\n')
    f.write(f'const MANIFEST = {manifest_js};\n')

size = os.path.getsize('_traj_data_js.txt')
print(f'JS data block: {size} bytes ({size/1024:.1f} KB)')
print('Done - wrote _traj_data_js.txt')
