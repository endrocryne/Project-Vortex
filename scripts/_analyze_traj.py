"""Quick analysis of the attached real trajectory data."""
import csv, os

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "optimization_20260222_120325", "trajectory.csv")
rows = list(csv.DictReader(open(path)))
# Strip column name whitespace
for r in rows:
    for k in list(r.keys()):
        if k != k.strip():
            r[k.strip()] = r[k]

print(f"Total rows: {len(rows)}")

# Find apogee
max_z = 0
apogee_i = 0
for i, r in enumerate(rows):
    z = float(r["Z"])
    if z > max_z:
        max_z = z
        apogee_i = i

ra = rows[apogee_i]
t_ap = float(ra["Time"])
print(f"Apogee: row {apogee_i}, t={t_ap:.3f}s, z={max_z:.2f}m, vz={float(ra['VZ']):.3f} m/s")

# Find ascent burn end (mass stops decreasing rapidly after initial burn)
for i in range(1, len(rows)):
    m = float(rows[i]["Mass"])
    mp = float(rows[i-1]["Mass"])
    t = float(rows[i]["Time"])
    if t > 1.5 and abs(m - mp) < 0.0001 and mp < 1.35:
        print(f"Ascent burn ends: row {i}, t={t:.3f}s, mass={m:.4f}")
        break

# Find descent ignition (mass starts decreasing again after coast)
for i in range(apogee_i, len(rows)-1):
    m = float(rows[i]["Mass"])
    mn = float(rows[i+1]["Mass"])
    if mn < m - 0.0001:
        r = rows[i]
        t_ign = float(r["Time"])
        z_ign = float(r["Z"])
        vz_ign = float(r["VZ"])
        print(f"Descent ignition: row {i}, t={t_ign:.3f}s, z={z_ign:.2f}m, vz={vz_ign:.3f} m/s, mass={m:.4f}")
        # Show descent burn duration
        for j in range(i, len(rows)):
            mj = float(rows[j]["Mass"])
            if j > i and mj == float(rows[j-1]["Mass"]):
                print(f"  Descent burn ends: row {j}, t={float(rows[j]['Time']):.3f}s")
                break
        break

# Key stats
r_last = rows[-1]
print(f"Landing: t={float(r_last['Time']):.3f}s, z={float(r_last['Z']):.2f}m, vz={float(r_last['VZ']):.3f} m/s, mass={float(r_last['Mass']):.4f}")

# Print descent detail
print("\n=== Descent detail (from row 740) ===")
for i in range(740, len(rows)):
    if i % 5 == 0 or i == len(rows)-1:
        r = rows[i]
        print(f"  [{i:4d}] t={float(r['Time']):7.3f} z={float(r['Z']):8.2f} vz={float(r['VZ']):8.3f} mass={float(r['Mass']):.4f}")

# Compute descent motor impulse from mass loss
m_at_ignition = None
for i in range(apogee_i, len(rows)-1):
    m = float(rows[i]["Mass"])
    mn = float(rows[i+1]["Mass"])
    if mn < m - 0.0001:
        m_at_ignition = m
        break
m_final = float(rows[-1]["Mass"])
print(f"\nDescent motor mass consumed: {m_at_ignition - m_final:.4f} kg (from {m_at_ignition:.4f} to {m_final:.4f})")
print(f"Coast mass: {float(rows[apogee_i]['Mass']):.4f} kg")
