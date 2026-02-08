import csv
import sys

filename = r'c:\Users\rishi\Documents\Vortex\Project-Vortex\results\single_run_20260127_113248\single_run.csv'
with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

times = [float(row['Time']) for row in data]
z = [float(row['Z']) for row in data]
vz = [float(row['VZ']) for row in data]

print(f"Max Z: {max(z)}")
print(f"Min VZ: {min(vz)}")
print(f"Initial Z: {z[0]}")
print(f"Final Z: {z[-1]}")
print(f"Final VZ: {vz[-1]}")

# Find index of apogee (max Z)
apogee_idx = z.index(max(z))
print(f"Apogee Time: {times[apogee_idx]}")

# Find ignition (when vz starts increasing in descent)
ignition_idx = -1
for i in range(apogee_idx + 1, len(vz) - 1):
    if vz[i] < vz[i-1] and vz[i+1] > vz[i]:
        ignition_idx = i
        break

if ignition_idx != -1:
    print(f"Ignition Time: {times[ignition_idx]}")
    print(f"Ignition Z: {z[ignition_idx]}")
    print(f"Ignition VZ: {vz[ignition_idx]}")
else:
    print("Ignition not found")
