import csv
import numpy as np

filename = r'c:\Users\rishi\Documents\Vortex\Project-Vortex\results\single_run_20260127_113248\single_run.csv'
with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

times = [float(row['Time']) for row in data]
z = [float(row['Z']) for row in data]
vz = [float(row['VZ']) for row in data]
qw = [float(row['QW']) for row in data]
qx = [float(row['QX']) for row in data]
qy = [float(row['QY']) for row in data]
qz = [float(row['QZ']) for row in data]

print(f"Apogee: {max(z):.2f}m")
apogee_idx = z.index(max(z))

# Find ignition
ignition_idx = -1
for i in range(apogee_idx + 1, len(vz) - 1):
    if vz[i] < vz[i-1] and vz[i+1] > vz[i]:
        ignition_idx = i
        break

if ignition_idx != -1:
    print(f"Ignition at T={times[ignition_idx]:.2f}s, Z={z[ignition_idx]:.2f}m, VZ={vz[ignition_idx]:.2f}m/s")
    # Check orientation (calculate pitch, yaw)
    w, x, y, z_q = qw[ignition_idx], qx[ignition_idx], qy[ignition_idx], qz[ignition_idx]
    # Pitch error (2*qy)
    print(f"Angle at ignition: Pitch error approx {np.degrees(2*y):.1f} deg, Yaw error approx {np.degrees(2*x):.1f} deg")
    
    # Check final state
    print(f"Final Z: {z[-1]:.2f}m, Final VZ: {vz[-1]:.2f}m/s")
    w, x, y, z_q = qw[-1], qx[-1], qy[-1], qz[-1]
    print(f"Final angle: Pitch error {np.degrees(2*y):.1f} deg, Yaw error {np.degrees(2*x):.1f} deg")
else:
    print("Ignition not detected by velocity change")
