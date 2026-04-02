import math, os, time
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "4"

# -------------------------------------
# Constants
# -------------------------------------
c0 = 299792458
um_scale = 1e-6

# -------------------------------------
# Materials
# -------------------------------------
sigma_meep = 0.000
n_eff = 2.9933
gaas = mp.Medium(epsilon= n_eff**2, D_conductivity=2*math.pi*1*sigma_meep/n_eff)
air = mp.Medium(epsilon=1)

# -------------------------------------
# Geometry
# -------------------------------------
disk_radius = 3.5
wg_length = 20
wg_width = 0.22
gap = 0.1

cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 10
cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(2.0)]
resolution = 36  # pixels/um

geometry = [
    mp.Cylinder(radius=disk_radius, height=mp.inf, center=mp.Vector3(0, 0), material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
             material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
             material=gaas)
]

# -------------------------------------
# Frequency sweep setup (input in THz)
# -------------------------------------
freqs_thz = np.linspace(322, 324, 800)  # THz sweep range
freqs_meep = freqs_thz * um_scale * 1e12 / c0

transmissions = []

# Source and flux positions (bottom waveguide)
src_pos = mp.Vector3(-wg_length/2 + 2, -disk_radius - gap - wg_width/2)
flux_pos = mp.Vector3(wg_length/2 - 2, -disk_radius - gap - wg_width/2)
flux_size = mp.Vector3(0, wg_width, 0)

# -------------------------------------
# Output directory setup
# -------------------------------------
base_dir = "data_cw"
os.makedirs(base_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{timestamp}_cw_sweep_r{disk_radius}_g{gap}_res{resolution}"
run_dir = os.path.join(base_dir, folder_name)
os.makedirs(run_dir, exist_ok=True)

# -------------------------------------
# Frequency sweep loop
# -------------------------------------
t_total_start = time.time()

for i, (f_thz, f_meep) in enumerate(zip(freqs_thz, freqs_meep)):
    print(f"\n[{i+1}/{len(freqs_thz)}] Running {f_thz:.3f} THz ...")
    t_start = time.time()

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=[mp.Source(mp.ContinuousSource(frequency=f_meep),
                           component=mp.Ez,
                           center=src_pos)]
    )

    trans_flux = sim.add_flux(f_meep, 0, 1, mp.FluxRegion(center=flux_pos, size=flux_size))

    sim.run(until=200 / f_meep)  # ~300 optical periods

    flux_val = mp.get_fluxes(trans_flux)[0]
    transmissions.append(flux_val)

    print(f"    Transmission = {flux_val:.3e}, runtime = {time.time()-t_start:.1f}s")

t_total = time.time() - t_total_start

# -------------------------------------
# Normalize and plot
# -------------------------------------
transmissions = np.array(transmissions)
trans_norm = transmissions / np.max(transmissions)

plt.figure(figsize=(12, 4))
plt.plot(freqs_thz, trans_norm, 'o-', lw=1.5)
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Transmission")
plt.title(f"Transmission spectrum — {folder_name}")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------
# Save results
# -------------------------------------
np.save(os.path.join(run_dir, "freqs_thz.npy"), freqs_thz)
np.save(os.path.join(run_dir, "transmission_raw.npy"), transmissions)
np.save(os.path.join(run_dir, "transmission_norm.npy"), trans_norm)

np.savetxt(os.path.join(run_dir, "spectrum.txt"),
           np.column_stack([freqs_thz, trans_norm]),
           header="Frequency (THz)\tNormalized Transmission")

params = {
    "disk_radius": disk_radius,
    "gap": gap,
    "wg_length": wg_length,
    "wg_width": wg_width,
    "resolution": resolution,
    "sigma_meep": sigma_meep,
    "freq_start_THz": freqs_thz[0],
    "freq_end_THz": freqs_thz[-1],
    "num_points": len(freqs_thz),
    "runtime_total_s": round(t_total, 2)
}

with open(os.path.join(run_dir, "params.txt"), "w") as f:
    for k, v in params.items():
        f.write(f"{k} = {v}\n")

print("\n✅ Sweep complete.")
print("Results saved to:", run_dir)
