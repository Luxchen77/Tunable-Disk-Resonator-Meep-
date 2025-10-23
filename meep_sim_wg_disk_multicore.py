

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from datetime import datetime

c0 = 299792458  # m/s
um_scale = 1e-6  # 1 µm in m

# -----------------------------
# Simulation parameters
# -----------------------------

# Materials
#gaas = mp.Medium(epsilon=np.square(3.2)) # not really gaas, just kept the name, now epsilon used from paper
gaas = mp.Medium(epsilon=12)
air = mp.Medium(epsilon=1)

# Disk and waveguide geometry
disk_radius = 3.5
wg_length = 20
wg_width = 0.22
gap = 0.1  # distance between disk and waveguides

# Simulation cell size
cell_x = wg_length + 10
cell_y = 2*(disk_radius + gap + wg_width/2) + 10
cell = mp.Vector3(cell_x, cell_y, 0)

# PML layers
pml_layers = [mp.PML(2.0)]

# -----------------------------
# Geometry: disk + two waveguides
# -----------------------------
geometry = [
    mp.Cylinder(radius=disk_radius, height=mp.inf, center=mp.Vector3(0,0), material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width/2),
             material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, -disk_radius - gap - wg_width/2),
             material=gaas)
]



# -----------------------------

# -----------------------------
# Source (broadband run)
# -----------------------------
source_x = -wg_length/2
source_y = disk_radius + gap + wg_width/2

f_thz = 318  # target resonance frequency in THz
f_cen = f_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)
# bandwidth for Gaussian source
df_thz = 20  # in THz
df = df_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)


#fmin = 0.3
#fmax = 0.35
#df = fmax - fmin

nfreq = 2000
resolution = 20  # pixels/um
field_decay = 1e-3  # field decay for stopping condition

sources = [mp.Source(mp.GaussianSource(frequency=f_cen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(source_x, source_y),
                     size=mp.Vector3(0, wg_width, 0))]

# -----------------------------
# Flux monitors
# -----------------------------
flux_region_bus = mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1,
                                                  disk_radius + gap + wg_width/2),
                                size=mp.Vector3(0, wg_width, 0))

# Drop port moved to left side of lower waveguide (same side as source, opposite waveguide)
flux_region_drop = mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1,
                                                   -disk_radius - gap - wg_width/2),
                                 size=mp.Vector3(0, wg_width, 0))

# -----------------------------
# First Simulation (spectra)
# -----------------------------
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=sources,
                    boundary_layers=pml_layers,
                    resolution=resolution)

trans_flux_bus = sim.add_flux(f_cen, df, nfreq, flux_region_bus)
trans_flux_drop = sim.add_flux(f_cen, df, nfreq, flux_region_drop)

sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Ez, mp.Vector3(), field_decay))


frequencies = np.array(mp.get_flux_freqs(trans_flux_bus))
flux_bus = np.array(mp.get_fluxes(trans_flux_bus))
flux_drop = np.array(mp.get_fluxes(trans_flux_drop))

res_freq = frequencies[np.argmax(np.abs(flux_drop))]



# -----------------------------
# Create data folder structure
# -----------------------------
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

# Use parameter-based folder name
folder_name = f"sim_r{disk_radius}_g{gap}_f{f_thz}_res{resolution}_decay{field_decay}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join(base_dir, folder_name)
os.makedirs(run_dir, exist_ok=True)


# -----------------------------
# Save parameters
# -----------------------------
params = {
    "disk_radius": disk_radius,
    "gap": gap,
    "wg_length": wg_length,
    "wg_width": wg_width,
    "cell_size": (cell_x, cell_y),
    "f_thz": f_thz,
    "df_thz": df_thz,
    "n_freq_points": nfreq,
    "resolution": resolution,
    "field_decay": field_decay,
}


params_path = os.path.join(run_dir, "params.txt")
with open(params_path, "w") as f:
    f.write("# Simulation parameters\n")
    f.write("# ----------------------\n")
    for key, value in params.items():
        f.write(f"{key} = {value}\n")

print(f"Parameters saved to {params_path}")



# -----------------------------
# Save flux data
# -----------------------------
flux_path = os.path.join(run_dir, "flux_data.h5")

with h5py.File(flux_path, "w") as f:
    f.create_dataset("frequency", data=frequencies)
    f.create_dataset("flux_bus", data=flux_bus)
    f.create_dataset("flux_drop", data=flux_drop)

print(f"Flux data saved to {flux_path}")


# -----------------------------
# Save field data (Ez and epsilon)
# -----------------------------
# Get Ez and epsilon arrays from the final simulation state
ez_field = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
eps_data = sim.get_epsilon()

ez_path = os.path.join(run_dir, "Ez_field.npy")
eps_path = os.path.join(run_dir, "epsilon.npy")

np.save(ez_path, ez_field)
np.save(eps_path, eps_data)

print(f"Ez field saved to {ez_path}")
print(f"Epsilon field saved to {eps_path}")



# =====================================================
# SECOND SIMULATION — Field profile at resonance
# =====================================================
print("\n Running second simulation at resonance frequency...\n")

sources_res = [mp.Source(mp.ContinuousSource(frequency=res_freq),
                         component=mp.Ez,
                         center=mp.Vector3(source_x, source_y),
                         size=mp.Vector3(0, wg_width, 0))]

sim_res = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        sources=sources_res,
                        boundary_layers=pml_layers,
                        resolution=resolution)

ez_data = []

def store_fields(sim):
    ez_data.append(sim.get_array(center=mp.Vector3(),
                                 size=cell,
                                 component=mp.Ez))

# store Ez every 20 time units, total duration 4000
sim_res.run(mp.at_every(20, store_fields), until=4000)

# -----------------------------
# Save Ez time evolution data
# -----------------------------
ez_data = np.array(ez_data)
ez_data_path = os.path.join(run_dir, "Ez_time_evolution.npy")
np.save(ez_data_path, ez_data)

# Optional lightweight txt version (for debugging / quick plotting)
# np.savetxt(os.path.join(run_dir, "Ez_time_evolution.txt"), ez_data.reshape(ez_data.shape[0], -1))

print(f"Time-evolving Ez field saved to {ez_data_path}")
print(f"All data successfully stored in: {run_dir}")