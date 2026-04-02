import math
import os

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import datetime
import time

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

c0 = 299792458  # m/s
um_scale = 1e-6  # 1 µm in m

# -----------------------------
# Simulation parameters
# -----------------------------

# Materials
#gaas = mp.Medium(epsilon=np.square(3.2)) # not really gaas, just kept the name, now epsilon used from paper
#gaas = mp.Medium(epsilon=12, E_susceptibilities=[mp.LorentzianSusceptibility(frequency=0, gamma=0.5, sigma=1)])
sigma_meep = 0.000
n_eff = 2.9933
gaas = mp.Medium(epsilon= n_eff**2, D_conductivity=2*math.pi*1*sigma_meep/n_eff)
#gaas = mp.Medium(epsilon=12)
air = mp.Medium(epsilon=1)

# Disk and waveguide geometry
disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02  # distance between disk and waveguides

# Simulation cell size
cell_x = wg_length
cell_y = 2*(disk_radius + gap + wg_width/2) + 3.5
cell = mp.Vector3(cell_x, cell_y, 0)

# PML layers
pml_layers = [mp.PML(0.8)]

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

geometry_norm = [
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width/2),
             material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, -disk_radius - gap - wg_width/2),
             material=gaas)
]


# -----------------------------
# Source (broadband run)
# -----------------------------
source_x = -wg_length/2 + 1.5
source_y = disk_radius + gap + wg_width/2


for resolution in [64, 100]:
    f_thz = 320
    f_cen = f_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)
    df_thz = 20  # n THz
    df = df_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)
    field_decay = 5e-4  # field decay for stopping condition
    #resolution = 48  # pixels/um
    nfreq = 20000  # single value



    # -----------------------------
    # Normalization Simulation (geometry_norm)
    # -----------------------------
    norm_sources = [mp.Source(mp.GaussianSource(frequency=f_cen, fwidth=df),
                            component=mp.Hz,
                            center=mp.Vector3(source_x, source_y),
                            size=mp.Vector3(0, wg_width, 0))]

    norm_flux_region = mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                                    size=mp.Vector3(0, wg_width, 0))

    norm_sim = mp.Simulation(cell_size=cell,
                            geometry=geometry_norm,
                            sources=norm_sources,
                            boundary_layers=pml_layers,
                            resolution=resolution)

    norm_flux = norm_sim.add_flux(f_cen, df, nfreq, norm_flux_region)

    print("Running normalization simulation...")
    norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))

    norm_freqs = np.array(mp.get_flux_freqs(norm_flux))
    norm_flux_data = np.array(mp.get_fluxes(norm_flux))


    # -----------------------------
    # Main Simulation (disk + two waveguides)
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

    sources = [mp.Source(mp.GaussianSource(frequency=f_cen, fwidth=df),
                        component=mp.Hz,
                        center=mp.Vector3(source_x, source_y),
                        size=mp.Vector3(0, wg_width, 0))]

    flux_region_bus = mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5,
                                                    disk_radius + gap + wg_width/2),
                                    size=mp.Vector3(0, wg_width, 0))

    flux_region_drop = mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1.5,
                                                    -disk_radius - gap - wg_width/2),
                                    size=mp.Vector3(0, wg_width, 0))

    sim = mp.Simulation(cell_size=cell,
                        eps_averaging=True,
                        geometry=geometry,
                        sources=sources,
                        boundary_layers=pml_layers,
                        resolution=resolution)

    trans_flux_bus = sim.add_flux(f_cen, df, nfreq, flux_region_bus)
    trans_flux_drop = sim.add_flux(f_cen, df, nfreq, flux_region_drop)

    print("Running main simulation...")
    start_time = time.time()
    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))
    runtime = time.time() - start_time

    frequencies = np.array(mp.get_flux_freqs(trans_flux_bus))
    flux_bus = np.array(mp.get_fluxes(trans_flux_bus))
    flux_drop = np.array(mp.get_fluxes(trans_flux_drop))

    if rank == 0:

        # -----------------------------
        # Create joint data folder for both runs
        # -----------------------------
        base_dir = "data"
        os.makedirs(base_dir, exist_ok=True)
        folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sim_norm_r{disk_radius}_g{gap}_f{f_thz}_res{resolution}_decay{field_decay}"
        run_dir = os.path.join(base_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save normalization data in joint folder
        np.save(os.path.join(run_dir, "norm_freqs.npy"), norm_freqs)
        np.save(os.path.join(run_dir, "norm_flux.npy"), norm_flux_data)
        print(f"Normalization data saved to {run_dir}")

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
            "runtime_seconds": runtime,
            "runtime_minutes": runtime/60,
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
