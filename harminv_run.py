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

tuner = False
gap_tune = 0.05
tw = 0.05  # tuner width

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


# -----------------------------------
# Helper for curved tuner
# -----------------------------------
def arc_prism(radius, width, angle_start, angle_end, npoints, material=gaas):
    r_in = radius
    r_out = radius + width
    outer = [mp.Vector3(r_out*np.cos(a), r_out*np.sin(a)) for a in np.linspace(angle_start, angle_end, npoints)]
    inner = [mp.Vector3(r_in*np.cos(a), r_in*np.sin(a)) for a in np.linspace(angle_end, angle_start, npoints)]
    vertices = outer + inner
    return mp.Prism(vertices=vertices, height=mp.inf, material=material)

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
# Source (broadband run)
# -----------------------------
source_x = -wg_length/2 + 1.5
source_y = disk_radius + gap + wg_width/2

for resolution in [48]:
    for gap_tune in [0.05]:
        for tw in [0.05]:
            f_thz = 320
            f_cen = f_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)
            df_thz = 20  # n THz
            df = df_thz * um_scale * 1e12 / c0  # convert to Meep freq (1/um)
            field_decay = 5e-2  # field decay for stopping condition
            sim_time = 30000
            nfreq = 30000  # single value


            geometry_tuned = [
                mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas, center=mp.Vector3()),
                mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                        center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                        material=gaas),
                mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                        center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
                        material=gaas),
                arc_prism(radius=disk_radius + gap_tune,
                        width=tw,
                        angle_start=-np.pi/3,
                        angle_end=np.pi/3,
                        npoints=256,
                        material=gaas)
            ]

            # -----------------------------
            sources = [mp.Source(mp.GaussianSource(frequency=f_cen, fwidth=df),
                                component=mp.Hz,
                                center=mp.Vector3(source_x, source_y),
                                size=mp.Vector3(0, wg_width, 0))]

            sim = mp.Simulation(cell_size=cell,
                                eps_averaging=True,
                                geometry=geometry,
                                sources=sources,
                                boundary_layers=pml_layers,
                                resolution=resolution)

            # Harminv analysis at disk center
            # Harminv analysis at 45-degree angle, slightly inside the disk edge
            angle_deg = 90
            angle_rad = math.radians(angle_deg)
            r_harminv = disk_radius - 0.1  # 0.1 µm inside the edge
            harminv_point = mp.Vector3(
                r_harminv * math.cos(angle_rad),
                r_harminv * math.sin(angle_rad)
            )
            harminv_obj = mp.Harminv(mp.Hz, harminv_point, fcen=f_cen, df=df)

            if tuner:
                harminv_tuner_point = mp.Vector3(disk_radius + gap_tune + tw / 2, 0)
                harminv_obj_tuner = mp.Harminv(mp.Hz, harminv_tuner_point, fcen=f_cen, df=df)

                print("Running Harminv analysis...")
                start_time = time.time()
                #sim.run(harminv_obj, harminv_obj_tuner, until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, harminv_point, field_decay))
                sim.run(harminv_obj, harminv_obj_tuner, until_after_sources=sim_time)
                runtime = time.time() - start_time
            else:
                print("Running Harminv analysis...")
                start_time = time.time()
                #sim.run(harminv_obj, until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, harminv_point, field_decay))
                sim.run(harminv_obj, until_after_sources=sim_time)
                runtime = time.time() - start_time

            if rank == 0:
                # -----------------------------
                # Create joint data folder for both runs
                # -----------------------------
                base_dir = "data_large_sweep"
                os.makedirs(base_dir, exist_ok=True)
                folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sim_norm_f{f_thz}_res{resolution}_simtime{sim_time}_harminv_tuner{tuner}_tw{tw}_gt{gap_tune}"
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
                    "runtime_seconds": runtime,
                    "runtime_minutes": runtime/60,
                    "tuner": tuner,
                    "tuner_gap": gap_tune,
                    "tuner_width": tw
                }

                params_path = os.path.join(run_dir, "params.txt")
                with open(params_path, "w") as f:
                    f.write("# Simulation parameters\n")
                    f.write("# ----------------------\n")
                    for key, value in params.items():
                        f.write(f"{key} = {value}\n")

                print(f"Parameters saved to {params_path}")

                # -----------------------------
                # Save Harminv data
                # -----------------------------

                def get_amplitude(mode):
                    if hasattr(mode, "amplitude"):
                        return abs(mode.amplitude)
                    elif hasattr(mode, "ampl"):
                        return abs(mode.ampl)
                    else:
                        return np.nan

                harminv_data = {
                    "freq_meep": np.array([m.freq for m in harminv_obj.modes]),
                    "Q": np.array([m.Q for m in harminv_obj.modes]),
                    "decay": np.array([m.decay for m in harminv_obj.modes]),
                    "amplitude": np.array([abs(m.amp) for m in harminv_obj.modes]),
                    "error": np.array([getattr(m, "err", np.nan) for m in harminv_obj.modes]), 
                }

                harminv_path = os.path.join(run_dir, "harminv_data.h5")
                with h5py.File(harminv_path, "w") as f:
                    for k, v in harminv_data.items():
                        f.create_dataset(k, data=v)
                print(f"Harminv data saved to {harminv_path}")
