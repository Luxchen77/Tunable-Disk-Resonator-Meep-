import meep as mp
import numpy as np
import os, math, time
from datetime import datetime
import matplotlib.pyplot as plt

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

main_run = False
video_run = True

# -----------------------------------
# Simulation parameters
# -----------------------------------
c0 = 299792458  # m/s
um_scale = 1e-6  # 1 µm

sigma_meep = 0.000
gaas = mp.Medium(epsilon=12, D_conductivity=2 * math.pi * 1 * sigma_meep / 12)
disk_radius = 3.5
wg_length = 20
wg_width = 0.22
gap = 0.1
gap_tune = 0.05


cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 8
cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(2.0)]

f_thz = 318
fcen = f_thz * um_scale * 1e12 / c0
df_thz = 2
fwidth = df_thz * um_scale * 1e12 / c0
nfreq = 3000
field_decay = 2e-4
resolution = 32

src_center = mp.Vector3(-wg_length / 2 + 4, disk_radius + gap + wg_width / 2)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                     component=mp.Hz,
                     center=src_center,
                     size=mp.Vector3(0, wg_width, 0))]

tw = 0.1  # tuner width


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


# -----------------------------------
# Geometry
# -----------------------------------
geometry = [
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


# -----------------------------------
# Simulation
# -----------------------------------
sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution
)

# Flux monitors
flux_top = sim.add_flux(
    fcen, fwidth, nfreq,
    mp.FluxRegion(center=mp.Vector3(wg_length/2 - 4, disk_radius + gap + wg_width/2),
                  size=mp.Vector3(0, wg_width, 0))
)
flux_bottom = sim.add_flux(
    fcen, fwidth, nfreq,
    mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 4, -disk_radius - gap - wg_width/2),
                  size=mp.Vector3(0, wg_width, 0))
)
flux_tuner = sim.add_flux(
    fcen, fwidth, nfreq,
    mp.FluxRegion(center=mp.Vector3(disk_radius + gap_tune + tw/2, 0),
                  size=mp.Vector3(tw, 0, 0))
)


if main_run:

    # Run the simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))

    # -----------------------------------
    # Extract flux data
    # -----------------------------------
    frequencies = np.array(mp.get_flux_freqs(flux_top)) * c0 / um_scale / 1e12  # in THz
    top_flux = np.array(mp.get_fluxes(flux_top))
    bottom_flux = np.array(mp.get_fluxes(flux_bottom))
    tuner_flux = np.array(mp.get_fluxes(flux_tuner))

    if rank == 0:
        # Base folder
        base_dir = "data"
        os.makedirs(base_dir, exist_ok=True)
        folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sim_r{disk_radius}_g{gap}_f{f_thz:.3f}_res{resolution}_decay{field_decay}_tunerthickness{tw:.3f}"
        run_dir = os.path.join(base_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save data
        np.save(os.path.join(run_dir, "freqs_thz.npy"), frequencies)
        np.save(os.path.join(run_dir, "flux_bus.npy"), top_flux)
        np.save(os.path.join(run_dir, "flux_drop.npy"), bottom_flux)
        np.save(os.path.join(run_dir, "flux_tuner.npy"), tuner_flux)

else:
    # If not main_run, still need to define base_dir, folder_name, run_dir for video output
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sim_r{disk_radius}_g{gap}_f{f_thz:.3f}_res{resolution}_decay{field_decay}_tunerthickness{tw:.3f}"
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)

if video_run:

    sim.reset_meep()
    # -----------------------------
    # Create the animation object
    # -----------------------------
    animate = mp.Animate2D(
        sim=sim,
        fields=mp.Hz,
        normalize=True,
        realtime=False,
        field_parameters={"alpha": 0.8, "cmap": "RdBu", "interpolation": "none"},
        boundary_parameters={"hatch": "o", "linewidth": 1.5, "facecolor": "y",
                            "edgecolor": "b", "alpha": 0.3},
        output_plane=mp.Volume(center=mp.Vector3(), size=cell)
    )

    # -----------------------------
    # Run the simulation, capturing frames
    # -----------------------------
    sim.run(mp.at_every(20, animate), until=8000)

    # -----------------------------
    # Export to MP4 in the data folder
    # -----------------------------
    mp4_filename = os.path.join(run_dir, "animation.mp4")
    animate.to_mp4(fps=15, filename=mp4_filename)
