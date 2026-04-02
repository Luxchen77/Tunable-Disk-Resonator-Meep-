import meep as mp
import numpy as np
import os, math, time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

main_run = False
video_run = True
flux_reading = False

# -----------------------------------
# Simulation parameters
# -----------------------------------
c0 = 299792458  # m/s
um_scale = 1e-6  # 1 µm

sigma_meep = 0.00
n_eff = 2.9933
gaas = mp.Medium(epsilon=n_eff**2, D_conductivity=2*math.pi*1*sigma_meep/n_eff)
disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02
gap_tune = 0.05
tw = 0.08  # tuner width

cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(0.8)]

nfreq = 3000
field_decay = 1e-4
resolution = 48

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

# ---------------------------------
# Base directory
# ---------------------------------
base_dir = "data"

# ---------------------------------
# Choose which simulation runs to plot
# ---------------------------------
selected_folders = [
    "20251212_1708_norm_r3.5_g0.02_f318.000_res48_decay0.0005_tw0.08/tunergap_0.050um",
]

# ---------------------------------
# Helper: load flux data from HDF5
# ---------------------------------
def load_flux_data(h5_file):
    with h5py.File(h5_file, "r") as f:
        freq = f["frequency"][:]       # Meep units (1/µm)
        flux_bus = f["flux_bus"][:]
        flux_drop = f["flux_drop"][:]
    return freq, flux_bus, flux_drop

# ---------------------------------
# Helper: parse parameters from folder name
# ---------------------------------
def parse_params_from_name(folder_name):
    pattern = (
        r"(?P<date>\d{8})_(?P<time>\d{4})_sim_r(?P<r>[\d\.]+)_g(?P<g>[\d\.]+)"
        r"_f(?P<f>[\d\.]+)_res(?P<res>\d+)_decay(?P<decay>[\deE\-\+\.]+)"
    )
    match = re.search(pattern, folder_name)
    return match.groupdict() if match else {}

# ---------------------------------
# Collect resonance data
# ---------------------------------
resonance_summary = []

# ---------------------------------
# Main logic for animation
# ---------------------------------
for folder in selected_folders:
    run_dir = os.path.join(base_dir, folder)
    top_freqs_thz = []

    if flux_reading:
        flux_file = os.path.join(run_dir, "flux_data.h5")
        if not os.path.exists(flux_file):
            print(f"No flux_data.h5 found in {run_dir}, skipping flux analysis.")
        else:
            freq_meep, flux_bus, flux_drop = load_flux_data(flux_file)
            norm_flux_path = os.path.join(run_dir, "norm_flux.npy")
            norm_freqs_path = os.path.join(run_dir, "norm_freqs.npy")
            if os.path.exists(norm_flux_path) and os.path.exists(norm_freqs_path):
                norm_flux = np.load(norm_flux_path)
                norm_freqs = np.load(norm_freqs_path)
                if not np.allclose(freq_meep, norm_freqs):
                    norm_flux = np.interp(freq_meep, norm_freqs, norm_flux)
                flux_drop_normalized = flux_drop / norm_flux
            else:
                print(f"No normalization data found in {run_dir}, using raw flux.")
                flux_drop_normalized = flux_drop
            freq_thz = freq_meep * (c0 / um_scale) / 1e12
            peaks, _ = find_peaks(-flux_drop_normalized, prominence=0.001)
            if len(peaks) > 0:
                peak_vals = np.abs(flux_drop_normalized[peaks])
                top_n = min(3, len(peak_vals))
                top_indices = np.argsort(peak_vals)[-top_n:][::-1]
                top_peaks = peaks[top_indices]
                top_freqs_thz = freq_thz[top_peaks]
            else:
                print(f"No resonance peak found in {folder}, using manual frequency.")
                top_freqs_thz = [311.060]
    else:
        # Manual frequency for animation if no flux data
        top_freqs_thz = [314.6629, 319.6251, 321.3204, 322.8625]
    # -----------------------------------
    # Set up new simulation for animation at resonance frequency
    # -----------------------------------
    for i, res_freq_thz in enumerate(top_freqs_thz):

        # Parse parameters from folder name (optional)
        params = parse_params_from_name(folder)
        param_text = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "params unavailable"

        src_center = mp.Vector3(-wg_length / 2 + 1.5, disk_radius + gap + wg_width / 2)
        sources = [mp.Source(mp.ContinuousSource(frequency=res_freq_thz * um_scale * 1e12 / c0),
                            component=mp.Hz,
                            center=src_center,
                            size=mp.Vector3(0, wg_width, 0))]

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
                    angle_start=-np.pi/4,
                    angle_end=np.pi/4,
                    npoints=256,
                    material=gaas)
        ]

        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=resolution
        )

        # -----------------------------------
        # Animation / MP4 + snapshot
        # -----------------------------------
        if video_run:

            sim.reset_meep()

            # ---- Video title ----
            video_title = (
                f"{folder}\n"
                f"{param_text}\n"
                f"Peak {i+1}: {res_freq_thz:.2f} THz"
            )

            animate = mp.Animate2D(
                sim=sim,
                fields=mp.Hz,
                normalize=True,
                realtime=False,
                field_parameters={"alpha": 0.8, "cmap": "RdBu", "interpolation": "none"},
                boundary_parameters={"hatch": "o", "linewidth": 1.5, "facecolor": "y",
                                    "edgecolor": "b", "alpha": 0.3},
                output_plane=mp.Volume(center=mp.Vector3(), size=cell),
                title=video_title
            )

            # --- Run simulation ---
            sim.run(mp.at_every(20, animate), until=1000)

            # --- Save MP4 ---
            mp4_filename = os.path.join(run_dir, f"animation_peak{i+1}_{res_freq_thz:.3f}THz_tw_{tw:.2f}_tg_{gap_tune:.3f}_res{resolution}_gap{gap:.3f}.mp4")
            animate.to_mp4(fps=15, filename=mp4_filename)

            # ---------------------------------------------------
            # Snapshot of Hz field at final time (fancier version)
            # ---------------------------------------------------
            hz_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)

            fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)

            im = ax.imshow(
                hz_data.T,
                interpolation="bicubic",
                cmap="RdBu",
                origin="lower",
                extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2]
            )

            # --- Overplot waveguides + disk outline ---
            # Disk
            disk = plt.Circle((0, 0), disk_radius, fill=False, color="black", linewidth=0.6)
            ax.add_patch(disk)

            # Waveguides
            ax.add_patch(plt.Rectangle(
                (-wg_length/2, disk_radius + gap),
                wg_length, wg_width,
                fill=False, edgecolor="black", linewidth=0.6
            ))
            ax.add_patch(plt.Rectangle(
                (-wg_length/2, -disk_radius - gap - wg_width),
                wg_length, wg_width,
                fill=False, edgecolor="black", linewidth=0.6
            ))

            # Tuner outer arc (approx)
            theta = np.linspace(-np.pi/4, np.pi/4, 300)
            r_t = disk_radius + gap_tune + tw
            r_i = disk_radius + gap_tune
            ax.plot(r_i*np.cos(theta), r_i*np.sin(theta), "k--", lw=0.5)
            ax.plot(r_t*np.cos(theta), r_t*np.sin(theta), "k--", lw=0.5)

            # --- Fancy title box ---
            param_text = "\n".join([f"{k}: {v}" for k, v in params.items()]) if params else "No parameters parsed"

            title_text = (
                f"{folder}\n"
                f"Resonance frequency: {res_freq_thz:.2f} THz\n\n"
                f"{param_text}"
            )

            ax.set_title(title_text, fontsize=10, loc="left", pad=15)

            # --- Colorbar (styled) ---
            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label("Hz field amplitude", fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_aspect("equal")
            ax.set_xlim(-cell_x/2, cell_x/2)
            ax.set_ylim(-cell_y/2, cell_y/2)
            plt.tight_layout()

            snapshot_filename = os.path.join(run_dir, f"snapshot_peak{i+1}_{res_freq_thz:.2f}THz.png")
            plt.savefig(snapshot_filename, dpi=200)
            plt.close()

