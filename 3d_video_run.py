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
n_eff = 3.5125
gaas = mp.Medium(epsilon=n_eff**2, D_conductivity=2*math.pi*1*sigma_meep/n_eff)
disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02
gap_tune = 0.01
tw = 0.05  # tuner width

# Structure thickness
slab_thickness = 0.16
pml_thickness_z = 0.5

cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
cell_z = slab_thickness + 2 * pml_thickness_z
cell = mp.Vector3(cell_x, cell_y, cell_z)
pml_layers = [mp.PML(0.8, direction=mp.X),
              mp.PML(0.8, direction=mp.Y),
              mp.PML(pml_thickness_z, direction=mp.Z)]

# Use z-mirror symmetry
symmetries = [mp.Mirror(mp.Z, phase=+1)]

nfreq = 3000
field_decay = 1e-4
resolution = 48

# -----------------------------------
# Helper for curved tuner (3D version)
# -----------------------------------
def arc_prism(radius, width, angle_start, angle_end, npoints, material=gaas):
    r_in = radius
    r_out = radius + width
    outer = [mp.Vector3(r_out*np.cos(a), r_out*np.sin(a)) for a in np.linspace(angle_start, angle_end, npoints)]
    inner = [mp.Vector3(r_in*np.cos(a), r_in*np.sin(a)) for a in np.linspace(angle_end, angle_start, npoints)]
    vertices = outer + inner
    return mp.Prism(vertices=vertices, height=slab_thickness, material=material, center=mp.Vector3(0, 0, 0))

# ---------------------------------
# Base directory
# ---------------------------------
base_dir = "data"

# ---------------------------------
# Choose which simulation runs to plot
# ---------------------------------
selected_folders = [
    "20260105_2001_3d_norm_r3.5_g0.02_f318.000_res48_decay0.02_tw0.05/tunergap_0.050um",
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
        r"(?P<date>\d{8})_(?P<time>\d{4})_3d_norm_r(?P<r>[\d\.]+)_g(?P<g>[\d\.]+)"
        r"_f(?P<f>[\d\.]+)_res(?P<res>\d+)_decay(?P<decay>[\deE\-\+\.]+)"
    )
    match = re.search(pattern, folder_name)
    return match.groupdict() if match else {}

# ---------------------------------
# Collect resonance data
# ---------------------------------
resonance_summary = []

# ---------------------------------
# Main logic for animation and visualization
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
        top_freqs_thz = [314.1213, 319.4106, 320.6986, 322.2747, 323.3118, 327.2]
    
    # -----------------------------------
    # Set up new simulation for animation at resonance frequency
    # -----------------------------------
    for i, res_freq_thz in enumerate(top_freqs_thz):

        # Parse parameters from folder name (optional)
        params = parse_params_from_name(folder)
        param_text = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "params unavailable"

        src_center = mp.Vector3(-wg_length / 2 + 1.5, disk_radius + gap + wg_width / 2, 0)
        sources = [mp.Source(mp.ContinuousSource(frequency=res_freq_thz * um_scale * 1e12 / c0),
                            component=mp.Hz,
                            center=src_center,
                            size=mp.Vector3(0, wg_width, slab_thickness))]

        geometry = [
            mp.Cylinder(radius=disk_radius, height=slab_thickness, material=gaas, center=mp.Vector3(0, 0, 0)),
            mp.Block(size=mp.Vector3(wg_length, wg_width, slab_thickness),
                    center=mp.Vector3(0, disk_radius + gap + wg_width / 2, 0),
                    material=gaas),
            mp.Block(size=mp.Vector3(wg_length, wg_width, slab_thickness),
                    center=mp.Vector3(0, -disk_radius - gap - wg_width / 2, 0),
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
            symmetries=symmetries,
            resolution=resolution
        )

        # -----------------------------------
        # Animation / MP4 for z=0 slice
        # -----------------------------------
        if video_run:

            sim.reset_meep()

            # ---- Video title ----
            video_title = (
                f"{folder}\n"
                f"{param_text}\n"
                f"Peak {i+1}: {res_freq_thz:.2f} THz (z=0 slice)"
            )
            # Animate at z = slab_thickness/4 (middle of upper half-slab due to symmetry)
            z_slice = slab_thickness / 4
            animate = mp.Animate2D(
                sim=sim,
                fields=mp.Hz,
                normalize=True,
                realtime=False,
                field_parameters={"alpha": 0.8, "cmap": "RdBu", "interpolation": "none"},
                boundary_parameters={"hatch": "o", "linewidth": 1.5, "facecolor": "y",
                                    "edgecolor": "b", "alpha": 0.3},
                output_plane=mp.Volume(center=mp.Vector3(0, 0, z_slice), size=mp.Vector3(cell_x, cell_y, 0)),
                title=video_title
            )

            # --- Run simulation ---
            sim.run(mp.at_every(20, animate), until=1000)

            # --- Save MP4 ---
            mp4_filename = os.path.join(run_dir, f"animation_3d_peak{i+1}_{res_freq_thz:.3f}THz_tw_{tw:.2f}_tg_{gap_tune:.3f}_res{resolution}_gap{gap:.3f}.mp4")
            animate.to_mp4(fps=15, filename=mp4_filename)

            # ---------------------------------------------------
            # Multi-slice snapshot (xy at z=0, xz at y=0, yz at x=0)
            # ---------------------------------------------------
            hz_xy = sim.get_array(center=mp.Vector3(0, 0, 0), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Hz)
            hz_xz = sim.get_array(center=mp.Vector3(0, 0, 0), size=mp.Vector3(cell_x, 0, cell_z), component=mp.Hz)
            hz_yz = sim.get_array(center=mp.Vector3(0, 0, 0), size=mp.Vector3(0, cell_y, cell_z), component=mp.Hz)

            fig = plt.figure(figsize=(15, 5), dpi=150)
            
            # XY slice (z=0)
            ax1 = fig.add_subplot(131)
            im1 = ax1.imshow(
                hz_xy.T,
                interpolation="bicubic",
                cmap="RdBu",
                origin="lower",
                extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2],
                vmin=-np.abs(hz_xy).max(), vmax=np.abs(hz_xy).max()
            )
            
            # Disk outline
            disk = plt.Circle((0, 0), disk_radius, fill=False, color="black", linewidth=0.6)
            ax1.add_patch(disk)
            
            # Waveguides
            ax1.add_patch(plt.Rectangle(
                (-wg_length/2, disk_radius + gap),
                wg_length, wg_width,
                fill=False, edgecolor="black", linewidth=0.6
            ))
            ax1.add_patch(plt.Rectangle(
                (-wg_length/2, -disk_radius - gap - wg_width),
                wg_length, wg_width,
                fill=False, edgecolor="black", linewidth=0.6
            ))
            
            # Tuner
            theta = np.linspace(-np.pi/4, np.pi/4, 300)
            r_t = disk_radius + gap_tune + tw
            r_i = disk_radius + gap_tune
            ax1.plot(r_i*np.cos(theta), r_i*np.sin(theta), "k--", lw=0.5)
            ax1.plot(r_t*np.cos(theta), r_t*np.sin(theta), "k--", lw=0.5)
            
            ax1.set_xlabel("x (µm)")
            ax1.set_ylabel("y (µm)")
            ax1.set_title(f"XY plane (z=0)\nPeak {i+1}: {res_freq_thz:.2f} THz")
            ax1.set_aspect("equal")
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # XZ slice (y=0)
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(
                hz_xz.T,
                interpolation="bicubic",
                cmap="RdBu",
                origin="lower",
                extent=[-cell_x/2, cell_x/2, -cell_z/2, cell_z/2],
                vmin=-np.abs(hz_xz).max(), vmax=np.abs(hz_xz).max()
            )
            
            # Show slab extent
            ax2.axhline(y=-slab_thickness/2, color='black', linestyle='--', linewidth=0.5)
            ax2.axhline(y=slab_thickness/2, color='black', linestyle='--', linewidth=0.5)
            
            ax2.set_xlabel("x (µm)")
            ax2.set_ylabel("z (µm)")
            ax2.set_title("XZ plane (y=0)")
            ax2.set_aspect("equal")
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            # YZ slice (x=0)
            ax3 = fig.add_subplot(133)
            im3 = ax3.imshow(
                hz_yz.T,
                interpolation="bicubic",
                cmap="RdBu",
                origin="lower",
                extent=[-cell_y/2, cell_y/2, -cell_z/2, cell_z/2],
                vmin=-np.abs(hz_yz).max(), vmax=np.abs(hz_yz).max()
            )
            
            # Show slab extent
            ax3.axhline(y=-slab_thickness/2, color='black', linestyle='--', linewidth=0.5)
            ax3.axhline(y=slab_thickness/2, color='black', linestyle='--', linewidth=0.5)
            
            ax3.set_xlabel("y (µm)")
            ax3.set_ylabel("z (µm)")
            ax3.set_title("YZ plane (x=0)")
            ax3.set_aspect("equal")
            plt.colorbar(im3, ax=ax3, shrink=0.8)
            
            plt.tight_layout()
            
            snapshot_filename = os.path.join(run_dir, f"snapshot_3d_slices_peak{i+1}_{res_freq_thz:.2f}THz.png")
            plt.savefig(snapshot_filename, dpi=200)
            plt.close()

            # ---------------------------------------------------
            # Additional: Top view with field intensity
            # ---------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='white')

            # Use a better colormap - 'viridis' is perceptually uniform and scientific
            # Other good options: 'plasma', 'magma', 'cividis', 'inferno'
            im = ax.imshow(
                np.abs(hz_xy.T),
                interpolation='bicubic',
                cmap='plasma',  
                origin='lower',
                extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2]
            )

            # Overlay geometry with better contrast
            # Disk resonator
            disk = plt.Circle((0, 0), disk_radius, fill=False, color='white', 
                            linewidth=1.5, linestyle='-', alpha=0.8)
            ax.add_patch(disk)

            # Waveguides - use white with slight transparency for better visibility
            ax.add_patch(plt.Rectangle(
                (-wg_length/2, disk_radius + gap),
                wg_length, wg_width,
                fill=False, edgecolor='white', linewidth=1.5, alpha=0.8
            ))
            ax.add_patch(plt.Rectangle(
                (-wg_length/2, -disk_radius - gap - wg_width),
                wg_length, wg_width,
                fill=False, edgecolor='white', linewidth=1.5, alpha=0.8
            ))

            # Tuner arcs
            theta = np.linspace(-np.pi/4, np.pi/4, 300)
            r_t = disk_radius + gap_tune + tw
            r_i = disk_radius + gap_tune
            ax.plot(r_i*np.cos(theta), r_i*np.sin(theta), 'w--', lw=1.0, alpha=0.7)
            ax.plot(r_t*np.cos(theta), r_t*np.sin(theta), 'w--', lw=1.0, alpha=0.7)

            # Title with better formatting
            param_text = "\n".join([f"{k}: {v}" for k, v in params.items()]) if params else "No parameters parsed"
            title_text = (
                f"{folder}\n"
                f"Resonance: {res_freq_thz:.2f} THz\n"
                f"Field intensity |Hz| (z=0 plane)"
            )
            ax.set_title(title_text, fontsize=11, loc='left', pad=15, fontweight='normal')

            # Improved colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label('|Hz| field amplitude', fontsize=11)
            cbar.ax.tick_params(labelsize=9)

            # Cleaner axes
            ax.set_xlabel('x (µm)', fontsize=11)
            ax.set_ylabel('y (µm)', fontsize=11)
            ax.tick_params(labelsize=9)
            ax.set_aspect('equal')
            ax.set_xlim(-cell_x/2, cell_x/2)
            ax.set_ylim(-cell_y/2, cell_y/2)

            # Optional: add a subtle grid
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

            plt.tight_layout()
            intensity_filename = os.path.join(run_dir, f"intensity_peak{i+1}_{res_freq_thz:.2f}THz.png")
            plt.savefig(intensity_filename, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Rank {rank}: Saved animations and snapshots for peak {i+1} at {res_freq_thz:.2f} THz")