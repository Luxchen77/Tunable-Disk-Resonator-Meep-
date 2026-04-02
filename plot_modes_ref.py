"""
Reference WGM mode field patterns — no tuner geometry.

Finds resonances from the drop port flux in the ref_notuner dataset,
runs CW simulations at selected peak ranks, and produces publication-quality
field snapshots showing different azimuthal mode orders.

Usage:
    mpirun -np 4 python plot_modes_ref.py
"""

import numpy as np
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks

try:
    import meep as mp
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    HAS_MEEP = True
except ImportError:
    rank = 0
    HAS_MEEP = False

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "thesis_data/20251211_1742_norm_r3.5_g0.02_f318.000_res48_decay0.0005_tw0.1/ref_notuner"

PEAK_INDICES = [10,14]  # 0-indexed ranks by drop-port peak height
FIELD_CMAP = "RdBu"
CW_RUN_TIME = 10000

# ============================================================
# Publication matplotlib style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

COL_WIDTH = 3.4
DBL_WIDTH = 7.0
c0 = 299792458
um_scale = 1e-6


# ============================================================
# Data loading
# ============================================================
def parse_params(params_path):
    params = {}
    with open(params_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            try:
                if "." in val or "e" in val.lower():
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = val
    return params


def find_resonances_from_drop(freqs, flux_drop, norm_flux, top_n=30,
                               prominence=0.01, min_distance_thz=0.5,
                               sort_by="height"):
    """Find resonances from abs(drop port) peaks."""
    drop_norm = np.abs(flux_drop / norm_flux)
    drop_norm = np.clip(drop_norm, 0, None)

    min_dist = max(1, int(min_distance_thz / (freqs[1] - freqs[0])))
    peaks, _ = find_peaks(drop_norm, prominence=prominence, distance=min_dist)

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    heights = drop_norm[peaks]
    order = np.argsort(heights)[::-1]
    peaks = peaks[order[:top_n]]
    heights = heights[order[:top_n]]

    if sort_by == "frequency":
        freq_order = np.argsort(freqs[peaks])
        peaks = peaks[freq_order]
        heights = heights[freq_order]

    return freqs[peaks], heights, peaks


# ============================================================
# Geometry (no tuner)
# ============================================================
def build_geometry(params):
    n_eff = params.get("n_eff", 2.9933)
    gaas = mp.Medium(epsilon=n_eff**2)

    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]

    geometry = [
        mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas,
                     center=mp.Vector3()),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                 material=gaas),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
                 material=gaas),
    ]

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
    cell = mp.Vector3(cell_x, cell_y, 0)

    return geometry, cell, gaas


# ============================================================
# CW simulation
# ============================================================
def run_cw_at_frequency(freq_thz, params, resolution):
    geometry, cell, gaas = build_geometry(params)

    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]

    f_meep = freq_thz * um_scale * 1e12 / c0
    src_center = mp.Vector3(-wg_length / 2 + 1.5, disk_radius + gap + wg_width / 2)

    sources = [mp.Source(mp.ContinuousSource(frequency=f_meep),
                         component=mp.Hz,
                         center=src_center,
                         size=mp.Vector3(0, wg_width, 0))]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(0.8)],
        resolution=resolution,
    )

    sim.run(until=CW_RUN_TIME)

    hz_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)
    sim.reset_meep()
    return hz_data


# ============================================================
# Plot: single field snapshot
# ============================================================
def plot_field_snapshot(hz_data, params, freq_thz, save_dir=None):
    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * cell_y / cell_x))

    vmax = np.max(np.abs(hz_data)) * 0.8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(hz_data.T, origin="lower", cmap=FIELD_CMAP,
                   norm=norm, interpolation="bicubic",
                   extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2])

    # Geometry overlays — disk + waveguides only
    ax.add_patch(Circle((0, 0), disk_radius, fill=False, edgecolor="k", linewidth=0.6))
    wg_y_top = disk_radius + gap
    ax.add_patch(Rectangle((-wg_length / 2, wg_y_top), wg_length, wg_width,
                            fill=False, edgecolor="k", linewidth=0.5))
    wg_y_bot = -disk_radius - gap - wg_width
    ax.add_patch(Rectangle((-wg_length / 2, wg_y_bot), wg_length, wg_width,
                            fill=False, edgecolor="k", linewidth=0.5))

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label(r"$H_z$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    ax.set_aspect("equal")
    ax.set_xlim(-cell_x / 2, cell_x / 2)
    ax.set_ylim(-cell_y / 2, cell_y / 2)

    plt.tight_layout()
    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"mode_ref_{freq_thz:.2f}THz.{ext}"))
    plt.close(fig)


# ============================================================
# Plot: multi-panel grid
# ============================================================
def plot_multi_panel(hz_arrays, params, freq_list, save_dir=None):
    n = len(hz_arrays)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(DBL_WIDTH, DBL_WIDTH * nrows / ncols * cell_y / cell_x))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    vmax = max(np.max(np.abs(h)) for h in hz_arrays) * 0.8
    labels = "abcdefghij"

    for idx, (hz_data, freq_thz) in enumerate(zip(hz_arrays, freq_list)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(hz_data.T, origin="lower", cmap=FIELD_CMAP,
                       norm=norm, interpolation="bicubic",
                       extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2])

        ax.add_patch(Circle((0, 0), disk_radius, fill=False, edgecolor="k", linewidth=0.5))
        wg_y_top = disk_radius + gap
        ax.add_patch(Rectangle((-wg_length / 2, wg_y_top), wg_length, wg_width,
                                fill=False, edgecolor="k", linewidth=0.4))
        wg_y_bot = -disk_radius - gap - wg_width
        ax.add_patch(Rectangle((-wg_length / 2, wg_y_bot), wg_length, wg_width,
                                fill=False, edgecolor="k", linewidth=0.4))

        ax.set_aspect("equal")
        ax.set_xlim(-cell_x / 2, cell_x / 2)
        ax.set_ylim(-cell_y / 2, cell_y / 2)
        ax.text(0.03, 0.95, f"({labels[idx]})",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                fontweight="bold")

        if row == nrows - 1:
            ax.set_xlabel(r"$x$ ($\mu$m)", fontsize=9)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(r"$y$ ($\mu$m)", fontsize=9)
        else:
            ax.set_yticklabels([])

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$H_z$", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"modes_ref_panel.{ext}"))
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reference mode field patterns (no tuner)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--peaks", type=str, default=None,
                        help="Comma-separated peak indices, e.g. '0,1,2,4,7'")
    args = parser.parse_args()

    data_dir = args.data_dir
    peak_indices = PEAK_INDICES
    if args.peaks:
        peak_indices = [int(x) for x in args.peaks.split(",")]

    if not HAS_MEEP:
        if rank == 0:
            print("Error: meep is not installed. This script requires meep for CW simulations.")
            print("Run on your meep cluster:")
            print(f"  mpirun -np 4 python plot_modes_ref.py --data-dir {data_dir}")
        return

    # Load data
    freqs = np.load(os.path.join(data_dir, "freqs_thz.npy"))
    flux_drop = np.load(os.path.join(data_dir, "flux_drop.npy"))
    norm_flux = np.load(os.path.join(data_dir, "norm_flux.npy"))
    params = parse_params(os.path.join(data_dir, "params.txt"))
    params["n_eff"] = 2.9933
    resolution = int(params.get("resolution", 48))

    # Find resonances from drop port
    n_needed = max(peak_indices) + 1
    all_freqs, all_heights, _ = find_resonances_from_drop(
        freqs, flux_drop, norm_flux, top_n=n_needed, sort_by="height")

    valid_indices = [i for i in peak_indices if i < len(all_freqs)]
    res_freqs = all_freqs[valid_indices]

    if rank == 0:
        print(f"Data: {data_dir}")
        print(f"Found {len(all_freqs)} peaks, selecting ranks {valid_indices}:")
        for i, idx in enumerate(valid_indices):
            print(f"  rank {idx}: {res_freqs[i]:.2f} THz (height={all_heights[idx]:.2f})")

    # Output directory
    out_dir = os.path.join(os.path.dirname(data_dir), "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Run CW sims
    hz_arrays = []
    for i, freq_thz in enumerate(res_freqs):
        if rank == 0:
            print(f"\n  [{i+1}/{len(res_freqs)}] f = {freq_thz:.2f} THz ...")

        hz = run_cw_at_frequency(freq_thz, params, resolution)
        hz_arrays.append(hz)

        if rank == 0:
            plot_field_snapshot(hz, params, freq_thz, save_dir=out_dir)

    # Multi-panel
    if rank == 0 and len(hz_arrays) > 1:
        plot_multi_panel(hz_arrays, params, res_freqs, save_dir=out_dir)
        print(f"\nField plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
