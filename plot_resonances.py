"""
Publication-quality resonance visualization for Meep disk resonator simulations.

Reads existing flux data from tuner gap sweeps, finds resonances automatically,
optionally runs CW simulations for field patterns, and produces thesis-ready figures.

Usage:
    python plot_resonances.py                          # spectra only (fast)
    mpirun -np 4 python plot_resonances.py --fields    # + CW field snapshots
    mpirun -np 4 python plot_resonances.py --videos    # + animations
"""

import numpy as np
import os
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks

# Meep and MPI are only needed for CW field simulations.
# Plotting from existing data works without them.
try:
    import meep as mp
    from mpi4py import MPI
    import h5py
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    HAS_MEEP = True
except ImportError:
    rank = 0
    HAS_MEEP = False

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "thesis_data/20251211_1742_norm_r3.5_g0.02_f318.000_res48_decay0.0005_tw0.1"

SELECTED_GAPS = None  # None = all, or e.g. [0.004, 0.010, 0.020, 0.050]
TOP_N_RESONANCES = 2  # how many resonance dips to annotate / visualize
CW_RUN_TIME = 3000    # Meep time units for CW steady-state
FIELD_CMAP = "RdYlBu"  # colormap for field plots ("RdYlBu", "twilight_shifted", "RdBu", ...)

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

COL_WIDTH = 3.4   # inches, single journal column
DBL_WIDTH = 7.0   # inches, double journal column

c0 = 299792458      # m/s
um_scale = 1e-6     # 1 um


# ============================================================
# Data loading
# ============================================================
def parse_params(params_path):
    """Parse a params.txt file into a dict of floats/strings."""
    params = {}
    with open(params_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            # Try to parse as number
            try:
                if "." in val or "e" in val.lower():
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                # Handle tuples like (12, 10.84)
                if val.startswith("("):
                    params[key] = val
                elif val in ("True", "False"):
                    params[key] = val == "True"
                else:
                    params[key] = val
    return params


def load_gap_data(gap_dir):
    """Load flux data from a single tunergap_* subfolder."""
    data = {}
    data["freqs"] = np.load(os.path.join(gap_dir, "freqs_thz.npy"))
    data["flux_bus"] = np.load(os.path.join(gap_dir, "flux_bus.npy"))

    for name in ["flux_drop", "flux_tuner", "flux_disk"]:
        path = os.path.join(gap_dir, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)

    # Load normalization
    norm_path = os.path.join(gap_dir, "norm_flux.npy")
    if os.path.exists(norm_path):
        data["norm_flux"] = np.load(norm_path)
    return data


def load_run(run_dir):
    """Load all tunergap subfolders and normalization data from a run directory."""
    run_dir = Path(run_dir)

    # Top-level normalization
    norm_flux = np.load(run_dir / "norm_flux.npy")
    norm_freqs = np.load(run_dir / "norm_freqs.npy")

    # Load CSV if available
    csv_data = None
    csv_files = list(run_dir.glob("gap_sweep_*.csv"))
    if csv_files:
        csv_data = np.genfromtxt(csv_files[0], delimiter=",", names=True)

    # Scan tunergap subfolders
    gap_dirs = sorted(run_dir.glob("tunergap_*um"))
    gaps = {}
    for gd in gap_dirs:
        gap_val = float(gd.name.replace("tunergap_", "").replace("um", ""))
        try:
            gdata = load_gap_data(str(gd))
            gdata["params"] = {}
            params_file = gd / "params.txt"
            if params_file.exists():
                gdata["params"] = parse_params(str(params_file))
            gaps[gap_val] = gdata
        except FileNotFoundError as e:
            if rank == 0:
                print(f"  Skipping {gd.name}: {e}")

    # Load ref_notuner if present
    ref_data = None
    ref_dir = run_dir / "ref_notuner"
    if ref_dir.exists():
        try:
            ref_data = load_gap_data(str(ref_dir))
        except FileNotFoundError:
            pass

    return {
        "norm_flux": norm_flux,
        "norm_freqs": norm_freqs,
        "csv": csv_data,
        "gaps": gaps,
        "ref": ref_data,
    }


# ============================================================
# Resonance finding
# ============================================================
def find_resonances_from_dips(freqs, transmission, top_n=4, prominence=0.005,
                              min_distance_thz=0.5):
    """Find resonance dips in normalized bus transmission.

    Returns arrays of (frequencies, depths, indices) sorted by frequency.
    """
    inverted = 1.0 - transmission
    min_dist = max(1, int(min_distance_thz / (freqs[1] - freqs[0])))

    peaks, _ = find_peaks(inverted, prominence=prominence, distance=min_dist)

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    depths = inverted[peaks]
    order = np.argsort(depths)[::-1]
    peaks = peaks[order[:top_n]]
    depths = depths[order[:top_n]]

    freq_order = np.argsort(freqs[peaks])
    peaks = peaks[freq_order]
    depths = depths[freq_order]

    return freqs[peaks], depths, peaks


def find_resonances_from_disk(freqs, flux_disk, norm_flux, top_n=4,
                               prominence=0.5, min_distance_thz=0.5,
                               sort_by="frequency"):
    """Find resonances from flux_disk peaks (intracavity power buildup).

    flux_disk has strong positive peaks at resonances — much cleaner than
    inverting bus transmission dips.

    sort_by: "frequency" (default, for spectra) or "height" (for selecting
             specific peak ranks like 1st, 5th, 8th biggest).

    Returns arrays of (frequencies, heights, indices) sorted accordingly.
    """
    disk_norm = flux_disk / norm_flux
    # Clip negative values (numerical noise)
    disk_norm = np.clip(disk_norm, 0, None)

    min_dist = max(1, int(min_distance_thz / (freqs[1] - freqs[0])))

    peaks, _ = find_peaks(disk_norm, prominence=prominence, distance=min_dist)

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    heights = disk_norm[peaks]
    order = np.argsort(heights)[::-1]
    peaks = peaks[order[:top_n]]
    heights = heights[order[:top_n]]

    if sort_by == "frequency":
        freq_order = np.argsort(freqs[peaks])
        peaks = peaks[freq_order]
        heights = heights[freq_order]

    return freqs[peaks], heights, peaks


def find_resonances(freqs, gap_data, top_n=4, min_distance_thz=0.5):
    """Find resonances using flux_disk if available, otherwise bus transmission dips."""
    if "flux_disk" in gap_data and gap_data["flux_disk"] is not None:
        return find_resonances_from_disk(
            freqs, gap_data["flux_disk"], gap_data["norm_flux"],
            top_n=top_n, min_distance_thz=min_distance_thz,
        )
    else:
        trans = gap_data["flux_bus"] / gap_data["norm_flux"]
        return find_resonances_from_dips(
            freqs, trans, top_n=top_n, min_distance_thz=min_distance_thz,
        )


def track_resonances(run_data, top_n=3, search_window_thz=1.5):
    """Track resonances across gaps by seeding from the largest gap.

    Starts at the largest tuner gap (cleanest modes), finds the top_n peaks,
    then follows each peak to smaller gaps by nearest-neighbor matching within
    a search window.  This prevents mode-hopping between different WGM orders.

    Returns dict  {res_idx: {"gaps": [...], "freqs": [...]}}  sorted by gap
    (smallest to largest).
    """
    gaps_dict = run_data["gaps"]
    sorted_gaps = sorted(gaps_dict.keys(), reverse=True)  # largest first

    # Seed: top_n peaks at the largest gap
    d0 = gaps_dict[sorted_gaps[0]]
    seed_freqs, _, _ = find_resonances(d0["freqs"], d0, top_n=top_n)
    seed_freqs = np.array(seed_freqs, dtype=float)  # mutable copy

    tracks = {i: {"gaps": [], "freqs": []} for i in range(len(seed_freqs))}

    for gap_val in sorted_gaps:
        d = gaps_dict[gap_val]
        # Find a generous set of candidate peaks
        all_freqs, _, _ = find_resonances(d["freqs"], d, top_n=20)

        for i in range(len(seed_freqs)):
            last_freq = seed_freqs[i]
            if np.isnan(last_freq):
                tracks[i]["gaps"].append(gap_val)
                tracks[i]["freqs"].append(np.nan)
                continue
            if len(all_freqs) == 0:
                tracks[i]["gaps"].append(gap_val)
                tracks[i]["freqs"].append(np.nan)
                continue
            dists = np.abs(all_freqs - last_freq)
            best = np.argmin(dists)
            if dists[best] < search_window_thz:
                matched = float(all_freqs[best])
                tracks[i]["gaps"].append(gap_val)
                tracks[i]["freqs"].append(matched)
                seed_freqs[i] = matched  # update for next gap
            else:
                tracks[i]["gaps"].append(gap_val)
                tracks[i]["freqs"].append(np.nan)

    # Reverse so gaps go small → large (natural x-axis order)
    for t in tracks.values():
        t["gaps"] = t["gaps"][::-1]
        t["freqs"] = t["freqs"][::-1]

    return tracks


def thz_to_nm(f_thz):
    """Convert frequency in THz to wavelength in nm."""
    return c0 / (f_thz * 1e12) * 1e9


def nm_to_thz(lam_nm):
    """Convert wavelength in nm to frequency in THz."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return c0 / (lam_nm * 1e-9) * 1e-12


# ============================================================
# Plot: Transmission spectrum
# ============================================================
def plot_transmission_spectrum(freqs, transmission, flux_disk_norm=None,
                               resonance_freqs=None, resonance_heights=None,
                               gap_nm=None, save_dir=None,
                               filename_prefix="transmission"):
    """Plot normalized bus transmission + intracavity power with resonance markers."""
    has_disk = flux_disk_norm is not None
    nrows = 2 if has_disk else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(DBL_WIDTH, 2.2 * nrows),
                              sharex=True, layout="constrained",
                              gridspec_kw={"hspace": 0.08})
    if nrows == 1:
        axes = [axes]

    # Top panel: bus transmission
    ax = axes[0]
    ax.plot(freqs, transmission, color="#1f77b4", linewidth=0.6)
    ax.set_ylabel("Normalized transmission")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(freqs[0], freqs[-1])

    # Mark resonances on both panels
    if resonance_freqs is not None:
        for f_res in resonance_freqs:
            ax.axvline(f_res, color="#d62728", linewidth=0.4, alpha=0.5)
        # Annotate on whichever panel is on top
        for i, f_res in enumerate(resonance_freqs):
            ax.text(f_res, 1.08, f"{f_res:.2f}",
                    fontsize=6, ha="center", va="bottom", color="#d62728",
                    rotation=90)

    if gap_nm is not None:
        ax.text(0.98, 0.95, f"gap = {gap_nm:.0f} nm",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.9))
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Secondary wavelength axis on top of first panel
    ax_top = ax.secondary_xaxis("top", functions=(thz_to_nm, nm_to_thz))
    ax_top.set_xlabel(r"Wavelength (nm)")

    # Bottom panel: intracavity power (flux_disk)
    if has_disk:
        ax2 = axes[1]
        ax2.plot(freqs, flux_disk_norm, color="#2ca02c", linewidth=0.6)
        ax2.set_ylabel(r"Intracavity power (norm.)")
        ax2.set_xlabel("Frequency (THz)")
        ax2.set_xlim(freqs[0], freqs[-1])
        ax2.grid(True, alpha=0.15, linewidth=0.5)

        if resonance_freqs is not None:
            for f_res in resonance_freqs:
                ax2.axvline(f_res, color="#d62728", linewidth=0.4, alpha=0.5)
    else:
        axes[0].set_xlabel("Frequency (THz)")

    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"{filename_prefix}.{ext}"))
    plt.close(fig)


# ============================================================
# Plot: Three-spectrum reference (bus + drop + disk)
# ============================================================
def plot_three_spectra(freqs, flux_bus, flux_drop, flux_disk, norm_flux,
                       save_dir=None, filename="spectra_reference"):
    """Three-panel stacked plot of bus, drop, and intracavity spectra."""
    trans_bus = flux_bus / norm_flux
    trans_drop = flux_drop / norm_flux
    disk_norm = np.clip(flux_disk / norm_flux, 0, None)

    # Absolute values (drop port flux is negative), then normalize to [0,1]
    trans_bus = np.abs(trans_bus)
    trans_drop = np.abs(trans_drop)
    for arr in [trans_bus, trans_drop, disk_norm]:
        amax = np.nanmax(arr)
        if amax > 0:
            arr /= amax
    data = [trans_bus, trans_drop, disk_norm]

    fig, axes = plt.subplots(3, 1, figsize=(DBL_WIDTH, 3.5),
                              sharex=True, layout="constrained")

    colors = ["#400F77", "#5B6DAE", "#6D1717"]
    labels = ["Bus port", "Drop port", "Intracavity"]
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, y, color, label, plabel in zip(axes, data, colors, labels, panel_labels):
        ax.plot(freqs, y, color=color, linewidth=0.6)
        ax.set_ylabel(label, fontsize=9)
        ax.set_yticks([])
        ax.text(0.02, 0.92, plabel, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # x-axis only on bottom panel
    axes[-1].set_xlabel("Frequency (THz)")
    axes[0].set_xlim(freqs[0], freqs[-1])

    # Secondary wavelength axis on top of first panel
    ax_top = axes[0].secondary_xaxis("top", functions=(thz_to_nm, nm_to_thz))
    ax_top.set_xlabel("Wavelength (nm)")

    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"{filename}.{ext}"))
    plt.close(fig)


# ============================================================
# Plot: Tuning summary (multi-gap overlay + resonance shift)
# ============================================================
def plot_tuning_summary(run_data, tracks, resonance_idx=0,
                        freq_window=2.0, save_dir=None):
    """Two-panel figure: overlaid spectra near one resonance + resonance freq vs gap."""
    gaps_dict = run_data["gaps"]
    csv_data = run_data["csv"]

    if resonance_idx not in tracks:
        if rank == 0:
            print(f"  Warning: no track for resonance index {resonance_idx}")
        return

    track = tracks[resonance_idx]
    track_gaps = np.array(track["gaps"])
    track_freqs = np.array(track["freqs"])
    valid = ~np.isnan(track_freqs)

    if not np.any(valid):
        if rank == 0:
            print(f"  Warning: all NaN for resonance index {resonance_idx}")
        return

    # Center frequency: use the largest-gap value (seed point)
    center_freq = track_freqs[valid][-1]  # last = largest gap

    # Pick a subset of gaps for the overlay (max ~6 curves)
    all_gaps = sorted(gaps_dict.keys())
    if len(all_gaps) > 6:
        indices = np.linspace(0, len(all_gaps) - 1, 6, dtype=int)
        overlay_gaps = [all_gaps[i] for i in indices]
    else:
        overlay_gaps = all_gaps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.8),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Left panel: overlaid spectra near one resonance
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(overlay_gaps)))

    for gap_val, color in zip(overlay_gaps, colors):
        d = gaps_dict[gap_val]
        trans = d["flux_bus"] / d["norm_flux"]
        mask = (d["freqs"] >= center_freq - freq_window) & (d["freqs"] <= center_freq + freq_window)
        ax1.plot(d["freqs"][mask], trans[mask], color=color, linewidth=0.8,
                 label=f"{gap_val*1000:.0f} nm")

    ax1.set_xlabel("Frequency (THz)")
    ax1.set_ylabel("Normalized transmission")
    ax1.legend(title="Tuner gap", fontsize=7, title_fontsize=8, ncol=2)
    ax1.set_xlim(center_freq - freq_window, center_freq + freq_window)
    ax1.grid(True, alpha=0.15, linewidth=0.5)

    # Right panel: tracked resonance frequency vs gap
    gap_nm = track_gaps[valid] * 1000
    freq_thz = track_freqs[valid]
    ax2.plot(gap_nm, freq_thz, "o-", color="#1f77b4", markersize=4, linewidth=1.2)
    ax2.set_xlabel("Tuner gap (nm)")
    ax2.set_ylabel("Resonance frequency (THz)")
    ax2.grid(True, alpha=0.15, linewidth=0.5)

    # Overlay CSV wavelength shift on secondary axis if available and matching
    if csv_data is not None and "lambda_shift_nm" in csv_data.dtype.names and resonance_idx == 0:
        ax2r = ax2.twinx()
        ax2r.plot(csv_data["gap_um"] * 1000, csv_data["lambda_shift_nm"],
                  "s--", color="#d62728", markersize=3, linewidth=0.8, alpha=0.7)
        ax2r.set_ylabel(r"$\Delta\lambda$ (nm)", color="#d62728")
        ax2r.tick_params(axis="y", labelcolor="#d62728")

    plt.tight_layout()
    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"tuning_summary_res{resonance_idx}.{ext}"))
    plt.close(fig)


# ============================================================
# Geometry helpers (for CW sim and overlays)
# ============================================================
def arc_prism(radius, width, angle_start, angle_end, npoints, material):
    r_in = radius
    r_out = radius + width
    outer = [mp.Vector3(r_out * np.cos(a), r_out * np.sin(a))
             for a in np.linspace(angle_start, angle_end, npoints)]
    inner = [mp.Vector3(r_in * np.cos(a), r_in * np.sin(a))
             for a in np.linspace(angle_end, angle_start, npoints)]
    return mp.Prism(vertices=outer + inner, height=mp.inf, material=material)


def build_geometry(params):
    """Build two-waveguide + disk + tuner geometry from params dict."""
    n_eff = params.get("n_eff", np.sqrt(params.get("epsilon", 8.96)))
    gaas = mp.Medium(epsilon=n_eff**2)

    disk_radius = params["disk_radius"]
    wg_length = params["wg_length"]
    wg_width = params["wg_width"]
    gap = params["gap"]
    gap_tune = params["tuner_gap"]
    tw = params["tuner_width"]
    theta = params.get("theta_rad", np.pi / 4)

    geometry = [
        mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas,
                     center=mp.Vector3()),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                 material=gaas),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
                 material=gaas),
        arc_prism(radius=disk_radius + gap_tune, width=tw,
                  angle_start=-theta, angle_end=theta,
                  npoints=256, material=gaas),
    ]

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
    cell = mp.Vector3(cell_x, cell_y, 0)

    return geometry, cell, gaas


# ============================================================
# Plot: Field snapshot (publication quality)
# ============================================================
def plot_field_snapshot(hz_data, params, freq_thz, gap_tune_um,
                       save_dir=None, filename_prefix="field"):
    """Plot Hz field with clean geometry overlays."""
    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]
    tw = params["tuner_width"]
    theta = params.get("theta_rad", np.pi / 4)

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * cell_y / cell_x))

    # Symmetric color normalization
    vmax = np.max(np.abs(hz_data)) * 0.8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(hz_data.T, origin="lower", cmap=FIELD_CMAP,
                   norm=norm, interpolation="bicubic",
                   extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2])

    # Geometry overlays — thin black outlines
    ax.add_patch(Circle((0, 0), disk_radius, fill=False,
                         edgecolor="k", linewidth=0.6))

    # Bus waveguide (top)
    wg_y_top = disk_radius + gap
    ax.add_patch(Rectangle((-wg_length / 2, wg_y_top), wg_length, wg_width,
                            fill=False, edgecolor="k", linewidth=0.5))
    # Drop waveguide (bottom)
    wg_y_bot = -disk_radius - gap - wg_width
    ax.add_patch(Rectangle((-wg_length / 2, wg_y_bot), wg_length, wg_width,
                            fill=False, edgecolor="k", linewidth=0.5))

    # Tuner arc outlines
    theta_arr = np.linspace(-theta, theta, 300)
    r_inner = disk_radius + gap_tune_um
    r_outer = r_inner + tw
    ax.plot(r_inner * np.cos(theta_arr), r_inner * np.sin(theta_arr),
            "k-", linewidth=0.4)
    ax.plot(r_outer * np.cos(theta_arr), r_outer * np.sin(theta_arr),
            "k-", linewidth=0.4)
    # Close the arc ends
    for ang in [-theta, theta]:
        ax.plot([r_inner * np.cos(ang), r_outer * np.cos(ang)],
                [r_inner * np.sin(ang), r_outer * np.sin(ang)],
                "k-", linewidth=0.4)

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
            fig.savefig(os.path.join(save_dir, f"{filename_prefix}_{freq_thz:.2f}THz.{ext}"))
    plt.close(fig)


# ============================================================
# Plot: Multi-resonance panel (2x2 grid)
# ============================================================
def plot_multi_resonance_panel(hz_arrays, params, freq_list, gap_tune_um,
                                save_dir=None, filename_prefix="modes_panel"):
    """2x2 grid of field snapshots for different resonance frequencies."""
    n = len(hz_arrays)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)

    disk_radius = params["disk_radius"]
    gap = params["gap"]
    wg_width = params["wg_width"]
    wg_length = params["wg_length"]
    tw = params["tuner_width"]
    theta = params.get("theta_rad", np.pi / 4)

    cell_x = wg_length
    cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(DBL_WIDTH, DBL_WIDTH * nrows / ncols * cell_y / cell_x))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    # Shared vmax across all panels
    vmax = max(np.max(np.abs(h)) for h in hz_arrays) * 0.8

    labels = "abcdefghij"
    for idx, (hz_data, freq_thz) in enumerate(zip(hz_arrays, freq_list)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(hz_data.T, origin="lower", cmap=FIELD_CMAP,
                       norm=norm, interpolation="bicubic",
                       extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2])

        # Geometry overlays
        ax.add_patch(Circle((0, 0), disk_radius, fill=False, edgecolor="k", linewidth=0.5))
        wg_y_top = disk_radius + gap
        ax.add_patch(Rectangle((-wg_length / 2, wg_y_top), wg_length, wg_width,
                                fill=False, edgecolor="k", linewidth=0.4))
        wg_y_bot = -disk_radius - gap - wg_width
        ax.add_patch(Rectangle((-wg_length / 2, wg_y_bot), wg_length, wg_width,
                                fill=False, edgecolor="k", linewidth=0.4))
        # Tuner arc
        theta_arr = np.linspace(-theta, theta, 200)
        r_inner = disk_radius + gap_tune_um
        r_outer = r_inner + tw
        ax.plot(r_inner * np.cos(theta_arr), r_inner * np.sin(theta_arr), "k-", lw=0.3)
        ax.plot(r_outer * np.cos(theta_arr), r_outer * np.sin(theta_arr), "k-", lw=0.3)

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

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$H_z$", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if save_dir:
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(save_dir, f"{filename_prefix}.{ext}"))
    plt.close(fig)


# ============================================================
# CW simulation runner
# ============================================================
def run_cw_at_frequency(freq_thz, params, resolution):
    """Run a CW simulation at a single frequency and return the Hz field array."""
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


def run_cw_animation(freq_thz, params, resolution, save_path):
    """Run a CW simulation and save an MP4 animation."""
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

    animate = mp.Animate2D(
        sim=sim,
        fields=mp.Hz,
        normalize=True,
        realtime=False,
        field_parameters={
            "alpha": 0.9,
            "cmap": FIELD_CMAP,
            "interpolation": "spline16",
        },
        boundary_parameters={
            "linewidth": 0.8,
            "facecolor": "none",
            "edgecolor": "gray",
            "alpha": 0.4,
        },
        output_plane=mp.Volume(center=mp.Vector3(), size=cell),
        title=f"$f$ = {freq_thz:.2f} THz",
    )

    sim.run(mp.at_every(15, animate), until=CW_RUN_TIME)
    animate.to_mp4(fps=20, filename=save_path)

    hz_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)
    sim.reset_meep()
    return hz_data


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot resonances from Meep disk resonator data")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to run directory")
    parser.add_argument("--fields", action="store_true", help="Run CW sims for field snapshots")
    parser.add_argument("--videos", action="store_true", help="Run CW sims and save MP4 animations")
    parser.add_argument("--gap", type=float, default=None,
                        help="Single tuner gap (um) for field/video runs (default: first available)")
    parser.add_argument("--top-n", type=int, default=TOP_N_RESONANCES,
                        help="Number of resonances to find/visualize")
    args = parser.parse_args()

    if rank == 0:
        print(f"Loading data from: {args.data_dir}")

    run_data = load_run(args.data_dir)
    gaps_dict = run_data["gaps"]

    if len(gaps_dict) == 0:
        if rank == 0:
            print("No tunergap data found!")
        return

    if rank == 0:
        print(f"Found {len(gaps_dict)} tuner gap values: "
              f"{', '.join(f'{g*1000:.0f} nm' for g in sorted(gaps_dict.keys()))}")

    # Select which gaps to process
    sel = SELECTED_GAPS
    if sel is not None:
        # Allow single float or list
        if isinstance(sel, (int, float)):
            sel = [sel]
        active_gaps = [g for g in sel if g in gaps_dict]
        if not active_gaps and sel:
            # Find closest match for each requested gap
            all_gaps = sorted(gaps_dict.keys())
            for s in sel:
                closest = min(all_gaps, key=lambda g: abs(g - s))
                if closest not in active_gaps:
                    active_gaps.append(closest)
            if rank == 0:
                print(f"  Requested gaps not exact match, using closest: {active_gaps}")
    else:
        active_gaps = sorted(gaps_dict.keys())

    # Output directory
    out_dir = os.path.join(args.data_dir, "figures")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # ---- Spectra (always, rank 0 only) ----
    if rank == 0:
        print("\n--- Generating transmission spectra ---")
        for gap_val in active_gaps:
            d = gaps_dict[gap_val]
            freqs = d["freqs"]
            trans = d["flux_bus"] / d["norm_flux"]

            res_freqs, res_heights, _ = find_resonances(freqs, d, top_n=args.top_n)
            print(f"  gap={gap_val*1000:.0f} nm: found {len(res_freqs)} resonances at "
                  f"{', '.join(f'{f:.2f}' for f in res_freqs)} THz")

            # Prepare flux_disk for plotting
            flux_disk_norm = None
            if "flux_disk" in d:
                flux_disk_norm = np.clip(d["flux_disk"] / d["norm_flux"], 0, None)

            plot_transmission_spectrum(
                freqs, trans,
                flux_disk_norm=flux_disk_norm,
                resonance_freqs=res_freqs, resonance_heights=res_heights,
                gap_nm=gap_val * 1000,
                save_dir=out_dir,
                filename_prefix=f"transmission_gap{gap_val*1000:.0f}nm",
            )

        # Three-spectrum reference plot at largest gap
        ref_gap = sorted(gaps_dict.keys())[-1]
        d_ref = gaps_dict[ref_gap]
        if all(k in d_ref for k in ("flux_bus", "flux_drop", "flux_disk")):
            print(f"\n--- Generating three-spectrum reference (gap = {ref_gap*1000:.0f} nm) ---")
            plot_three_spectra(
                d_ref["freqs"], d_ref["flux_bus"], d_ref["flux_drop"],
                d_ref["flux_disk"], d_ref["norm_flux"],
                save_dir=out_dir,
            )

        # Track resonances across all gaps (seed from largest gap)
        print("\n--- Tracking resonances across gaps ---")
        tracks = track_resonances(run_data, top_n=args.top_n)
        for i, t in tracks.items():
            valid = [f for f in t["freqs"] if not np.isnan(f)]
            if valid:
                print(f"  Resonance {i}: {valid[-1]:.2f} THz (at largest gap) → {valid[0]:.2f} THz (at smallest gap)")

        # Tuning summary using tracked resonances
        print("\n--- Generating tuning summary ---")
        for res_idx in range(min(args.top_n, len(tracks))):
            plot_tuning_summary(run_data, tracks=tracks,
                                resonance_idx=res_idx, save_dir=out_dir)

        print(f"\nSpectra saved to: {out_dir}/")

        if not args.fields and not args.videos:
            print("\nHint: For H-field snapshots, run on your meep cluster:")
            print(f"  mpirun -np 4 python plot_resonances.py --fields --data-dir {args.data_dir}")

    # ---- Field snapshots and videos (requires Meep CW sim) ----
    if (args.fields or args.videos) and not HAS_MEEP:
        if rank == 0:
            print("\nError: meep is not installed in this Python environment.")
            print("H-field snapshots require running CW simulations via meep.")
            print("Run this script on your meep cluster with:")
            print(f"  mpirun -np 4 python plot_resonances.py --fields --data-dir {args.data_dir}")
        return

    if args.fields or args.videos:
        # Pick which gap to simulate
        if args.gap is not None:
            sim_gap = args.gap
        else:
            sim_gap = active_gaps[-1]  # largest gap by default

        if sim_gap not in gaps_dict:
            if rank == 0:
                print(f"Gap {sim_gap} not in data, using closest available.")
            sim_gap = min(active_gaps, key=lambda g: abs(g - sim_gap))

        d = gaps_dict[sim_gap]
        freqs = d["freqs"]
        params = d["params"]
        params["n_eff"] = np.sqrt(params.get("epsilon", 8.96))
        resolution = int(params.get("resolution", 48))

        res_freqs, _, _ = find_resonances(freqs, d, top_n=args.top_n)

        if rank == 0:
            print(f"\n--- Running CW simulations at gap={sim_gap*1000:.0f} nm ---")
            print(f"    Resonances: {', '.join(f'{f:.2f} THz' for f in res_freqs)}")
            print(f"    Resolution: {resolution}")

        hz_arrays = []
        for i, freq_thz in enumerate(res_freqs):
            if rank == 0:
                print(f"\n  [{i+1}/{len(res_freqs)}] f = {freq_thz:.2f} THz ...")

            if args.videos:
                mp4_path = os.path.join(out_dir,
                    f"animation_{freq_thz:.2f}THz_gap{sim_gap*1000:.0f}nm.mp4")
                hz = run_cw_animation(freq_thz, params, resolution, mp4_path)
            else:
                hz = run_cw_at_frequency(freq_thz, params, resolution)

            hz_arrays.append(hz)

            if rank == 0:
                plot_field_snapshot(hz, params, freq_thz, sim_gap,
                                   save_dir=out_dir,
                                   filename_prefix=f"field_gap{sim_gap*1000:.0f}nm")

        # Multi-panel figure
        if rank == 0 and len(hz_arrays) > 1:
            plot_multi_resonance_panel(hz_arrays, params, res_freqs, sim_gap,
                                       save_dir=out_dir)
            print(f"\nField plots saved to: {out_dir}/")

    if rank == 0:
        print("\nDone.")


if __name__ == "__main__":
    main()
