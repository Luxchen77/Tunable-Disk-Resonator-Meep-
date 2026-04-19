"""
Resolution & runtime convergence study for the disk resonator with tuner.

Sweeps over resolution values, running the full simulation at each one,
and saves spectra + resonance frequencies + wall-clock times for thesis plots.

Usage:
    mpirun -np 4 python convergence_study.py
"""

import meep as mp
import numpy as np
import os
import math
import time
import csv
from datetime import datetime
from scipy.signal import find_peaks
from meep_utils import arc_prism

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ============================================================
# Fixed simulation parameters
# ============================================================
c0 = 299792458          # m/s
um_scale = 1e-6         # 1 µm

sigma_meep = 0.0
n_eff = 2.9933
gaas = mp.Medium(epsilon=n_eff**2,
                 D_conductivity=2 * math.pi * 1 * sigma_meep / n_eff)

disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02              # disk-waveguide gap
gap_tune = 0.02         # tuner gap (20 nm)
tw = 0.1                # tuner width (100 nm)
theta = np.pi / 4       # tuner arc half-angle

cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(0.8)]

f_thz = 318
fcen = f_thz * um_scale * 1e12 / c0
df_thz = 20
fwidth = df_thz * um_scale * 1e12 / c0
nfreq = 20000
field_decay = 5e-4

# Source position — bus waveguide input
src_center = mp.Vector3(-wg_length / 2 + 1.5,
                        disk_radius + gap + wg_width / 2)

# ============================================================
# Resolution sweep
# ============================================================
resolutions = [96, 128]
#resolutions = [8,12]
TOP_N_PEAKS = 5  # number of resonance peaks to track

# ============================================================
# Output directory
# ============================================================
if rank == 0:
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    study_dir = os.path.join(base_dir, "convergence_study")
    os.makedirs(study_dir, exist_ok=True)
    print(f"Convergence study output: {study_dir}")
    print(f"Resolutions: {resolutions}")
    print(f"Geometry: disk r={disk_radius}, gap={gap}, "
          f"tuner gap={gap_tune}, tuner width={tw}")
    print("=" * 60)
else:
    study_dir = None

study_dir = comm.bcast(study_dir, root=0)


# ============================================================
# Helper: find resonance peaks from disk flux
# ============================================================
def find_resonances(freqs, flux_disk, norm_flux, top_n=5):
    """Find top_n resonance peaks from normalized disk flux."""
    norm = np.where(np.abs(norm_flux) > 1e-20, norm_flux, 1e-20)
    spectrum = np.abs(flux_disk / norm)
    peaks, props = find_peaks(spectrum, prominence=0.5,
                              distance=int(0.5 / (freqs[1] - freqs[0])))
    if len(peaks) == 0:
        return np.array([]), np.array([])
    prominences = props["prominences"]
    order = np.argsort(prominences)[::-1][:top_n]
    peak_idx = peaks[order]
    peak_idx = np.sort(peak_idx)  # sort by frequency
    return freqs[peak_idx], spectrum[peak_idx]


# ============================================================
# Main convergence loop
# ============================================================
summary_rows = []

for res in resolutions:
    t_start = time.time()

    if rank == 0:
        res_dir = os.path.join(study_dir, f"res_{res}")
        os.makedirs(res_dir, exist_ok=True)
        print(f"\n--- Resolution {res} px/µm ---")
    else:
        res_dir = None
    res_dir = comm.bcast(res_dir, root=0)

    # ----------------------------------------------------------
    # Normalization: bare waveguides (no disk, no tuner)
    # ----------------------------------------------------------
    geometry_norm = [
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                 material=gaas),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, -(disk_radius + gap + wg_width / 2)),
                 material=gaas),
    ]

    norm_sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry_norm,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                           component=mp.Hz,
                           center=src_center,
                           size=mp.Vector3(0, wg_width, 0))],
        boundary_layers=pml_layers,
        resolution=res,
    )

    norm_flux_mon = norm_sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(wg_length / 2 - 1.5,
                                        disk_radius + gap + wg_width / 2),
                      size=mp.Vector3(0, wg_width, 0))
    )

    if rank == 0:
        print(f"  Normalization sim...")
    norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(
        100, mp.Hz, mp.Vector3(), field_decay))

    norm_freqs = np.array(mp.get_flux_freqs(norm_flux_mon)) * c0 / um_scale / 1e12
    norm_flux_data = np.array(mp.get_fluxes(norm_flux_mon))
    norm_sim.reset_meep()

    # ----------------------------------------------------------
    # Main simulation: disk + bus + drop + tuner
    # ----------------------------------------------------------
    geometry = [
        mp.Cylinder(radius=disk_radius, height=mp.inf,
                     material=gaas, center=mp.Vector3()),
        # Bus waveguide (top)
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                 material=gaas),
        # Drop waveguide (bottom)
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                 center=mp.Vector3(0, -(disk_radius + gap + wg_width / 2)),
                 material=gaas),
        # Tuner arc (right side, matching thesis data geometry)
        arc_prism(radius=disk_radius + gap_tune,
                  width=tw,
                  angle_start=-theta,
                  angle_end=theta,
                  npoints=256,
                  material=gaas),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                           component=mp.Hz,
                           center=src_center,
                           size=mp.Vector3(0, wg_width, 0))],
        boundary_layers=pml_layers,
        resolution=res,
    )

    # Flux monitors
    flux_bus = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(wg_length / 2 - 1.5,
                                        disk_radius + gap + wg_width / 2),
                      size=mp.Vector3(0, wg_width, 0))
    )

    flux_drop = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(-wg_length / 2 + 1.5,
                                        -(disk_radius + gap + wg_width / 2)),
                      size=mp.Vector3(0, wg_width, 0))
    )

    flux_disk = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(-disk_radius + 0.15, 0),
                      size=mp.Vector3(0.3, 0, 0))
    )

    flux_tuner = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(disk_radius + gap_tune + tw / 2, 0),
                      size=mp.Vector3(tw, 0, 0))
    )

    if rank == 0:
        print(f"  Main sim...")
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        100, mp.Hz, mp.Vector3(), field_decay))

    # Extract data
    freqs_thz = np.array(mp.get_flux_freqs(flux_bus)) * c0 / um_scale / 1e12
    bus_data = np.array(mp.get_fluxes(flux_bus))
    drop_data = np.array(mp.get_fluxes(flux_drop))
    disk_data = np.array(mp.get_fluxes(flux_disk))
    tuner_data = np.array(mp.get_fluxes(flux_tuner))

    sim.reset_meep()

    t_elapsed = time.time() - t_start

    # Find resonances
    peak_freqs, peak_heights = find_resonances(freqs_thz, disk_data,
                                                norm_flux_data,
                                                top_n=TOP_N_PEAKS)

    # Save data
    if rank == 0:
        np.save(os.path.join(res_dir, "freqs_thz.npy"), freqs_thz)
        np.save(os.path.join(res_dir, "flux_bus.npy"), bus_data)
        np.save(os.path.join(res_dir, "flux_drop.npy"), drop_data)
        np.save(os.path.join(res_dir, "flux_disk.npy"), disk_data)
        np.save(os.path.join(res_dir, "flux_tuner.npy"), tuner_data)
        np.save(os.path.join(res_dir, "norm_flux.npy"), norm_flux_data)

        with open(os.path.join(res_dir, "runtime_seconds.txt"), "w") as f:
            f.write(f"{t_elapsed:.1f}\n")

        params = {
            "disk_radius": disk_radius,
            "gap": gap,
            "wg_length": wg_length,
            "wg_width": wg_width,
            "n_eff": n_eff,
            "cell_size": f"({cell_x}, {cell_y})",
            "f_thz": f_thz,
            "df_thz": df_thz,
            "n_freq_points": nfreq,
            "resolution": res,
            "field_decay": field_decay,
            "tuner_width": tw,
            "tuner_gap": gap_tune,
            "theta_rad": theta,
        }
        with open(os.path.join(res_dir, "params.txt"), "w") as f:
            for key, val in params.items():
                f.write(f"{key} = {val}\n")

        # Summary row
        row = {"resolution": res, "runtime_s": round(t_elapsed, 1)}
        for i, freq in enumerate(peak_freqs):
            row[f"peak_{i+1}_thz"] = round(freq, 4)
        summary_rows.append(row)

        peak_str = ", ".join(f"{f:.2f}" for f in peak_freqs)
        print(f"  Done in {t_elapsed:.0f}s | "
              f"Peaks: [{peak_str}] THz")

# ============================================================
# Save convergence summary CSV
# ============================================================
if rank == 0:
    # Determine all column names
    all_keys = ["resolution", "runtime_s"]
    max_peaks = max(len([k for k in row if k.startswith("peak_")])
                    for row in summary_rows) if summary_rows else 0
    for i in range(max_peaks):
        all_keys.append(f"peak_{i+1}_thz")

    csv_path = os.path.join(study_dir, "convergence_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n" + "=" * 60)
    print(f"Convergence study complete.")
    print(f"Results: {study_dir}")
    print(f"Summary: {csv_path}")
    print("=" * 60)

    # Print summary table
    print(f"\n{'Res':>5} | {'Time (s)':>9} | Resonance frequencies (THz)")
    print("-" * 60)
    for row in summary_rows:
        peaks = [row.get(f"peak_{i+1}_thz", "") for i in range(max_peaks)]
        peak_str = "  ".join(f"{p:.2f}" if p else "  --  " for p in peaks)
        print(f"{row['resolution']:>5} | {row['runtime_s']:>9.1f} | {peak_str}")
