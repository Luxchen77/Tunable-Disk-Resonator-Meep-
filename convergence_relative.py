"""
Relative convergence study: tuner gap sweep at res=48 vs res=128.

Compares the resonance shift (the key thesis parameter) at two resolutions
to validate that res=48 is sufficient for the tuning curve analysis.

Usage:
    mpirun -np 4 python convergence_relative.py
"""

import meep as mp
import numpy as np
import os
import math
import time
import csv
from scipy.signal import find_peaks
from meep_utils import arc_prism

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ============================================================
# Fixed simulation parameters (matching thesis data exactly)
# ============================================================
c0 = 299792458
um_scale = 1e-6

sigma_meep = 0.0
n_eff = 2.9933
gaas = mp.Medium(epsilon=n_eff**2,
                 D_conductivity=2 * math.pi * 1 * sigma_meep / n_eff)

disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02
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
nfreq = 30000
field_decay = 5e-4

src_center = mp.Vector3(-wg_length / 2 + 1.5,
                        disk_radius + gap + wg_width / 2)

# ============================================================
# Sweep parameters
# ============================================================
resolutions = [80]
gap_tunes = [0.035, 0.07]  # µm
TOP_N_PEAKS = 3

# ============================================================
# Output directory
# ============================================================
if rank == 0:
    base_dir = os.path.join("data", "convergence_relative")
    os.makedirs(base_dir, exist_ok=True)
    print(f"Relative convergence study")
    print(f"Resolutions: {resolutions}")
    print(f"Tuner gaps: {[g*1000 for g in gap_tunes]} nm")
    print(f"Tuner width: {tw*1000:.0f} nm")
    print("=" * 60)
else:
    base_dir = None

base_dir = comm.bcast(base_dir, root=0)

# ============================================================
# Helper: find resonance peaks
# ============================================================
def find_resonances(freqs, flux_disk, norm_flux, top_n=3):
    norm = np.where(np.abs(norm_flux) > 1e-20, norm_flux, 1e-20)
    spectrum = np.abs(flux_disk / norm)
    peaks, props = find_peaks(spectrum, prominence=0.5,
                              distance=int(0.5 / (freqs[1] - freqs[0])))
    if len(peaks) == 0:
        return np.array([])
    order = np.argsort(props["prominences"])[::-1][:top_n]
    peak_idx = np.sort(peaks[order])
    return freqs[peak_idx]


# ============================================================
# Main loop: resolution × tuner gap
# ============================================================
all_results = []  # list of dicts for CSV

for res in resolutions:
    t_res_start = time.time()

    if rank == 0:
        res_dir = os.path.join(base_dir, f"res_{res}")
        os.makedirs(res_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Resolution {res} px/µm")
        print(f"{'='*60}")
    else:
        res_dir = None
    res_dir = comm.bcast(res_dir, root=0)

    # ----------------------------------------------------------
    # Normalization: bare waveguides (no disk, no tuner)
    # ----------------------------------------------------------
    norm_flux_path = os.path.join(res_dir, "norm_flux.npy")
    norm_freqs_path = os.path.join(res_dir, "norm_freqs.npy")
    skip_norm = comm.bcast(
        os.path.exists(norm_flux_path) and os.path.exists(norm_freqs_path),
        root=0)

    if skip_norm:
        norm_flux_data = np.load(norm_flux_path)
        if rank == 0:
            print(f"  Normalization: loaded from disk (skipping sim)")
    else:
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
            print(f"  Normalization...")
        norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(
            100, mp.Hz, mp.Vector3(), field_decay))

        norm_freqs = np.array(mp.get_flux_freqs(norm_flux_mon)) * c0 / um_scale / 1e12
        norm_flux_data = np.array(mp.get_fluxes(norm_flux_mon))
        norm_sim.reset_meep()

        if rank == 0:
            np.save(norm_freqs_path, norm_freqs)
            np.save(norm_flux_path, norm_flux_data)

    # ----------------------------------------------------------
    # Reference run: disk + waveguides, NO tuner
    # ----------------------------------------------------------
    if rank == 0:
        ref_dir = os.path.join(res_dir, "ref_notuner")
        os.makedirs(ref_dir, exist_ok=True)
    else:
        ref_dir = None
    ref_dir = comm.bcast(ref_dir, root=0)

    skip_ref = comm.bcast(os.path.exists(os.path.join(ref_dir, "flux_disk.npy")), root=0)

    if skip_ref:
        if rank == 0:
            print(f"  Reference (no tuner): loaded from disk (skipping sim)")
    else:
        if rank == 0:
            print(f"  Reference (no tuner)...")

        geometry_ref = [
            mp.Cylinder(radius=disk_radius, height=mp.inf,
                         material=gaas, center=mp.Vector3()),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                     material=gaas),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, -(disk_radius + gap + wg_width / 2)),
                     material=gaas),
        ]

        ref_sim = mp.Simulation(
            cell_size=cell, geometry=geometry_ref,
            sources=[mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                               component=mp.Hz, center=src_center,
                               size=mp.Vector3(0, wg_width, 0))],
            boundary_layers=pml_layers, resolution=res,
        )

        ref_flux_bus = ref_sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5,
                          disk_radius + gap + wg_width/2),
                          size=mp.Vector3(0, wg_width, 0)))
        ref_flux_drop = ref_sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1.5,
                          -(disk_radius + gap + wg_width/2)),
                          size=mp.Vector3(0, wg_width, 0)))
        ref_flux_disk = ref_sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-disk_radius + 0.15, 0),
                          size=mp.Vector3(0.3, 0, 0)))

        ref_sim.run(until_after_sources=mp.stop_when_fields_decayed(
            100, mp.Hz, mp.Vector3(), field_decay))

        ref_freqs_thz = np.array(mp.get_flux_freqs(ref_flux_bus)) * c0 / um_scale / 1e12
        ref_bus = np.array(mp.get_fluxes(ref_flux_bus))
        ref_drop = np.array(mp.get_fluxes(ref_flux_drop))
        ref_disk = np.array(mp.get_fluxes(ref_flux_disk))
        ref_sim.reset_meep()

        if rank == 0:
            np.save(os.path.join(ref_dir, "freqs_thz.npy"), ref_freqs_thz)
            np.save(os.path.join(ref_dir, "flux_bus.npy"), ref_bus)
            np.save(os.path.join(ref_dir, "flux_drop.npy"), ref_drop)
            np.save(os.path.join(ref_dir, "flux_disk.npy"), ref_disk)
            np.save(os.path.join(ref_dir, "norm_flux.npy"), norm_flux_data)

            ref_peaks = find_resonances(ref_freqs_thz, ref_disk, norm_flux_data)
            print(f"    Ref peaks: {[f'{p:.2f}' for p in ref_peaks]} THz")

    # ----------------------------------------------------------
    # Tuner gap sweep
    # ----------------------------------------------------------
    for gap_tune in gap_tunes:
        t_gap_start = time.time()

        if rank == 0:
            gap_dir = os.path.join(res_dir, f"tunergap_{gap_tune:.3f}um")
            os.makedirs(gap_dir, exist_ok=True)
        else:
            gap_dir = None
        gap_dir = comm.bcast(gap_dir, root=0)

        skip_gap = comm.bcast(os.path.exists(os.path.join(gap_dir, "flux_disk.npy")), root=0)
        if skip_gap:
            if rank == 0:
                print(f"  Tuner gap = {gap_tune*1000:.0f} nm: loaded from disk (skipping sim)")
                freqs_thz = np.load(os.path.join(gap_dir, "freqs_thz.npy"))
                disk_data  = np.load(os.path.join(gap_dir, "flux_disk.npy"))
                peak_freqs = find_resonances(freqs_thz, disk_data, norm_flux_data,
                                              top_n=TOP_N_PEAKS)
                row = {"resolution": res, "gap_nm": gap_tune * 1000, "runtime_s": 0.0}
                for i, pf in enumerate(peak_freqs):
                    row[f"peak_{i+1}_thz"] = round(pf, 4)
                all_results.append(row)
                peak_str = ", ".join(f"{p:.2f}" for p in peak_freqs)
                print(f"    Peaks: [{peak_str}] THz")
            continue

        if rank == 0:
            print(f"  Tuner gap = {gap_tune*1000:.0f} nm...")

        geometry = [
            mp.Cylinder(radius=disk_radius, height=mp.inf,
                         material=gaas, center=mp.Vector3()),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                     material=gaas),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, -(disk_radius + gap + wg_width / 2)),
                     material=gaas),
            arc_prism(radius=disk_radius + gap_tune, width=tw,
                      angle_start=-theta, angle_end=theta,
                      npoints=256, material=gaas),
        ]

        sim = mp.Simulation(
            cell_size=cell, geometry=geometry,
            sources=[mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                               component=mp.Hz, center=src_center,
                               size=mp.Vector3(0, wg_width, 0))],
            boundary_layers=pml_layers, resolution=res,
        )

        flux_bus = sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5,
                          disk_radius + gap + wg_width/2),
                          size=mp.Vector3(0, wg_width, 0)))
        flux_drop = sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1.5,
                          -(disk_radius + gap + wg_width/2)),
                          size=mp.Vector3(0, wg_width, 0)))
        flux_tuner = sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(disk_radius + gap_tune + tw/2, 0),
                          size=mp.Vector3(tw, 0, 0)))
        flux_disk = sim.add_flux(fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-disk_radius + 0.15, 0),
                          size=mp.Vector3(0.3, 0, 0)))

        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            100, mp.Hz, mp.Vector3(), field_decay))

        freqs_thz = np.array(mp.get_flux_freqs(flux_bus)) * c0 / um_scale / 1e12
        bus_data = np.array(mp.get_fluxes(flux_bus))
        drop_data = np.array(mp.get_fluxes(flux_drop))
        tuner_data = np.array(mp.get_fluxes(flux_tuner))
        disk_data = np.array(mp.get_fluxes(flux_disk))
        sim.reset_meep()

        t_gap_elapsed = time.time() - t_gap_start

        # Find peaks
        peak_freqs = find_resonances(freqs_thz, disk_data, norm_flux_data,
                                      top_n=TOP_N_PEAKS)

        if rank == 0:
            np.save(os.path.join(gap_dir, "freqs_thz.npy"), freqs_thz)
            np.save(os.path.join(gap_dir, "flux_bus.npy"), bus_data)
            np.save(os.path.join(gap_dir, "flux_drop.npy"), drop_data)
            np.save(os.path.join(gap_dir, "flux_tuner.npy"), tuner_data)
            np.save(os.path.join(gap_dir, "flux_disk.npy"), disk_data)
            np.save(os.path.join(gap_dir, "norm_flux.npy"), norm_flux_data)

            with open(os.path.join(gap_dir, "params.txt"), "w") as f:
                for k, v in {"disk_radius": disk_radius, "gap": gap,
                              "wg_length": wg_length, "wg_width": wg_width,
                              "n_eff": n_eff, "resolution": res,
                              "field_decay": field_decay, "nfreq": nfreq,
                              "tuner_width": tw, "tuner_gap": gap_tune,
                              "theta_rad": theta}.items():
                    f.write(f"{k} = {v}\n")

            row = {"resolution": res, "gap_nm": gap_tune * 1000,
                   "runtime_s": round(t_gap_elapsed, 1)}
            for i, pf in enumerate(peak_freqs):
                row[f"peak_{i+1}_thz"] = round(pf, 4)
            all_results.append(row)

            peak_str = ", ".join(f"{p:.2f}" for p in peak_freqs)
            print(f"    Done in {t_gap_elapsed:.0f}s | Peaks: [{peak_str}] THz")

    t_res_total = time.time() - t_res_start
    if rank == 0:
        print(f"\n  Resolution {res} total: {t_res_total/60:.1f} min")

# ============================================================
# Save summary CSV and print comparison
# ============================================================
if rank == 0:
    # Determine columns
    max_peaks = max(len([k for k in r if k.startswith("peak_")])
                    for r in all_results) if all_results else 0
    fieldnames = ["resolution", "gap_nm", "runtime_s"]
    for i in range(max_peaks):
        fieldnames.append(f"peak_{i+1}_thz")

    csv_path = os.path.join(base_dir, "convergence_relative_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    # Print comparison table
    print("\n" + "=" * 70)
    print("RELATIVE CONVERGENCE COMPARISON")
    print("=" * 70)
    print(f"{'Gap (nm)':>10} | {'Res 48 (THz)':>14} | {'Res 128 (THz)':>14} | {'Diff (GHz)':>12}")
    print("-" * 55)

    for gap_tune in gap_tunes:
        gap_nm = gap_tune * 1000
        r48 = [r for r in all_results if r["resolution"] == 48 and r["gap_nm"] == gap_nm]
        r128 = [r for r in all_results if r["resolution"] == 128 and r["gap_nm"] == gap_nm]
        if r48 and r128:
            f48 = r48[0].get("peak_1_thz", np.nan)
            f128 = r128[0].get("peak_1_thz", np.nan)
            if f48 and f128:
                diff_ghz = abs(f48 - f128) * 1000
                print(f"{gap_nm:>10.0f} | {f48:>14.4f} | {f128:>14.4f} | {diff_ghz:>10.1f}")

    print("=" * 70)
    print(f"Summary saved: {csv_path}")
