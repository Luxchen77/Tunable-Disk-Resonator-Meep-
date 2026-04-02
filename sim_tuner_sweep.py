import meep as mp
import numpy as np
import os, math, time
from datetime import datetime
import matplotlib.pyplot as plt

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# -----------------------------------
# Simulation parameters
# -----------------------------------
c0 = 299792458  # m/s
um_scale = 1e-6  # 1 µm

sigma_meep = 0.000
gaas = mp.Medium(epsilon=12, D_conductivity=2 * math.pi * 1 * sigma_meep / 12)
disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.1

cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
cell = mp.Vector3(cell_x, cell_y, 0)
pml_layers = [mp.PML(0.8)]

f_thz = 322
fcen = f_thz * um_scale * 1e12 / c0
df_thz = 20
fwidth = df_thz * um_scale * 1e12 / c0
nfreq = 3000
field_decay = 1e-4
resolution = 24

src_center = mp.Vector3(-wg_length / 2 + 1.5, disk_radius + gap + wg_width / 2)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                     component=mp.Hz,
                     center=src_center,
                     size=mp.Vector3(0, wg_width, 0))]


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
# PART 1: FLUX ANALYSIS - Find tuner widths with acceptable loss
# -----------------------------------
print("\n" + "="*70)
print("PART 1: FLUX ANALYSIS - Screening tuner widths")
print("="*70)

# Tuner width sweep for flux analysis
tuner_widths = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12]
gap_tune = 0.03  # Fixed gap for initial screening

# Run normalization ONCE
if rank == 0:
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    analysis_folder = f"{timestamp}_tuner_optimization_f{f_thz:.1f}THz"
    analysis_dir = os.path.join(base_dir, analysis_folder)
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"\nResults will be saved to: {analysis_dir}\n")

geometry_norm = [
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
             material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
             material=gaas)
]

norm_sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                          component=mp.Hz,
                          center=src_center,
                          size=mp.Vector3(0, wg_width, 0))]

norm_flux_region = mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                                size=mp.Vector3(0, wg_width, 0))

if rank == 0:
    print("Running normalization simulation...")
    
norm_sim = mp.Simulation(cell_size=cell,
                        geometry=geometry_norm,
                        sources=norm_sources,
                        boundary_layers=pml_layers,
                        resolution=resolution)
norm_flux = norm_sim.add_flux(fcen, fwidth, nfreq, norm_flux_region)
norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))
norm_freqs = np.array(mp.get_flux_freqs(norm_flux)) * c0 / um_scale / 1e12  # THz
norm_flux_data = np.array(mp.get_fluxes(norm_flux))

if rank == 0:
    norm_dir = os.path.join(analysis_dir, "normalization")
    os.makedirs(norm_dir, exist_ok=True)
    np.save(os.path.join(norm_dir, "norm_freqs.npy"), norm_freqs)
    np.save(os.path.join(norm_dir, "norm_flux.npy"), norm_flux_data)
    print(f"Normalization complete. Saved to {norm_dir}\n")

# Sweep tuner widths
flux_results = []

for tw in tuner_widths:
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Simulating tuner width = {tw:.3f} µm")
        print(f"{'='*70}")
    
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

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=resolution
    )

    # Add flux monitors
    flux_bus_thru = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                      size=mp.Vector3(0, wg_width, 0))
    )
    flux_drop = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1.5, -disk_radius - gap - wg_width/2),
                      size=mp.Vector3(0, wg_width, 0))
    )
    # Monitor flux going INTO tuner (radial direction)
    flux_tuner = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, disk_radius + gap_tune + tw/2, 0),
                      size=mp.Vector3(0, tw, 0),
                      direction=mp.Y)
    )

    start_time = time.time()
    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))
    runtime = time.time() - start_time

    frequencies = np.array(mp.get_flux_freqs(flux_bus_thru)) * c0 / um_scale / 1e12  # THz
    bus_flux = np.array(mp.get_fluxes(flux_bus_thru))
    drop_flux = np.array(mp.get_fluxes(flux_drop))
    tuner_flux = np.abs(np.array(mp.get_fluxes(flux_tuner)))  # Take absolute value

    if rank == 0:
        # Normalize
        bus_trans = bus_flux / norm_flux_data
        drop_trans = drop_flux / norm_flux_data
        tuner_coupling = tuner_flux / norm_flux_data
        
        # Calculate total accounted power
        total_out = bus_trans + drop_trans + tuner_coupling
        other_loss = 1.0 - total_out
        
        # Average over frequency range of interest (e.g., 315-325 THz)
        freq_mask = (frequencies >= 315) & (frequencies <= 325)
        avg_bus = np.mean(bus_trans[freq_mask])
        avg_drop = np.mean(drop_trans[freq_mask])
        avg_tuner = np.mean(tuner_coupling[freq_mask])
        avg_other = np.mean(other_loss[freq_mask])
        
        flux_results.append({
            'tuner_width': tw,
            'frequencies': frequencies,
            'bus_trans': bus_trans,
            'drop_trans': drop_trans,
            'tuner_coupling': tuner_coupling,
            'other_loss': other_loss,
            'avg_bus': avg_bus,
            'avg_drop': avg_drop,
            'avg_tuner': avg_tuner,
            'avg_other': avg_other,
            'runtime': runtime
        })
        
        # Save data
        width_dir = os.path.join(analysis_dir, "flux_analysis", f"width_{tw:.3f}um")
        os.makedirs(width_dir, exist_ok=True)
        np.save(os.path.join(width_dir, "freqs_thz.npy"), frequencies)
        np.save(os.path.join(width_dir, "bus_trans.npy"), bus_trans)
        np.save(os.path.join(width_dir, "drop_trans.npy"), drop_trans)
        np.save(os.path.join(width_dir, "tuner_coupling.npy"), tuner_coupling)
        
        print(f"\nResults for width {tw:.3f} µm:")
        print(f"  Average bus throughput:    {avg_bus*100:.2f}%")
        print(f"  Average drop transmission: {avg_drop*100:.2f}%")
        print(f"  Average tuner coupling:    {avg_tuner*100:.2f}%")
        print(f"  Average other losses:      {avg_other*100:.2f}%")
        print(f"  Runtime: {runtime/60:.2f} minutes")

# -----------------------------------
# Plot flux analysis results
# -----------------------------------
if rank == 0:
    print(f"\n{'='*70}")
    print("PART 1 COMPLETE - Generating plots...")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    widths = [r['tuner_width'] for r in flux_results]
    avg_bus = [r['avg_bus'] for r in flux_results]
    avg_drop = [r['avg_drop'] for r in flux_results]
    avg_tuner = [r['avg_tuner'] for r in flux_results]
    avg_other = [r['avg_other'] for r in flux_results]
    
    # Plot 1: Power distribution vs width
    ax = axes[0, 0]
    ax.plot(widths, avg_bus, 'o-', label='Bus throughput', linewidth=2, markersize=8)
    ax.plot(widths, avg_drop, 's-', label='Drop port', linewidth=2, markersize=8)
    ax.plot(widths, avg_tuner, '^-', label='Tuner coupling', linewidth=2, markersize=8, color='red')
    ax.plot(widths, avg_other, 'd-', label='Other losses', linewidth=2, markersize=8, color='gray')
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, linewidth=2, label='5% threshold')
    ax.set_xlabel('Tuner width (µm)', fontsize=12)
    ax.set_ylabel('Power fraction', fontsize=12)
    ax.set_title('Power Budget vs Tuner Width', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Drop efficiency (ignoring tuner loss)
    ax = axes[0, 1]
    efficiency = np.array(avg_drop) / (np.array(avg_drop) + np.array(avg_tuner))
    ax.plot(widths, efficiency, 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='90% efficiency')
    ax.set_xlabel('Tuner width (µm)', fontsize=12)
    ax.set_ylabel('Drop efficiency (drop / (drop + tuner))', fontsize=12)
    ax.set_title('How Much Tuner Degrades Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spectral detail for a few widths
    ax = axes[1, 0]
    for i, idx in enumerate([0, len(flux_results)//2, -1]):
        r = flux_results[idx]
        ax.plot(r['frequencies'], r['bus_trans'], 
                label=f"Bus (w={r['tuner_width']:.2f}µm)", linewidth=2)
        ax.plot(r['frequencies'], r['drop_trans'], '--',
                label=f"Drop (w={r['tuner_width']:.2f}µm)", linewidth=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('Normalized transmission', fontsize=12)
    ax.set_title('Spectral Transmission (selected widths)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([315, 325])
    
    # Plot 4: Tuner coupling detail
    ax = axes[1, 1]
    for r in flux_results[::2]:  # Plot every other width
        ax.plot(r['frequencies'], r['tuner_coupling'],
                label=f"w={r['tuner_width']:.2f}µm", linewidth=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('Tuner coupling (normalized)', fontsize=12)
    ax.set_title('Power Coupled to Tuner vs Frequency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([315, 325])
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "flux_analysis_summary.png"), dpi=300, bbox_inches='tight')
    print(f"Saved flux analysis plots to {analysis_dir}/flux_analysis_summary.png")
    plt.close()
    
    # Determine good candidates
    max_tuner_loss = 0.08  # 8% threshold
    good_widths = [w for w, t in zip(widths, avg_tuner) if t < max_tuner_loss]
    
    print(f"\n{'='*70}")
    print(f"Candidate widths with < {max_tuner_loss*100:.0f}% tuner coupling:")
    print(f"{'='*70}")
    for w in good_widths:
        idx = widths.index(w)
        print(f"  {w:.3f} µm: Bus={avg_bus[idx]*100:.1f}%, Drop={avg_drop[idx]*100:.1f}%, Tuner={avg_tuner[idx]*100:.1f}%")
    
    if len(good_widths) == 0:
        print("\nWARNING: No widths meet criterion! Consider higher threshold or different tuner design.")
        good_widths = sorted(widths)[:3]  # Take 3 smallest
        print(f"Proceeding with 3 smallest widths: {good_widths}")


# -----------------------------------
# PART 2: HARMINV ANALYSIS - Tuning effectiveness for good candidates
# -----------------------------------
if rank == 0:
    print(f"\n\n{'='*70}")
    print("PART 2: HARMINV ANALYSIS - Measuring tuning effectiveness")
    print(f"{'='*70}\n")

# Select promising widths from Part 1
selected_widths = good_widths[:3] if rank == 0 else None  # Test top 3 candidates
selected_widths = comm.bcast(selected_widths, root=0)

# Gap sweep for each selected width
tuner_gaps = np.linspace(0.01, 0.1, 6)  # Fewer points for speed
harminv_results = {}

for tw in selected_widths:
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Harminv analysis for tuner width = {tw:.3f} µm")
        print(f"{'='*70}")
    
    gap_freqs = []
    gap_Qs = []
    gap_amps = []
    
    for tg in tuner_gaps:
        if rank == 0:
            print(f"\n  Gap = {tg:.3f} µm...")
        
        geometry = [
            mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas, center=mp.Vector3()),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                     material=gaas),
            mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                     center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
                     material=gaas),
            arc_prism(radius=disk_radius + tg,
                      width=tw,
                      angle_start=-np.pi/3,
                      angle_end=np.pi/3,
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
        
        # Harminv at disk edge
        harminv_point = mp.Vector3(disk_radius * 0.7, 0)
        harminv_obj = mp.Harminv(mp.Hz, harminv_point, fcen, fwidth)
        
        sim.run(mp.after_sources(harminv_obj),
                until_after_sources=5000)
        
        # Extract modes
        modes = [m for m in harminv_obj.modes if m.Q > 1000 and m.err < 0.01]
        
        if rank == 0:
            if len(modes) > 0:
                # Store all modes
                freqs_thz = [m.freq * c0 / um_scale / 1e12 for m in modes]
                Qs = [m.Q for m in modes]
                amps = [abs(m.amp) for m in modes]
                
                gap_freqs.append(freqs_thz)
                gap_Qs.append(Qs)
                gap_amps.append(amps)
                
                # Report strongest mode
                max_Q_idx = np.argmax(Qs)
                print(f"    Found {len(modes)} modes. Strongest: f={freqs_thz[max_Q_idx]:.4f} THz, Q={Qs[max_Q_idx]:.0f}")
            else:
                print(f"    WARNING: No modes found!")
                gap_freqs.append([])
                gap_Qs.append([])
                gap_amps.append([])
    
    if rank == 0:
        harminv_results[tw] = {
            'gaps': tuner_gaps,
            'freqs': gap_freqs,
            'Qs': gap_Qs,
            'amps': gap_amps
        }
        
        # Save raw data
        harminv_dir = os.path.join(analysis_dir, "harminv_analysis", f"width_{tw:.3f}um")
        os.makedirs(harminv_dir, exist_ok=True)
        with h5py.File(os.path.join(harminv_dir, "harminv_data.h5"), "w") as f:
            f.create_dataset("tuner_gaps", data=tuner_gaps)
            # Save as ragged arrays (different number of modes per gap)
            for i, tg in enumerate(tuner_gaps):
                grp = f.create_group(f"gap_{tg:.4f}")
                grp.create_dataset("frequencies_THz", data=gap_freqs[i] if gap_freqs[i] else [])
                grp.create_dataset("Q_factors", data=gap_Qs[i] if gap_Qs[i] else [])
                grp.create_dataset("amplitudes", data=gap_amps[i] if gap_amps[i] else [])

# -----------------------------------
# Plot Harminv results
# -----------------------------------
if rank == 0:
    print(f"\n{'='*70}")
    print("PART 2 COMPLETE - Generating tuning analysis plots...")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Frequency vs gap for each width
    ax = axes[0, 0]
    for tw in selected_widths:
        data = harminv_results[tw]
        # Track strongest mode at each gap
        main_freqs = []
        main_gaps = []
        for i, (gap, freqs, Qs) in enumerate(zip(data['gaps'], data['freqs'], data['Qs'])):
            if len(Qs) > 0:
                max_Q_idx = np.argmax(Qs)
                main_freqs.append(freqs[max_Q_idx])
                main_gaps.append(gap)
        
        if len(main_freqs) > 0:
            ax.plot(main_gaps, main_freqs, 'o-', label=f'w={tw:.2f}µm', 
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Tuner gap (µm)', fontsize=12)
    ax.set_ylabel('Resonance frequency (THz)', fontsize=12)
    ax.set_title('Frequency Tuning vs Gap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Frequency shift from reference
    ax = axes[0, 1]
    sensitivities = []
    for tw in selected_widths:
        data = harminv_results[tw]
        main_freqs = []
        main_gaps = []
        for gap, freqs, Qs in zip(data['gaps'], data['freqs'], data['Qs']):
            if len(Qs) > 0:
                max_Q_idx = np.argmax(Qs)
                main_freqs.append(freqs[max_Q_idx])
                main_gaps.append(gap)
        
        if len(main_freqs) > 1:
            shifts = np.array(main_freqs) - main_freqs[0]
            ax.plot(main_gaps, shifts, 's-', label=f'w={tw:.2f}µm',
                   linewidth=2, markersize=8)
            
            # Calculate sensitivity (linear fit)
            slope = np.polyfit(main_gaps, shifts, 1)[0]
            sensitivities.append((tw, slope))
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tuner gap (µm)', fontsize=12)
    ax.set_ylabel('Frequency shift from smallest gap (THz)', fontsize=12)
    ax.set_title('Tuning Response', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-factor degradation
    ax = axes[1, 0]
    for tw in selected_widths:
        data = harminv_results[tw]
        main_Qs = []
        main_gaps = []
        for gap, freqs, Qs in zip(data['gaps'], data['freqs'], data['Qs']):
            if len(Qs) > 0:
                max_Q_idx = np.argmax(Qs)
                main_Qs.append(Qs[max_Q_idx])
                main_gaps.append(gap)
        
        if len(main_Qs) > 0:
            ax.plot(main_gaps, np.array(main_Qs)/1000, '^-', label=f'w={tw:.2f}µm',
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Tuner gap (µm)', fontsize=12)
    ax.set_ylabel('Q-factor (×10³)', fontsize=12)
    ax.set_title('Q-factor vs Tuner Gap', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Figure of Merit
    ax = axes[1, 1]
    if len(sensitivities) > 0:
        fom_widths = []
        fom_values = []
        for tw, sensitivity in sensitivities:
            # Find corresponding tuner loss from Part 1
            idx = widths.index(tw)
            tuner_loss = avg_tuner[idx]
            if tuner_loss > 0:
                fom = abs(sensitivity) / tuner_loss  # THz/µm per % loss
                fom_widths.append(tw)
                fom_values.append(fom)
        
        ax.bar(fom_widths, fom_values, color='steelblue', alpha=0.7, width=0.02)
        ax.set_xlabel('Tuner width (µm)', fontsize=12)
        ax.set_ylabel('FOM: |Sensitivity| / Tuner Loss', fontsize=12)
        ax.set_title('Figure of Merit (Higher is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mark best
        if len(fom_values) > 0:
            best_idx = np.argmax(fom_values)
            ax.bar(fom_widths[best_idx], fom_values[best_idx], 
                  color='gold', alpha=0.9, width=0.02, label='Best')
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "harminv_analysis_summary.png"), dpi=300, bbox_inches='tight')
    print(f"Saved Harminv analysis plots to {analysis_dir}/harminv_analysis_summary.png")
    plt.close()
    
    # Final summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}\n")
    
    print("Tuning sensitivities:")
    for tw, sens in sensitivities:
        idx = widths.index(tw)
        print(f"  Width {tw:.3f} µm: {sens:.4f} THz/µm shift, {avg_tuner[idx]*100:.2f}% tuner loss")
    
    if len(fom_values) > 0:
        best_idx = np.argmax(fom_values)
        best_width = fom_widths[best_idx]
        print(f"\n✓ RECOMMENDED: Tuner width = {best_width:.3f} µm")
        print(f"  - Highest figure of merit: {fom_values[best_idx]:.2f}")
        print(f"  - Tuning sensitivity: {sensitivities[best_idx][1]:.4f} THz/µm")
        idx = widths.index(best_width)
        print(f"  - Tuner coupling loss: {avg_tuner[idx]*100:.2f}%")
        print(f"  - Drop transmission: {avg_drop[idx]*100:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {analysis_dir}")
    print(f"{'='*70}\n")