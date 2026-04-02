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

sigma_meep = 0.00
n_eff = 2.9933
gaas = mp.Medium(epsilon= n_eff**2, D_conductivity=2*math.pi*1*sigma_meep/n_eff)

disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02
gap_tune = 0.02


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
resolution = 48
theta = np.pi/4

tuner = True  # Set according to your simulation

src_center = mp.Vector3(-wg_length / 2 + 1.5, disk_radius + gap + wg_width / 2)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                     component=mp.Hz,
                     center=src_center,
                     size=mp.Vector3(0, wg_width, 0))]

tw = 0.08  # tuner width


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
geometry_norm = [
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
             material=gaas)
]

norm_sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                          component=mp.Hz,
                          center=src_center,
                          size=mp.Vector3(0, wg_width, 0))]

norm_flux_region = mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                                size=mp.Vector3(0, wg_width, 0))


# -----------------------------
# Create main run directory
# -----------------------------
if rank == 0:
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    main_folder = f"{datetime.now().strftime('%Y%m%d_%H%M')}_theta_sweep_r{disk_radius}_g{gap}_f{f_thz:.3f}_res{resolution}"
    main_dir = os.path.join(base_dir, main_folder)
    os.makedirs(main_dir, exist_ok=True)
    print(f"Main directory: {main_dir}")
else:
    main_dir = None

# Broadcast main_dir to all ranks
main_dir = comm.bcast(main_dir, root=0)

# Run normalization ONCE and save
norm_sim = mp.Simulation(cell_size=cell,
                        geometry=geometry_norm,
                        sources=norm_sources,
                        boundary_layers=pml_layers,
                        resolution=resolution)
norm_flux = norm_sim.add_flux(fcen, fwidth, nfreq, norm_flux_region)
if rank == 0:
    print("Running normalization simulation...")
norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))
norm_freqs = np.array(mp.get_flux_freqs(norm_flux)) * c0 / um_scale / 1e12  # THz
norm_flux_data = np.array(mp.get_fluxes(norm_flux))
if rank == 0:
    np.save(os.path.join(main_dir, "norm_freqs.npy"), norm_freqs)
    np.save(os.path.join(main_dir, "norm_flux.npy"), norm_flux_data)
    print("Normalization complete.")


# -----------------------------
# Sweep over theta values
# -----------------------------------
#theta_values = [1.2/2*np.pi, 1.1/2*np.pi, 1.0/2*np.pi, 0.9/2*np.pi, 0.8/2*np.pi, 0.7/2*np.pi, 0.6/2*np.pi, 0.5/2*np.pi, 0.4/2*np.pi]  # Add more theta values here
gap_tune = 0.02  # Fixed gap_tune value
res = [72, 60, 48, 36, 24] 


for resolution in res:
    # Create subfolder for this theta value
    if rank == 0:
        theta_deg = theta * 180 / np.pi
        theta_folder = f"theta_{theta_deg:.1f}deg"
        theta_dir = os.path.join(main_dir, theta_folder)
        os.makedirs(theta_dir, exist_ok=True)
        print(f"\n=== Running theta = {theta_deg:.1f}° ===")
    else:
        theta_dir = None
    
    # Broadcast theta_dir to all ranks
    theta_dir = comm.bcast(theta_dir, root=0)
    
    # Geometry with tuner on BOTTOM side
    geometry = [
        mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas, center=mp.Vector3()),
        mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
                center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
                material=gaas),
        arc_prism(radius=disk_radius + gap_tune,
                width=tw,
                angle_start=-np.pi/2 - theta,
                angle_end=-np.pi/2 + theta,
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

    flux_bus = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                    size=mp.Vector3(0, wg_width, 0))
    )
    
    flux_tuner = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, -(disk_radius + gap_tune + tw/2)),
                    size=mp.Vector3(0, tw, 0))
    )
    
    flux_disk = sim.add_flux(
        fcen, fwidth, nfreq,
        mp.FluxRegion(center=mp.Vector3(-disk_radius+0.15,0),
                    size=mp.Vector3(0.3,0,0))
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))

    frequencies = np.array(mp.get_flux_freqs(flux_bus)) * c0 / um_scale / 1e12
    bus_flux = np.array(mp.get_fluxes(flux_bus))
    tuner_flux = np.array(mp.get_fluxes(flux_tuner))
    disk_flux = np.array(mp.get_fluxes(flux_disk))

    print(f"Rank {rank} completed simulation for theta = {theta*180/np.pi:.1f}°")

    if rank == 0:
        # Save data in theta subfolder
        np.save(os.path.join(theta_dir, "freqs_thz.npy"), frequencies)
        np.save(os.path.join(theta_dir, "flux_bus.npy"), bus_flux)
        np.save(os.path.join(theta_dir, "flux_tuner.npy"), tuner_flux)
        np.save(os.path.join(theta_dir, "flux_disk.npy"), disk_flux)
        
        # Copy norm data to theta folder for convenience
        np.save(os.path.join(theta_dir, "norm_freqs.npy"), norm_freqs)
        np.save(os.path.join(theta_dir, "norm_flux.npy"), norm_flux_data)
        
        # Save parameters for this run
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
            "tuner": tuner,
            "tuner_width": tw,
            "tuner_gap": gap_tune,
            "theta_deg": theta*180/np.pi,
            "theta_rad": theta,
            "epsilon": n_eff**2
        }
        
        params_path = os.path.join(theta_dir, "params.txt")
        with open(params_path, "w") as f:
            f.write("# Simulation parameters\n")
            f.write("# ----------------------\n")
            for key, value in params.items():
                f.write(f"{key} = {value}\n")
        
        print(f"Completed theta = {theta*180/np.pi:.1f}°")

if rank == 0:
    print(f"\n=== All simulations complete ===")
    print(f"Results saved in: {main_dir}")