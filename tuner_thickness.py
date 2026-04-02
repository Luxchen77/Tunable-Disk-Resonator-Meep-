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
field_decay = 1e-3
resolution = 48
theta = np.pi/4  # tuner angle extent

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
geometry_base = [
    mp.Cylinder(radius=disk_radius, height=mp.inf, material=gaas, center=mp.Vector3()),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, disk_radius + gap + wg_width / 2),
             material=gaas),
    mp.Block(size=mp.Vector3(wg_length, wg_width, mp.inf),
             center=mp.Vector3(0, -disk_radius - gap - wg_width / 2),
             material=gaas)
]


# -----------------------------
# Normalization Simulation (geometry_norm)
# -----------------------------
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


# -----------------------------
# Sweep over tuner widths
# -----------------------------------

for theta in [np.pi/4]:

    gap_tunes = [0.005, 0.010, 0.020, 0.040]
    #gap_tunes = [0.014, 0.016]

    # Run normalization ONCE and save
    if rank == 0:
        base_dir = "data"
        os.makedirs(base_dir, exist_ok=True)
        norm_folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_norm_r{disk_radius}_g{gap}_f{f_thz:.3f}_res{resolution}_decay{field_decay}_tw{tw}"
        norm_run_dir = os.path.join(base_dir, norm_folder_name)
        os.makedirs(norm_run_dir, exist_ok=True)

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
        np.save(os.path.join(norm_run_dir, "norm_freqs.npy"), norm_freqs)
        np.save(os.path.join(norm_run_dir, "norm_flux.npy"), norm_flux_data)

    # Sweep loop
    for gap_tune in gap_tunes:
        # Geometry with current tuner width
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
                    angle_start=-theta,
                    angle_end=theta,
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

        flux_top = sim.add_flux(
            fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(wg_length/2 - 1.5, disk_radius + gap + wg_width/2),
                        size=mp.Vector3(0, wg_width, 0))
        )
        flux_bottom = sim.add_flux(
            fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-wg_length/2 + 1.5, -disk_radius - gap - wg_width/2),
                        size=mp.Vector3(0, wg_width, 0))
        )
        flux_tuner = sim.add_flux(
            fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(disk_radius + gap_tune + tw/2, 0),
                        size=mp.Vector3(tw, 0, 0))
        )
        flux_disk = sim.add_flux(
            fcen, fwidth, nfreq,
            mp.FluxRegion(center=mp.Vector3(-disk_radius+0.15,0),
                        size=mp.Vector3(0.3,0,0))
        )

        #DFT MOnitotr Test -----------------------------

        # Sample field at 5 points around disk edge (evenly spaced)
        # Slightly inside edge to avoid numerical artifacts
        sample_radius = disk_radius - 0.15
        n_sample_points = 5
        angles = np.linspace(0, 2*np.pi, n_sample_points, endpoint=False)
        
        dft_disk_points = []
        for i, angle in enumerate(angles):
            x = sample_radius * np.cos(angle)
            y = sample_radius * np.sin(angle)
            dft = sim.add_dft_fields([mp.Hz], fcen, fwidth, nfreq,
                                     center=mp.Vector3(x, y),
                                     size=mp.Vector3(0, 0, 0))
            dft_disk_points.append(dft)

        sim.run(until_after_sources=mp.stop_when_fields_decayed(100, mp.Hz, mp.Vector3(), field_decay))

        frequencies = np.array(mp.get_flux_freqs(flux_top)) * c0 / um_scale / 1e12  # in THz
        top_flux = np.array(mp.get_fluxes(flux_top))
        bottom_flux = np.array(mp.get_fluxes(flux_bottom))
        tuner_flux = np.array(mp.get_fluxes(flux_tuner))
        disk_flux = np.array(mp.get_fluxes(flux_disk))

        # Extract Hz field from all 5 sample points
        disk_hz_fields = []
        for dft in dft_disk_points:
            hz_array = sim.get_dft_array(dft, mp.Hz, 0)
            disk_hz_fields.append(np.abs(hz_array)**2)  # Store intensity
        
        # Take maximum intensity across all points (avoids nodes)
        disk_intensity = np.max(disk_hz_fields, axis=0)
        
        # Optionally also save average for comparison
        disk_intensity_avg = np.mean(disk_hz_fields, axis=0)


        if rank == 0:
            folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_sim_r{disk_radius}_g{gap}_f{f_thz:.3f}_res{resolution}_decay{field_decay}_tunergap_{gap_tune:.3f}um"
            run_dir = os.path.join(base_dir, norm_folder_name, f"tunergap_{gap_tune:.3f}um")
            os.makedirs(run_dir, exist_ok=True)
            np.save(os.path.join(run_dir, "freqs_thz.npy"), frequencies)
            np.save(os.path.join(run_dir, "flux_bus.npy"), top_flux)
            np.save(os.path.join(run_dir, "flux_drop.npy"), bottom_flux)
            np.save(os.path.join(run_dir, "flux_tuner.npy"), tuner_flux)
            np.save(os.path.join(run_dir, "flux_disk.npy"), disk_flux)

            # Save disk field intensity (max and average)
            np.save(os.path.join(run_dir, "disk_intensity_max.npy"), disk_intensity)
            np.save(os.path.join(run_dir, "disk_intensity_avg.npy"), disk_intensity_avg)
            
            # Optionally save individual point data for debugging
            for i, hz_field in enumerate(disk_hz_fields):
                np.save(os.path.join(run_dir, f"disk_point_{i}.npy"), hz_field)
            


            # Save normalization data (same for all)
            np.save(os.path.join(run_dir, "norm_freqs.npy"), norm_freqs)
            np.save(os.path.join(run_dir, "norm_flux.npy"), norm_flux_data)

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
                "tuner": tuner,
                "tuner_gap": gap_tune,
                "tuner_width": tw,
                "epsilon": n_eff**2
            }

            params_path = os.path.join(run_dir, "params.txt")
            with open(params_path, "w") as f:
                f.write("# Simulation parameters\n")
                f.write("# ----------------------\n")
                for key, value in params.items():
                    f.write(f"{key} = {value}\n")

            print(f"Parameters saved to {params_path}")

