import meep as mp
import numpy as np
import os
from mpi4py import MPI

# RUN WITH: mpirun -np 4 python paraview_mpi_vtk.py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# =========================================================
# PARAMETERS
# =========================================================
c0 = 299792458
um_scale = 1e-6

n_eff = 3.5125
gaas = mp.Medium(epsilon=n_eff**2)

disk_radius = 3.5
wg_length = 12
wg_width = 0.3
gap = 0.02
gap_tune = 0.05
tw = 0.05
slab_thickness = 0.16
pml_thickness_z = 0.5

# WITH SYMMETRY
cell_x = wg_length
cell_y = 2 * (disk_radius + gap + wg_width / 2) + 3.5
cell_z = slab_thickness + 2 * pml_thickness_z
cell = mp.Vector3(cell_x, cell_y, cell_z)

pml_layers = [
    mp.PML(0.8, direction=mp.X),
    mp.PML(0.8, direction=mp.Y),
    mp.PML(pml_thickness_z, direction=mp.Z),
]

# EVEN SYMMETRY for fundamental mode
symmetries = [mp.Mirror(mp.Z, phase=+1)]

# GOOD RESOLUTION
resolution = 48

# =========================================================
# GEOMETRY
# =========================================================
def arc_prism(radius, width, a0, a1, npts, material):
    r_in = radius
    r_out = radius + width
    outer = [mp.Vector3(r_out*np.cos(a), r_out*np.sin(a)) for a in np.linspace(a0, a1, npts)]
    inner = [mp.Vector3(r_in*np.cos(a), r_in*np.sin(a)) for a in np.linspace(a1, a0, npts)]
    return mp.Prism(vertices=outer + inner, height=slab_thickness, material=material)

# =========================================================
# VTK WRITER
# =========================================================
def write_vtk(filename, data, origin, spacing):
    """Write a structured grid VTK file (ASCII format)"""
    nx, ny, nz = data.shape
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MEEP Hz field data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        f.write("SCALARS Hz float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{data[i,j,k]}\n")

# =========================================================
# SETUP
# =========================================================
f_thz = 315.5
freq = f_thz * um_scale * 1e12 / c0

output_dir = "paraview_mpi_vtk_test"

if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {resolution}")
    print(f"Using symmetry: Even (phase=+1)")
    print(f"MPI ranks: {comm.Get_size()}")

comm.Barrier()

src = mp.Source(
    mp.ContinuousSource(frequency=freq),
    component=mp.Hz,
    center=mp.Vector3(-wg_length/2 + 1.5, disk_radius + gap + wg_width/2, 0),
    size=mp.Vector3(0, wg_width, slab_thickness)
)

geometry = [
    mp.Cylinder(radius=disk_radius, height=slab_thickness, material=gaas),
    mp.Block(mp.Vector3(wg_length, wg_width, slab_thickness),
             center=mp.Vector3(0, disk_radius + gap + wg_width/2, 0),
             material=gaas),
    mp.Block(mp.Vector3(wg_length, wg_width, slab_thickness),
             center=mp.Vector3(0, -disk_radius - gap - wg_width/2, 0),
             material=gaas),
    arc_prism(disk_radius + gap_tune, tw, -np.pi/4, np.pi/4, 128, gaas)
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=[src],
    boundary_layers=pml_layers,
    symmetries=symmetries,
    resolution=resolution
)

if rank == 0:
    print("Running simulation to steady state...")

# Run to steady state (all ranks)
sim.run(until=500)

if rank == 0:
    print("At steady state. Capturing time evolution...")

comm.Barrier()

# Capture 10 snapshots
n_snapshots = 10
time_interval = 10

for t in range(n_snapshots):
    # All ranks run
    sim.run(until=time_interval)
    
    # All ranks call get_array (MPI-safe)
    field = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)
    
    # Only rank 0 processes and saves
    if rank == 0:
        # Check if we got half-volume from symmetry
        expected_nz = int(cell_z * resolution)
        actual_nz = field.shape[2]
        
        if actual_nz < expected_nz * 0.7:
            # Got half volume - mirror it (even symmetry: same sign)
            field = np.concatenate([field[:, :, ::-1][:, :, :-1], field], axis=2)
        
        max_val = np.abs(field).max()
        print(f"  t={t}, shape={field.shape}, max={max_val:.3e}")
        
        # Check for corruption
        if max_val > 1000:
            print(f"    WARNING: Unusually large values detected!")
        
        # Write VTK
        nx, ny, nz = field.shape
        dx = cell_x / (nx - 1) if nx > 1 else cell_x
        dy = cell_y / (ny - 1) if ny > 1 else cell_y
        dz = cell_z / (nz - 1) if nz > 1 else cell_z
        
        origin = [-cell_x/2, -cell_y/2, -cell_z/2]
        spacing = [dx, dy, dz]
        
        vtk_file = os.path.join(output_dir, f"hz_{t:03d}.vtk")
        write_vtk(vtk_file, field, origin, spacing)
    
    comm.Barrier()

# Write PVD file (only rank 0)
if rank == 0:
    pvd_file = os.path.join(output_dir, "timeseries.pvd")
    with open(pvd_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        for t in range(n_snapshots):
            f.write(f'    <DataSet timestep="{t*time_interval}" file="hz_{t:03d}.vtk"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nOpen in ParaView:")
    print(f"  {os.path.abspath(pvd_file)}")
    print("\nIn ParaView:")
    print("  1. Open timeseries.pvd")
    print("  2. Click Apply")
    print("  3. Change Representation to 'Surface'")
    print("  4. Hit Play to see time evolution")
    print("\nData should show values around 0.01 - 10")
    print("If you see 10^60, the MPI+symmetry combination is still buggy")
    print("="*60)