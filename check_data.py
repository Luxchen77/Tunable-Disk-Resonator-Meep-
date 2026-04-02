#!/usr/bin/env python3
"""
Quick diagnostic: Check if ParaView data is valid
Usage: python check_data.py path/to/volume3d_folder
"""

import h5py
import numpy as np
import sys
import os


folder = "/home/jonah/Desktop/meep/Tunable-Disk-Resonator-Meep-/paraview_simple_test"

print("="*60)
print("PARAVIEW DATA DIAGNOSTIC")
print("="*60)

# Check first file
first_file = os.path.join(folder, "hz_t000.h5")

if not os.path.exists(first_file):
    print(f"❌ File not found: {first_file}")
    print("\nLooking for available files...")
    import glob
    files = glob.glob(os.path.join(folder, "*.h5"))
    if files:
        print(f"Found {len(files)} HDF5 files:")
        for f in files[:5]:
            print(f"  - {os.path.basename(f)}")
    else:
        print("No HDF5 files found!")
    sys.exit(1)

print(f"Checking: {first_file}\n")

with h5py.File(first_file, 'r') as f:
    data = f['hz'][:]
    
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min: {data.min():.6e}")
    print(f"Max: {data.max():.6e}")
    print(f"Mean: {data.mean():.6e}")
    print(f"Std: {data.std():.6e}")
    print(f"Non-zero elements: {np.count_nonzero(data)} / {data.size}")
    
    abs_data = np.abs(data)
    print(f"\nAbsolute values:")
    print(f"  Max |Hz|: {abs_data.max():.6e}")
    print(f"  Mean |Hz|: {abs_data.mean():.6e}")
    
    # Check for issues
    print("\n" + "="*60)
    if np.all(data == 0):
        print("❌ PROBLEM: All data is ZERO!")
        print("\nThis means fields weren't captured properly.")
        print("Possible causes:")
        print("  1. Simulation didn't run long enough")
        print("  2. Wrong symmetry (Hz is zero at symmetry plane)")
        print("  3. Capturing at wrong location")
    elif abs_data.max() < 1e-10:
        print("⚠️  WARNING: Data values extremely small")
        print(f"   Max: {abs_data.max():.3e}")
        print("\nFields exist but might be hard to see in ParaView.")
        print("Try: Rescale to Data Range in ParaView")
    else:
        print("✅ DATA LOOKS GOOD!")
        print(f"   Field range: {data.min():.3e} to {data.max():.3e}")
        print("\nIf you can't see it in ParaView:")
        print("  1. Click 'Apply' button")
        print("  2. Change Representation to 'Surface'")
        print("  3. Click the Rescale button (⟳)")
        print("  4. Try adding a 'Threshold' filter")
        print(f"     - Set minimum to {abs_data.max()*0.1:.3e}")
        
    # Show a slice
    print("\n" + "="*60)
    print("MIDDLE SLICE CHECK (XY at z-center)")
    print("="*60)
    z_mid = data.shape[2] // 2
    slice_data = data[:, :, z_mid]
    print(f"Slice shape: {slice_data.shape}")
    print(f"Max in slice: {np.abs(slice_data).max():.6e}")
    print(f"Non-zero in slice: {np.count_nonzero(slice_data)}")
    
    if np.abs(slice_data).max() > 0:
        print("✅ Fields visible in middle z-slice")
    else:
        print("❌ No fields in middle z-slice")
        # Check other slices
        for z in [0, data.shape[2]//4, 3*data.shape[2]//4, data.shape[2]-1]:
            slice_z = data[:, :, z]
            max_z = np.abs(slice_z).max()
            if max_z > 0:
                print(f"   ✓ Found fields at z-index {z}: max = {max_z:.3e}")

print("\n" + "="*60)