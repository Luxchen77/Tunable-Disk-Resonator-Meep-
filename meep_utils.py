"""Shared utilities for Meep disk resonator simulations."""

import meep as mp
import numpy as np
import h5py


def arc_prism(radius, width, angle_start, angle_end, npoints, material,
              height=mp.inf, center=None):
    """Create an arc-shaped prism (tuner element).

    Parameters
    ----------
    height : float
        Prism height. Use mp.inf for 2D sims, slab_thickness for 3D.
    center : mp.Vector3 or None
        Prism center. Only needed for 3D sims.
    """
    r_in = radius
    r_out = radius + width
    outer = [mp.Vector3(r_out*np.cos(a), r_out*np.sin(a))
             for a in np.linspace(angle_start, angle_end, npoints)]
    inner = [mp.Vector3(r_in*np.cos(a), r_in*np.sin(a))
             for a in np.linspace(angle_end, angle_start, npoints)]
    vertices = outer + inner
    kwargs = dict(vertices=vertices, height=height, material=material)
    if center is not None:
        kwargs["center"] = center
    return mp.Prism(**kwargs)


def load_flux_data(h5_file):
    """Load frequency and flux arrays from an HDF5 file."""
    with h5py.File(h5_file, "r") as f:
        freq = f["frequency"][:]       # Meep units (1/µm)
        flux_bus = f["flux_bus"][:]
        flux_drop = f["flux_drop"][:]
    return freq, flux_bus, flux_drop
