
"""
Staggered Cartesian grid for the Thermal Rotating Shallow Water model.

Scalar fields D and theta are stored at cell centres.
The velocity component u is stored on vertical faces.
The velocity component v is stored on horizontal faces.
"""

import numpy as np

from config import ModelConfig


def create_staggered_grid(config: ModelConfig):
    """
    Construct a uniform staggered grid for a rectangular domain.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing domain lengths and grid sizes.

    Returns
    -------
    dict
        Dictionary containing coordinate arrays, mesh arrays, and
        grid spacings.

        Cell centres:
            xc, yc, Xc, Yc

        Vertical faces (where u is stored):
            xf, yf_u, Xu, Yu

        Horizontal faces (where v is stored):
            xf_v, yf, Xv, Yv

        Spacings:
            dx, dy

        Shapes:
            Xc, Yc : (Ny, Nx)
            Xu, Yu : (Ny, Nx + 1)
            Xv, Yv : (Ny + 1, Nx)
    """
    Nx = config.Nx
    Ny = config.Ny

    dx = config.Lx / Nx
    dy = config.Ly / Ny

    # Cell-centre coordinates for scalar fields D and theta.
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy

    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    # Vertical faces for the x-velocity u.
    xf = np.linspace(0.0, config.Lx, Nx + 1)
    yf_u = yc.copy()

    Xu, Yu = np.meshgrid(xf, yf_u, indexing="xy")

    # Horizontal faces for the y-velocity v.
    xf_v = xc.copy()
    yf = np.linspace(0.0, config.Ly, Ny + 1)

    Xv, Yv = np.meshgrid(xf_v, yf, indexing="xy")

    return {
        "xc": xc,
        "yc": yc,
        "Xc": Xc,
        "Yc": Yc,
        "xf": xf,
        "yf_u": yf_u,
        "Xu": Xu,
        "Yu": Yu,
        "xf_v": xf_v,
        "yf": yf,
        "Xv": Xv,
        "Yv": Yv,
        "dx": dx,
        "dy": dy,
    }
