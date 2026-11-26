"""
Depth fluxes for the staggered TRSW grid.

Depth D is cell-centred.
u is defined on vertical faces.
v is defined on horizontal faces.

Depth is interpolated to faces using arithmetic averages.
The resulting fluxes are D_face * velocity_face.
"""

import numpy as np


def compute_depth_fluxes(D, u, v):
    """
    Compute depth fluxes on the staggered grid.

    Parameters
    ----------
    D : ndarray, shape (Ny, Nx)
        Fluid depth at cell centres.

    u : ndarray, shape (Ny, Nx+1)
        Horizontal velocity at vertical faces.

    v : ndarray, shape (Ny+1, Nx)
        Vertical velocity at horizontal faces.

    Returns
    -------
    Qx : ndarray, shape (Ny, Nx+1)
        Depth flux across vertical faces.

    Qy : ndarray, shape (Ny+1, Nx)
        Depth flux across horizontal faces.
    """

    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Validate array dimensions.
    if D.ndim != 2:
        raise ValueError("D must be a 2D array.")

    Ny, Nx = D.shape

    if Nx < 2 or Ny < 2:
        raise ValueError("D must have at least 2 cells in each direction.")

    if u.shape != (Ny, Nx + 1):
        raise ValueError("u must have shape (Ny, Nx+1).")

    if v.shape != (Ny + 1, Nx):
        raise ValueError("v must have shape (Ny+1, Nx).")

    # Validate finite values.
    if not np.all(np.isfinite(D)):
        raise ValueError("D must contain only finite values.")

    if not np.all(np.isfinite(u)):
        raise ValueError("u must contain only finite values.")

    if not np.all(np.isfinite(v)):
        raise ValueError("v must contain only finite values.")

    # Depth must be strictly positive.
    if np.any(D <= 0):
        raise ValueError("D must be strictly positive.")

    # Reconstruct depth at vertical faces.
    D_x = np.empty((Ny, Nx + 1))

    D_x[:, 1:-1] = 0.5 * (D[:, :-1] + D[:, 1:])

    # Boundary values do not affect the flux because u=0 there.
    D_x[:, 0] = D[:, 0]
    D_x[:, -1] = D[:, -1]

    # Reconstruct depth at horizontal faces.
    D_y = np.empty((Ny + 1, Nx))

    D_y[1:-1, :] = 0.5 * (D[:-1, :] + D[1:, :])

    # Boundary values do not affect the flux because v=0 there.
    D_y[0, :] = D[0, :]
    D_y[-1, :] = D[-1, :]

    # Compute depth fluxes.
    Qx = D_x * u
    Qy = D_y * v

    return Qx, Qy