"""
Buoyancy RHS for the staggered TRSW grid.

Buoyancy theta is cell-centred.
u and v are defined on their respective staggered faces.

The buoyancy equation is:

    D(theta)/Dt = 0

or equivalently:

    dtheta/dt + u*dtheta/dx + v*dtheta/dy = 0.
"""

import numpy as np


def compute_buoyancy_rhs(theta, u, v, dx, dy):
    """
    Compute the buoyancy tendency at cell centres.

    Parameters
    ----------
    theta : ndarray, shape (Ny, Nx)
        Buoyancy at cell centres.

    u : ndarray, shape (Ny, Nx+1)
        Horizontal velocity at vertical faces.

    v : ndarray, shape (Ny+1, Nx)
        Vertical velocity at horizontal faces.

    dx, dy : float
        Grid spacings.

    Returns
    -------
    dthetadt : ndarray, shape (Ny, Nx)
        Buoyancy time derivative at cell centres.
    """

    theta = np.asarray(theta, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Validate buoyancy array.
    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array.")

    Ny, Nx = theta.shape

    if Ny < 3 or Nx < 3:
        raise ValueError("Grid must have at least 3 cells in each direction.")

    # Validate velocity shapes.
    if u.shape != (Ny, Nx + 1):
        raise ValueError("u must have shape (Ny, Nx+1).")

    if v.shape != (Ny + 1, Nx):
        raise ValueError("v must have shape (Ny+1, Nx).")

    # Validate finite values.
    for name, field in (("theta", theta), ("u", u), ("v", v)):
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must contain only finite values.")

    # Validate grid spacings.
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    # Interpolate velocity components to cell centres.
    u_center = 0.5 * (u[:, :-1] + u[:, 1:])
    v_center = 0.5 * (v[:-1, :] + v[1:, :])

    # Compute buoyancy gradients at cell centres.
    dtheta_dx = np.gradient(theta, dx, axis=1, edge_order=2)
    dtheta_dy = np.gradient(theta, dy, axis=0, edge_order=2)

    # Advective buoyancy tendency.
    dthetadt = -(u_center * dtheta_dx + v_center * dtheta_dy)

    return dthetadt