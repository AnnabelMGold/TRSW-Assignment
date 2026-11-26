"""
Depth equation RHS for the staggered TRSW grid.

The continuity equation is:

    dD/dt + div(D*u) = 0

Depth is cell-centred and the depth fluxes are defined
on the cell faces.
"""

import numpy as np

from fluxes import compute_depth_fluxes


def compute_depth_rhs(D, u, v, dx, dy):
    """
    Compute the time derivative of depth using the
    conservative finite-volume form of continuity.

    Parameters
    ----------
    D : ndarray, shape (Ny, Nx)
        Depth at cell centres.

    u : ndarray, shape (Ny, Nx+1)
        Horizontal velocity at vertical faces.

    v : ndarray, shape (Ny+1, Nx)
        Vertical velocity at horizontal faces.

    dx : float
        Grid spacing in x.

    dy : float
        Grid spacing in y.

    Returns
    -------
    dDdt : ndarray, shape (Ny, Nx)
        Depth time derivative at cell centres.
    """

    # Validate grid spacings.
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    # Compute fluxes on cell faces.
    Qx, Qy = compute_depth_fluxes(D, u, v)

    # Compute divergence of the fluxes.
    divergence_x = (Qx[:, 1:] - Qx[:, :-1]) / dx

    divergence_y = (Qy[1:, :] - Qy[:-1, :]) / dy

    # Continuity equation.
    dDdt = -(divergence_x + divergence_y)

    return dDdt