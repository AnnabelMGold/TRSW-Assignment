"""
Buoyancy RHS for the staggered TRSW grid.

D is cell-centred.
theta is vertex-centred.
u is located on vertical faces.
v is located on horizontal faces.

The buoyancy equation is:

    D(theta)/Dt = 0

or equivalently:

    dtheta/dt + u*dtheta/dx + v*dtheta/dy = 0.

The buoyancy tendency is calculated at interior vertices.
The boundary buoyancy tendency is set to zero to enforce
the selected buoyancy boundary condition.
"""

import numpy as np


def compute_buoyancy_rhs(theta, u, v, dx, dy):
    """
    Compute the vertex-centred buoyancy tendency.

    Parameters
    ----------
    theta : ndarray, shape (Ny+1, Nx+1)
        Buoyancy at grid vertices.

    u : ndarray, shape (Ny, Nx+1)
        Horizontal velocity at vertical faces.

    v : ndarray, shape (Ny+1, Nx)
        Vertical velocity at horizontal faces.

    dx, dy : float
        Grid spacings.

    Returns
    -------
    dthetadt : ndarray, shape (Ny+1, Nx+1)
        Buoyancy time derivative at grid vertices.
    """

    theta = np.asarray(theta, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Validate buoyancy array.
    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array.")

    Ny = theta.shape[0] - 1
    Nx = theta.shape[1] - 1

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

    # Interpolate u to vertices.
    u_vertex = np.empty_like(theta)
    u_vertex[1:-1, :] = 0.5 * (u[:-1, :] + u[1:, :])
    u_vertex[0, :] = u[0, :]
    u_vertex[-1, :] = u[-1, :]

    # Interpolate v to vertices.
    v_vertex = np.empty_like(theta)
    v_vertex[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    v_vertex[:, 0] = v[:, 0]
    v_vertex[:, -1] = v[:, -1]

    # Compute buoyancy gradients at vertices.
    dtheta_dx = np.gradient(theta, dx, axis=1, edge_order=1)
    dtheta_dy = np.gradient(theta, dy, axis=0, edge_order=1)

    # Advective buoyancy tendency.
    dthetadt = -(
        u_vertex * dtheta_dx
        + v_vertex * dtheta_dy
    )

    # Enforce zero buoyancy tendency on the boundary.
    dthetadt[0, :] = 0.0
    dthetadt[-1, :] = 0.0
    dthetadt[:, 0] = 0.0
    dthetadt[:, -1] = 0.0

    return dthetadt