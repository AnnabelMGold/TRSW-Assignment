"""
Momentum equation RHS for the staggered TRSW grid.

D and theta are cell-centred.
u is located on vertical faces.
v is located on horizontal faces.

The momentum equation is:

    epsilon * Du/Dt + f * z x u + gamma * grad(D*gamma) = 0

where gamma = sqrt(theta) / F.

The returned tendencies have the same locations and shapes
as their respective velocity components.
"""

import numpy as np


def _cell_to_x_faces(q):
    """Interpolate a cell-centred field onto vertical faces."""
    Ny, Nx = q.shape
    qf = np.empty((Ny, Nx + 1))

    qf[:, 1:-1] = 0.5 * (q[:, :-1] + q[:, 1:])
    qf[:, 0] = q[:, 0]
    qf[:, -1] = q[:, -1]

    return qf


def _cell_to_y_faces(q):
    """Interpolate a cell-centred field onto horizontal faces."""
    Ny, Nx = q.shape
    qf = np.empty((Ny + 1, Nx))

    qf[1:-1, :] = 0.5 * (q[:-1, :] + q[1:, :])
    qf[0, :] = q[0, :]
    qf[-1, :] = q[-1, :]

    return qf


def _u_to_cells(u):
    """Interpolate u from vertical faces to cell centres."""
    return 0.5 * (u[:, :-1] + u[:, 1:])


def _v_to_cells(v):
    """Interpolate v from horizontal faces to cell centres."""
    return 0.5 * (v[:-1, :] + v[1:, :])


def compute_momentum_rhs(D, u, v, theta, dx, dy, epsilon, f, F):
    """
    Compute the staggered-grid momentum tendencies.

    Parameters
    ----------
    D : ndarray, shape (Ny, Nx)
        Depth at cell centres.

    u : ndarray, shape (Ny, Nx+1)
        Horizontal velocity at vertical faces.

    v : ndarray, shape (Ny+1, Nx)
        Vertical velocity at horizontal faces.

    theta : ndarray, shape (Ny, Nx)
        Buoyancy at cell centres.

    dx, dy : float
        Grid spacings.

    epsilon : float
        Rossby number.

    f : float
        Coriolis parameter.

    F : float
        Parameter in gamma = sqrt(theta)/F.

    Returns
    -------
    dudt : ndarray, shape (Ny, Nx+1)
        Horizontal velocity tendency.

    dvdt : ndarray, shape (Ny+1, Nx)
        Vertical velocity tendency.
    """

    D = np.asarray(D, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # Validate scalar fields.
    if D.ndim != 2:
        raise ValueError("D must be a 2D array.")

    Ny, Nx = D.shape

    if Ny < 3 or Nx < 3:
        raise ValueError("Grid must have at least 3 cells in each direction.")

    if theta.shape != D.shape:
        raise ValueError("theta must have the same shape as D.")

    # Validate staggered velocity shapes.
    if u.shape != (Ny, Nx + 1):
        raise ValueError("u must have shape (Ny, Nx+1).")

    if v.shape != (Ny + 1, Nx):
        raise ValueError("v must have shape (Ny+1, Nx).")

    # Validate finite values.
    for name, field in (("D", D), ("u", u), ("v", v), ("theta", theta)):
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must contain only finite values.")

    if np.any(D <= 0):
        raise ValueError("D must be strictly positive.")

    if np.any(theta <= 0):
        raise ValueError("theta must be strictly positive.")

    # Validate numerical parameters.
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    if not np.isfinite(f):
        raise ValueError("f must be finite.")

    if not np.isfinite(F) or F <= 0:
        raise ValueError("F must be finite and positive.")

    # Compute gamma and the pressure-like field S = D*gamma.
    gamma = np.sqrt(theta) / F
    S = D * gamma

    # Interpolate gamma onto the velocity faces.
    gamma_u = _cell_to_x_faces(gamma)
    gamma_v = _cell_to_y_faces(gamma)

    # Compute the gradient of S at the velocity locations.
    dS_dx = np.zeros_like(u)
    dS_dy = np.zeros_like(v)

    dS_dx[:, 1:-1] = (S[:, 1:] - S[:, :-1]) / dx
    dS_dy[1:-1, :] = (S[1:, :] - S[:-1, :]) / dy

    # Interpolate the cross-component velocity to each face.
    v_at_u = _cell_to_x_faces(_v_to_cells(v))
    u_at_v = _cell_to_y_faces(_u_to_cells(u))

    # Compute spatial derivatives of the velocity components.
    du_dx = np.gradient(u, dx, axis=1, edge_order=1)
    du_dy = np.gradient(u, dy, axis=0, edge_order=1)

    dv_dx = np.gradient(v, dx, axis=1, edge_order=1)
    dv_dy = np.gradient(v, dy, axis=0, edge_order=1)

    # Horizontal velocity tendency.
    dudt = (
        -u * du_dx
        -v_at_u * du_dy
        +(f / epsilon) * v_at_u
        -(gamma_u / epsilon) * dS_dx
    )

    # Vertical velocity tendency.
    dvdt = (
        -u_at_v * dv_dx
        -v * dv_dy
        -(f / epsilon) * u_at_v
        -(gamma_v / epsilon) * dS_dy
    )

    # Enforce zero tendency on the normal velocity boundary faces.
    dudt[:, 0] = 0.0
    dudt[:, -1] = 0.0

    dvdt[0, :] = 0.0
    dvdt[-1, :] = 0.0

    return dudt, dvdt