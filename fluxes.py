import numpy as np


def compute_depth_fluxes(D, u, v):
    """
    Compute numerical depth fluxes across cell faces.

    Uses averaged velocities at interior faces and first-order
    upwind reconstruction of the depth. Normal fluxes at the
    outer domain boundaries are set to zero, enforcing the
    impermeable-wall condition for the depth equation.

    Parameters
    ----------
    D : numpy.ndarray
        Fluid depth at cell centres, shape (Ny, Nx).
    u : numpy.ndarray
        Horizontal velocity at cell centres, shape (Ny, Nx).
    v : numpy.ndarray
        Vertical (y-direction) velocity at cell centres,
        shape (Ny, Nx).

    Returns
    -------
    Qx : numpy.ndarray
        Depth flux across vertical faces, shape (Ny, Nx+1).
    Qy : numpy.ndarray
        Depth flux across horizontal faces, shape (Ny+1, Nx).

    Notes
    -----
    Qx is positive in the x-direction.
    Qy is positive in the y-direction.

    Boundary fluxes are explicitly set to zero.
    """

    # ---------------------------------------------------------
    # 1. Check input arrays
    # ---------------------------------------------------------

    # All fields must be two-dimensional.
    if D.ndim != 2 or u.ndim != 2 or v.ndim != 2:
        raise ValueError("D, u and v must be 2D arrays.")

    # All fields must have the same shape.
    if D.shape != u.shape or D.shape != v.shape:
        raise ValueError("D, u and v must have identical shapes.")

    Ny, Nx = D.shape

    # Require at least 3 cells in each direction.
    if Nx < 3 or Ny < 3:
        raise ValueError("Grid must have at least 3 cells in each direction.")

    # Reject non-finite input values.
    if not np.all(np.isfinite(D)):
        raise ValueError("D contains non-finite values.")

    if not np.all(np.isfinite(u)):
        raise ValueError("u contains non-finite values.")

    if not np.all(np.isfinite(v)):
        raise ValueError("v contains non-finite values.")

    # Depth must be strictly positive.
    if np.any(D <= 0):
        raise ValueError("Depth D must be strictly positive.")

    # ---------------------------------------------------------
    # 2. Allocate face-flux arrays
    # ---------------------------------------------------------

    # Qx: one more column than the cell-centred fields.
    # Columns correspond to the vertical faces.
    Qx = np.zeros((Ny, Nx + 1))

    # Qy: one more row than the cell-centred fields.
    # Rows correspond to the horizontal faces.
    Qy = np.zeros((Ny + 1, Nx))

    # ---------------------------------------------------------
    # 3. Compute x-direction fluxes at interior faces
    # ---------------------------------------------------------

    # Average the horizontal velocity across each interior face.
    u_face = 0.5 * (u[:, :-1] + u[:, 1:])

    # First-order upwind reconstruction of depth:
    # use the left cell if flow is towards the right,
    # and the right cell if flow is towards the left.
    D_face = np.where(
        u_face >= 0,
        D[:, :-1],
        D[:, 1:]
    )

    # Compute the depth flux at interior vertical faces.
    Qx[:, 1:-1] = u_face * D_face

    # The outer left and right boundary fluxes remain zero.
    # This enforces no normal flow through those boundaries.

    # ---------------------------------------------------------
    # 4. Compute y-direction fluxes at interior faces
    # ---------------------------------------------------------

    # Average the vertical velocity across each interior face.
    v_face = 0.5 * (v[:-1, :] + v[1:, :])

    # First-order upwind reconstruction of depth:
    # use the lower cell if flow is towards increasing y,
    # and the upper cell if flow is towards decreasing y.
    D_face = np.where(
        v_face >= 0,
        D[:-1, :],
        D[1:, :]
    )

    # Compute the depth flux at interior horizontal faces.
    Qy[1:-1, :] = v_face * D_face

    # The outer bottom and top boundary fluxes remain zero.
    # This enforces no normal flow through those boundaries.

    # ---------------------------------------------------------
    # 5. Return the face fluxes
    # ---------------------------------------------------------

    return Qx, Qy