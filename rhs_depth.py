import numpy as np

from fluxes import compute_depth_fluxes


def compute_depth_rhs(D, u, v, dx, dy):
    """
    Compute the time derivative of fluid depth using a
    conservative finite-volume discretisation.

    The continuity equation is:

        dD/dt = -div(D*u_vector)

    The divergence is computed using numerical fluxes
    across cell faces, obtained from compute_depth_fluxes.

    Parameters
    ----------
    D : numpy.ndarray
        Fluid depth at cell centres, shape (Ny, Nx).
    u : numpy.ndarray
        Horizontal velocity, shape (Ny, Nx).
    v : numpy.ndarray
        Vertical (y-direction) velocity, shape (Ny, Nx).
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    Returns
    -------
    dDdt : numpy.ndarray
        Time derivative of depth, shape (Ny, Nx).

    Notes
    -----
    Zero normal depth fluxes are imposed at the outer
    domain boundaries by compute_depth_fluxes.

    The finite-volume formulation is conservative:
    interior fluxes cancel when summed over the domain.
    """

    # ---------------------------------------------------------
    # 1. Check grid spacing
    # ---------------------------------------------------------

    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be a finite positive number.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be a finite positive number.")

    # ---------------------------------------------------------
    # 2. Compute numerical fluxes at cell faces
    # ---------------------------------------------------------

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # ---------------------------------------------------------
    # 3. Compute the discrete divergence of the fluxes
    # ---------------------------------------------------------

    # X-direction flux difference:
    # right face minus left face, divided by dx.
    div_x = (Qx[:, 1:] - Qx[:, :-1]) / dx

    # Y-direction flux difference:
    # upper face minus lower face, divided by dy.
    div_y = (Qy[1:, :] - Qy[:-1, :]) / dy

    # ---------------------------------------------------------
    # 4. Compute the depth RHS
    # ---------------------------------------------------------

    # Continuity equation:
    # dD/dt = -div(D*u_vector).
    dDdt = -(div_x + div_y)

    # ---------------------------------------------------------
    # 5. Return the result
    # ---------------------------------------------------------

    return dDdt