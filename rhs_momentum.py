import numpy as np


def compute_momentum_rhs(D, u, v, theta, dx, dy, epsilon, f, F):
    """
    Compute the time derivatives of the two velocity components
    using a finite-difference discretisation of the TRSW momentum
    equations.

    The governing momentum equation is:

        epsilon * D(u_vector)/Dt
        + f * (z cross u_vector)
        + gamma * grad(D * gamma) = 0

    The velocity components are evaluated using:
        - Centred finite differences in the interior.
        - Second-order one-sided differences at the edges,
          as provided by numpy.gradient.

    Parameters
    ----------
    D : numpy.ndarray
        Fluid depth, shape (Ny, Nx).
    u : numpy.ndarray
        Horizontal velocity, shape (Ny, Nx).
    v : numpy.ndarray
        Vertical (y-direction) velocity, shape (Ny, Nx).
    theta : numpy.ndarray
        Buoyancy field, shape (Ny, Nx), strictly positive.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    epsilon : float
        Rossby number. Must be strictly positive.
    f : float
        Coriolis parameter.
    F : float
        Parameter in gamma = sqrt(theta) / F.
        Must be strictly positive.

    Returns
    -------
    dudt : numpy.ndarray
        Time derivative of horizontal velocity, shape (Ny, Nx).
    dvdt : numpy.ndarray
        Time derivative of vertical velocity, shape (Ny, Nx).

    Notes
    -----
    The momentum equations implemented are:

        dudt = -u * du/dx - v * du/dy
               + (f/epsilon) * v
               - (gamma/epsilon) * d(D*gamma)/dx

        dvdt = -u * dv/dx - v * dv/dy
               - (f/epsilon) * u
               - (gamma/epsilon) * d(D*gamma)/dy

    This function computes the RHS only. It does not advance
    the solution in time.

    Boundary derivatives are approximated using numpy.gradient
    with edge_order=2. This does not independently enforce the
    physical velocity boundary conditions.
    """

    # ---------------------------------------------------------
    # 1. Validate the input arrays
    # ---------------------------------------------------------

    # Every field must be a two-dimensional array.
    if any(field.ndim != 2 for field in (D, u, v, theta)):
        raise ValueError("D, u, v and theta must all be 2D arrays.")

    # All fields must have exactly the same shape.
    if D.shape != u.shape or D.shape != v.shape or D.shape != theta.shape:
        raise ValueError("D, u, v and theta must have identical shapes.")

    Ny, Nx = D.shape

    # At least 3 cells in each direction are needed for the
    # second-order edge derivative formulas.
    if Nx < 3 or Ny < 3:
        raise ValueError("Grid must have at least 3 cells in each direction.")

    # All input fields must contain finite values.
    for name, field in zip(("D", "u", "v", "theta"), (D, u, v, theta)):
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} contains non-finite values.")

    # Depth and buoyancy must be strictly positive.
    if np.any(D <= 0):
        raise ValueError("Depth D must be strictly positive.")

    if np.any(theta <= 0):
        raise ValueError("Buoyancy theta must be strictly positive.")

    # ---------------------------------------------------------
    # 2. Validate the scalar parameters
    # ---------------------------------------------------------

    # Grid spacing must be finite and positive.
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    # Rossby number and F must be finite and positive.
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    if not np.isfinite(F) or F <= 0:
        raise ValueError("F must be finite and positive.")

    # Coriolis parameter must be finite.
    if not np.isfinite(f):
        raise ValueError("f must be finite.")

    # ---------------------------------------------------------
    # 3. Calculate gamma and the auxiliary field D*gamma
    # ---------------------------------------------------------

    # gamma is diagnosed from the buoyancy field.
    gamma = np.sqrt(theta) / F

    # Define the auxiliary scalar S = D * gamma.
    S = D * gamma

    # ---------------------------------------------------------
    # 4. Calculate velocity gradients
    # ---------------------------------------------------------

    # numpy.gradient uses centred differences in the interior
    # and second-order one-sided differences at the edges.
    #
    # Array axis 0 corresponds to y.
    # Array axis 1 corresponds to x.

    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=2)

    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=2)

    # ---------------------------------------------------------
    # 5. Calculate gradients of the auxiliary scalar S
    # ---------------------------------------------------------

    dS_dy, dS_dx = np.gradient(S, dy, dx, edge_order=2)

    # ---------------------------------------------------------
    # 6. Calculate the advection contributions
    # ---------------------------------------------------------

    # Horizontal momentum advection.
    advection_u = -(u * du_dx + v * du_dy)

    # Vertical (y-direction) momentum advection.
    advection_v = -(u * dv_dx + v * dv_dy)

    # ---------------------------------------------------------
    # 7. Calculate the Coriolis contributions
    # ---------------------------------------------------------

    # z cross u_vector = (-v, u).
    coriolis_u = (f / epsilon) * v
    coriolis_v = -(f / epsilon) * u

    # ---------------------------------------------------------
    # 8. Calculate the pressure/buoyancy contributions
    # ---------------------------------------------------------

    # Preserve the governing term gamma * grad(D*gamma).
    pressure_u = -(gamma / epsilon) * dS_dx
    pressure_v = -(gamma / epsilon) * dS_dy

    # ---------------------------------------------------------
    # 9. Combine the contributions
    # ---------------------------------------------------------

    dudt = advection_u + coriolis_u + pressure_u
    dvdt = advection_v + coriolis_v + pressure_v

    # ---------------------------------------------------------
    # 10. Return the momentum RHS
    # ---------------------------------------------------------

    return dudt, dvdt