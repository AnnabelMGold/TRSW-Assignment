
"""
Buoyancy RHS module for the Thermal Rotating Shallow Water
(TRSW) model.

This module calculates the time derivative of the buoyancy
field, theta.

It implements the buoyancy transport equation:

    dtheta/dt = -u * dtheta/dx - v * dtheta/dy

Spatial derivatives are calculated using the shared gradient
function in operators.py.

This function calculates the buoyancy tendency only. It does
not advance the solution in time.
"""

import numpy as np

from operators import gradient


def compute_buoyancy_rhs(theta, u, v, dx, dy):
    """
    Calculate the right-hand side of the buoyancy equation.

    Inputs:
        theta (numpy.ndarray): Buoyancy field. Shape (Ny, Nx).
        u (numpy.ndarray): x-component of velocity. Shape (Ny, Nx).
        v (numpy.ndarray): y-component of velocity. Shape (Ny, Nx).
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.

    Outputs:
        dtheta_dt (numpy.ndarray): Time derivative of buoyancy.
            Shape (Ny, Nx).

    Description:
        Calculates the transport of buoyancy by the horizontal
        fluid velocity.

        The governing equation is:

            dtheta/dt + u*dtheta/dx + v*dtheta/dy = 0

        The function uses second-order finite differences through
        the shared gradient function.
    """

    # ---------------------------------------------------------
    # 1. Check that all input fields are 2D NumPy arrays
    # ---------------------------------------------------------

    fields = [theta, u, v]

    for field in fields:
        if not isinstance(field, np.ndarray) or field.ndim != 2:
            raise ValueError("All input fields must be 2D NumPy arrays.")

    # ---------------------------------------------------------
    # 2. Check that all fields have the same shape
    # ---------------------------------------------------------

    if not (theta.shape == u.shape == v.shape):
        raise ValueError("All input fields must have the same shape.")

    # ---------------------------------------------------------
    # 3. Check that the grid has enough points
    # ---------------------------------------------------------

    Ny, Nx = theta.shape

    if Nx < 3 or Ny < 3:
        raise ValueError("The grid must have at least 3 points in each direction.")

    # ---------------------------------------------------------
    # 4. Check that all field values are finite
    # ---------------------------------------------------------

    for field in fields:
        if not np.all(np.isfinite(field)):
            raise ValueError("Input fields must contain only finite values.")

    # ---------------------------------------------------------
    # 5. Check that buoyancy is strictly positive
    # ---------------------------------------------------------

    if np.any(theta <= 0):
        raise ValueError("Buoyancy theta must be strictly positive.")

    # ---------------------------------------------------------
    # 6. Check grid spacing
    # ---------------------------------------------------------

    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    # ---------------------------------------------------------
    # 7. Calculate the spatial derivatives of buoyancy
    # ---------------------------------------------------------

    dtheta_dx, dtheta_dy = gradient(theta, dx, dy)

    # ---------------------------------------------------------
    # 8. Calculate the buoyancy transport term
    # ---------------------------------------------------------

    dtheta_dt = -(u * dtheta_dx + v * dtheta_dy)

    return dtheta_dt
