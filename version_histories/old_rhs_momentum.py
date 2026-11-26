
"""
Momentum RHS module for the Thermal Rotating Shallow Water
(TRSW) model.

This module calculates the time derivatives of the horizontal
velocity components, u and v.

It implements the momentum equations:

    du/dt = -u * du/dx - v * du/dy
            + (f/epsilon) * v
            - (gamma/epsilon) * d(D*gamma)/dx

    dv/dt = -u * dv/dx - v * dv/dy
            - (f/epsilon) * u
            - (gamma/epsilon) * d(D*gamma)/dy

where:

    gamma = sqrt(theta) / F

Spatial derivatives are calculated using the shared functions
in operators.py.
"""

import numpy as np

from operators import gradient
from boundary_conditions import compute_velocity_gradients


def compute_momentum_rhs(D, u, v, theta, dx, dy, epsilon, f, F):
    """
    Calculate the right-hand sides of the two momentum equations.

    Inputs:
        D (numpy.ndarray): Fluid depth. Shape (Ny, Nx).
        u (numpy.ndarray): x-component of velocity. Shape (Ny, Nx).
        v (numpy.ndarray): y-component of velocity. Shape (Ny, Nx).
        theta (numpy.ndarray): Buoyancy field. Shape (Ny, Nx).
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        epsilon (float): Rossby number; must be positive.
        f (float): Coriolis parameter.
        F (float): Parameter in gamma = sqrt(theta) / F;
            must be positive.

    Outputs:
        dudt (numpy.ndarray): Time derivative of u.
            Shape (Ny, Nx).
        dvdt (numpy.ndarray): Time derivative of v.
            Shape (Ny, Nx).

    Description:
        Calculates the momentum tendencies from advection,
        Coriolis acceleration and the buoyancy-related pressure
        gradient.

        This function calculates tendencies only. It does not
        advance the solution in time.
    """

    # ---------------------------------------------------------
    # 1. Check that all input fields are 2D arrays
    # ---------------------------------------------------------

    fields = [D, u, v, theta]

    for field in fields:
        if not isinstance(field, np.ndarray) or field.ndim != 2:
            raise ValueError("All input fields must be 2D NumPy arrays.")

    # ---------------------------------------------------------
    # 2. Check that all fields have the same shape
    # ---------------------------------------------------------

    if not (D.shape == u.shape == v.shape == theta.shape):
        raise ValueError("All input fields must have the same shape.")

    # ---------------------------------------------------------
    # 3. Check that the grid has enough points
    # ---------------------------------------------------------

    Ny, Nx = D.shape

    if Nx < 3 or Ny < 3:
        raise ValueError("The grid must have at least 3 points in each direction.")

    # ---------------------------------------------------------
    # 4. Check that all field values are finite
    # ---------------------------------------------------------

    for field in fields:
        if not np.all(np.isfinite(field)):
            raise ValueError("Input fields must contain only finite values.")

    # ---------------------------------------------------------
    # 5. Check that depth and buoyancy are strictly positive
    # ---------------------------------------------------------

    if np.any(D <= 0):
        raise ValueError("Fluid depth D must be strictly positive.")

    if np.any(theta <= 0):
        raise ValueError("Buoyancy theta must be strictly positive.")

    # ---------------------------------------------------------
    # 6. Check numerical parameters
    # ---------------------------------------------------------

    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")

    if not np.isfinite(dy) or dy <= 0:
        raise ValueError("dy must be finite and positive.")

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    if not np.isfinite(F) or F <= 0:
        raise ValueError("F must be finite and positive.")

    if not np.isfinite(f):
        raise ValueError("f must be finite.")

    # ---------------------------------------------------------
    # 7. Calculate gamma and D*gamma
    # ---------------------------------------------------------

    # Diagnostic calculation of gamma from the buoyancy field
    gamma = np.sqrt(theta) / F

    # Quantity appearing inside the spatial gradient
    D_gamma = D * gamma

    # ---------------------------------------------------------
    # 8. Calculate the spatial derivatives
    # ---------------------------------------------------------

    # Velocity derivatives
    du_dx, du_dy = gradient(u, dx, dy)
    dv_dx, dv_dy = gradient(v, dx, dy)

    #TRIED TO DO du_dx, du_dy, dv_dx, dv_dy = compute_velocity_gradients(u, v, dx, dy)


    # Derivatives of D*gamma
    dD_gamma_dx, dD_gamma_dy = gradient(D_gamma, dx, dy)

    # ---------------------------------------------------------
    # 9. Calculate the advection terms
    # ---------------------------------------------------------

    advection_u = -(u * du_dx + v * du_dy)
    advection_v = -(u * dv_dx + v * dv_dy)

    # ---------------------------------------------------------
    # 10. Calculate the Coriolis terms
    # ---------------------------------------------------------

    coriolis_u = (f / epsilon) * v
    coriolis_v = -(f / epsilon) * u

    # ---------------------------------------------------------
    # 11. Calculate the pressure/buoyancy terms
    # ---------------------------------------------------------

    pressure_u = -(gamma / epsilon) * dD_gamma_dx
    pressure_v = -(gamma / epsilon) * dD_gamma_dy

    # ---------------------------------------------------------
    # 12. Combine all contributions
    # ---------------------------------------------------------

    dudt = advection_u + coriolis_u + pressure_u
    dvdt = advection_v + coriolis_v + pressure_v

    return dudt, dvdt
