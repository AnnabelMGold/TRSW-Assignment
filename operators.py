
"""
Spatial operators module for the Thermal Rotating Shallow Water
(TRSW) model.

This module implements numerical spatial derivatives on a uniform,
cell-centred Cartesian grid.
"""

import numpy as np


def derivative_x(q, dx):
    """
    Compute the first spatial derivative of a scalar field in x.

    Inputs:
        q (numpy.ndarray): 2D scalar field on the computational grid.
            Shape: (Ny, Nx).
        dx (float): Uniform grid spacing in the x-direction.

    Outputs:
        dqdx (numpy.ndarray): Numerical approximation to the x-derivative
            of q. Shape: (Ny, Nx).

    Description:
        Uses second-order finite differences.

        Centred differences are used at interior points, while
        second-order one-sided differences are used at the boundaries.

        The derivative is calculated along axis 1, corresponding
        to the x-direction under our array convention.
    """

    # Check that the input field is a 2D array
    if q.ndim != 2:
        raise ValueError("q must be a 2D array.")

    # Check that the grid spacing is positive
    if dx <= 0:
        raise ValueError("dx must be positive.")

    # At least three points are needed for second-order
    # one-sided boundary differences
    if q.shape[1] < 3:
        raise ValueError("q must contain at least 3 points in x.")

    # Calculate the derivative along the x-direction
    dqdx = np.gradient(q, dx, axis=1, edge_order=2)

    return dqdx


def derivative_y(q, dy):
    """
    Compute the first spatial derivative of a scalar field in y.

    Inputs:
        q (numpy.ndarray): 2D scalar field on the computational grid.
            Shape: (Ny, Nx).
        dy (float): Uniform grid spacing in the y-direction.

    Outputs:
        dqdy (numpy.ndarray): Numerical approximation to the y-derivative
            of q. Shape: (Ny, Nx).

    Description:
        Uses second-order finite differences.

        Centred differences are used at interior points, while
        second-order one-sided differences are used at the boundaries.

        The derivative is calculated along axis 0, corresponding
        to the y-direction under our array convention.
    """

    # Check that the input field is a 2D array
    if q.ndim != 2:
        raise ValueError("q must be a 2D array.")

    # Check that the grid spacing is positive
    if dy <= 0:
        raise ValueError("dy must be positive.")

    # At least three points are needed for second-order
    # one-sided boundary differences
    if q.shape[0] < 3:
        raise ValueError("q must contain at least 3 points in y.")

    # Calculate the derivative along the y-direction
    dqdy = np.gradient(q, dy, axis=0, edge_order=2)

    return dqdy


def gradient(q, dx, dy):
    """
    Compute the first spatial derivatives of a scalar field in x and y.

    Inputs:
        q (numpy.ndarray): 2D scalar field on the computational grid.
            Shape: (Ny, Nx).
        dx (float): Uniform grid spacing in the x-direction.
        dy (float): Uniform grid spacing in the y-direction.

    Outputs:
        dqdx (numpy.ndarray): Numerical x-derivative of q.
            Shape: (Ny, Nx).
        dqdy (numpy.ndarray): Numerical y-derivative of q.
            Shape: (Ny, Nx).

    Description:
        Calls derivative_x and derivative_y to calculate the
        corresponding spatial derivatives.

        The output arrays follow our convention:
        rows correspond to y and columns correspond to x.
    """

    # Calculate the derivatives in each direction
    dqdx = derivative_x(q, dx)
    dqdy = derivative_y(q, dy)

    return dqdx, dqdy
