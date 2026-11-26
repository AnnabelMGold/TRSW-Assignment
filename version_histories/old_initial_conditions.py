
"""
Initial conditions module for the Thermal Rotating Shallow Water (TRSW) model.

This module defines and assembles the initial fields:
    D  - fluid depth
    u  - x-component of horizontal velocity
    v  - y-component of horizontal velocity
    theta - buoyancy

Each field can be specified as a constant or as a function of the
2D coordinate arrays X and Y.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class InitialConditions:
    """
    Store the initial fields for the TRSW model.

    Attributes:
        D (numpy.ndarray): Initial fluid depth. Shape (Ny, Nx).
        u (numpy.ndarray): Initial x-component of velocity.
            Shape (Ny, Nx).
        v (numpy.ndarray): Initial y-component of velocity.
            Shape (Ny, Nx).
        theta (numpy.ndarray): Initial buoyancy field.
            Shape (Ny, Nx).
    """

    D: np.ndarray
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray


def evaluate_field(field, X, Y, field_name):
    """
    Evaluate an initial field on the computational grid.

    Inputs:
        field (float or callable):
            Either a constant value or a function of X and Y.
            If callable, it must accept X and Y as arguments and return
            an array of values on the grid.
        X (numpy.ndarray): 2D array of x-coordinates. Shape (Ny, Nx).
        Y (numpy.ndarray): 2D array of y-coordinates. Shape (Ny, Nx).
        field_name (str): Name of the field, used in error messages.

    Outputs:
        field_values (numpy.ndarray): Evaluated field on the grid.
            Shape (Ny, Nx).

    Description:
        Constant values are expanded to arrays of the grid shape.
        Functions are evaluated directly on the coordinate arrays.

        The resulting field must have the same shape as X and Y,
        and all values must be finite.
    """

    # Evaluate the field if a function has been supplied
    if callable(field):
        field_values = np.asarray(field(X, Y), dtype=float)
    else:
        # Expand a constant value across the entire grid
        field_values = np.full(X.shape, field, dtype=float)

    # Check that the field has the correct shape
    if field_values.shape != X.shape:
        raise ValueError(
            f"{field_name} must have shape {X.shape}, "
            f"but received shape {field_values.shape}."
        )

    # Check that all values are finite
    if not np.all(np.isfinite(field_values)):
        raise ValueError(f"{field_name} contains non-finite values.")

    return field_values


def create_initial_conditions(
    X,
    Y,
    D0=1.0,
    u0=0.0,
    v0=0.0,
    theta0=1.0,
):
    """
    Construct the initial fields for the TRSW simulation.

    Inputs:
        X (numpy.ndarray): 2D array of x-coordinates. Shape (Ny, Nx).
        Y (numpy.ndarray): 2D array of y-coordinates. Shape (Ny, Nx).
        D0 (float or callable): Initial depth, either a constant or
            a function D0(X, Y).
        u0 (float or callable): Initial x-velocity, either a constant
            or a function u0(X, Y).
        v0 (float or callable): Initial y-velocity, either a constant
            or a function v0(X, Y).
        theta0 (float or callable): Initial buoyancy, either a constant
            or a function theta0(X, Y).

    Outputs:
        initial_conditions (InitialConditions):
            Object containing the four initial fields, each represented
            by a numpy.ndarray of shape (Ny, Nx).

    Description:
        Each initial field is evaluated on the computational grid.
        Depth and buoyancy must be strictly positive everywhere.
        Velocity components may take any finite values.
    """

    # Evaluate each initial field on the grid
    D = evaluate_field(D0, X, Y, "D0")
    u = evaluate_field(u0, X, Y, "u0")
    v = evaluate_field(v0, X, Y, "v0")
    theta = evaluate_field(theta0, X, Y, "theta0")

    # Depth must be strictly positive
    if np.any(D <= 0):
        raise ValueError("Initial depth D0 must be strictly positive.")

    # Buoyancy must be strictly positive because gamma = sqrt(theta) / F
    if np.any(theta <= 0):
        raise ValueError("Initial buoyancy theta0 must be strictly positive.")

    # Assemble the initial fields into a single object
    return InitialConditions(D=D, u=u, v=v, theta=theta)
