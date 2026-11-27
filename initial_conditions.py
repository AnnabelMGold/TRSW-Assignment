
"""
Initial-condition utilities for the staggered TRSW model.

D is evaluated at cell centres.
theta is evaluated at grid vertices.
u is evaluated on vertical faces.
v is evaluated on horizontal faces.

The normal velocity is set to zero on the outer boundary faces.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class InitialConditions:
    """
    Container for the four initial TRSW fields on their staggered locations.

    Attributes
    ----------
    D : numpy.ndarray
        Depth at cell centres, shape (Ny, Nx).
    u : numpy.ndarray
        x-velocity on vertical faces, shape (Ny, Nx+1).
    v : numpy.ndarray
        y-velocity on horizontal faces, shape (Ny+1, Nx).
    theta : numpy.ndarray
        Buoyancy at grid vertices, shape (Ny+1, Nx+1).
    """

    D: np.ndarray
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray


def evaluate_field(field, X, Y, field_name):
    """
    Evaluate an initial field on a specified coordinate mesh.

    A field may be supplied as a scalar constant or as a callable
    function of the coordinates, field(X, Y).

    Parameters
    ----------
    field : float or callable
        Constant value or function to evaluate.
    X, Y : numpy.ndarray
        Coordinate meshes of matching shape.
    field_name : str
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Evaluated field with the same shape as X and Y.
    """
    if callable(field):
        values = np.asarray(field(X, Y), dtype=float)
    else:
        values = np.full(X.shape, field, dtype=float)

    if values.shape != X.shape:
        raise ValueError(
            f"{field_name} must have shape {X.shape}; "
            f"received {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} must contain only finite values.")

    return values


def create_initial_conditions(
    grid,
    D0=1.0,
    u0=0.0,
    v0=0.0,
    theta0=1.0,
):
    """
    Evaluate and validate initial fields on a staggered grid.

    Depth and buoyancy are evaluated on the cell-centre mesh.
    Velocities are evaluated on their respective face meshes.

    The normal velocities on the physical boundary faces are
    explicitly set to zero to satisfy the impermeable-wall condition.

    Parameters
    ----------
    grid : dict
        Dictionary returned by create_staggered_grid().
    D0, u0, v0, theta0 : float or callable
        Initial fields. Callable fields must accept two arguments,
        X and Y, and return an array matching the corresponding
        coordinate mesh.

    Returns
    -------
    InitialConditions
        Validated staggered initial fields.
    """
    # Evaluate depth at cell centres.
    D = evaluate_field(D0, grid["Xc"], grid["Yc"], "D0")

    # Evaluate buoyancy at grid vertices.
    theta = evaluate_field(
        theta0, grid["Xtheta"], grid["Ytheta"], "theta0"
    )

    # Evaluate each velocity component at its own face locations.
    u = evaluate_field(u0, grid["Xu"], grid["Yu"], "u0")
    v = evaluate_field(v0, grid["Xv"], grid["Yv"], "v0")

    # Depth and buoyancy must be strictly positive.
    if np.any(D <= 0):
        raise ValueError("Initial depth D must be strictly positive.")

    if np.any(theta <= 0):
        raise ValueError("Initial buoyancy theta must be strictly positive.")

    # Enforce zero normal velocity at the physical boundary faces.
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    return InitialConditions(
        D=D,
        u=u,
        v=v,
        theta=theta,
    )
