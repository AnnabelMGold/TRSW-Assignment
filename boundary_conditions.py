
"""
Boundary-condition reconstruction utilities for the TRSW model.

The velocity components are stored at cell centres. These utilities
construct ghost-cell values for the normal velocity boundary condition
and reconstruct buoyancy traces on the physical boundary faces.

Governing boundary conditions:
    n · u = 0
    n × grad(theta) = 0

These functions do not advance the solution in time.
"""

import numpy as np


def reconstruct_velocity_ghost_cells(u, v):
    """
    Construct one layer of ghost cells for the velocity components.

    Normal velocity components are reflected oddly across their walls,
    giving zero normal velocity at the wall under linear reconstruction.

    Tangential components are copied across the walls (even reflection).
    This is a numerical closure assumption for the derivative stencil,
    not an additional physical boundary condition specified by the model.

    Parameters
    ----------
    u : numpy.ndarray
        x-velocity at cell centres, shape (Ny, Nx).
    v : numpy.ndarray
        y-velocity at cell centres, shape (Ny, Nx).

    Returns
    -------
    u_pad : numpy.ndarray
        Padded x-velocity, shape (Ny + 2, Nx + 2).
    v_pad : numpy.ndarray
        Padded y-velocity, shape (Ny + 2, Nx + 2).
    """
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("u and v must be two-dimensional arrays.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    if min(u.shape) < 2:
        raise ValueError("Each grid dimension must contain at least 2 cells.")

    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Velocity fields must contain only finite values.")

    Ny, Nx = u.shape

    # Initially extend both arrays using edge-value copying.
    # This provides the tangential-component closure.
    u_pad = np.pad(u, 1, mode="edge")
    v_pad = np.pad(v, 1, mode="edge")

    # Odd reflection for the normal x-velocity at left and right walls.
    u_pad[1:-1, 0] = -u[:, 0]
    u_pad[1:-1, -1] = -u[:, -1]

    # Odd reflection for the normal y-velocity at bottom and top walls.
    v_pad[0, 1:-1] = -v[0, :]
    v_pad[-1, 1:-1] = -v[-1, :]

    return u_pad, v_pad


def reconstruct_buoyancy_wall_traces(theta):
    """
    Reconstruct buoyancy traces on the four physical wall faces.

    Each wall's buoyancy trace is approximated as a constant equal
    to the mean of the adjacent interior cell-centred buoyancy values
    along that wall.

    This approximates the prescribed zero tangential derivative
    condition on each wall. The four wall values are reconstructed
    independently; corner compatibility is not imposed here.

    Parameters
    ----------
    theta : numpy.ndarray
        Buoyancy at cell centres, shape (Ny, Nx).

    Returns
    -------
    theta_left : numpy.ndarray
        Buoyancy trace along the left wall, shape (Ny,).
    theta_right : numpy.ndarray
        Buoyancy trace along the right wall, shape (Ny,).
    theta_bottom : numpy.ndarray
        Buoyancy trace along the bottom wall, shape (Nx,).
    theta_top : numpy.ndarray
        Buoyancy trace along the top wall, shape (Nx,).
    """
    if theta.ndim != 2:
        raise ValueError("theta must be a two-dimensional array.")

    if min(theta.shape) < 2:
        raise ValueError("Each grid dimension must contain at least 2 cells.")

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values.")

    if np.any(theta <= 0):
        raise ValueError("theta must be strictly positive.")

    Ny, Nx = theta.shape

    # Each wall trace is reconstructed independently as a constant.
    theta_left = np.full(Ny, np.mean(theta[:, 0]))
    theta_right = np.full(Ny, np.mean(theta[:, -1]))

    theta_bottom = np.full(Nx, np.mean(theta[0, :]))
    theta_top = np.full(Nx, np.mean(theta[-1, :]))

    return theta_left, theta_right, theta_bottom, theta_top
