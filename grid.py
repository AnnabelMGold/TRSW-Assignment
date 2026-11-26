
"""
Grid module for the Thermal Rotating Shallow Water (TRSW) model.

This module constructs a uniform, cell-centred Cartesian grid
over a rectangular computational domain.
"""

import numpy as np
from config import ModelConfig


def create_grid(config: ModelConfig):
    """
    Construct a uniform, cell-centred Cartesian grid.

    Inputs:
        config (ModelConfig): Configuration object containing the domain
            dimensions (Lx, Ly) and grid resolution (Nx, Ny).

    Outputs:
        x (numpy.ndarray): 1D array of x-coordinate cell centres.
            Shape: (Nx,).
        y (numpy.ndarray): 1D array of y-coordinate cell centres.
            Shape: (Ny,).
        X (numpy.ndarray): 2D array of x-coordinates across the grid.
            Shape: (Ny, Nx).
        Y (numpy.ndarray): 2D array of y-coordinates across the grid.
            Shape: (Ny, Nx).
        dx (float): Uniform grid spacing in the x-direction.
        dy (float): Uniform grid spacing in the y-direction.

    Description:
        The grid points are located at the centres of rectangular cells.
        The physical domain boundaries are located at the cell faces.

        The coordinate arrays X and Y are constructed using np.meshgrid
        with indexing='xy', so that the first array dimension corresponds
        to the y-direction and the second array dimension corresponds
        to the x-direction.
    """

    # Calculate uniform grid spacing
    dx = config.Lx / config.Nx
    dy = config.Ly / config.Ny

    # Generate cell-centre coordinates in each direction
    x = (np.arange(config.Nx) + 0.5) * dx
    y = (np.arange(config.Ny) + 0.5) * dy

    # Construct 2D coordinate arrays
    X, Y = np.meshgrid(x, y, indexing="xy")

    return x, y, X, Y, dx, dy
