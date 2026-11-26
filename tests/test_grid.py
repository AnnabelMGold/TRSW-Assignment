
"""
Automated tests for grid.py.

These tests check the shape, spacing and coordinates of
the computational grid.
"""

import numpy as np

from config import ModelConfig
from grid import create_grid


def test_grid_shape():
    """Check that the grid arrays have shape (Ny, Nx)."""
    config = ModelConfig(Nx=20, Ny=15)

    x, y, X, Y, dx, dy = create_grid(config)

    assert X.shape == (15, 20)
    assert Y.shape == (15, 20)


def test_grid_spacing():
    """Check that grid spacing is correctly calculated."""
    config = ModelConfig(Lx=2.0, Ly=3.0, Nx=20, Ny=30)

    x, y, X, Y, dx, dy = create_grid(config)

    assert dx == 0.1
    assert dy == 0.1


def test_grid_coordinates():
    """Check that coordinates correspond to cell centres."""
    config = ModelConfig(Lx=2.0, Ly=1.0, Nx=4, Ny=2)

    x, y, X, Y, dx, dy = create_grid(config)

    expected_x = np.array([0.25, 0.75, 1.25, 1.75])
    expected_y = np.array([0.25, 0.75])

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)


def test_grid_mesh_orientation():
    """Check that the mesh uses the xy array convention."""
    config = ModelConfig(Nx=5, Ny=3)

    x, y, X, Y, dx, dy = create_grid(config)

    assert np.allclose(X[0, :], x)
    assert np.allclose(Y[:, 0], y)
