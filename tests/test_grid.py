
"""
Tests for the staggered TRSW grid.
"""

import numpy as np

from config import ModelConfig
from grid import create_staggered_grid


def test_staggered_grid_shapes():
    """Grid coordinate arrays should have the correct shapes."""
    config = ModelConfig(Nx=7, Ny=5)

    grid = create_staggered_grid(config)

    assert grid["Xc"].shape == (5, 7)
    assert grid["Yc"].shape == (5, 7)

    assert grid["Xu"].shape == (5, 8)
    assert grid["Yu"].shape == (5, 8)

    assert grid["Xv"].shape == (6, 7)
    assert grid["Yv"].shape == (6, 7)


def test_grid_spacing():
    """Grid spacings should match domain lengths and grid sizes."""
    config = ModelConfig(
        Lx=2.0,
        Ly=3.0,
        Nx=10,
        Ny=15,
    )

    grid = create_staggered_grid(config)

    assert np.isclose(grid["dx"], 0.2)
    assert np.isclose(grid["dy"], 0.2)


def test_cell_centres():
    """Scalar grid points should be located at cell centres."""
    config = ModelConfig(
        Lx=2.0,
        Ly=1.0,
        Nx=4,
        Ny=2,
    )

    grid = create_staggered_grid(config)

    expected_x = np.array([0.25, 0.75, 1.25, 1.75])
    expected_y = np.array([0.25, 0.75])

    np.testing.assert_allclose(grid["xc"], expected_x)
    np.testing.assert_allclose(grid["yc"], expected_y)


def test_u_faces_include_domain_boundaries():
    """Vertical velocity faces should include both x-domain boundaries."""
    config = ModelConfig(Lx=2.0, Ly=1.0, Nx=4, Ny=2)

    grid = create_staggered_grid(config)

    np.testing.assert_allclose(
        grid["xf"],
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )


def test_v_faces_include_domain_boundaries():
    """Horizontal velocity faces should include both y-domain boundaries."""
    config = ModelConfig(Lx=2.0, Ly=1.0, Nx=4, Ny=2)

    grid = create_staggered_grid(config)

    np.testing.assert_allclose(
        grid["yf"],
        np.array([0.0, 0.5, 1.0]),
    )


def test_coordinate_meshes():
    """Coordinate mesh arrays should correspond to their 1D coordinates."""
    config = ModelConfig(Lx=2.0, Ly=1.0, Nx=4, Ny=2)

    grid = create_staggered_grid(config)

    np.testing.assert_allclose(grid["Xc"][0, :], grid["xc"])
    np.testing.assert_allclose(grid["Yc"][:, 0], grid["yc"])

    np.testing.assert_allclose(grid["Xu"][0, :], grid["xf"])
    np.testing.assert_allclose(grid["Yu"][:, 0], grid["yf_u"])

    np.testing.assert_allclose(grid["Xv"][0, :], grid["xf_v"])
    np.testing.assert_allclose(grid["Yv"][:, 0], grid["yf"])
