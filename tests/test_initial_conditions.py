
"""
Tests for staggered initial-condition generation.
"""

import numpy as np
import pytest

from config import ModelConfig
from grid import create_staggered_grid
from initial_conditions import (
    create_initial_conditions,
    evaluate_field,
)


@pytest.fixture
def grid():
    """Provide a small staggered grid for testing."""
    config = ModelConfig(Lx=2.0, Ly=1.0, Nx=6, Ny=4)
    return create_staggered_grid(config)


def test_uniform_initial_conditions(grid):
    """Constant initial fields should be evaluated correctly."""
    ic = create_initial_conditions(
        grid,
        D0=2.0,
        u0=3.0,
        v0=4.0,
        theta0=5.0,
    )

    np.testing.assert_allclose(ic.D, 2.0)
    np.testing.assert_allclose(ic.theta, 5.0)

    # Boundary-normal velocities are zeroed.
    np.testing.assert_allclose(ic.u[:, 0], 0.0)
    np.testing.assert_allclose(ic.u[:, -1], 0.0)
    np.testing.assert_allclose(ic.v[0, :], 0.0)
    np.testing.assert_allclose(ic.v[-1, :], 0.0)

    # Interior face velocities retain their prescribed values.
    np.testing.assert_allclose(ic.u[:, 1:-1], 3.0)
    np.testing.assert_allclose(ic.v[1:-1, :], 4.0)


def test_staggered_field_shapes(grid):
    """Fields should have the correct staggered array dimensions."""
    ic = create_initial_conditions(grid)

    Ny, Nx = grid["Xc"].shape

    assert ic.D.shape == (Ny, Nx)
    assert ic.theta.shape == (Ny, Nx)
    assert ic.u.shape == (Ny, Nx + 1)
    assert ic.v.shape == (Ny + 1, Nx)


def test_callable_scalar_fields(grid):
    """Scalar initial fields should use cell-centre coordinates."""
    ic = create_initial_conditions(
        grid,
        D0=lambda X, Y: 1.0 + X,
        theta0=lambda X, Y: 2.0 + Y,
    )

    np.testing.assert_allclose(ic.D, 1.0 + grid["Xc"])
    np.testing.assert_allclose(ic.theta, 2.0 + grid["Yc"])


def test_callable_velocity_fields(grid):
    """Velocity fields should be evaluated on their respective face meshes."""
    ic = create_initial_conditions(
        grid,
        u0=lambda X, Y: 1.0 + X + Y,
        v0=lambda X, Y: 2.0 + X - Y,
    )

    expected_u = 1.0 + grid["Xu"] + grid["Yu"]
    expected_v = 2.0 + grid["Xv"] - grid["Yv"]

    # Apply the same wall constraints to the expected fields.
    expected_u[:, 0] = 0.0
    expected_u[:, -1] = 0.0
    expected_v[0, :] = 0.0
    expected_v[-1, :] = 0.0

    np.testing.assert_allclose(ic.u, expected_u)
    np.testing.assert_allclose(ic.v, expected_v)


@pytest.mark.parametrize(
    "D0,theta0",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_nonpositive_scalar_fields_rejected(grid, D0, theta0):
    """Nonpositive depth or buoyancy should be rejected."""
    with pytest.raises(ValueError):
        create_initial_conditions(grid, D0=D0, theta0=theta0)


def test_wrong_field_shape_rejected(grid):
    """Callable fields with incorrect shapes should raise ValueError."""
    with pytest.raises(ValueError):
        create_initial_conditions(
            grid,
            D0=lambda X, Y: np.ones((3, 3)),
        )


def test_nonfinite_field_rejected(grid):
    """Nonfinite initial values should raise ValueError."""
    with pytest.raises(ValueError):
        create_initial_conditions(grid, theta0=np.nan)


def test_evaluate_field_scalar():
    """A constant field should expand to the coordinate shape."""
    X = np.ones((3, 4))
    Y = np.ones((3, 4))

    result = evaluate_field(2.5, X, Y, "test_field")

    assert result.shape == X.shape
    np.testing.assert_allclose(result, 2.5)


def test_evaluate_field_callable():
    """Callable fields should return the correct evaluated array."""
    X, Y = np.meshgrid(
        np.arange(4),
        np.arange(3),
        indexing="xy",
    )

    result = evaluate_field(
        lambda X, Y: X + 2.0 * Y,
        X,
        Y,
        "test_field",
    )

    np.testing.assert_allclose(result, X + 2.0 * Y)
