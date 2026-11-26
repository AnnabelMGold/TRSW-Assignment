
"""
Automated tests for initial_conditions.py.

These tests check uniform and spatially varying initial conditions,
as well as invalid fields.
"""

import numpy as np
import pytest

from config import ModelConfig
from version_histories.old_grid import create_grid
from version_histories.old_initial_conditions import create_initial_conditions, evaluate_field


@pytest.fixture
def mesh():
    """Create a small grid for testing initial conditions."""
    config = ModelConfig(Nx=10, Ny=8)

    _, _, X, Y, dx, dy = create_grid(config)

    return X, Y


def test_uniform_initial_conditions(mesh):
    """Check that constant initial conditions are correctly assigned."""
    X, Y = mesh

    initial = create_initial_conditions(
        X, Y,
        D0=2.0,
        u0=0.5,
        v0=-0.3,
        theta0=4.0,
    )

    assert np.all(initial.D == 2.0)
    assert np.all(initial.u == 0.5)
    assert np.all(initial.v == -0.3)
    assert np.all(initial.theta == 4.0)


def test_function_initial_conditions(mesh):
    """Check that spatially varying initial conditions work."""
    X, Y = mesh

    initial = create_initial_conditions(
        X, Y,
        D0=lambda X, Y: 1.0 + 0.1 * X,
        u0=lambda X, Y: X + Y,
        v0=lambda X, Y: X - Y,
        theta0=lambda X, Y: 2.0 + 0.2 * Y,
    )

    assert np.allclose(initial.D, 1.0 + 0.1 * X)
    assert np.allclose(initial.u, X + Y)
    assert np.allclose(initial.v, X - Y)
    assert np.allclose(initial.theta, 2.0 + 0.2 * Y)


def test_evaluate_field_shape(mesh):
    """Check that evaluated fields have the grid shape."""
    X, Y = mesh

    field = evaluate_field(
        lambda X, Y: X + Y,
        X, Y,
        field_name="test_field",
    )

    assert field.shape == X.shape


def test_nonpositive_depth_rejected(mesh):
    """Check that zero or negative depth is rejected."""
    X, Y = mesh

    with pytest.raises(ValueError):
        create_initial_conditions(X, Y, D0=0.0)

    with pytest.raises(ValueError):
        create_initial_conditions(X, Y, D0=-1.0)


def test_nonpositive_buoyancy_rejected(mesh):
    """Check that zero or negative buoyancy is rejected."""
    X, Y = mesh

    with pytest.raises(ValueError):
        create_initial_conditions(X, Y, theta0=0.0)

    with pytest.raises(ValueError):
        create_initial_conditions(X, Y, theta0=-1.0)


def test_wrong_field_shape_rejected(mesh):
    """Check that fields with an incompatible shape are rejected."""
    X, Y = mesh

    with pytest.raises(ValueError):
        evaluate_field(
            np.ones((3, 3)),
            X, Y,
            field_name="test_field",
        )


def test_nonfinite_field_rejected(mesh):
    """Check that fields containing NaN or infinity are rejected."""
    X, Y = mesh

    with pytest.raises(ValueError):
        evaluate_field(
            np.nan,
            X, Y,
            field_name="test_field",
        )
