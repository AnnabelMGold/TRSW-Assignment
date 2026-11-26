
"""
Automated tests for operators.py.

These tests verify the numerical derivatives of known functions
and check that invalid inputs are rejected.
"""

import numpy as np
import pytest

from config import ModelConfig
from version_histories.old_grid import create_grid
from operators import derivative_x, derivative_y, gradient


@pytest.fixture
def quadratic_grid():
    """Create a grid and a quadratic function with known derivatives."""
    config = ModelConfig(
        Lx=2.0,
        Ly=3.0,
        Nx=30,
        Ny=40,
    )

    _, _, X, Y, dx, dy = create_grid(config)

    q = X**2 + 3.0 * Y**2

    return X, Y, q, dx, dy


def test_derivative_x(quadratic_grid):
    """Check the x derivative of q = X^2 + 3Y^2."""
    X, Y, q, dx, dy = quadratic_grid

    dq_dx = derivative_x(q, dx)

    assert dq_dx.shape == q.shape
    assert np.allclose(dq_dx, 2.0 * X)


def test_derivative_y(quadratic_grid):
    """Check the y derivative of q = X^2 + 3Y^2."""
    X, Y, q, dx, dy = quadratic_grid

    dq_dy = derivative_y(q, dy)

    assert dq_dy.shape == q.shape
    assert np.allclose(dq_dy, 6.0 * Y)


def test_gradient(quadratic_grid):
    """Check that gradient returns both correct derivatives."""
    X, Y, q, dx, dy = quadratic_grid

    dq_dx, dq_dy = gradient(q, dx, dy)

    assert dq_dx.shape == q.shape
    assert dq_dy.shape == q.shape

    assert np.allclose(dq_dx, 2.0 * X)
    assert np.allclose(dq_dy, 6.0 * Y)


def test_derivative_constant_field(quadratic_grid):
    """Check that the derivatives of a constant field are zero."""
    X, Y, q, dx, dy = quadratic_grid

    constant = np.ones_like(q) * 5.0

    dq_dx = derivative_x(constant, dx)
    dq_dy = derivative_y(constant, dy)

    assert np.allclose(dq_dx, 0.0)
    assert np.allclose(dq_dy, 0.0)


def test_invalid_spacing(quadratic_grid):
    """Check that non-positive grid spacing is rejected."""
    X, Y, q, dx, dy = quadratic_grid

    with pytest.raises(ValueError):
        derivative_x(q, 0.0)

    with pytest.raises(ValueError):
        derivative_y(q, -1.0)


def test_invalid_array_dimensions():
    """Check that one-dimensional arrays are rejected."""
    q = np.ones(10)

    with pytest.raises(ValueError):
        derivative_x(q, 0.1)

    with pytest.raises(ValueError):
        derivative_y(q, 0.1)

def test_array_too_small():
    """Check that arrays too small in the relevant direction are rejected."""

    # Only 2 columns: insufficient for the x derivative.
    q_x_too_small = np.ones((5, 2))

    with pytest.raises(ValueError):
        derivative_x(q_x_too_small, 0.1)

    # Only 2 rows: insufficient for the y derivative.
    q_y_too_small = np.ones((2, 5))

    with pytest.raises(ValueError):
        derivative_y(q_y_too_small, 0.1)

