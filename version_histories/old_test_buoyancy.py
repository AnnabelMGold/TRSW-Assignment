
import numpy as np
import pytest

from rhs_buoyancy import compute_buoyancy_rhs


def test_uniform_buoyancy_has_zero_rhs():
    """Uniform buoyancy has no spatial gradient, so its RHS is zero."""

    theta = np.ones((5, 6)) * 2.0
    u = np.ones((5, 6)) * 1.0
    v = np.ones((5, 6)) * 0.5

    result = compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)

    assert np.allclose(result, 0.0)


def test_x_direction_buoyancy_transport():
    """Check transport of a linear buoyancy profile in x."""

    x = np.arange(6) * 0.2
    y = np.arange(5) * 0.3
    X, Y = np.meshgrid(x, y)

    theta = 2.0 + 3.0 * X
    u = np.ones_like(theta) * 2.0
    v = np.zeros_like(theta)

    result = compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)

    # dtheta/dx = 3, so dtheta/dt = -2*3 = -6
    assert np.allclose(result, -6.0)


def test_y_direction_buoyancy_transport():
    """Check transport of a linear buoyancy profile in y."""

    x = np.arange(6) * 0.2
    y = np.arange(5) * 0.3
    X, Y = np.meshgrid(x, y)

    theta = 2.0 + 4.0 * Y
    u = np.zeros_like(theta)
    v = np.ones_like(theta) * 0.5

    result = compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)

    # dtheta/dy = 4, so dtheta/dt = -0.5*4 = -2
    assert np.allclose(result, -2.0)


def test_buoyancy_rhs_shape():
    """The output must have the same shape as the input fields."""

    theta = np.ones((5, 6))
    u = np.ones((5, 6))
    v = np.ones((5, 6))

    result = compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)

    assert result.shape == theta.shape


def test_nonpositive_buoyancy_raises_error():
    """Buoyancy must be strictly positive."""

    theta = np.ones((5, 6))
    theta[2, 3] = 0.0

    u = np.ones((5, 6))
    v = np.ones((5, 6))

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)


def test_inconsistent_field_shapes_raise_error():
    """All fields must have matching shapes."""

    theta = np.ones((5, 6))
    u = np.ones((5, 6))
    v = np.ones((5, 5))

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(theta, u, v, 0.2, 0.3)


def test_invalid_grid_spacing_raises_error():
    """Grid spacing must be positive."""

    theta = np.ones((5, 6))
    u = np.ones((5, 6))
    v = np.ones((5, 6))

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(theta, u, v, 0.0, 0.3)
