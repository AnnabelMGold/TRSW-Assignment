import numpy as np
import pytest

from rhs_buoyancy import compute_buoyancy_rhs


@pytest.fixture
def fields():
    Ny, Nx = 5, 6

    theta = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    return theta, u, v


def test_rhs_shape(fields):
    theta, u, v = fields

    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx=0.2, dy=0.25
    )

    assert dthetadt.shape == theta.shape


def test_uniform_buoyancy_and_zero_velocity(fields):
    theta, u, v = fields

    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx=0.2, dy=0.25
    )

    assert np.allclose(dthetadt, 0.0)


def test_uniform_buoyancy_with_velocity(fields):
    theta, u, v = fields

    u[:, 1:-1] = 1.0
    v[1:-1, :] = -0.5

    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx=0.2, dy=0.25
    )

    # A uniform buoyancy field has zero gradient.
    assert np.allclose(dthetadt, 0.0)


def test_x_advection():
    Ny, Nx = 5, 6

    dx = 0.5
    dy = 0.5

    x = (np.arange(Nx) + 0.5) * dx
    theta = np.tile(2.0 + 3.0 * x, (Ny, 1))

    u = np.full((Ny, Nx + 1), 2.0)
    v = np.zeros((Ny + 1, Nx))

    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx, dy
    )

    # dtheta/dx = 3 and u = 2.
    expected = -6.0

    assert np.allclose(dthetadt, expected)


def test_y_advection():
    Ny, Nx = 5, 6

    dx = 0.5
    dy = 0.5

    y = (np.arange(Ny) + 0.5) * dy
    theta = np.tile((2.0 + 4.0 * y)[:, None], (1, Nx))

    u = np.zeros((Ny, Nx + 1))
    v = np.full((Ny + 1, Nx), 1.5)

    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx, dy
    )

    # dtheta/dy = 4 and v = 1.5.
    expected = -6.0

    assert np.allclose(dthetadt, expected)


def test_invalid_velocity_shapes():
    theta = np.ones((4, 5))
    u = np.zeros((4, 5))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(
            theta, u, v, dx=1.0, dy=1.0
        )


def test_invalid_spacing():
    theta = np.ones((4, 5))
    u = np.zeros((4, 6))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(
            theta, u, v, dx=0.0, dy=1.0
        )

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(
            theta, u, v, dx=1.0, dy=-1.0
        )


def test_nonfinite_values():
    theta = np.ones((4, 5))
    u = np.zeros((4, 6))
    v = np.zeros((5, 5))

    theta[1, 1] = np.nan

    with pytest.raises(ValueError):
        compute_buoyancy_rhs(
            theta, u, v, dx=1.0, dy=1.0
        )