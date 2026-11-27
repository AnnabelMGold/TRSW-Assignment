import numpy as np
import pytest

from rhs_momentum import compute_momentum_rhs


@pytest.fixture
def fields():
    Ny, Nx = 5, 6

    D = np.ones((Ny, Nx))
    theta = np.ones((Ny+1, Nx+1))

    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    return D, u, v, theta


def test_rhs_shapes(fields):
    D, u, v, theta = fields

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert dudt.shape == u.shape
    assert dvdt.shape == v.shape


def test_uniform_state_zero_velocity(fields):
    D, u, v, theta = fields

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert np.allclose(dudt, 0.0)
    assert np.allclose(dvdt, 0.0)


def test_wall_tendencies_are_zero(fields):
    D, u, v, theta = fields

    # Set nonzero interior velocities.
    u[:, 1:-1] = 1.0
    v[1:-1, :] = -0.5

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert np.allclose(dudt[:, 0], 0.0)
    assert np.allclose(dudt[:, -1], 0.0)

    assert np.allclose(dvdt[0, :], 0.0)
    assert np.allclose(dvdt[-1, :], 0.0)


def test_pressure_gradient_x():
    Ny, Nx = 5, 6

    D = np.tile(np.arange(Nx, dtype=float), (Ny, 1)) + 2.0
    theta = np.ones((Ny+1, Nx+1))

    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    dx = 0.5
    dy = 0.5
    epsilon = 2.0
    F = 1.0

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=dx, dy=dy,
        epsilon=epsilon, f=0.0, F=F
    )

    # S = D*gamma, gamma = 1.
    # Interior dS/dx = 1/dx.
    expected = -(1.0 / epsilon) * (1.0 / dx)

    assert np.allclose(dudt[:, 1:-1], expected)
    assert np.allclose(dudt[:, 0], 0.0)
    assert np.allclose(dudt[:, -1], 0.0)

    # No y-gradient.
    assert np.allclose(dvdt, 0.0)


def test_invalid_velocity_shapes():
    D = np.ones((4, 5))
    theta = np.ones((5, 6))

    u = np.zeros((4, 5))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=1.0, dy=1.0,
            epsilon=1.0, f=1.0, F=1.0
        )


def test_nonpositive_theta():
    D = np.ones((4, 5))
    theta = np.ones((5, 6))
    theta[0, 0] = 0.0

    u = np.zeros((4, 6))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=1.0, dy=1.0,
            epsilon=1.0, f=1.0, F=1.0
        )


def test_invalid_epsilon():
    D = np.ones((4, 5))
    theta = np.ones((5, 6))

    u = np.zeros((4, 6))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=1.0, dy=1.0,
            epsilon=0.0, f=1.0, F=1.0
        )
        