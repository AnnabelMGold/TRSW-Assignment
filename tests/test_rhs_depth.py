import numpy as np
import pytest

from rhs_depth import compute_depth_rhs


@pytest.fixture
def fields():
    Ny, Nx = 4, 6

    D = np.full((Ny, Nx), 2.0)
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    return D, u, v


def test_rhs_shape(fields):
    D, u, v = fields

    dDdt = compute_depth_rhs(D, u, v, dx=0.2, dy=0.25)

    assert dDdt.shape == D.shape


def test_zero_velocity(fields):
    D, u, v = fields

    dDdt = compute_depth_rhs(D, u, v, dx=0.2, dy=0.25)

    assert np.allclose(dDdt, 0.0)


def test_uniform_depth_and_velocity():
    Ny, Nx = 4, 6

    D = np.full((Ny, Nx), 3.0)

    u = np.full((Ny, Nx + 1), 2.0)
    v = np.full((Ny + 1, Nx), -1.0)

    # Impermeable walls.
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    dDdt = compute_depth_rhs(D, u, v, dx=1.0, dy=1.0)

    # Expected divergence contributions from the x-direction.
    expected_x = np.zeros((Ny, Nx))
    expected_x[:, 0] = -6.0
    expected_x[:, -1] = 6.0

    # Expected divergence contributions from the y-direction.
    expected_y = np.zeros((Ny, Nx))
    expected_y[0, :] = 3.0
    expected_y[-1, :] = -3.0

    expected = expected_x + expected_y

    assert np.allclose(dDdt, expected)


def test_mass_conservation():
    Ny, Nx = 5, 8

    D = np.ones((Ny, Nx))

    rng = np.random.default_rng(42)

    u = rng.normal(size=(Ny, Nx + 1))
    v = rng.normal(size=(Ny + 1, Nx))

    # Enforce impermeable boundaries.
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    dx = 0.3
    dy = 0.2

    dDdt = compute_depth_rhs(D, u, v, dx, dy)

    # Integral of the depth tendency over the domain.
    total_change = np.sum(dDdt) * dx * dy

    # Internal fluxes cancel and wall fluxes are zero.
    assert np.isclose(total_change, 0.0, atol=1e-12)


def test_invalid_spacing():
    D = np.ones((3, 4))
    u = np.zeros((3, 5))
    v = np.zeros((4, 4))

    with pytest.raises(ValueError):
        compute_depth_rhs(D, u, v, dx=0.0, dy=1.0)

    with pytest.raises(ValueError):
        compute_depth_rhs(D, u, v, dx=1.0, dy=-1.0)