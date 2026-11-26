import numpy as np
import pytest

from fluxes import compute_depth_fluxes


@pytest.fixture
def fields():
    Ny, Nx = 4, 6

    D = np.full((Ny, Nx), 2.0)
    u = np.full((Ny, Nx + 1), 3.0)
    v = np.full((Ny + 1, Nx), 4.0)

    # Enforce zero normal velocity at the boundaries.
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    return D, u, v


def test_flux_shapes(fields):
    D, u, v = fields

    Qx, Qy = compute_depth_fluxes(D, u, v)

    Ny, Nx = D.shape

    assert Qx.shape == (Ny, Nx + 1)
    assert Qy.shape == (Ny + 1, Nx)


def test_uniform_depth_and_velocity(fields):
    D, u, v = fields

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # Uniform depth multiplied by face velocity.
    assert np.allclose(Qx, D[0, 0] * u)
    assert np.allclose(Qy, D[0, 0] * v)


def test_zero_boundary_fluxes(fields):
    D, u, v = fields

    Qx, Qy = compute_depth_fluxes(D, u, v)

    assert np.allclose(Qx[:, 0], 0.0)
    assert np.allclose(Qx[:, -1], 0.0)

    assert np.allclose(Qy[0, :], 0.0)
    assert np.allclose(Qy[-1, :], 0.0)


def test_arithmetic_depth_reconstruction():
    D = np.array([
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ])

    u = np.ones((2, 4))
    v = np.ones((3, 3))

    # Enforce zero normal velocity at the walls.
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    v[0, :] = 0.0
    v[-1, :] = 0.0

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # Arithmetic averages of neighbouring cell depths.
    expected_x = np.array([
        [2.0, 4.0],
        [3.0, 5.0],
    ])

    expected_y = np.array([
        [1.5, 3.5, 5.5],
    ])

    assert np.allclose(Qx[:, 1:-1], expected_x)
    assert np.allclose(Qy[1:-1, :], expected_y)


def test_invalid_depth():
    D = np.ones((3, 4))
    D[0, 0] = 0.0

    u = np.zeros((3, 5))
    v = np.zeros((4, 4))

    with pytest.raises(ValueError):
        compute_depth_fluxes(D, u, v)


def test_incorrect_velocity_shapes():
    D = np.ones((3, 4))
    u = np.zeros((3, 4))
    v = np.zeros((4, 4))

    with pytest.raises(ValueError):
        compute_depth_fluxes(D, u, v)


def test_nonfinite_values():
    D = np.ones((3, 4))
    u = np.zeros((3, 5))
    v = np.zeros((4, 4))

    u[1, 1] = np.nan

    with pytest.raises(ValueError):
        compute_depth_fluxes(D, u, v)