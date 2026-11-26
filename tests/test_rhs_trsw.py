import numpy as np

from rhs_trsw import compute_trsw_rhs
from rhs_depth import compute_depth_rhs
from rhs_momentum import compute_momentum_rhs
from rhs_buoyancy import compute_buoyancy_rhs


def create_fields():
    Ny, Nx = 5, 6

    D = np.ones((Ny, Nx))
    theta = np.ones((Ny, Nx))

    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))

    return D, u, v, theta


def test_rhs_shapes():
    D, u, v, theta = create_fields()

    dDdt, dudt, dvdt, dthetadt = compute_trsw_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert dDdt.shape == D.shape
    assert dudt.shape == u.shape
    assert dvdt.shape == v.shape
    assert dthetadt.shape == theta.shape


def test_uniform_stationary_state():
    D, u, v, theta = create_fields()

    tendencies = compute_trsw_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    for tendency in tendencies:
        assert np.allclose(tendency, 0.0)


def test_combined_rhs_matches_individual_functions():
    D, u, v, theta = create_fields()

    # Introduce nonuniform fields and interior velocities.
    D[2, 2] = 1.5
    theta[2, 3] = 2.0

    u[:, 1:-1] = 0.1
    v[1:-1, :] = -0.2

    dx = 0.2
    dy = 0.25
    epsilon = 0.5
    f = 1.0
    F = 1.0

    combined = compute_trsw_rhs(
        D, u, v, theta,
        dx, dy, epsilon, f, F
    )

    expected_depth = compute_depth_rhs(
        D, u, v, dx, dy
    )

    expected_u, expected_v = compute_momentum_rhs(
        D, u, v, theta,
        dx, dy, epsilon, f, F
    )

    expected_theta = compute_buoyancy_rhs(
        theta, u, v, dx, dy
    )

    expected = (
        expected_depth,
        expected_u,
        expected_v,
        expected_theta,
    )

    for actual, reference in zip(combined, expected):
        assert np.allclose(actual, reference)


def test_wall_velocity_tendencies():
    D, u, v, theta = create_fields()

    u[:, 1:-1] = 0.5
    v[1:-1, :] = -0.5

    _, dudt, dvdt, _ = compute_trsw_rhs(
        D, u, v, theta,
        dx=0.2, dy=0.25,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert np.allclose(dudt[:, 0], 0.0)
    assert np.allclose(dudt[:, -1], 0.0)

    assert np.allclose(dvdt[0, :], 0.0)
    assert np.allclose(dvdt[-1, :], 0.0)