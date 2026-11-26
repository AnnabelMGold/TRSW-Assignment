import numpy as np
import pytest

from rhs_momentum import compute_momentum_rhs


# -------------------------------------------------------------
# Test helper
# -------------------------------------------------------------


def make_uniform_fields(Ny=5, Nx=5):
    """
    Create uniform fields for testing the momentum RHS.

    Parameters
    ----------
    Ny : int
        Number of grid cells in the y-direction.
    Nx : int
        Number of grid cells in the x-direction.

    Returns
    -------
    D, u, v, theta : numpy.ndarray
        Uniform fields with shape (Ny, Nx).
    """

    D = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    theta = np.ones((Ny, Nx))

    return D, u, v, theta


# -------------------------------------------------------------
# Test 1: stationary uniform fields
# -------------------------------------------------------------


def test_uniform_stationary_fields_give_zero_rhs():
    """Uniform stationary fields should give zero momentum RHS."""

    D, u, v, theta = make_uniform_fields()

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.1, dy=0.1,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert np.allclose(dudt, 0.0)
    assert np.allclose(dvdt, 0.0)


# -------------------------------------------------------------
# Test 2: Coriolis contributions
# -------------------------------------------------------------


def test_coriolis_contributions():
    """Check the signs and values of the Coriolis terms."""

    D, u, v, theta = make_uniform_fields()

    # Set uniform nonzero velocities.
    u[:] = 2.0
    v[:] = 3.0

    epsilon = 0.5
    f = 4.0

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.1, dy=0.1,
        epsilon=epsilon, f=f, F=1.0
    )

    # With uniform D, theta, u and v, the spatial gradients
    # vanish, so only the Coriolis contributions remain.
    expected_dudt = (f / epsilon) * v
    expected_dvdt = -(f / epsilon) * u

    assert np.allclose(dudt, expected_dudt)
    assert np.allclose(dvdt, expected_dvdt)


# -------------------------------------------------------------
# Test 3: x-direction advection
# -------------------------------------------------------------


def test_x_advection_of_u():
    """Check the x-direction advection of horizontal velocity."""

    Ny, Nx = 5, 7
    dx, dy = 0.2, 0.3

    # Construct a field that varies linearly in x.
    x = (np.arange(Nx) + 0.5) * dx
    X = np.tile(x, (Ny, 1))

    D = np.ones((Ny, Nx))
    u = X.copy()
    v = np.zeros_like(u)
    theta = np.ones_like(D)

    # Disable Coriolis and pressure contributions.
    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=dx, dy=dy,
        epsilon=1.0, f=0.0, F=1.0
    )

    # du/dx = 1, du/dy = 0.
    # Thus advection_u = -u.
    assert np.allclose(dudt, -u)

    # v is identically zero, so dvdt is zero.
    assert np.allclose(dvdt, 0.0)


# -------------------------------------------------------------
# Test 4: y-direction advection
# -------------------------------------------------------------


def test_y_advection_of_v():
    """Check the y-direction advection of vertical velocity."""

    Ny, Nx = 7, 5
    dx, dy = 0.2, 0.3

    # Construct a field that varies linearly in y.
    y = (np.arange(Ny) + 0.5) * dy
    Y = np.tile(y[:, None], (1, Nx))

    D = np.ones((Ny, Nx))
    u = np.zeros_like(Y)
    v = Y.copy()
    theta = np.ones_like(D)

    # Disable Coriolis and pressure contributions.
    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=dx, dy=dy,
        epsilon=1.0, f=0.0, F=1.0
    )

    # dv/dy = 1, dv/dx = 0.
    # Thus advection_v = -v.
    assert np.allclose(dvdt, -v)

    # u is identically zero, so dudt is zero.
    assert np.allclose(dudt, 0.0)


# -------------------------------------------------------------
# Test 5: pressure/buoyancy contribution
# -------------------------------------------------------------


def test_pressure_contribution():
    """Check the gamma * grad(D*gamma) contribution."""

    Ny, Nx = 7, 5
    dx, dy = 0.2, 0.3

    D = np.ones((Ny, Nx))
    u = np.zeros_like(D)
    v = np.zeros_like(D)

    # Make theta vary linearly in x while remaining positive.
    x = (np.arange(Nx) + 0.5) * dx
    X = np.tile(x, (Ny, 1))

    theta = 2.0 + X

    epsilon = 2.0
    F = 3.0

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=dx, dy=dy,
        epsilon=epsilon, f=0.0, F=F
    )

    # Independently calculate the expected pressure term.
    gamma = np.sqrt(theta) / F
    S = D * gamma

    _, dS_dx = np.gradient(S, dy, dx, edge_order=2)
    dS_dy, _ = np.gradient(S, dy, dx, edge_order=2)

    expected_dudt = -(gamma / epsilon) * dS_dx
    expected_dvdt = -(gamma / epsilon) * dS_dy

    assert np.allclose(dudt, expected_dudt)
    assert np.allclose(dvdt, expected_dvdt)


# -------------------------------------------------------------
# Test 6: shape and output checks
# -------------------------------------------------------------


def test_rhs_shapes():
    """Check that the momentum RHS arrays match the input shape."""

    D, u, v, theta = make_uniform_fields(Ny=6, Nx=8)

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx=0.1, dy=0.2,
        epsilon=1.0, f=1.0, F=1.0
    )

    assert dudt.shape == D.shape
    assert dvdt.shape == D.shape


# -------------------------------------------------------------
# Test 7: invalid inputs
# -------------------------------------------------------------


def test_nonpositive_buoyancy_is_rejected():
    """Check that nonpositive buoyancy is rejected."""

    D, u, v, theta = make_uniform_fields()

    theta[2, 2] = 0.0

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=0.1, dy=0.1,
            epsilon=1.0, f=1.0, F=1.0
        )


def test_nonpositive_depth_is_rejected():
    """Check that nonpositive depth is rejected."""

    D, u, v, theta = make_uniform_fields()

    D[2, 2] = -1.0

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=0.1, dy=0.1,
            epsilon=1.0, f=1.0, F=1.0
        )


def test_invalid_parameters_are_rejected():
    """Check that invalid scalar parameters are rejected."""

    D, u, v, theta = make_uniform_fields()

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=0.0, dy=0.1,
            epsilon=1.0, f=1.0, F=1.0
        )

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=0.1, dy=0.1,
            epsilon=0.0, f=1.0, F=1.0
        )

    with pytest.raises(ValueError):
        compute_momentum_rhs(
            D, u, v, theta,
            dx=0.1, dy=0.1,
            epsilon=1.0, f=1.0, F=0.0
        )