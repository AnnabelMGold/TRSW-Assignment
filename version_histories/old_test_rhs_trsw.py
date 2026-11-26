
"""
Tests for the complete TRSW right-hand-side module.
"""

import numpy as np

from rhs_trsw import compute_trsw_rhs


def test_uniform_stationary_state():
    """
    A spatially uniform state with zero velocity should
    have zero tendencies in all four equations.
    """

    # Create a uniform grid of fields
    Ny, Nx = 5, 6

    D = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    theta = np.ones((Ny, Nx))

    # Numerical parameters
    dx = 0.2
    dy = 0.25
    epsilon = 1.0
    f = 1.0
    F = 1.0

    # Calculate all four tendencies
    dDdt, dudt, dvdt, dthetadt = compute_trsw_rhs(
        D, u, v, theta, dx, dy, epsilon, f, F
    )

    # All tendencies should be zero
    assert np.allclose(dDdt, 0.0)
    assert np.allclose(dudt, 0.0)
    assert np.allclose(dvdt, 0.0)
    assert np.allclose(dthetadt, 0.0)


def test_output_shapes():
    """
    Check that all four returned tendencies have the
    same shape as the input fields.
    """

    Ny, Nx = 6, 8

    D = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    theta = np.ones((Ny, Nx))

    dx = 0.1
    dy = 0.2
    epsilon = 1.0
    f = 1.0
    F = 1.0

    results = compute_trsw_rhs(
        D, u, v, theta, dx, dy, epsilon, f, F
    )

    # Check that exactly four arrays are returned
    assert len(results) == 4

    # Check the shape of each returned array
    for result in results:
        assert result.shape == (Ny, Nx)


def test_combined_rhs_matches_individual_modules():
    """
    Check that the combined RHS returns the same results
    as calling each individual equation module separately.
    """

    from rhs_depth import compute_depth_rhs
    from rhs_momentum import compute_momentum_rhs
    from rhs_buoyancy import compute_buoyancy_rhs

    Ny, Nx = 6, 7

    D = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    theta = np.ones((Ny, Nx))

    dx = 0.2
    dy = 0.3
    epsilon = 1.0
    f = 0.5
    F = 1.0

    # Calculate tendencies individually
    expected_D = compute_depth_rhs(D, u, v, dx, dy)

    expected_u, expected_v = compute_momentum_rhs(
        D, u, v, theta, dx, dy, epsilon, f, F
    )

    expected_theta = compute_buoyancy_rhs(
        theta, u, v, dx, dy
    )

    # Calculate tendencies using the combined module
    actual = compute_trsw_rhs(
        D, u, v, theta, dx, dy, epsilon, f, F
    )

    # Compare all four results
    expected = (expected_D, expected_u, expected_v, expected_theta)

    for actual_field, expected_field in zip(actual, expected):
        assert np.allclose(actual_field, expected_field)
