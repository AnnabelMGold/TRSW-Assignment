import numpy as np
import pytest

from fluxes import compute_depth_fluxes
from rhs_depth import compute_depth_rhs


# -------------------------------------------------------------
# Tests for numerical depth fluxes
# -------------------------------------------------------------


def test_flux_shapes():
    """Check that flux arrays have the correct dimensions."""

    Ny, Nx = 5, 7

    D = np.ones((Ny, Nx))
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))

    Qx, Qy = compute_depth_fluxes(D, u, v)

    assert Qx.shape == (Ny, Nx + 1)
    assert Qy.shape == (Ny + 1, Nx)


def test_zero_velocity_gives_zero_flux():
    """Check that zero velocities produce zero fluxes."""

    D = np.ones((5, 7))
    u = np.zeros((5, 7))
    v = np.zeros((5, 7))

    Qx, Qy = compute_depth_fluxes(D, u, v)

    assert np.allclose(Qx, 0.0)
    assert np.allclose(Qy, 0.0)


def test_boundary_fluxes_are_zero():
    """Check that all outer boundary fluxes are zero."""

    D = np.ones((5, 7))
    u = np.ones((5, 7))
    v = np.ones((5, 7))

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # Left and right boundary fluxes.
    assert np.allclose(Qx[:, 0], 0.0)
    assert np.allclose(Qx[:, -1], 0.0)

    # Bottom and top boundary fluxes.
    assert np.allclose(Qy[0, :], 0.0)
    assert np.allclose(Qy[-1, :], 0.0)


def test_upwind_flux_x():
    """Check that the x-flux uses the correct upwind depth."""

    D = np.array([
        [1.0, 3.0, 5.0],
        [1.0, 3.0, 5.0],
        [1.0, 3.0, 5.0]
    ])

    # Positive velocity: upwind depth comes from the left.
    u = np.ones_like(D)
    v = np.zeros_like(D)

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # At the face between columns 0 and 1:
    # averaged velocity = 1 and upwind depth = 1.
    assert np.allclose(Qx[:, 1], 1.0)

    # At the face between columns 1 and 2:
    # averaged velocity = 1 and upwind depth = 3.
    assert np.allclose(Qx[:, 2], 3.0)


def test_upwind_flux_y():
    """Check that the y-flux uses the correct upwind depth."""

    D = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [5.0, 5.0, 5.0]
    ])

    # Positive velocity: upwind depth comes from the lower row.
    u = np.zeros_like(D)
    v = np.ones_like(D)

    Qx, Qy = compute_depth_fluxes(D, u, v)

    # At the face between rows 0 and 1:
    # averaged velocity = 1 and upwind depth = 1.
    assert np.allclose(Qy[1, :], 1.0)

    # At the face between rows 1 and 2:
    # averaged velocity = 1 and upwind depth = 3.
    assert np.allclose(Qy[2, :], 3.0)


# -------------------------------------------------------------
# Tests for the discrete depth RHS
# -------------------------------------------------------------


def test_zero_velocity_gives_zero_rhs():
    """Check that stationary fluid produces no depth evolution."""

    D = np.ones((5, 7))
    u = np.zeros((5, 7))
    v = np.zeros((5, 7))

    dDdt = compute_depth_rhs(D, u, v, dx=0.1, dy=0.1)

    assert np.allclose(dDdt, 0.0)


def test_mass_conservation():
    """
    Check that the spatial depth RHS conserves total mass.

    The sum of the depth RHS multiplied by cell area
    should be zero when all outer boundary fluxes vanish.
    """

    Ny, Nx = 8, 10

    # Create a nonuniform positive depth field.
    D = np.ones((Ny, Nx))
    D[3:5, 4:6] = 2.0

    # Create arbitrary velocity fields.
    rng = np.random.default_rng(42)

    u = rng.normal(size=(Ny, Nx))
    v = rng.normal(size=(Ny, Nx))

    dx = 0.2
    dy = 0.3

    dDdt = compute_depth_rhs(D, u, v, dx, dy)

    # Total mass change per unit time.
    mass_rate = np.sum(dDdt) * dx * dy

    # Internal fluxes cancel and outer fluxes are zero.
    assert mass_rate == pytest.approx(0.0, abs=1e-12)


def test_rhs_shape():
    """Check that the depth RHS has the same shape as D."""

    D = np.ones((5, 7))
    u = np.zeros((5, 7))
    v = np.zeros((5, 7))

    dDdt = compute_depth_rhs(D, u, v, dx=0.1, dy=0.2)

    assert dDdt.shape == D.shape


# -------------------------------------------------------------
# Tests for invalid input
# -------------------------------------------------------------


def test_invalid_grid_spacing():
    """Check that invalid grid spacings are rejected."""

    D = np.ones((5, 5))
    u = np.zeros((5, 5))
    v = np.zeros((5, 5))

    with pytest.raises(ValueError):
        compute_depth_rhs(D, u, v, dx=0.0, dy=0.1)

    with pytest.raises(ValueError):
        compute_depth_rhs(D, u, v, dx=0.1, dy=-1.0)


def test_nonpositive_depth_is_rejected():
    """Check that zero and negative depths are rejected."""

    D = np.ones((5, 5))
    u = np.zeros((5, 5))
    v = np.zeros((5, 5))

    D[2, 2] = 0.0

    with pytest.raises(ValueError):
        compute_depth_rhs(D, u, v, dx=0.1, dy=0.1)