
"""
Tests for the TRSW boundary-condition reconstruction functions.
"""

import numpy as np
import pytest

from boundary_conditions import (
    reconstruct_velocity_ghost_cells,
    reconstruct_buoyancy_wall_traces,
)


def test_velocity_ghost_cell_shapes():
    """Padded velocity arrays should have one ghost layer."""
    u = np.ones((5, 6))
    v = np.ones((5, 6))

    u_pad, v_pad = reconstruct_velocity_ghost_cells(u, v)

    assert u_pad.shape == (7, 8)
    assert v_pad.shape == (7, 8)


def test_normal_velocity_odd_reflection():
    """Normal velocity ghost cells should be odd reflections."""
    u = np.ones((4, 5)) * 2.0
    v = np.ones((4, 5)) * 3.0

    u_pad, v_pad = reconstruct_velocity_ghost_cells(u, v)

    # Left and right x-velocity ghost cells.
    np.testing.assert_allclose(u_pad[1:-1, 0], -u[:, 0])
    np.testing.assert_allclose(u_pad[1:-1, -1], -u[:, -1])

    # Bottom and top y-velocity ghost cells.
    np.testing.assert_allclose(v_pad[0, 1:-1], -v[0, :])
    np.testing.assert_allclose(v_pad[-1, 1:-1], -v[-1, :])


def test_wall_normal_velocity_reconstruction_is_zero():
    """Linear reconstruction should give zero normal velocity at walls."""
    u = np.random.default_rng(1).random((4, 5))
    v = np.random.default_rng(2).random((4, 5))

    u_pad, v_pad = reconstruct_velocity_ghost_cells(u, v)

    # Average the interior and ghost-cell values at the wall.
    u_left_wall = 0.5 * (u_pad[1:-1, 0] + u_pad[1:-1, 1])
    u_right_wall = 0.5 * (u_pad[1:-1, -1] + u_pad[1:-1, -2])

    v_bottom_wall = 0.5 * (v_pad[0, 1:-1] + v_pad[1, 1:-1])
    v_top_wall = 0.5 * (v_pad[-1, 1:-1] + v_pad[-2, 1:-1])

    np.testing.assert_allclose(u_left_wall, 0.0)
    np.testing.assert_allclose(u_right_wall, 0.0)
    np.testing.assert_allclose(v_bottom_wall, 0.0)
    np.testing.assert_allclose(v_top_wall, 0.0)


def test_tangential_velocity_copying():
    """Tangential ghost cells should be copied by edge extension."""
    u = np.arange(20, dtype=float).reshape(4, 5)
    v = np.arange(20, dtype=float).reshape(4, 5)

    u_pad, v_pad = reconstruct_velocity_ghost_cells(u, v)

    # Tangential u at bottom and top walls.
    np.testing.assert_allclose(u_pad[0, 1:-1], u[0, :])
    np.testing.assert_allclose(u_pad[-1, 1:-1], u[-1, :])

    # Tangential v at left and right walls.
    np.testing.assert_allclose(v_pad[1:-1, 0], v[:, 0])
    np.testing.assert_allclose(v_pad[1:-1, -1], v[:, -1])


def test_buoyancy_wall_trace_shapes():
    """Buoyancy wall traces should have the appropriate dimensions."""
    theta = np.ones((5, 6))

    left, right, bottom, top = (
        reconstruct_buoyancy_wall_traces(theta)
    )

    assert left.shape == (5,)
    assert right.shape == (5,)
    assert bottom.shape == (6,)
    assert top.shape == (6,)


def test_buoyancy_wall_traces_are_constant():
    """The reconstructed buoyancy trace should be constant per wall."""
    theta = np.arange(1, 31, dtype=float).reshape(5, 6)

    left, right, bottom, top = (
        reconstruct_buoyancy_wall_traces(theta)
    )

    np.testing.assert_allclose(left, np.mean(theta[:, 0]))
    np.testing.assert_allclose(right, np.mean(theta[:, -1]))
    np.testing.assert_allclose(bottom, np.mean(theta[0, :]))
    np.testing.assert_allclose(top, np.mean(theta[-1, :]))


def test_buoyancy_uniform_state():
    """Uniform buoyancy should remain uniform on all wall traces."""
    theta = np.full((5, 6), 2.5)

    left, right, bottom, top = (
        reconstruct_buoyancy_wall_traces(theta)
    )

    np.testing.assert_allclose(left, 2.5)
    np.testing.assert_allclose(right, 2.5)
    np.testing.assert_allclose(bottom, 2.5)
    np.testing.assert_allclose(top, 2.5)


@pytest.mark.parametrize(
    "bad_theta",
    [
        np.zeros((4, 4)),
        -np.ones((4, 4)),
        np.full((4, 4), np.nan),
    ],
)
def test_buoyancy_invalid_values(bad_theta):
    """Invalid buoyancy fields should raise ValueError."""
    with pytest.raises(ValueError):
        reconstruct_buoyancy_wall_traces(bad_theta)


def test_velocity_invalid_shapes():
    """Velocity fields with inconsistent shapes should raise ValueError."""
    u = np.ones((4, 5))
    v = np.ones((5, 4))

    with pytest.raises(ValueError):
        reconstruct_velocity_ghost_cells(u, v)
