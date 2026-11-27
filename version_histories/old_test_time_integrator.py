
"""
Tests for the SSP-RK3 time integrator.
"""

import numpy as np
import pytest

from version_histories.old_time_integrator import ssprk3_step


def exponential_rhs(q, rate):
    """
    Test equation:

        dq/dt = rate * q

    Its exact solution is:

        q(t) = q(0) * exp(rate * t)
    """

    return (rate * q,)


def test_ssprk3_exponential_equation():
    """
    Check that SSP-RK3 gives a third-order approximation
    to the solution of a simple exponential equation.
    """

    q0 = np.array([[1.0]])
    rate = -1.0
    dt = 0.1

    new_state = ssprk3_step(
        (q0,), dt, exponential_rhs, rate
    )

    # Third-order Runge-Kutta amplification factor
    expected = 1.0 - dt + dt**2 / 2 - dt**3 / 6

    assert np.allclose(new_state[0], expected)


def test_ssprk3_zero_rhs():
    """
    If every tendency is zero, the state must remain unchanged.
    """

    def zero_rhs(a, b):
        return np.zeros_like(a), np.zeros_like(b)

    a = np.ones((4, 5))
    b = 2.0 * np.ones((4, 5))

    new_state = ssprk3_step(
        (a, b), 0.1, zero_rhs
    )

    assert np.allclose(new_state[0], a)
    assert np.allclose(new_state[1], b)


def test_ssprk3_does_not_modify_original_state():
    """
    Check that the original input arrays are not modified.
    """

    def simple_rhs(q):
        return (np.ones_like(q),)

    q = np.ones((4, 4))
    original = q.copy()

    ssprk3_step((q,), 0.1, simple_rhs)

    assert np.array_equal(q, original)


def test_ssprk3_returns_correct_shapes():
    """
    Check that all updated fields retain their original shapes.
    """

    def zero_rhs(a, b):
        return np.zeros_like(a), np.zeros_like(b)

    a = np.ones((5, 6))
    b = np.ones((5, 6))

    new_state = ssprk3_step(
        (a, b), 0.1, zero_rhs
    )

    assert len(new_state) == 2
    assert new_state[0].shape == (5, 6)
    assert new_state[1].shape == (5, 6)


def test_ssprk3_rejects_invalid_timestep():
    """
    Check that zero or negative timesteps are rejected.
    """

    q = np.ones((3, 3))

    def zero_rhs(a):
        return (np.zeros_like(a),)

    with pytest.raises(ValueError):
        ssprk3_step((q,), 0.0, zero_rhs)

    with pytest.raises(ValueError):
        ssprk3_step((q,), -0.1, zero_rhs)


def test_ssprk3_rejects_wrong_number_of_tendencies():
    """
    Check that the RHS must return one tendency per field.
    """

    q = np.ones((3, 3))

    def incorrect_rhs(a):
        return ()

    with pytest.raises(ValueError):
        ssprk3_step((q,), 0.1, incorrect_rhs)
