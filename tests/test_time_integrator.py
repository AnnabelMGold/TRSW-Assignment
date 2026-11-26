import numpy as np
import pytest

from time_integrator import ssprk3_step


def constant_rhs(*state):
    """Return a constant tendency for every state component."""
    return tuple(np.ones_like(q) for q in state)


def test_constant_rhs():
    state = (np.zeros((3, 4)),)
    dt = 0.1

    result = ssprk3_step(state, dt, constant_rhs)

    assert np.allclose(result[0], dt)


def test_multiple_components():
    state = (
        np.zeros((3, 4)),
        np.zeros((4, 5)),
    )

    result = ssprk3_step(state, 0.1, constant_rhs)

    assert result[0].shape == (3, 4)
    assert result[1].shape == (4, 5)

    assert np.allclose(result[0], 0.1)
    assert np.allclose(result[1], 0.1)


def test_staggered_state_shapes():
    # Mimics the shapes of the TRSW staggered state.
    state = (
        np.ones((5, 6)),       # D
        np.ones((5, 7)),       # u
        np.ones((6, 6)),       # v
        np.ones((5, 6)),       # theta
    )

    result = ssprk3_step(state, 0.01, constant_rhs)

    expected_shapes = (
        (5, 6),
        (5, 7),
        (6, 6),
        (5, 6),
    )

    for q, expected_shape in zip(result, expected_shapes):
        assert q.shape == expected_shape


def test_input_state_not_modified():
    state = (
        np.zeros((3, 4)),
        np.zeros((4, 5)),
    )

    original = tuple(q.copy() for q in state)

    ssprk3_step(state, 0.1, constant_rhs)

    for q, q_original in zip(state, original):
        assert np.array_equal(q, q_original)


def test_invalid_dt():
    state = (np.zeros((3, 4)),)

    with pytest.raises(ValueError):
        ssprk3_step(state, 0.0, constant_rhs)

    with pytest.raises(ValueError):
        ssprk3_step(state, -0.1, constant_rhs)


def test_wrong_rhs_component_shape():
    state = (
        np.zeros((3, 4)),
        np.zeros((4, 5)),
    )

    def bad_rhs(*state):
        return (
            np.zeros((3, 4)),
            np.zeros((3, 5)),
        )

    with pytest.raises(ValueError):
        ssprk3_step(state, 0.1, bad_rhs)


def test_wrong_number_of_tendencies():
    state = (
        np.zeros((3, 4)),
        np.zeros((4, 5)),
    )

    def bad_rhs(*state):
        return (np.zeros((3, 4)),)

    with pytest.raises(ValueError):
        ssprk3_step(state, 0.1, bad_rhs)


def test_nonfinite_state():
    state = (np.zeros((3, 4)),)
    state[0][0, 0] = np.nan

    with pytest.raises(ValueError):
        ssprk3_step(state, 0.1, constant_rhs)