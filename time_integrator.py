"""
SSP-RK3 time integrator.

Supports model states consisting of multiple NumPy arrays
that may have different shapes, as in the staggered TRSW model.
"""


import numpy as np


def _validate_state(state):
    """Validate a state consisting of multiple NumPy arrays."""

    if not isinstance(state, (tuple, list)):
        raise TypeError("State must be a tuple or list of arrays.")

    if len(state) == 0:
        raise ValueError("State must contain at least one array.")

    validated = []

    for array in state:
        array = np.asarray(array, dtype=float)

        if array.ndim != 2:
            raise ValueError("Each state component must be 2D.")

        if not np.all(np.isfinite(array)):
            raise ValueError("State arrays must contain finite values.")

        validated.append(array)

    return tuple(validated)


def _evaluate_rhs(state, rhs_function, rhs_args):
    """Evaluate the RHS and validate its output."""

    rhs = rhs_function(*state, *rhs_args)

    if not isinstance(rhs, (tuple, list)):
        raise TypeError("RHS function must return a tuple or list.")

    if len(rhs) != len(state):
        raise ValueError("RHS must return one tendency per state component.")

    validated_rhs = []

    for component, tendency in zip(state, rhs):
        tendency = np.asarray(tendency, dtype=float)

        if tendency.shape != component.shape:
            raise ValueError("RHS component shape does not match state.")

        if not np.all(np.isfinite(tendency)):
            raise ValueError("RHS tendencies must contain finite values.")

        validated_rhs.append(tendency)

    return tuple(validated_rhs)


def ssprk3_step(state, dt, rhs_function, *rhs_args):
    """
    Perform one SSP-RK3 time step.

    Parameters
    ----------
    state : tuple or list of ndarray
        Model state. Components may have different shapes.

    dt : float
        Time-step size.

    rhs_function : callable
        Function returning one tendency per state component.

    *rhs_args
        Additional arguments passed to the RHS function.

    Returns
    -------
    tuple
        Updated model state.
    """

    state = _validate_state(state)

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    # Stage 1.
    rhs1 = _evaluate_rhs(state, rhs_function, rhs_args)

    stage1 = tuple(
        q + dt * dq
        for q, dq in zip(state, rhs1)
    )

    # Stage 2.
    rhs2 = _evaluate_rhs(stage1, rhs_function, rhs_args)

    stage2 = tuple(
        0.75 * q + 0.25 * (q1 + dt * dq2)
        for q, q1, dq2 in zip(state, stage1, rhs2)
    )

    # Stage 3.
    rhs3 = _evaluate_rhs(stage2, rhs_function, rhs_args)

    state_new = tuple(
        (1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * dq3)
        for q, q2, dq3 in zip(state, stage2, rhs3)
    )

    return state_new