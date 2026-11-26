
"""
Time integration module for the TRSW model.

This module implements the third-order Strong Stability
Preserving Runge-Kutta method (SSP-RK3).

The integrator is independent of the physical equations.
It accepts a state consisting of multiple NumPy arrays
and a function that calculates their time derivatives.

For the TRSW model, the state contains:

    (D, u, v, theta)

The RHS function is compute_trsw_rhs().
"""


import numpy as np


def ssprk3_step(state, dt, rhs_function, *rhs_args):
    """
    Advance a collection of fields by one timestep using SSP-RK3.

    Inputs:
        state (tuple of numpy.ndarray):
            Current solution fields. All arrays must have
            the same shape.

        dt (float):
            Timestep. Must be finite and positive.

        rhs_function (callable):
            Function that calculates the time derivative
            of every field in state.

            It must return a tuple of arrays with the
            same number and shapes as state.

        *rhs_args:
            Additional arguments passed to rhs_function.

    Outputs:
        new_state (tuple of numpy.ndarray):
            Updated solution fields after one timestep.

    Description:
        Implements the three stages of SSP-RK3:

            Q1 = Qn + dt * L(Qn)

            Q2 = 3/4 Qn + 1/4 * (Q1 + dt * L(Q1))

            Qnew = 1/3 Qn + 2/3 * (Q2 + dt * L(Q2))

        The input state is not modified.
    """

    # ---------------------------------------------------------
    # 1. Validate the timestep
    # ---------------------------------------------------------

    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    # ---------------------------------------------------------
    # 2. Validate the input state
    # ---------------------------------------------------------

    if not isinstance(state, (tuple, list)) or len(state) == 0:
        raise ValueError("state must be a non-empty tuple or list.")

    for field in state:
        if not isinstance(field, np.ndarray) or field.ndim != 2:
            raise ValueError("Every state field must be a 2D NumPy array.")

        if not np.all(np.isfinite(field)):
            raise ValueError("State fields must contain only finite values.")

    # All fields must have the same shape
    reference_shape = state[0].shape

    for field in state:
        if field.shape != reference_shape:
            raise ValueError("All state fields must have the same shape.")

    if not callable(rhs_function):
        raise ValueError("rhs_function must be callable.")

    # ---------------------------------------------------------
    # 3. Define a helper to evaluate and validate the RHS
    # ---------------------------------------------------------

    def evaluate_rhs(current_state):
        """
        Evaluate the RHS function and check its outputs.

        Returns:
            tuple: Time derivatives of all state fields.
        """

        tendencies = rhs_function(*current_state, *rhs_args)

        if not isinstance(tendencies, (tuple, list)):
            raise ValueError("The RHS function must return a tuple or list.")

        if len(tendencies) != len(state):
            raise ValueError("The RHS must return one tendency per state field.")

        for tendency in tendencies:
            if not isinstance(tendency, np.ndarray):
                raise ValueError("Every tendency must be a NumPy array.")

            if tendency.shape != reference_shape:
                raise ValueError("RHS output arrays must match the state shape.")

            if not np.all(np.isfinite(tendency)):
                raise ValueError("RHS outputs must contain only finite values.")

        return tuple(tendencies)

    # ---------------------------------------------------------
    # 4. SSP-RK3: Stage 1
    # ---------------------------------------------------------

    # Evaluate the tendencies at the original state
    rhs_1 = evaluate_rhs(state)

    # Calculate the first intermediate state
    state_1 = tuple(
        q + dt * dq
        for q, dq in zip(state, rhs_1)
    )

    # ---------------------------------------------------------
    # 5. SSP-RK3: Stage 2
    # ---------------------------------------------------------

    # Evaluate the tendencies at the first intermediate state
    rhs_2 = evaluate_rhs(state_1)

    # Calculate the second intermediate state
    state_2 = tuple(
        0.75 * q + 0.25 * (q1 + dt * dq)
        for q, q1, dq in zip(state, state_1, rhs_2)
    )

    # ---------------------------------------------------------
    # 6. SSP-RK3: Stage 3
    # ---------------------------------------------------------

    # Evaluate the tendencies at the second intermediate state
    rhs_3 = evaluate_rhs(state_2)

    # Calculate the final state at the next time level
    new_state = tuple(
        (1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * dq)
        for q, q2, dq in zip(state, state_2, rhs_3)
    )

    return new_state
