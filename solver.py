"""
Main simulation driver for the staggered-grid TRSW model.

This module assembles the grid, initial conditions, RHS and
time integrator into a complete simulation.
"""

import numpy as np

from grid import create_staggered_grid
from initial_conditions import create_initial_conditions
from rhs_trsw import compute_trsw_rhs
from time_integrator import ssprk3_step


def run_simulation(
    config,
    D0=1.0,
    u0=0.0,
    v0=0.0,
    theta0=1.0,
):
    """
    Run the TRSW model from t=0 to t_end.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.

    D0 : float or callable
        Initial depth.

    u0 : float or callable
        Initial horizontal velocity.

    v0 : float or callable
        Initial vertical velocity.

    theta0 : float or callable
        Initial buoyancy.

    Returns
    -------
    dict
        Simulation results containing the grid, final state
        and time information.
    """

    # Create the staggered grid.
    grid = create_staggered_grid(config)

    # Generate the initial conditions on the grid.
    initial = create_initial_conditions(
        grid,
        D0=D0,
        u0=u0,
        v0=v0,
        theta0=theta0,
    )

    # Assemble the state in the order expected by the RHS.
    state = (
        initial.D.copy(),
        initial.u.copy(),
        initial.v.copy(),
        initial.theta.copy(),
    )

    # Time-step size.
    dt = config.compute_dt()

    # Store the initial state for reference.
    initial_state = tuple(q.copy() for q in state)

    # Model parameters passed to the combined RHS.
    rhs_args = (
        grid["dx"],
        grid["dy"],
        config.epsilon,
        config.f,
        config.F,
    )

    # Advance the model in time.
    for _ in range(config.Nt):
        state = ssprk3_step(
            state,
            dt,
            compute_trsw_rhs,
            *rhs_args,
        )

    # Final time.
    final_time = config.Nt * dt

    # Package the results.
    return {
        "grid": grid,
        "initial_state": initial_state,
        "final_state": state,
        "dt": dt,
        "final_time": final_time,
    }