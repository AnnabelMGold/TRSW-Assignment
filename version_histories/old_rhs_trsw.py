
"""
Complete TRSW right-hand-side module.

This module combines the depth, momentum and buoyancy
equations into one function.

It calculates the time derivatives of all four evolving
fields:

    D     : fluid depth
    u     : x-component of velocity
    v     : y-component of velocity
    theta : buoyancy

The function calculates tendencies only. It does not
advance the solution in time.
"""

from version_histories.old_rhs_depth import compute_depth_rhs
from version_histories.old_rhs_momentum import compute_momentum_rhs
from version_histories.old_rhs_buoyancy import compute_buoyancy_rhs


def compute_trsw_rhs(D, u, v, theta, dx, dy, epsilon, f, F):
    """
    Calculate the complete right-hand side of the TRSW system.

    Inputs:
        D (numpy.ndarray): Fluid depth. Shape (Ny, Nx).
        u (numpy.ndarray): x-component of velocity. Shape (Ny, Nx).
        v (numpy.ndarray): y-component of velocity. Shape (Ny, Nx).
        theta (numpy.ndarray): Buoyancy field. Shape (Ny, Nx).
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        epsilon (float): Rossby number.
        f (float): Coriolis parameter.
        F (float): Parameter in gamma = sqrt(theta) / F.

    Outputs:
        dDdt (numpy.ndarray): Time derivative of depth.
        dudt (numpy.ndarray): Time derivative of x-velocity.
        dvdt (numpy.ndarray): Time derivative of y-velocity.
        dthetadt (numpy.ndarray): Time derivative of buoyancy.

        All output arrays have shape (Ny, Nx).

    Description:
        Calls the individual equation modules to calculate
        the tendencies of depth, velocity and buoyancy.

        The returned tendencies can subsequently be passed
        to a time integration method such as SSP-RK3.
    """

    # ---------------------------------------------------------
    # 1. Calculate the depth tendency
    # ---------------------------------------------------------

    dDdt = compute_depth_rhs(D, u, v, dx, dy)

    # ---------------------------------------------------------
    # 2. Calculate the momentum tendencies
    # ---------------------------------------------------------

    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta, dx, dy, epsilon, f, F
    )

    # ---------------------------------------------------------
    # 3. Calculate the buoyancy tendency
    # ---------------------------------------------------------

    dthetadt = compute_buoyancy_rhs(theta, u, v, dx, dy)

    # ---------------------------------------------------------
    # 4. Return all four tendencies
    # ---------------------------------------------------------

    return dDdt, dudt, dvdt, dthetadt
