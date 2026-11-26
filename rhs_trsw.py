"""
Combined TRSW right-hand-side function.

Combines the depth, momentum and buoyancy equations
for the staggered-grid TRSW model.

State ordering:
    (D, u, v, theta)

Tendency ordering:
    (dDdt, dudt, dvdt, dthetadt)
"""

from rhs_depth import compute_depth_rhs
from rhs_momentum import compute_momentum_rhs
from rhs_buoyancy import compute_buoyancy_rhs


def compute_trsw_rhs(
    D, u, v, theta, dx, dy, epsilon, f, F
):
    """
    Compute all four TRSW tendencies.

    Parameters
    ----------
    D : ndarray
        Cell-centred depth.

    u : ndarray
        Horizontal velocity on vertical faces.

    v : ndarray
        Vertical velocity on horizontal faces.

    theta : ndarray
        Cell-centred buoyancy.

    dx, dy : float
        Grid spacings.

    epsilon : float
        Rossby number.

    f : float
        Coriolis parameter.

    F : float
        Parameter in gamma = sqrt(theta)/F.

    Returns
    -------
    tuple
        (dDdt, dudt, dvdt, dthetadt)
    """

    # Depth tendency.
    dDdt = compute_depth_rhs(D, u, v, dx, dy)

    # Momentum tendencies.
    dudt, dvdt = compute_momentum_rhs(
        D, u, v, theta,
        dx, dy, epsilon, f, F
    )

    # Buoyancy tendency.
    dthetadt = compute_buoyancy_rhs(
        theta, u, v, dx, dy
    )

    return dDdt, dudt, dvdt, dthetadt