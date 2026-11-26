
"""
Configuration module for the Thermal Rotating Shallow Water (TRSW) model.

This module defines the model parameters, computational domain,
grid resolution and time-stepping configuration.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Store the configuration parameters for the TRSW simulation.

    Attributes / Inputs:
        Lx (float): Length of the rectangular domain in the x-direction.
        Ly (float): Length of the rectangular domain in the y-direction.
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        epsilon (float): Rossby number in the TRSW momentum equation.
        f (float): Coriolis parameter.
        F (float): Parameter in the definition gamma = sqrt(theta) / F.
        t_end (float): Final time of the simulation.
        Nt (int): Number of time steps in the simulation.

    Output:
        ModelConfig: An object containing the model configuration.
    """

    # Domain dimensions
    Lx: float = 1.0
    Ly: float = 1.0

    # Grid resolution
    Nx: int = 100
    Ny: int = 100

    # TRSW model parameters
    epsilon: float = 1.0
    f: float = 1.0
    F: float = 1.0

    # Time-stepping parameters
    t_end: float = 1.0
    Nt: int = 1000

    def __post_init__(self):
        """
        Validate the configuration parameters after object creation.

        Inputs:
            None: Uses the parameters stored in this configuration object.

        Outputs:
            None: Raises ValueError if a parameter is invalid.
        """

        # Domain dimensions must be positive
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must be positive.")

        # Grid resolution must contain at least two points per direction
        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must be at least 2.")

        # Model parameters
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        if self.F <= 0:
            raise ValueError("F must be positive.")

        # Time-stepping parameters
        if self.t_end <= 0:
            raise ValueError("t_end must be positive.")

        if self.Nt < 1:
            raise ValueError("Nt must be at least 1.")

    def compute_dt(self) -> float:
        """
        Compute the time-step size from the simulation duration.

        Inputs:
            None: Uses t_end and Nt stored in the configuration.

        Outputs:
            dt (float): Time-step size, calculated as t_end / Nt.
        """

        return self.t_end / self.Nt
