
"""
Grid-refinement study for the Thermal Rotating Shallow Water model.

Compares the final depth fields at three grid resolutions.
"""

import numpy as np

from config import ModelConfig
from solver import run_simulation


def restrict_to_coarse(fine_field):
    """
    Average a fine-grid cell-centred field onto a grid half its resolution.

    Inputs:
        fine_field (np.ndarray): Field with shape (2*Ny, 2*Nx).

    Outputs:
        np.ndarray: Coarse field with shape (Ny, Nx).
    """
    ny_fine, nx_fine = fine_field.shape

    if ny_fine % 2 != 0 or nx_fine % 2 != 0:
        raise ValueError("Fine-grid dimensions must be even.")

    return fine_field.reshape(
        ny_fine // 2, 2, nx_fine // 2, 2
    ).mean(axis=(1, 3))


def run_case(nx, ny, nt):
    """
    Run one TRSW simulation at a specified resolution.

    Inputs:
        nx (int): Number of cells in the x direction.
        ny (int): Number of cells in the y direction.
        nt (int): Number of time steps.

    Outputs:
        np.ndarray: Final cell-centred depth field.
    """
    config = ModelConfig(
        Lx=2.0,
        Ly=1.0,
        Nx=nx,
        Ny=ny,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.01,
        Nt=nt,
    )

    def initial_depth(X, Y):
        return (
            1.0
            + 0.05
            * np.cos(2.0 * np.pi * X / config.Lx)
            * np.cos(2.0 * np.pi * Y / config.Ly)
        )

    def initial_u(X, Y):
        return (
            0.1
            * np.sin(np.pi * X / config.Lx)
            * np.cos(2.0 * np.pi * Y / config.Ly)
        )

    def initial_v(X, Y):
        return (
            0.1
            * np.cos(2.0 * np.pi * X / config.Lx)
            * np.sin(np.pi * Y / config.Ly)
        )

    result = run_simulation(
        config,
        D0=initial_depth,
        u0=initial_u,
        v0=initial_v,
        theta0=1.0,
    )

    return result["final_state"][0]


def main():
    """Run the grid-refinement study and print depth-field differences."""

    print("Running coarse simulation...")
    depth_16 = run_case(16, 8, 100)

    print("Running medium simulation...")
    depth_32 = run_case(32, 16, 200)

    print("Running fine simulation...")
    depth_64 = run_case(64, 32, 400)

    # Restrict the finer solutions to the coarser grid.
    depth_32_on_16 = restrict_to_coarse(depth_32)
    depth_64_on_32 = restrict_to_coarse(depth_64)

    # Calculate root-mean-square differences.
    error_16_32 = np.sqrt(
        np.mean((depth_16 - depth_32_on_16) ** 2)
    )

    error_32_64 = np.sqrt(
        np.mean((depth_32 - depth_64_on_32) ** 2)
    )

    print("\nGrid-refinement results")
    print("-----------------------")
    print(f"Difference (16x8 vs 32x16): {error_16_32:.8e}")
    print(f"Difference (32x16 vs 64x32): {error_32_64:.8e}")

    if error_32_64 < error_16_32:
        print("\nThe differences decrease with grid refinement.")
    else:
        print("\nThe differences do not decrease with grid refinement.")
        print("We need to investigate the numerical results.")


if __name__ == "__main__":
    main()
