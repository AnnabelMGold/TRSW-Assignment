import numpy as np

from config import ModelConfig
from solver import run_simulation


def test_solver_shapes():
    config = ModelConfig(
        Lx=2.0,
        Ly=1.0,
        Nx=6,
        Ny=5,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.1,
        Nt=10,
    )

    result = run_simulation(config)

    D, u, v, theta = result["final_state"]

    assert D.shape == (5, 6)
    assert u.shape == (5, 7)
    assert v.shape == (6, 6)
    assert theta.shape == (5, 6)


def test_stationary_uniform_state():
    config = ModelConfig(
        Lx=1.0,
        Ly=1.0,
        Nx=6,
        Ny=5,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.1,
        Nt=10,
    )

    result = run_simulation(config)

    D, u, v, theta = result["final_state"]

    assert np.allclose(D, 1.0)
    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)
    assert np.allclose(theta, 1.0)


def test_initial_and_final_state():
    config = ModelConfig(
        Lx=1.0,
        Ly=1.0,
        Nx=6,
        Ny=5,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.2,
        Nt=20,
    )

    result = run_simulation(config)

    initial_state = result["initial_state"]
    final_state = result["final_state"]

    assert len(initial_state) == 4
    assert len(final_state) == 4

    for q_initial, q_final in zip(initial_state, final_state):
        assert q_initial.shape == q_final.shape


def test_final_time_and_dt():
    config = ModelConfig(
        Lx=1.0,
        Ly=1.0,
        Nx=6,
        Ny=5,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.2,
        Nt=20,
    )

    result = run_simulation(config)

    assert np.isclose(result["dt"], 0.01)
    assert np.isclose(result["final_time"], config.t_end)


def test_initial_conditions_passed_to_solver():
    config = ModelConfig(
        Lx=2.0,
        Ly=1.0,
        Nx=6,
        Ny=5,
        epsilon=1.0,
        f=1.0,
        F=1.0,
        t_end=0.1,
        Nt=10,
    )

    result = run_simulation(
        config,
        D0=2.0,
        u0=0.0,
        v0=0.0,
        theta0=3.0,
    )

    D, u, v, theta = result["initial_state"]

    assert np.allclose(D, 2.0)
    assert np.allclose(theta, 3.0)
    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)