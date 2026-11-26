
"""
Automated tests for config.py.

These tests check parameter validation and timestep calculation
for the TRSW model configuration.
"""

import pytest

from config import ModelConfig


def test_default_configuration():
    """Check that the default model configuration is valid."""
    config = ModelConfig()

    assert config.Lx == 1.0
    assert config.Ly == 1.0
    assert config.Nx == 100
    assert config.Ny == 100
    assert config.t_end == 1.0
    assert config.Nt == 1000


def test_compute_dt():
    """Check that dt is calculated as t_end / Nt."""
    config = ModelConfig(t_end=2.0, Nt=500)

    dt = config.compute_dt()

    assert dt == pytest.approx(0.004)


def test_invalid_domain_length():
    """Check that non-positive domain lengths are rejected."""
    with pytest.raises(ValueError):
        ModelConfig(Lx=0.0)

    with pytest.raises(ValueError):
        ModelConfig(Ly=-1.0)


def test_invalid_grid_resolution():
    """Check that grid resolutions below 2 are rejected."""
    with pytest.raises(ValueError):
        ModelConfig(Nx=1)

    with pytest.raises(ValueError):
        ModelConfig(Ny=1)


def test_invalid_model_parameters():
    """Check that invalid physical parameters are rejected."""
    with pytest.raises(ValueError):
        ModelConfig(epsilon=0.0)

    with pytest.raises(ValueError):
        ModelConfig(F=0.0)


def test_invalid_time_parameters():
    """Check that invalid time parameters are rejected."""
    with pytest.raises(ValueError):
        ModelConfig(t_end=0.0)

    with pytest.raises(ValueError):
        ModelConfig(Nt=0)
