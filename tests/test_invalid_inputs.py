"""Tests that NaN and inf inputs raise an error instead of giving a silent result."""

import h5py
import numpy as np
import pytest

import boss_bench as bb
from boss_bench import heat_flux
from test_heat_flux import random_target, torus_fields

BAD_VALUES = [np.nan, np.inf]


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_loss_quantities(bad):
    """Loss quantities reject non-finite confinement times.

    Args:
        bad (float): the non-finite value.
    """
    ct, s_init = np.array([0.05, bad]), np.array([0.1, 0.1])
    for f in (bb.loss_fraction, bb.energy_loss, bb.normalized_energy_loss, bb.mean_confinement_time):
        with pytest.raises(ValueError):
            f(ct, 0.1, s_init)


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_heat_flux_prediction(bad):
    """Predicted strike points and loss fractions must be finite.

    Args:
        bad (float): the non-finite value.
    """
    t = random_target(50)
    zeta = t["zeta"].copy()
    zeta[0] = bad
    with pytest.raises(ValueError):
        heat_flux._sample_metrics(zeta, t["theta"], t["energy"], t["loss_fraction"], t)
    with pytest.raises(ValueError):
        heat_flux._sample_metrics(t["zeta"], t["theta"], t["energy"], bad, t)
    with pytest.raises(ValueError):
        bb.binned_heat_flux(zeta, t["theta"], t["energy"], 1000, torus_fields())


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_trajectory_predictions(trajectory_file, bad):
    """Predicted states of scored orbits must be finite, and loss times not infinite.

    Args:
        trajectory_file: fixture, see conftest.py.
        bad (float): the non-finite value.
    """
    with h5py.File(trajectory_file, "r") as f:
        t_loss = bb.read_trajectories(f, 0)["t_loss"]
    z = np.zeros((3, 3, 5))
    z[0, 0, 0] = bad
    with pytest.raises(ValueError):
        bb.trajectory_metrics([trajectory_file], {"config-0": (z, t_loss)})
    with pytest.raises(ValueError):
        bb.trajectory_metrics([trajectory_file], {"config-0": (np.zeros((3, 3, 5)), np.array([np.inf, 1e-4, 1e-4]))})


def test_unscored_orbits_may_be_nan(trajectory_file):
    """The orbit lost before its first sample is not scored, so its prediction may be NaN.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    z = np.zeros((3, 3, 5))
    z[2] = np.nan
    bb.trajectory_metrics([trajectory_file], {"config-0": (z, np.full(3, np.nan))})


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_design_metrics(bad):
    """design_metrics rejects non-finite inputs.

    Args:
        bad (float): the non-finite value.
    """
    with pytest.raises(ValueError):
        bb.design_metrics([0.1, bad], [1.5, 1.5], [1.5, 1.5], [5.0, 5.0], [5.0, 5.0])
