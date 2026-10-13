"""Tests that NaN and inf inputs raise an error instead of giving a silent result."""

from functools import partial

import numpy as np
import pytest

import boss_bench as bb
from test_heat_flux import random_target

BAD_VALUES = [np.nan, np.inf]


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_loss_quantities(bad):
    """Loss quantities reject non-finite confinement times.

    Args:
        bad: the non-finite value.
    """
    ct, s_init = np.array([0.05, bad]), np.array([0.1, 0.1])
    for f in (bb.loss_fraction, bb.energy_loss, bb.normalized_energy_loss):
        with pytest.raises(ValueError):
            f(ct, 0.1, s_init)


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_heat_flux_prediction(bad):
    """heat_flux_metrics rejects non-finite predicted strike points and loss fractions.

    Args:
        bad: the non-finite value.
    """
    t = random_target(50)
    zeta = t["zeta"].copy()
    zeta[0] = bad
    with pytest.raises(ValueError):
        bb.heat_flux_metrics(zeta, t["theta"], t["energy"], t["loss_fraction"], t)
    with pytest.raises(ValueError):
        bb.heat_flux_metrics(t["zeta"], t["theta"], t["energy"], bad, t)


def bad_rollout(z0, record, n_steps, bad):
    """A rollout model whose output is all non-finite.

    Args:
        z0: initial states, (n, 5).
        record, n_steps: as for a rollout model.
        bad: the non-finite value to return.

    Returns:
        (n, n_steps, 5) array filled with bad.
    """
    return np.full((len(z0), n_steps, 5), bad)


def bad_predict_at(z0, tau, record, bad):
    """A direct model whose output is all non-finite.

    Args:
        z0: initial states, (n, 5).
        tau, record: as for a direct model.
        bad: the non-finite value to return.

    Returns:
        (n, 5) array filled with bad.
    """
    return np.full((len(z0), 5), bad)


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_trajectory_models(trajectory_file, bad):
    """evaluate_trajectories rejects models that return non-finite states.

    Args:
        trajectory_file: fixture, see conftest.py.
        bad: the non-finite value.
    """
    with pytest.raises(ValueError):
        bb.evaluate_trajectories([trajectory_file], rollout=partial(bad_rollout, bad=bad))
    with pytest.raises(ValueError):
        bb.evaluate_trajectories([trajectory_file], predict_at=partial(bad_predict_at, bad=bad))


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_design_metrics(bad):
    """design_metrics rejects non-finite inputs.

    Args:
        bad: the non-finite value.
    """
    with pytest.raises(ValueError):
        bb.design_metrics([0.1, bad], [1.5, 1.5], [1.5, 1.5], [5.0, 5.0], [5.0, 5.0])
