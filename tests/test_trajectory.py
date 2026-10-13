"""Tests of the trajectory benchmark on three hand-made orbits."""

import h5py
import numpy as np
import pytest

import boss_bench as bb
from boss_bench import trajectory


def read_truth(trajectory_file):
    """The true orbits of the one configuration of the example file.

    Args:
        trajectory_file (str): the file.

    Returns:
        dict: as from ``read_trajectories``.
    """
    with h5py.File(trajectory_file, "r") as f:
        return bb.read_trajectories(f, 0)


def test_read_trajectories(trajectory_file):
    """On-grid samples and loss times are read; the off-grid loss sample is not a state.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    truth = read_truth(trajectory_file)
    assert list(truth["n_valid"]) == [1000, 500, 0]
    np.testing.assert_allclose(truth["t_loss"], [np.nan, 500.4e-6, 0.5e-6])
    z = truth["z"]
    assert z.shape == (3, 1000, 5)
    np.testing.assert_allclose(z[0, 0], bb.embed_state(*np.array([0.3, 0.0, 0.0, 1e6])))
    assert np.isnan(z[1, 500:]).all() and np.isfinite(z[1, :500]).all()


def test_true_predictions(trajectory_file):
    """The true states and loss times score zero on every metric. The lost orbit's true
    state at 999 us does not exist, so any prediction there is not scored, and the orbit
    lost before its first sample is ignored.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    truth = read_truth(trajectory_file)
    z = np.nan_to_num(truth["z"][:, trajectory.TABLE_STEPS], nan=0.5)
    m = bb.trajectory_metrics([trajectory_file], {"config-0": (z, truth["t_loss"])})
    assert m["nrmse_tau10us"] == m["nrmse_tau100us"] == m["nrmse_tau999us"] == 0
    assert m["loss_fraction_mae"] == 0 and m["w1_us"] == 0
    assert m["n_configurations"] == 1


def test_no_motion(trajectory_file):
    """Predicting z(t) = z0 and no losses: the loss-fraction error is 1/2 and
    W1 = (1000 - 500.4) / 2 us over the two scored orbits.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    truth = read_truth(trajectory_file)
    z = np.repeat(np.nan_to_num(truth["z"][:, :1], nan=0.5), 3, axis=1)
    t_loss = np.full(3, np.nan)
    m = bb.trajectory_metrics([trajectory_file], {"config-0": (z, t_loss)})
    assert m["loss_fraction_mae"] == 0.5
    np.testing.assert_allclose(m["w1_us"], (1000 - 500.4) / 2)
    assert m["nrmse_tau10us"] > 0


def test_loss_at_the_last_sample_counts():
    """A predicted loss at t = 1 ms counts as a loss; one after it does not."""
    truth = {"z": np.zeros((2, 1000, 5)), "n_valid": np.array([1000, 1000]), "t_loss": np.full(2, np.nan)}
    z = np.zeros((2, 3, 5))
    scores = trajectory._score_configuration(truth, z, np.array([trajectory.T_MAX, 1.1e-3]))
    assert scores["f_pred"] == 0.5


def test_predictions_must_fit(trajectory_file):
    """Predictions of the wrong shape, or for an unknown configuration, are errors.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    with pytest.raises(ValueError):
        bb.trajectory_metrics([trajectory_file], {"config-0": (np.zeros((2, 3, 5)), np.zeros(2))})
    with pytest.raises(KeyError):
        bb.trajectory_metrics([trajectory_file], {"no-such-uuid": (np.zeros((3, 3, 5)), np.zeros(3))})
