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
    t_pred = np.array([trajectory.T_MAX, 1.1e-3])
    assert bb.loss_fraction_mae([t_pred], [np.full(2, np.nan)]) == 0.5


def test_nrmse_by_hand():
    """One configuration, one particle, an error of size a in the x channel only:
    nRMSE = a / (4 sigma_x) at every lead time."""
    a = 0.2
    z_true = np.zeros((1, 3, 5))
    z_pred = z_true.copy()
    z_pred[..., 0] = a
    sigma_x = trajectory.CHANNEL_GROUPS["pol_x"][1]
    np.testing.assert_allclose(bb.nrmse([z_pred], [z_true]), a / (4 * sigma_x))


def test_nrmse_skips_lost_particles():
    """A particle with no true state at a lead time is not scored there."""
    z_true = np.zeros((2, 1, 5))
    z_true[1] = np.nan
    z_pred = np.zeros((2, 1, 5))
    z_pred[1] = 0.7  # any prediction for the lost particle
    assert bb.nrmse([z_pred], [z_true])[0] == 0


def test_confinement_time_w1_by_hand():
    """Sorted confinement times (100, 1000) us against (300, 1000) us: W1 = 200 / 2 = 100 us."""
    t_pred = np.array([np.nan, 300e-6])
    t_true = np.array([100e-6, np.nan])
    np.testing.assert_allclose(bb.confinement_time_w1([t_pred], [t_true]), 100)


def test_read_by_uuid(trajectory_file):
    """A configuration can be read by its uuid as well as by its index.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    with h5py.File(trajectory_file, "r") as f:
        by_index = bb.read_trajectories(f, 0)
        by_uuid = bb.read_trajectories(f, "config-0")
        with pytest.raises(KeyError):
            bb.read_trajectories(f, "no-such-uuid")
    np.testing.assert_array_equal(by_index["n_valid"], by_uuid["n_valid"])


def test_predictions_must_fit(trajectory_file):
    """Predictions of the wrong shape, or for an unknown configuration, are errors.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    with pytest.raises(ValueError):
        bb.trajectory_metrics([trajectory_file], {"config-0": (np.zeros((2, 3, 5)), np.zeros(2))})
    with pytest.raises(KeyError):
        bb.trajectory_metrics([trajectory_file], {"no-such-uuid": (np.zeros((3, 3, 5)), np.zeros(3))})
