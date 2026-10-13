"""Tests of the trajectory benchmark on three hand-made orbits."""

import h5py
import numpy as np
import pytest

import boss_bench as bb
from boss_bench import trajectory

S_RATE = 1234.0  # ds/dt of `linear_s_model` [1/s]


def state(s):
    """Embedded states at flux labels s, with all angles and vpar zero.

    Args:
        s: flux labels.

    Returns:
        Array of shape s.shape + (5,).
    """
    s = np.asarray(s, dtype=float)
    return bb.embed(s, 0 * s, 0 * s, 0 * s)


def persistence(z0, record, n_steps):
    """A rollout model that predicts no motion.

    Args:
        z0: initial states, (n, 5).
        record: unused.
        n_steps: number of lead times.

    Returns:
        z0 repeated at every lead time, (n, n_steps, 5).
    """
    return np.repeat(z0[:, None], n_steps, axis=1)


def linear_s_model(z0, tau, record):
    """A direct model in which s rises at the rate S_RATE.

    Args:
        z0: initial states, (n, 5).
        tau: lead times [s], (n,).
        record: unused.

    Returns:
        Predicted states, (n, 5).
    """
    s0 = z0[:, 0] ** 2 + z0[:, 1] ** 2
    return state(np.minimum(s0 + S_RATE * tau, 1.0))


def oracle_rollout(trajectory_file):
    """Build a rollout model that returns the true orbits, at s = 0.99 after a loss.

    Args:
        trajectory_file: the file the model will be evaluated on.

    Returns:
        A rollout model, as `evaluate_trajectories` expects.
    """
    with h5py.File(trajectory_file, "r") as f:
        device = bb.read_device(f, 0)
    z = device["z"][device["n_valid"] >= 1]
    z[np.isnan(z[..., 0])] = state(0.99)

    def rollout(z0, record, n_steps):
        """Return the stored orbits.

        Args:
            z0, record: unused; the orbits are those of the file.
            n_steps: number of lead times.

        Returns:
            The stored states at lead times 1, ..., n_steps, (n, n_steps, 5).
        """
        return z[:, 1:n_steps + 1]
    return rollout


def test_read_device(trajectory_file):
    """On-grid samples and loss times are read; the off-grid loss sample is not a state.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    with h5py.File(trajectory_file, "r") as f:
        device = bb.read_device(f, 0)
    assert list(device["n_valid"]) == [1000, 500, 0]
    np.testing.assert_allclose(device["t_loss"], [np.nan, 500.4e-6, 0.5e-6])
    z = device["z"]
    assert z.shape == (3, 1000, 5)
    np.testing.assert_allclose(z[0, 0], bb.embed(*np.array([0.3, 0.0, 0.0, 1e6])))
    assert np.isnan(z[1, 500:]).all() and np.isfinite(z[1, :500]).all()


def test_oracle_rollout(trajectory_file):
    """The true orbits score nRMSE 0 and the right loss fraction.

    The predicted loss is at the first sample after the true one, 501 us against
    500.4 us, so with two scored orbits W1 = 0.6 / 2 = 0.3 us. The orbit with no
    initial state is not scored.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    m = bb.evaluate_trajectories([trajectory_file], rollout=oracle_rollout(trajectory_file))
    assert m["nrmse_tau10us"] == m["nrmse_tau100us"] == m["nrmse_tau999us"] == 0
    assert m["loss_fraction_mae"] == 0
    np.testing.assert_allclose(m["w1_us"], 0.3, rtol=1e-9)


def test_persistence(trajectory_file):
    """Predicting no motion loses nothing: loss-fraction error 1/2, W1 = (1000 - 500.4) / 2 us.

    Args:
        trajectory_file: fixture, see conftest.py.
    """
    m = bb.evaluate_trajectories([trajectory_file], rollout=persistence)
    assert m["loss_fraction_mae"] == 0.5
    np.testing.assert_allclose(m["w1_us"], (1000 - 500.4) / 2)
    assert m["nrmse_tau10us"] > 0


def test_direct_model_loss_time_is_bisected():
    """s reaches 0.98 at a known time; bisection finds it to within 1 us / 2^N_BISECT."""
    s0 = np.array([0.5, 0.9])
    _, t_loss = trajectory._predict_direct(linear_s_model, state(s0), record=None)
    expected = trajectory.DT + (0.98 - s0) / S_RATE  # absolute time: z0 is at t = 1 us
    assert np.all(t_loss >= expected - 1e-15)
    assert np.all(t_loss - expected <= trajectory.DT / 2 ** trajectory.N_BISECT)


def test_exactly_one_model():
    """evaluate_trajectories needs exactly one of rollout and predict_at."""
    with pytest.raises(ValueError):
        bb.evaluate_trajectories([])
