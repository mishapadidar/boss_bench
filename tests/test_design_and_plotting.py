"""Tests of the inverse-design metrics, and that every plot draws."""

import h5py
import matplotlib
import pytest

import boss_bench as bb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_design_metrics():
    """Four designs; thresholds are inclusive."""
    m = bb.design_metrics(loss_fraction=[0.05, 0.10, 0.15, 0.30],
                          mirror_ratio=[1.4, 1.5, 1.6, 2.0], mirror_target=[1.5] * 4,
                          aspect_ratio=[5.0, 6.0, 9.0, 4.0], aspect_target=[5.0, 5.0, 10.0, 5.0])
    assert m["low_loss_rate_10"] == 50
    assert m["low_loss_rate_15"] == 75
    assert m["mirror_constraint"] == 50
    assert m["mirror_constraint_1.1"] == 75
    assert m["aspect_error_median"] == pytest.approx(15)  # median of 0%, 20%, 10%, 20%


def test_plots_draw(confinement_file, trajectory_file):
    """Each plotting function runs on the example data and returns its figure or axis.

    Args:
        confinement_file: fixture, see conftest.py.
        trajectory_file: fixture, see conftest.py.
    """
    with h5py.File(confinement_file, "r") as f:
        record = bb.load_record(f, 0)
    with h5py.File(trajectory_file, "r") as f:
        z = bb.read_trajectories(f, 0)["z"][:2]
    predictions = {}
    for s0 in (0.1, 0.25, 0.5):
        zeta, theta, energy, n_launched = bb.lost_particles(record, s0)
        predictions[s0] = (zeta, theta, energy, len(zeta) / n_launched)

    assert bb.plot_device(record) is not None
    assert bb.plot_modB_boozer(record) is not None
    assert bb.plot_loss_fraction_curves(record) is not None
    assert bb.plot_strike_points(record, predictions) is not None
    assert bb.plot_heat_flux(record, predictions) is not None
    assert bb.plot_rollouts(z, z + 0.01) is not None
    plt.close("all")
