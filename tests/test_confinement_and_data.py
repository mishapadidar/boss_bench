"""Tests of data loading and of the particle-loss quantities."""

import h5py
import numpy as np

import boss_bench as bb
from conftest import CONFINEMENT_TIMES, T_MAX


def test_list_and_load(confinement_file):
    """Files are found by kind and split, and fields are stacked over files.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    data_dir = confinement_file.parent
    assert bb.list_files(data_dir, "confinement", "test") == [str(confinement_file)]
    assert bb.list_files(data_dir, "trajectory", "test") == []
    data = bb.load([confinement_file, confinement_file], ["nfp", "uuid"])
    assert data["nfp"].shape == (4,)
    assert list(data["uuid"]) == ["config-0", "config-1"] * 2


def test_load_record(confinement_file):
    """A record holds one configuration, with strings decoded, found by index or uuid.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    with h5py.File(confinement_file, "r") as f:
        record = bb.load_record(f, 1)
        assert bb.load_record(f, "config-1", ["nfp"])["nfp"] == record["nfp"]
    assert record["uuid"] == "config-1"
    assert record["confinement_time"].shape == (12,)


def test_load_record_fields(confinement_file):
    """With fields given, only those fields are read.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    with h5py.File(confinement_file, "r") as f:
        record = bb.load_record(f, 0, ["nfp", "t_max"])
    assert set(record) == {"nfp", "t_max"}


def test_loss_fraction():
    """Two of the four particles are lost; t_max - 1e-15 is within the loss tolerance."""
    ct = np.array(CONFINEMENT_TIMES)
    assert list(bb.is_lost(ct, T_MAX)) == [False, True, False, True]
    s_init = np.array([0.1, 0.1, 0.1, 0.1])
    assert bb.loss_fraction(ct, T_MAX, s_init, s0=0.1) == 0.5


def test_loss_fraction_per_surface_and_batch():
    """Only particles born on s0 count, and a batch of configurations gives one value each."""
    ct = np.array([[0.05, T_MAX, 0.05, 0.05], [T_MAX, T_MAX, 0.05, T_MAX]])
    s_init = np.array([[0.1, 0.1, 0.5, 0.5]] * 2)
    t_max = np.array([T_MAX, T_MAX])
    np.testing.assert_allclose(bb.loss_fraction(ct, t_max, s_init, s0=0.1), [0.5, 0.0])
    np.testing.assert_allclose(bb.loss_fraction(ct, t_max, s_init, s0=0.5), [1.0, 0.5])


def test_energy_loss():
    """A confined particle carries 3.5 exp(-2) MeV and one lost at once 3.5 MeV;
    normalized, these are 0 and 1."""
    ct = np.array([T_MAX, 0.0])
    s_init = np.array([0.1, 0.1])
    np.testing.assert_allclose(bb.energy_loss(ct, T_MAX, s_init), (3.5 * np.exp(-2) + 3.5) / 2)
    np.testing.assert_allclose(bb.normalized_energy_loss(ct, T_MAX, s_init), 0.5)
    np.testing.assert_allclose(bb.normalized_energy_loss(ct[:1], T_MAX, s_init[:1]), 0.0, atol=1e-15)


def test_mean_confinement_time():
    """Confined particles count as t_max; only particles born on s0 count."""
    ct = np.array([[0.05, T_MAX, 0.01, 0.03], [T_MAX, T_MAX, 0.02, T_MAX]])
    s_init = np.array([[0.1, 0.1, 0.5, 0.5]] * 2)
    t_max = np.array([T_MAX, T_MAX])
    np.testing.assert_allclose(bb.mean_confinement_time(ct, t_max, s_init, s0=0.1), [(0.05 + T_MAX) / 2, T_MAX])
    np.testing.assert_allclose(bb.mean_confinement_time(ct, t_max, s_init, s0=0.5), [0.02, (0.02 + T_MAX) / 2])
