"""Tests of the heat-flux benchmark on a circular torus."""

import h5py
import numpy as np
import pytest

import boss_bench as bb
from boss_bench import heat_flux
from conftest import A, NFP, R0, T_MAX, torus_fields

N_LAUNCHED = 1000
PARTICLE_FIELDS = ["uuid", "confinement_time", "t_max", "s_init", "zeta_end", "theta_end"]


def torus_target(zeta, theta, energy):
    """A heat-flux target on the circular torus with the given strike points.

    Args:
        zeta, theta (np.ndarray): Boozer angles of the strike points [rad].
        energy (np.ndarray): their energies [MeV].

    Returns:
        dict: a target, as from ``heat_flux._target``.
    """
    record = torus_fields()
    geometry = {key: heat_flux._boundary_slice(record, name) for key, name in heat_flux.BOOZER_FIELDS.items()}
    return {"s_init": 0.1, "n_launched": N_LAUNCHED, "loss_fraction": len(zeta) / N_LAUNCHED,
            "zeta": zeta, "theta": theta, "energy": energy, "nfp": NFP, **geometry}


def random_target(n_lost, seed=0):
    """A target with n_lost strike points in a band around theta = 3, with random energies.

    Args:
        n_lost (int): number of strike points.
        seed (int): random seed.

    Returns:
        dict: a target.
    """
    rng = np.random.default_rng(seed)
    return torus_target(rng.uniform(0, 2 * np.pi, n_lost), 3 + 0.2 * rng.standard_normal(n_lost),
                        rng.uniform(0.5, 3.5, n_lost))


def true_predictions(path):
    """Predictions equal to the truth for every sample of a confinement file.

    Args:
        path (str): the confinement file.

    Returns:
        dict: maps (uuid, s0) to (zeta, theta, energy, loss_fraction).
    """
    predictions = {}
    with h5py.File(path, "r") as f:
        for i in range(len(f["uuid"])):
            record = bb.load_record(f, i, PARTICLE_FIELDS)
            for s0 in (0.1, 0.25, 0.5):
                zeta, theta, energy, n_launched = bb.lost_particles(
                    record["confinement_time"], record["t_max"], record["s_init"],
                    record["zeta_end"], record["theta_end"], s0)
                predictions[record["uuid"], s0] = (zeta, theta, energy, len(zeta) / n_launched)
    return predictions


def test_lost_particles(confinement_file):
    """Each birth surface loses two of its four particles, with energy 3.5 exp(-2 tau / t_max).

    Args:
        confinement_file: fixture, see conftest.py.
    """
    with h5py.File(confinement_file, "r") as f:
        record = bb.load_record(f, 0, PARTICLE_FIELDS)
    for s0 in (0.1, 0.25, 0.5):
        zeta, theta, energy, n_launched = bb.lost_particles(
            record["confinement_time"], record["t_max"], record["s_init"],
            record["zeta_end"], record["theta_end"], s0)
        assert n_launched == 4 and len(zeta) == 2
        np.testing.assert_allclose(energy, 3.5 * np.exp(-2 * np.array([0.05, 0.001]) / T_MAX))
        assert np.all((zeta >= 0) & (zeta < 2 * np.pi) & (theta >= 0) & (theta < 2 * np.pi))


def test_bin_areas_sum_to_boundary_area():
    """The bins of one field period add up to 1/nfp of the torus area 4 pi^2 R0 A."""
    t = random_target(10)
    area = heat_flux._period_bin_areas(t["R"], t["Z"], t["nu"], NFP, 16)
    np.testing.assert_allclose(area.sum() * NFP, 4 * np.pi ** 2 * R0 * A, rtol=1e-3)


def test_binned_heat_flux_integrates_to_energy_lost():
    """Integrated over the boundary, the heat flux is the energy lost per launched particle."""
    t = random_target(500)
    q = bb.binned_heat_flux(t["zeta"], t["theta"], t["energy"], N_LAUNCHED, torus_fields(), resolution=16)
    area = heat_flux._period_bin_areas(t["R"], t["Z"], t["nu"], NFP, 16)
    assert q.shape == (16, 16)
    np.testing.assert_allclose(NFP * (q * area).sum(), t["energy"].sum() / N_LAUNCHED)


def test_perfect_prediction():
    """Predicting the true strike points gives HF = 0, no loss-fraction error, and the
    energy distance of a cloud with itself, sqrt(2 / n)."""
    t = random_target(200)
    m = heat_flux._sample_metrics(t["zeta"], t["theta"], t["energy"], t["loss_fraction"], t)
    assert m["HF_16"] == pytest.approx(0, abs=1e-12) and m["HF_32"] == pytest.approx(0, abs=1e-12)
    assert m["loss_fraction_error"] == 0
    np.testing.assert_allclose(m["ED"], np.sqrt(2 / 200))


def test_heat_flux_error_of_a_single_hot_spot():
    """Every particle lost at once into one bin, predicted as no loss at all:
    HF^2 = (boundary area) / (nfp * bin area)."""
    m = 16
    t = torus_target(np.full(N_LAUNCHED, 0.01), np.full(N_LAUNCHED, 0.01), np.full(N_LAUNCHED, 3.5))
    area = heat_flux._period_bin_areas(t["R"], t["Z"], t["nu"], NFP, m)
    empty = np.array([])
    hf = heat_flux._heat_flux_error(empty, empty, empty, 0.0, t, m)
    np.testing.assert_allclose(hf, np.sqrt(area.sum() / area[0, 0]))


def test_energy_distance_needs_predicted_points():
    """With enough true losses for an ED, an empty prediction is an error, not a NaN to skip."""
    t = random_target(50)
    empty = np.array([])
    with pytest.raises(ValueError):
        heat_flux._sample_metrics(empty, empty, empty, 0.0, t)


def test_energy_distance_grows_with_displacement():
    """A band of strike points displaced farther from the truth has a larger energy distance."""
    t = random_target(300)
    eds = [heat_flux._sample_metrics(t["zeta"], t["theta"] + shift, t["energy"], t["loss_fraction"], t)["ED"]
           for shift in (0.0, 0.3, 1.0)]
    assert eds[0] < eds[1] < eds[2]


def test_few_losses_have_no_energy_distance():
    """ED is NaN below MIN_LOST_ED lost particles, and the summary skips it."""
    few, many = random_target(5), random_target(50)
    results = [heat_flux._sample_metrics(t["zeta"], t["theta"], t["energy"], t["loss_fraction"], t)
               for t in (few, many)]
    assert np.isnan(results[0]["ED"])
    summary = heat_flux._summarize(results)
    assert summary["ED_median"] == pytest.approx(results[1]["ED"])
    assert summary["HF_16_mean"] == pytest.approx(0, abs=1e-12)


def test_metrics_of_perfect_predictions(confinement_file):
    """The true strike points score HF 0 and no loss-fraction error on every sample; with
    only two lost particles per sample, no sample has an energy distance.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    summary = bb.heat_flux_metrics([confinement_file], true_predictions(confinement_file))
    assert summary["n_samples"] == 6
    for key in ("HF_16_median", "HF_16_mean", "HF_32_median", "HF_32_mean",
                "loss_fraction_median_abs_error", "loss_fraction_mean_abs_error"):
        assert summary[key] == pytest.approx(0, abs=1e-12)
    assert np.isnan(summary["ED_median"])


def test_metrics_match_sample_by_sample(confinement_file):
    """Reading the file in blocks gives the same scores as scoring each record on its own.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    rng = np.random.default_rng(0)
    predictions = {key: (z + 0.3, t, 0.5 * e, f + 0.1 * rng.standard_normal())
                   for key, (z, t, e, f) in true_predictions(confinement_file).items()}
    results = []
    with h5py.File(confinement_file, "r") as f:
        for i in range(len(f["uuid"])):
            record = heat_flux._with_boundary(bb.load_record(f, i))
            for s0 in (0.1, 0.25, 0.5):
                t = heat_flux._target(record, s0)
                results.append(heat_flux._sample_metrics(*predictions[record["uuid"], s0], t))
    expected = heat_flux._summarize(results)
    summary = bb.heat_flux_metrics([confinement_file], predictions)
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, nan_ok=True)


def test_metrics_of_some_samples(confinement_file):
    """Only the predicted samples are scored, and a prediction for an unknown sample is an error.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    predictions = true_predictions(confinement_file)
    some = {key: value for key, value in predictions.items() if key[0] == "config-1"}
    assert bb.heat_flux_metrics([confinement_file], some)["n_samples"] == 3
    with pytest.raises(KeyError):
        bb.heat_flux_metrics([confinement_file], {**some, ("no-such-uuid", 0.1): some["config-1", 0.1]})


def many_losses_record(n_lost=200, seed=0):
    """A torus configuration whose particles born on s = 0.1 are half lost, in a band around theta = 3.

    Args:
        n_lost (int): number of lost particles.
        seed (int): random seed.

    Returns:
        dict: a record with the fields read by ``lost_particles`` and ``binned_heat_flux``.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_lost
    return {**torus_fields(), "t_max": T_MAX, "s_init": np.full(n, 0.1),
            "confinement_time": np.r_[rng.uniform(1e-4, 0.05, n_lost), np.full(n_lost, T_MAX)],
            "zeta_end": rng.uniform(0, 2 * np.pi, n), "theta_end": 3 + 0.2 * rng.standard_normal(n)}


def test_standalone_metrics_of_the_truth():
    """Predicting the true lost particles gives HF = 0 and ED = sqrt(2 / n_lost).

    The ED value is the energy distance of a cloud with itself, whose self terms skip
    the zero distance of each point to itself.
    """
    record = many_losses_record(n_lost=200)
    zeta, theta, energy, n_launched = bb.lost_particles(
        record["confinement_time"], record["t_max"], record["s_init"],
        record["zeta_end"], record["theta_end"], 0.1)
    prediction = (zeta, theta, energy, len(zeta) / n_launched)
    assert bb.heat_flux_error(prediction, record, 0.1, resolution=16) == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(bb.energy_distance(prediction, record, 0.1), np.sqrt(2 / 200))
    assert bb.energy_distance((zeta, theta), record, 0.1) == bb.energy_distance(prediction, record, 0.1)


def test_standalone_metrics_match_the_benchmark(confinement_file):
    """heat_flux_error and energy_distance give the per-sample scores used by heat_flux_metrics,
    and ED is NaN with fewer than 10 lost particles.

    Args:
        confinement_file: fixture, see conftest.py.
    """
    with h5py.File(confinement_file, "r") as f:
        record = bb.load_record(f, 0)
    zeta, theta, energy, n_launched = bb.lost_particles(
        record["confinement_time"], record["t_max"], record["s_init"],
        record["zeta_end"], record["theta_end"], 0.25)
    prediction = (zeta + 0.4, theta, 0.5 * energy, 0.3)
    expected = heat_flux._sample_metrics(*prediction, heat_flux._target(heat_flux._with_boundary(record), 0.25))
    assert bb.heat_flux_error(prediction, record, 0.25, resolution=32) == pytest.approx(expected["HF_32"])
    assert np.isnan(bb.energy_distance(prediction, record, 0.25))
