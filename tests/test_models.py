"""Tests of the model code on randomly initialized models; the trained weights are not needed."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from boss_bench import models  # noqa: E402
from boss_bench.models import _common, heat_flux, inverse_design, loss_fraction  # noqa: E402
from conftest import N_ANGLE, NFP, NS, torus_fields  # noqa: E402


@pytest.fixture(scope="module")
def random_models():
    """One model of each kind, randomly initialized, in evaluation mode.

    Returns:
        dict: maps each model name to its model.
    """
    torch.manual_seed(0)
    return {name: cls().eval() for name, cls in models.MODELS.items()}


def torus_record():
    """The fields of the circular torus that the models read.

    Returns:
        dict: the fields of one configuration.
    """
    record = torus_fields()
    s = record["s1d"]
    record.update(pressure=1e5 * (1 - s), d_pressure_d_s=np.full(NS, -1e5), toroidal_current=np.zeros(NS),
                  volume_avg_modB=5.0, minor_radius=1.0, toroidal_flux_edge=1.5, aspect_ratio=3.0, min_modB=4.0,
                  mirror_ratio=1.5, vmec_ncurr=1, loss_fraction=0.1)
    return record


def stack(record, n):
    """A batch of n copies of one configuration.

    Args:
        record (dict): the fields of one configuration.
        n (int): the number of copies.

    Returns:
        dict: the fields, with one row per copy.
    """
    return {name: np.stack([np.asarray(value)] * n) for name, value in record.items()}


def test_canonicalize_theta():
    """Shifting or reversing the poloidal angle of the torus is undone."""
    xyz = torch.as_tensor(torus_fields()["boundary_xyz"]).reshape(1, N_ANGLE, N_ANGLE, 3)
    shifted = torch.roll(xyz, 7, dims=2)
    reversed_ = torch.roll(torch.flip(xyz, dims=[2]), 3, dims=2)
    for x in (xyz, shifted, reversed_):
        assert torch.equal(loss_fraction._canonicalize_theta(x), xyz)


def test_loss_fraction_predict(random_models):
    """The prediction is in [0, 1], the same alone, in a batch and on tensors, and independent of the theta origin.

    Args:
        random_models: fixture, the models.
    """
    model = random_models["loss_fraction"]
    record = torus_record()
    single = model.predict(record)
    assert isinstance(single, float) and 0 <= single <= 1
    np.testing.assert_allclose(model.predict(stack(record, 3)), single, rtol=1e-6)
    xyz = record["boundary_xyz"].reshape(N_ANGLE, N_ANGLE, 3)
    shifted = dict(record, boundary_xyz=np.roll(xyz[:, ::-1], 5, axis=1).ravel())
    assert model.predict(shifted) == pytest.approx(single, rel=1e-6)

    # called on tensors, the prediction is the same and has gradients
    tensors = {name: torch.as_tensor(stack(record, 1)[name]) for name in loss_fraction.FIELDS}
    tensors["boundary_xyz"].requires_grad_(True)
    f = model(**tensors)
    assert f.item() == pytest.approx(single, rel=1e-6)
    f.sum().backward()
    assert tensors["boundary_xyz"].grad.shape == (1, N_ANGLE * N_ANGLE * 3)
    assert torch.all(torch.isfinite(tensors["boundary_xyz"].grad))


def test_loss_fraction_invalid_inputs(random_models):
    """Missing fields, NaN and an out-of-range nfp raise errors.

    Args:
        random_models: fixture, the models.
    """
    model = random_models["loss_fraction"]
    record = torus_record()
    with pytest.raises(KeyError):
        model.predict({name: value for name, value in record.items() if name != "pressure"})
    with pytest.raises(ValueError):
        model.predict(dict(record, volume_avg_modB=np.nan))
    with pytest.raises(ValueError):
        model.predict(dict(record, nfp=13))


def test_heat_flux_sample(random_models, monkeypatch):
    """There are max(1, round(5000 f)) strike points, with angles in [0, 2 pi) and positive energies, reproducibly.

    Args:
        random_models: fixture, the models.
        monkeypatch: pytest's monkeypatch fixture.
    """
    model = random_models["heat_flux"]
    monkeypatch.setattr(heat_flux, "N_STEPS", 2)
    record = torus_record()
    zeta, theta, energy, f = model.sample(record, s0=0.25, seed=0)
    assert f == pytest.approx(model.predict_loss_fraction(record, s0=0.25))
    assert len(zeta) == len(theta) == len(energy) == max(1, round(f * 5000))
    assert np.all((zeta >= 0) & (zeta < 2 * np.pi) & (theta >= 0) & (theta < 2 * np.pi) & (energy > 0))
    again = model.sample(record, s0=0.25, seed=0)
    np.testing.assert_array_equal(again[0], zeta)

    # integrated one configuration at a time, the batch gives one sample per configuration
    monkeypatch.setattr(heat_flux, "MAX_POINTS", len(zeta))
    samples = model.sample(stack(record, 3), s0=0.25, seed=0)
    assert len(samples) == 3
    assert all(len(s[0]) == len(zeta) and s[3] == pytest.approx(f) for s in samples)


def test_inverse_design_sample(random_models, monkeypatch):
    """A design is a boundary laid out as boundary_xyz, on the toroidal angles of one field period.

    Args:
        random_models: fixture, the models.
        monkeypatch: pytest's monkeypatch fixture.
    """
    model = random_models["inverse_design"]
    monkeypatch.setattr(inverse_design, "N_STEPS", 2)
    record = torus_fields()
    xyz = record["boundary_xyz"].reshape(N_ANGLE, N_ANGLE, 3)
    rz = np.stack([np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]], axis=-1)
    monkeypatch.setattr(model, "design_mean", torch.as_tensor(rz.ravel(), dtype=torch.float32))  # near the torus
    monkeypatch.setattr(model, "design_std", torch.full_like(model.design_std, 1e-3))
    conditions = {name: torus_record()[name] for name in inverse_design.CONDITIONS}

    design = model.sample(conditions, seed=0)
    assert design.shape == (N_ANGLE * N_ANGLE * 3,)
    out = design.reshape(N_ANGLE, N_ANGLE, 3)
    phi = np.mod(np.arctan2(out[..., 1], out[..., 0]), 2 * np.pi)
    np.testing.assert_allclose(phi, np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2 * np.pi), atol=1e-9)
    np.testing.assert_allclose(out, xyz, atol=0.1)
    monkeypatch.setattr(inverse_design, "BATCH_SIZE", 1)
    assert model.sample(stack(conditions, 2), seed=0).shape == (2, N_ANGLE * N_ANGLE * 3)

    for nfp in (0, 7, 2.5):
        with pytest.raises(ValueError):
            model.sample(dict(conditions, nfp=nfp))


def test_save_and_load(random_models, tmp_path):
    """A saved model loads with the same weights and statistics, and only into its own class.

    Args:
        random_models: fixture, the models.
        tmp_path: pytest's temporary directory.
    """
    for name, model in random_models.items():
        state = {key: value.clone() for key, value in model.state_dict().items()}
        for key, value in state.items():
            if value.is_floating_point() and value.dim() == 1:
                value.add_(0.5)  # statistics and biases differ from a fresh model's
        model.load_state_dict(state)
        models._save(model, tmp_path / name, description="test")
        loaded = type(model).load(tmp_path / name)
        assert not loaded.training
        for key, value in state.items():
            assert torch.equal(loaded.state_dict()[key], value), (name, key)
    with pytest.raises(ValueError):
        models.HeatFluxModel.load(tmp_path / "loss_fraction")


def test_integrate():
    """Euler and midpoint steps on dx/dt = -x match their closed forms."""
    x0 = torch.ones(1, dtype=torch.float64)
    h, n = 0.25, 4
    euler = _common.integrate(lambda x, t: -x, x0, 0.0, 1.0, n, "euler")
    midpoint = _common.integrate(lambda x, t: -x, x0, 0.0, 1.0, n, "midpoint")
    assert euler.item() == pytest.approx((1 - h) ** n)
    assert midpoint.item() == pytest.approx((1 - h + h ** 2 / 2) ** n)
    with pytest.raises(ValueError):
        _common.integrate(lambda x, t: -x, x0, 0.0, 1.0, n, "rk4")
    assert NFP == 2  # the toroidal angles above assume the conftest torus


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the download from Hugging Face by writing the URL to the file.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        list: the URLs downloaded, in order.
    """
    urls = []

    def retrieve(url, path):
        urls.append(url)
        with open(path, "w") as f:
            f.write(url)

    monkeypatch.setattr(models.urllib.request, "urlretrieve", retrieve)
    return urls


def test_download(tmp_path, fake_download):
    """The files of the model are saved in the folder, and only once.

    Args:
        tmp_path: pytest's temporary directory.
        fake_download: fixture, the URLs downloaded.
    """
    path = models.download("heat_flux", tmp_path / "heat_flux")
    assert path == tmp_path / "heat_flux"
    assert sorted(os.listdir(path)) == ["config.json", "model.safetensors"]
    assert fake_download == [f"https://huggingface.co/mishapadidar/boss/resolve/main/heat_flux/{file}"
                             for file in ("config.json", "model.safetensors")]
    models.download("heat_flux", path)
    assert len(fake_download) == 2  # the files were already there
    with pytest.raises(ValueError):
        models.download("pointnet", tmp_path / "pointnet")
    assert len(fake_download) == 2
