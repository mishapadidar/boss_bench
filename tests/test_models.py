"""Tests of the model code on small, randomly initialized models; the pretrained weights are not needed."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from boss_bench import models  # noqa: E402
from boss_bench.models import _common, inverse_design, loss_fraction  # noqa: E402
from conftest import N_ANGLE, NFP, NS, torus_fields  # noqa: E402


def small_models():
    """One small model of each kind, randomly initialized, in evaluation mode.

    Returns:
        dict: maps each model name to its model.
    """
    torch.manual_seed(0)
    return {
        "loss_fraction": models.LossFractionModel(conv_channels=(4, 8), surf_dim=8, profile_dim=4, scalar_dim=4,
                                                  hidden_dim=16, n_layers=1).eval(),
        "heat_flux": models.HeatFluxModel(embed_dim=16, hidden_dim=16, n_blocks=1).eval(),
        "inverse_design": models.InverseDesignModel(base_ch=8, ch_mult=(1, 2), num_res_blocks=1, attn_levels=(1,),
                                                    num_heads=2, time_dim=8).eval(),
    }


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


def test_loss_fraction_predict():
    """The prediction lies in [0, 1], is the same alone and in a batch, and does not depend on the theta origin."""
    model = small_models()["loss_fraction"]
    record = torus_record()
    single = model.predict(record)
    assert isinstance(single, float) and 0 <= single <= 1
    np.testing.assert_allclose(model.predict(stack(record, 3), batch_size=2), single, rtol=1e-6)
    xyz = record["boundary_xyz"].reshape(N_ANGLE, N_ANGLE, 3)
    shifted = dict(record, boundary_xyz=np.roll(xyz[:, ::-1], 5, axis=1).ravel())
    assert model.predict(shifted) == pytest.approx(single, rel=1e-6)


def test_loss_fraction_invalid_inputs():
    """Missing fields, NaN and an out-of-range nfp raise errors."""
    model = small_models()["loss_fraction"]
    record = torus_record()
    with pytest.raises(KeyError):
        model.predict({name: value for name, value in record.items() if name != "pressure"})
    with pytest.raises(ValueError):
        model.predict(dict(record, volume_avg_modB=np.nan))
    with pytest.raises(ValueError):
        model.predict(dict(record, nfp=13))


def test_heat_flux_sample():
    """There are max(1, round(f N)) strike points, with angles in [0, 2 pi) and positive energies, reproducibly."""
    model = small_models()["heat_flux"]
    record = torus_record()
    zeta, theta, energy, f = model.sample(record, s0=0.25, n_launched=100, n_steps=2, seed=0)
    assert f == pytest.approx(model.predict_loss_fraction(record, s0=0.25))
    assert len(zeta) == len(theta) == len(energy) == max(1, round(f * 100))
    assert np.all((zeta >= 0) & (zeta < 2 * np.pi) & (theta >= 0) & (theta < 2 * np.pi) & (energy > 0))
    again = model.sample(record, s0=0.25, n_launched=100, n_steps=2, seed=0)
    np.testing.assert_array_equal(again[0], zeta)

    # integrated a few configurations at a time, the batch gives one sample per configuration
    samples = model.sample(stack(record, 3), s0=0.25, n_launched=100, n_steps=2, seed=0, max_points=len(zeta))
    assert len(samples) == 3
    assert all(len(s[0]) == len(zeta) and s[3] == pytest.approx(f) for s in samples)


def test_inverse_design_sample():
    """A design is a boundary laid out as boundary_xyz, on the toroidal angles of one field period."""
    model = small_models()["inverse_design"]
    record = torus_fields()
    xyz = record["boundary_xyz"].reshape(N_ANGLE, N_ANGLE, 3)
    rz = np.stack([np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]], axis=-1)
    model.design_mean.copy_(torch.as_tensor(rz.ravel()))  # designs near the torus
    model.design_std.fill_(1e-3)
    conditions = {name: torus_record()[name] for name in inverse_design.CONDITIONS}

    design = model.sample(conditions, n_steps=2, seed=0)
    assert design.shape == (N_ANGLE * N_ANGLE * 3,)
    out = design.reshape(N_ANGLE, N_ANGLE, 3)
    phi = np.mod(np.arctan2(out[..., 1], out[..., 0]), 2 * np.pi)
    np.testing.assert_allclose(phi, np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2 * np.pi), atol=1e-9)
    np.testing.assert_allclose(out, xyz, atol=0.1)
    assert model.sample(stack(conditions, 2), n_steps=2, seed=0, batch_size=1).shape == (2, N_ANGLE * N_ANGLE * 3)

    for nfp in (0, 7, 2.5):
        with pytest.raises(ValueError):
            model.sample(dict(conditions, nfp=nfp), n_steps=2)


def test_save_and_load(tmp_path):
    """A saved model loads with the same weights and statistics.

    Args:
        tmp_path: pytest's temporary directory.
    """
    for name, model in small_models().items():
        with torch.no_grad():
            for buffer in model.buffers():
                if buffer.is_floating_point():
                    buffer.add_(0.5)
        models._save(model, tmp_path / name, description="test")
        loaded = models.load(tmp_path / name)
        assert type(loaded) is type(model) and not loaded.training
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value), (name, key)


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
    """The files of the model are saved in its folder, and only once.

    Args:
        tmp_path: pytest's temporary directory.
        fake_download: fixture, the URLs downloaded.
    """
    folder = models.download("heat_flux", tmp_path)
    assert folder == os.path.join(tmp_path, "heat_flux")
    assert sorted(os.listdir(folder)) == ["config.json", "model.safetensors"]
    assert fake_download == [f"https://huggingface.co/mishapadidar/boss/resolve/main/heat_flux/{file}"
                             for file in ("config.json", "model.safetensors")]
    models.download("heat_flux", tmp_path, revision="v1")
    assert len(fake_download) == 2  # the files were already there
    models.download("loss_fraction", tmp_path, revision="v1")
    assert fake_download[-1].endswith("/resolve/v1/loss_fraction/model.safetensors")
    with pytest.raises(ValueError):
        models.download("pointnet", tmp_path)
    assert len(fake_download) == 4
