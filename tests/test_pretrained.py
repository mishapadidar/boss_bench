"""Tests of downloading the pretrained models, without the network."""

import os

import pytest

import boss_bench as bb
from boss_bench import pretrained


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the download by writing the URL to the file.

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

    monkeypatch.setattr(pretrained.urllib.request, "urlretrieve", retrieve)
    return urls


def test_download_model(tmp_path, fake_download):
    """The files of the model are saved in its folder, and only once.

    Args:
        tmp_path: pytest's temporary directory.
        fake_download: fixture, the URLs downloaded.
    """
    folder = bb.download_model("heat_flux", tmp_path)
    assert folder == os.path.join(tmp_path, "heat_flux")
    assert sorted(os.listdir(folder)) == ["config.json", "model.safetensors"]
    assert fake_download[0] == "https://huggingface.co/mishapadidar/boss/resolve/main/heat_flux/config.json"

    bb.download_model("heat_flux", tmp_path, onnx=True, revision="v1")
    assert sorted(os.listdir(folder)) == ["config.json", "decoder.onnx", "encoder.onnx", "model.safetensors",
                                          "velocity.onnx"]
    assert len(fake_download) == 5  # config.json and model.safetensors were already there
    assert fake_download[-1].endswith("/resolve/v1/heat_flux/decoder.onnx")


def test_download_unknown_model(tmp_path, fake_download):
    """An unknown model name raises an error before anything is downloaded.

    Args:
        tmp_path: pytest's temporary directory.
        fake_download: fixture, the URLs downloaded.
    """
    with pytest.raises(ValueError):
        bb.download_model("pointnet", tmp_path)
    assert fake_download == []
