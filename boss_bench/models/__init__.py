"""Trained models from the BoSS paper, in PyTorch."""

import json
import os
import urllib.request

try:
    import torch  # noqa: F401
    from safetensors.torch import save_file
except ImportError as e:
    raise ImportError('boss_bench.models needs PyTorch and safetensors: pip install "boss-bench[models]"') from e

from .heat_flux import HeatFluxModel
from .inverse_design import InverseDesignModel
from .loss_fraction import LossFractionModel

MODELS = {"loss_fraction": LossFractionModel, "heat_flux": HeatFluxModel, "inverse_design": InverseDesignModel}
HF_REPO = "mishapadidar/boss"  # the Hugging Face repository of the trained models
ONNX_FILES = {"loss_fraction": ("model.onnx",), "heat_flux": ("encoder.onnx", "velocity.onnx", "decoder.onnx"),
              "inverse_design": ("encoder.onnx", "velocity.onnx", "decoder.onnx")}


def download(name, path, onnx=False):
    """Download a trained PyTorch or ONNX model baseline from Hugging Face.

    Three models are available:

    - ``"loss_fraction"`` (``LossFractionModel``), a neural surrogate which predicts the
      loss fraction of the alpha particles born on :math:`s_0 = 0.1` from the plasma
      boundary and profiles, without a VMEC run.
    - ``"heat_flux"`` (``HeatFluxModel``), a flow matching model which samples the
      locations at which lost fast ions strike the boundary.
    - ``"inverse_design"`` (``InverseDesignModel``), a flow matching model which generates
      plasma boundaries with a desired loss fraction, shape, field and profiles.

    This function downloads the model's ``config.json`` and ``model.safetensors`` into the
    folder ``path``, which the model's ``load`` then reads, e.g. ``HeatFluxModel.load(path)``.
    With ``onnx=True``, it also downloads the model's ONNX export, to run with ONNX Runtime:
    ``model.onnx``, or ``encoder.onnx``, ``velocity.onnx`` and ``decoder.onnx`` for the two
    flow matching models; ``config.json`` describes their inputs. Files already in the
    folder are not downloaded again.

    Args:
        name (str): ``"loss_fraction"``, ``"heat_flux"`` or ``"inverse_design"``.
        path (str): the folder to download the model into; it is created if needed.
        onnx (bool): whether to also download the ONNX files.

    Returns:
        str: ``path``.

    Raises:
        ValueError: if the model name is unknown.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    os.makedirs(path, exist_ok=True)
    for file in ("config.json", "model.safetensors", *(ONNX_FILES[name] if onnx else ())):
        target = os.path.join(path, file)
        if not os.path.exists(target):
            urllib.request.urlretrieve(f"https://huggingface.co/{HF_REPO}/resolve/main/{name}/{file}", target + ".part")
            os.replace(target + ".part", target)  # a partial download never looks complete
    return path


def _save(model, path, **metadata):
    """Save a model to a folder that its ``load`` reads.

    Args:
        model (torch.nn.Module): the model.
        path (str): the folder; it is created if needed.
        **metadata: further entries of ``config.json``, such as the model's provenance.
    """
    name = next(key for key, cls in MODELS.items() if isinstance(model, cls))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model": name, **metadata}, f, indent=2)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"))
