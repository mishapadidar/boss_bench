"""Trained models from the BoSS paper, in PyTorch."""

import json
import os
import urllib.request

try:
    import torch  # noqa: F401
    from safetensors.torch import load_file, save_file
except ImportError as e:
    raise ImportError('boss_bench.models needs PyTorch and safetensors: pip install "boss-bench[models]"') from e

from .heat_flux import HeatFluxModel
from .inverse_design import InverseDesignModel
from .loss_fraction import LossFractionModel

MODELS = {"loss_fraction": LossFractionModel, "heat_flux": HeatFluxModel, "inverse_design": InverseDesignModel}
HF_REPO = "mishapadidar/boss"  # the Hugging Face repository of the trained models


def download(name, path):
    """Download a trained PyTorch model of the BoSS benchmark from Hugging Face.

    Three models are available:

    - ``"loss_fraction"`` (``LossFractionModel``) predicts the alpha-particle loss
      fraction of a configuration from its boundary and profiles, without VMEC.
    - ``"heat_flux"`` (``HeatFluxModel``) samples where the lost alpha particles strike
      the boundary, as in the heat-flux benchmark.
    - ``"inverse_design"`` (``InverseDesignModel``) generates boundaries with a chosen
      loss fraction, shape, field and profiles.

    This function downloads the model's ``config.json`` and ``model.safetensors`` into the
    folder ``path``, which ``load`` then reads. Files already in the folder are not
    downloaded again. The ONNX files of the models are not downloaded; get them from
    Hugging Face directly.

    Args:
        name (str): ``"loss_fraction"``, ``"heat_flux"`` or ``"inverse_design"``.
        path (str): the folder to download the model into; it is created if needed.

    Returns:
        str: ``path``.

    Raises:
        ValueError: if the model name is unknown.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    os.makedirs(path, exist_ok=True)
    for file in ("config.json", "model.safetensors"):
        target = os.path.join(path, file)
        if not os.path.exists(target):
            urllib.request.urlretrieve(f"https://huggingface.co/{HF_REPO}/resolve/main/{name}/{file}", target + ".part")
            os.replace(target + ".part", target)  # a partial download never looks complete
    return path


def load(path, device="cpu"):
    """Load a trained model from its folder.

    Args:
        path (str): the folder, holding ``config.json`` and ``model.safetensors``, e.g.
            from ``download``.
        device (str or torch.device): the device to put the model on.

    Returns:
        torch.nn.Module: the model, in evaluation mode: a ``LossFractionModel``,
        ``HeatFluxModel`` or ``InverseDesignModel``.

    Raises:
        ValueError: if the folder holds an unknown kind of model.
    """
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    if config["model"] not in MODELS:
        raise ValueError(f"unknown model {config['model']!r} in {path}")
    model = MODELS[config["model"]](**config["architecture"])
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
    return model.to(device).eval()


def _save(model, path, **metadata):
    """Save a model to a folder that ``load`` reads.

    Args:
        model (torch.nn.Module): the model.
        path (str): the folder; it is created if needed.
        **metadata: further entries of ``config.json``, such as the model's provenance.
    """
    name = next(key for key, cls in MODELS.items() if isinstance(model, cls))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model": name, "architecture": model.config, **metadata}, f, indent=2)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"))
