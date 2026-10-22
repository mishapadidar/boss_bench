"""Pretrained models from the BoSS paper, in PyTorch.

- ``LossFractionModel`` predicts the alpha-particle loss fraction of a configuration from
  its boundary and profiles, without VMEC.
- ``HeatFluxModel`` samples where the lost alpha particles strike the boundary, as in the
  heat-flux benchmark.
- ``InverseDesignModel`` generates boundaries with a chosen loss fraction, shape, field
  and profiles.

Each pretrained model is a folder holding ``config.json`` and ``model.safetensors``:
download it with ``boss_bench.download_model`` and load it with ``load``. The models need
PyTorch and safetensors: ``pip install "boss-bench[models]"``.
"""

import json
import os

try:
    import torch  # noqa: F401
    from safetensors.torch import load_file, save_file
except ImportError as e:
    raise ImportError('boss_bench.models needs PyTorch and safetensors: pip install "boss-bench[models]"') from e

from .heat_flux import HeatFluxModel
from .inverse_design import InverseDesignModel
from .loss_fraction import LossFractionModel

MODELS = {"loss_fraction": LossFractionModel, "heat_flux": HeatFluxModel, "inverse_design": InverseDesignModel}


def load(path, device="cpu"):
    """Load a pretrained model from its folder.

    Args:
        path (str): the folder, holding ``config.json`` and ``model.safetensors``, e.g.
            from ``boss_bench.download_model``.
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
