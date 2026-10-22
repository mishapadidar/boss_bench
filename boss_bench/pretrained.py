"""Download the pretrained models of the paper from Hugging Face."""

import os
import urllib.request

HF_REPO = "mishapadidar/boss"
MODEL_FILES = {  # the PyTorch files of each model, then its ONNX files
    "loss_fraction": (("config.json", "model.safetensors"), ("model.onnx",)),
    "heat_flux": (("config.json", "model.safetensors"), ("encoder.onnx", "velocity.onnx", "decoder.onnx")),
    "inverse_design": (("config.json", "model.safetensors"), ("encoder.onnx", "velocity.onnx", "decoder.onnx")),
}


def download_model(name, models_dir, onnx=False, revision="main"):
    """Download a pretrained model from the BoSS repository on Hugging Face.

    The files are saved in ``models_dir/name``; files already there are not downloaded
    again. Load the PyTorch model with ``models.load`` on the returned folder. The ONNX
    files, ``model.onnx`` or ``encoder.onnx``, ``velocity.onnx`` and ``decoder.onnx``, run
    without PyTorch, for example with onnxruntime; ``config.json`` describes their inputs.

    Args:
        name (str): ``"loss_fraction"``, ``"heat_flux"`` or ``"inverse_design"``.
        models_dir (str): the folder to save the models in.
        onnx (bool): whether to also download the ONNX files.
        revision (str): the branch, tag or commit of the Hugging Face repository.

    Returns:
        str: the folder of the model, ``models_dir/name``.

    Raises:
        ValueError: if the model name is unknown.
    """
    if name not in MODEL_FILES:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_FILES)}")
    torch_files, onnx_files = MODEL_FILES[name]
    folder = os.path.join(models_dir, name)
    os.makedirs(folder, exist_ok=True)
    for file in torch_files + (onnx_files if onnx else ()):
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            url = f"https://huggingface.co/{HF_REPO}/resolve/{revision}/{name}/{file}"
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)  # a partial download never looks complete
    return folder
