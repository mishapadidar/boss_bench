"""Run the trained models with ONNX Runtime, without PyTorch.

Predicts the loss fraction of a test configuration with the surrogate, and samples the
strike points of its lost particles with the heat-flux model, integrating the flow with
numpy. The inputs of the ONNX files are BoSS fields, float64 except int64 nfp. Download
the two models from Hugging Face into MODELS_DIR first (pip install huggingface_hub):

    hf download mishapadidar/boss --include "loss_fraction/*" --include "heat_flux/*" --local-dir BoSS-models
    python onnx_models.py DATA_DIR [MODELS_DIR]

DATA_DIR is a directory containing at least one BoSS confinement data file from the test split,
e.g. confinement_data_test_split_00000.hdf5. It is searched recursively. MODELS_DIR defaults
to BoSS-models. Needs ONNX Runtime: pip install onnxruntime.
"""

import json
import os
import sys

import h5py
import numpy as np
import onnxruntime as ort

import boss_bench as bb

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
models_dir = sys.argv[2] if len(sys.argv) > 2 else "BoSS-models"
paths = bb.list_files(data_dir, "confinement", "test")[:1]
if not paths:
    sys.exit(f"No confinement test files found in {data_dir}. Download the BoSS data first.")
for name in ("loss_fraction", "heat_flux"):
    if not os.path.isdir(os.path.join(models_dir, name)):
        sys.exit(f"No {name} model in {models_dir}. Download it first: hf download mishapadidar/boss "
                 f'--include "{name}/*" --local-dir {models_dir}')
with h5py.File(paths[0], "r") as f:
    record = bb.load_record(f, 0)


def run(session, **inputs):
    """Run an ONNX model on inputs given by name, as batches of one configuration."""
    dtypes = {"tensor(double)": np.float64, "tensor(float)": np.float32, "tensor(int64)": np.int64}
    feed = {i.name: np.asarray(inputs[i.name], dtype=dtypes[i.type]) for i in session.get_inputs()}
    return session.run(None, feed)


one = {name: np.asarray(value)[None] for name, value in record.items()}  # a batch of one configuration

# loss-fraction surrogate
folder = os.path.join(models_dir, "loss_fraction")
(loss_fraction,) = run(ort.InferenceSession(os.path.join(folder, "model.onnx")), **one)
true = bb.loss_fraction(record["confinement_time"], record["t_max"], record["s_init"], s0=0.1)
print(f"loss fraction at s0 = 0.1: predicted {loss_fraction[0]:.3f}, true {true:.3f}")

# heat-flux model: encode the configuration, integrate the flow, decode the strike points
folder = os.path.join(models_dir, "heat_flux")
encoder, velocity, decoder = (ort.InferenceSession(os.path.join(folder, name))
                              for name in ("encoder.onnx", "velocity.onnx", "decoder.onnx"))
with open(os.path.join(folder, "config.json")) as f:
    sampling = json.load(f)["onnx"]["velocity.onnx"]["sampling"]  # t0, t1, n_steps, method

s0 = 0.1
embedding, f_pred = run(encoder, **one, s0=[s0])
n_points = max(1, round(f_pred[0] * 5000))  # the predicted number of lost particles, of 5000 launched
x = np.random.default_rng(0).standard_normal((n_points, 5)).astype(np.float32)
embedding = np.repeat(embedding, n_points, axis=0)
dt = (sampling["t1"] - sampling["t0"]) / sampling["n_steps"]
for i in range(sampling["n_steps"]):  # Euler steps
    (v,) = run(velocity, x=x, t=[sampling["t0"] + i * dt], embedding=embedding)
    x = x + dt * v
zeta, theta, energy = run(decoder, x=x)

prediction = (zeta, theta, energy, float(f_pred[0]))
print(f"heat-flux model at s0 = {s0}: {n_points} strike points, predicted loss fraction {f_pred[0]:.3f}, "
      f"heat-flux error (32 x 32) {bb.heat_flux_error(prediction, record, s0):.3f}")
