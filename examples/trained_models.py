"""Run the trained models of the paper on test configurations.

Predicts loss fractions with the surrogate, scores heat-flux samples with the benchmark,
and generates boundaries with a lower loss fraction than a test configuration's. The
models are downloaded from Hugging Face into MODELS_DIR the first time.

    python trained_models.py DATA_DIR [MODELS_DIR]

DATA_DIR is a directory containing at least one BoSS confinement data file from the test split,
e.g. confinement_data_test_split_00000.hdf5. It is searched recursively. MODELS_DIR defaults
to BoSS-models. The models need PyTorch: pip install "boss-bench[models]".
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

import boss_bench as bb
from boss_bench import models
from boss_bench.models import HeatFluxModel, InverseDesignModel, LossFractionModel

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
models_dir = sys.argv[2] if len(sys.argv) > 2 else "BoSS-models"
paths = bb.list_files(data_dir, "confinement", "test")[:1]
if not paths:
    sys.exit(f"No confinement test files found in {data_dir}. Download the BoSS data first.")
n_configurations = 100

# the fields the three models read, and the tracing outcomes to compare with
fields = {"uuid", "confinement_time", "t_max", "s_init", *models.loss_fraction.FIELDS, *models.heat_flux.FIELDS,
          *models.inverse_design.CONDITIONS} - {"loss_fraction"}
with h5py.File(paths[0], "r") as f:
    records = [bb.load_record(f, i, sorted(fields)) for i in range(n_configurations)]
data = {name: np.stack([r[name] for r in records]) for name in fields}
true_loss_fraction = bb.loss_fraction(data["confinement_time"], data["t_max"], data["s_init"], s0=0.1)

# loss-fraction surrogate: no VMEC run needed
surrogate = LossFractionModel.load(models.download("loss_fraction", os.path.join(models_dir, "loss_fraction")))
predicted = surrogate.predict(data)
print(f"loss-fraction surrogate, {n_configurations} configurations: "
      f"mean absolute error {np.mean(np.abs(predicted - true_loss_fraction)):.3f}")

# heat-flux model: one sample of strike points per configuration and birth surface, scored
heat_flux = HeatFluxModel.load(models.download("heat_flux", os.path.join(models_dir, "heat_flux")))
predictions = {}
for s0 in (0.1, 0.25, 0.5):
    samples = heat_flux.sample({name: data[name][:10] for name in fields}, s0=s0, seed=0)
    for uuid, sample in zip(data["uuid"][:10], samples):
        predictions[uuid, s0] = sample
scores = bb.heat_flux_metrics(paths, predictions)
print("heat-flux model, 10 configurations:")
for name, value in scores.items():
    print(f"  {name:32s} {value:.4g}")

# inverse design: three boundaries like configuration 0, but with a tenth of its loss fraction
generator = InverseDesignModel.load(models.download("inverse_design", os.path.join(models_dir, "inverse_design")))
conditions = {name: np.repeat(data[name][:1], 3, axis=0) for name in models.inverse_design.CONDITIONS
              if name != "loss_fraction"}
conditions["loss_fraction"] = np.full(3, 0.1 * true_loss_fraction[0])
designs = generator.sample(conditions, seed=0)

# cross sections at the start and the middle of a field period, of configuration 0 and the designs
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, i_phi, title in zip(axes, (0, 20), ("phi = 0", "half a field period")):
    for k, xyz in enumerate([data["boundary_xyz"][0], *designs]):
        xyz = xyz.reshape(41, 41, 3)[i_phi]
        R, Z = np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]
        ax.plot(np.append(R, R[0]), np.append(Z, Z[0]), "k" if k == 0 else f"C{k}",
                label=f"configuration 0, loss fraction {true_loss_fraction[0]:.2f}" if k == 0 else f"design {k}")
    ax.set(title=title, xlabel="R [m]", ylabel="Z [m]", aspect="equal")
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()
