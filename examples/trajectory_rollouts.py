"""Load the orbits of one test configuration, plot predicted against true rollouts, and score them.

The "model" is made up: it predicts that particles do not move and are never lost.

    python trajectory_rollouts.py DATA_DIR

DATA_DIR is a directory containing at least one BoSS trajectory data file from the test split,
e.g. trajectory_data_test_split_00000.hdf5. It is searched recursively.
"""

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

import boss_bench as bb
from boss_bench.trajectory import TABLE_STEPS

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
paths = bb.list_files(data_dir, "trajectory", "test")
if not paths:
    sys.exit(f"No trajectory test files found in {data_dir}. Download the BoSS data first.")
path = paths[0]
with h5py.File(path, "r") as f:
    uuid = f["uuid"].asstr()[0]
    truth = bb.read_trajectories(f, 0)
z_true = truth["z"]  # (n_orbits, 1000, 5), row k at lead time k us
print(f"configuration {uuid}: {len(z_true)} orbits, "
      f"{np.isfinite(truth['t_loss']).mean():.3f} of them lost within 1 ms")

# the prediction: every state stays at the first sample (z0), and no particle is lost
z_pred = np.repeat(z_true[:, :1], z_true.shape[1], axis=1)
t_loss_pred = np.full(len(z_true), np.nan)

# plot a few orbits that have an initial state
orbits = np.flatnonzero(truth["n_valid"] >= 1)[:3]
bb.plot_rollouts(z_true[orbits], z_pred[orbits])

# score the predicted states at the scored lead times (10, 100 and 999 us) and loss times
predictions = {uuid: (np.nan_to_num(z_pred[:, TABLE_STEPS]), t_loss_pred)}
for name, value in bb.trajectory_metrics([path], predictions).items():
    print(f"{name:30s} {value:.4g}")
plt.show()
