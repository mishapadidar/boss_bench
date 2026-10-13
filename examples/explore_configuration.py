"""Explore one stellarator configuration: its shape, |B|, particle losses and heat flux.

Compares the true heat flux with a made-up prediction.

    python explore_configuration.py DATA_DIR

DATA_DIR is a directory containing at least one BoSS confinement data file from the test split,
e.g. confinement_data_test_split_00000.hdf5. It is searched recursively.
"""

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

import boss_bench as bb

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
paths = bb.list_files(data_dir, "confinement", "test")
if not paths:
    sys.exit(f"No confinement test files found in {data_dir}. Download the BoSS data first.")
path = paths[0]
with h5py.File(path, "r") as f:
    record = bb.load_record(f, 0)
nfp = int(record["nfp"])
print(f"configuration {record['uuid']} from {record['source_dataset']}, nfp = {nfp}")
for s0 in (0.1, 0.25, 0.5):
    print(f"  loss fraction of particles born on s = {s0}: "
          f"{bb.loss_fraction(record['confinement_time'], record['t_max'], record['s_init'], s0):.3f}")

# shape, |B| and losses
fig = plt.figure(figsize=(16, 4.5))
bb.plot_device(record, ax=fig.add_subplot(1, 3, 1, projection="3d"))
bb.plot_modB_boozer(record, s=0.5, ax=fig.add_subplot(1, 3, 2))
bb.plot_loss_fraction_curves(record, ax=fig.add_subplot(1, 3, 3))
fig.tight_layout()

# a made-up prediction: the true strike points, shifted and blurred, with the true loss fraction
rng = np.random.default_rng(0)
predictions = {}
for s0 in (0.1, 0.25, 0.5):
    zeta, theta, energy, n_launched = bb.lost_particles(record, s0)
    predictions[s0] = (zeta + 0.2 + 0.1 * rng.standard_normal(len(zeta)),
                       theta + 0.1 * rng.standard_normal(len(zeta)), energy, len(zeta) / n_launched)

# true vs predicted strike points and heat flux, one row per birth surface
bb.plot_strike_points(record, predictions)
bb.plot_heat_flux(record, predictions)
plt.show()
