"""Explore one stellarator configuration: its shape, |B|, particle losses and heat flux.

Compares the true heat flux with a made-up prediction.

    python explore_configuration.py DATA_DIR

DATA_DIR is the folder the BoSS data was downloaded to.
"""

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

import boss_bench as bb

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
path = bb.list_files(data_dir, "confinement", "test")[0]
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

# a made-up prediction: the true strike points, shifted and blurred
rng = np.random.default_rng(0)
predictions = {}
for s0 in (0.1, 0.25, 0.5):
    zeta, theta, energy, _ = bb.lost_particles(record, s0)
    predictions[s0] = (zeta + 0.2 + 0.1 * rng.standard_normal(len(zeta)),
                       theta + 0.1 * rng.standard_normal(len(zeta)), energy)
bb.plot_strike_points(record, predictions)

# true vs predicted heat flux of the particles born on s = 0.1, on the same color scale
zeta, theta, energy, n_launched = bb.lost_particles(record, 0.1)
q_true = bb.binned_heat_flux(zeta, theta, energy, n_launched, record)
q_pred = bb.binned_heat_flux(*predictions[0.1], n_launched, record)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bb.plot_heat_flux(q_true, nfp, ax=axs[0], vmax=q_true.max())
bb.plot_heat_flux(q_pred, nfp, ax=axs[1], vmax=q_true.max())
axs[0].set_title("True heat flux")
axs[1].set_title("Predicted heat flux")
fig.tight_layout()
plt.show()
