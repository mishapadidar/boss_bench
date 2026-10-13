"""Score heat-flux predictions on a small batch of test configurations.

The "model" is made up: it predicts that 20% of the particles are lost, striking the
boundary at random angles with the birth energy. The benchmark itself scores every
sample of the test files.

    python heat_flux_benchmark.py DATA_DIR

DATA_DIR is a directory containing at least one BoSS confinement data file from the test split,
e.g. confinement_data_test_split_00000.hdf5. It is searched recursively.
"""

import sys

import h5py
import numpy as np

import boss_bench as bb

data_dir = sys.argv[1] if len(sys.argv) > 1 else "BoSS"
paths = bb.list_files(data_dir, "confinement", "test")[:1]
if not paths:
    sys.exit(f"No confinement test files found in {data_dir}. Download the BoSS data first.")
n_configurations = 10
rng = np.random.default_rng(0)

# one prediction per (configuration, birth surface); a real model would read the fields
# it needs here, e.g. ["uuid", "boundary_xyz", "iota", "pressure"]
predictions = {}
with h5py.File(paths[0], "r") as f:
    for i in range(n_configurations):
        record = bb.load_record(f, i, ["uuid", "nfp"])
        for s0 in (0.1, 0.25, 0.5):
            zeta = rng.uniform(0, 2 * np.pi, 1000)
            theta = rng.uniform(0, 2 * np.pi, 1000)
            energy = np.full(1000, 3.5)
            predictions[record["uuid"], s0] = (zeta, theta, energy, 0.2)

scores = bb.heat_flux_metrics(paths, predictions)
for name, value in scores.items():
    print(f"{name:32s} {value:.4g}")
