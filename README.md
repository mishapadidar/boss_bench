# boss-bench

[![PyPI](https://img.shields.io/pypi/v/boss-bench)](https://pypi.org/project/boss-bench/)
[![Python](https://img.shields.io/pypi/pyversions/boss-bench)](https://pypi.org/project/boss-bench/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mishapadidar/boss_bench/blob/main/LICENSE)
[![Data](https://img.shields.io/badge/Globus-data-2d6db5.svg)](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F)
[![Models](https://img.shields.io/badge/Hugging_Face-models-ffcc4d.svg)](https://huggingface.co/mishapadidar/boss)
<!-- [![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](<PAPER_URL>) -->

The **Benchmark of Stellarator Simulations (the BoSS)** is a database of stellarator
configurations paired with physics simulations: ideal-MHD equilibria and the
collisionless guiding-center orbits of fusion-born alpha particles. This package
accompanies the BoSS benchmark paper.

`boss-bench` loads the data and computes the benchmark metrics exactly as they are
defined in the paper.

<!-- once the paper is out, add: | [Paper](<PAPER_URL>) -->
[Documentation](https://boss.flatironinstitute.org/) | [Data](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F) | [Models](https://huggingface.co/mishapadidar/boss)

## Installation

`boss-bench` needs Python `3.9` or newer.

```bash
pip install boss-bench
```

From source:

```bash
git clone https://github.com/mishapadidar/boss_bench.git
cd boss_bench
pip install .
```

## Downloading the data

The package works on local files, so download the data before using it. The data is
distributed through [Globus](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F) as HDF5 files of two kinds, each
split into `train` and `test`:

- `confinement` files hold up to `3000` configurations each, with `15000` traced alpha
  particles per configuration, `5000` born on each of the surfaces `s0 = 0.1, 0.25, 0.5`.
- `trajectory` files hold up to `40` configurations each, with `6000` full orbits per configuration.

Open the Globus link, log in with your institution, Google or ORCID account, select the
files and click **Download**. For large transfers, use **Transfer or Sync to...** to send
them to your own Globus endpoint, such as a cluster or a laptop running
[Globus Connect Personal](https://www.globus.org/globus-connect-personal).

Put the downloaded files in one folder, for example `BoSS/`; any layout inside it works.
The examples below use that folder.

## Loading the data

Row `i` of every per-configuration field belongs to configuration `i`.

```python
import h5py
import boss_bench as bb

files = bb.list_files("BoSS", "confinement", "test")   # files under the download folder
with h5py.File(files[0], "r") as f:
    record = bb.load_record(f, 0)                       # every field of configuration 0 (or give its uuid)

data = bb.load(files, ["nfp", "aspect_ratio"])          # fields stacked over files

with h5py.File(bb.list_files("BoSS", "trajectory", "test")[0], "r") as f:
    orbit = bb.load_trajectory(f, 0, 5)                 # orbit 5 of configuration 0: t, s, theta, zeta, vpar
    record = bb.load_record(f, 0)
xyz = bb.to_xyz(record, orbit["s"], orbit["theta"], orbit["zeta"])   # its Cartesian positions [m]
```

## Particle losses and heat flux

A particle is lost if `confinement_time < t_max`. Each quantity is computed over
the particles born on one surface `s0`:

```python
ct, t_max, s_init = record["confinement_time"], record["t_max"], record["s_init"]
bb.loss_fraction(ct, t_max, s_init, s0=0.1)           # fraction lost
bb.energy_loss(ct, t_max, s_init, s0=0.1)             # mean of 3.5 MeV * exp(-2 ct / t_max)
bb.normalized_energy_loss(ct, t_max, s_init, s0=0.1)  # the same, rescaled to [0, 1]
bb.mean_confinement_time(ct, t_max, s_init, s0=0.1)   # mean of the confinement times [s]

# where the lost particles strike the boundary, and their energies
zeta_end, theta_end = record["zeta_end"], record["theta_end"]
zeta, theta, energy, n_launched = bb.lost_particles(ct, t_max, s_init, zeta_end, theta_end, s0=0.1)
# heat flux per bin; only the boundary geometry is read from the record
q = bb.binned_heat_flux(zeta, theta, energy, n_launched, record, resolution=32)
```

## Benchmarks

Each benchmark takes your predictions and returns the paper's metrics.

**Surrogate models.** A prediction is a per-configuration quantity on the test split,
compared with its true value, such as `bb.loss_fraction` or `bb.normalized_energy_loss`
at `s0 = 0.1`.

**Heat flux.** A prediction for each test configuration and birth surface `s0` is a
tuple `(zeta, theta, energy, loss_fraction)`: the Boozer angles and energies of the
predicted strike points of the lost particles on the plasma boundary (arrays of any
length), and the predicted loss fraction. With `predictions` a dict from
`(uuid, s0)` to these tuples,

```python
paths = bb.list_files("BoSS", "confinement", "test")
bb.heat_flux_metrics(paths, predictions)
```

The per-sample metrics are also available on their own, with their formulas in the
documentation: `bb.heat_flux_error(prediction, record, s0)` and
`bb.energy_distance(prediction, record, s0)`.

**Trajectories.** A prediction for each test configuration is a tuple `(z, t_loss)`
covering all its orbits: `z` holds the predicted states at lead times `10`, `100` and
`999` us, of shape `(n_orbits, 3, 5)`, in the embedded form of `bb.embed_state`, and
`t_loss` the predicted loss times, of shape `(n_orbits,)` (`NaN` if not lost). With
`predictions` a dict from `uuid` to these tuples,

```python
paths = bb.list_files("BoSS", "trajectory", "test")
bb.trajectory_metrics(paths, predictions)
```

Each nRMSE in the report is followed by its breakdown by channel group (`pol_x`, `pol_y`,
`phi`, `lambda`), whose mean it is. The metrics are also available on their own, with
their formulas in the documentation: `bb.nrmse` (with `by_group=True` for the breakdown),
`bb.loss_fraction_mae` and `bb.confinement_time_w1`.

**Inverse design.** Evaluate each design with VMEC and particle tracing at ARIES-CS scale,
then score the set:

```python
bb.design_metrics(loss_fraction, mirror_ratio, mirror_target, aspect_ratio, aspect_target)
```

## Plotting

`predictions` is optional, and maps each birth surface `s0` to a predicted
`(zeta, theta, energy, loss_fraction)`, drawn next to the truth.

```python
bb.plot_device(record)                          # plasma boundary colored by |B|
bb.plot_modB_boozer(record, s=0.5)              # |B| on a flux surface in Boozer coordinates
bb.plot_loss_fraction_curves(record)            # fraction lost vs time, per birth surface
bb.plot_strike_points(record, predictions)      # strike points and density of lost particles, per s0
bb.plot_heat_flux(record, predictions)          # heat flux over one field period, per s0
bb.plot_rollouts(z_true, z_pred)                # trajectories vs lead time, per channel
```

## Trained models

Three models from the paper are available on
[Hugging Face](https://huggingface.co/mishapadidar/boss), in PyTorch and ONNX:

- `loss_fraction`: predicts the loss fraction at `s0 = 0.1` from the boundary and
  profiles, without VMEC. This is the surrogate used in the paper's surrogate-based
  optimization.
- `heat_flux`: samples the strike points of the lost particles, as in the heat-flux
  benchmark.
- `inverse_design`: generates plasma boundaries with a chosen loss fraction, shape,
  field and profiles.

The PyTorch models need `pip install "boss-bench[models]"`. `models.download` saves a
model from Hugging Face in a folder, and `models.load` reads it. The models read the
BoSS fields by name, from one record or from a batch of configurations:

```python
from boss_bench import models

surrogate = models.load(models.download("loss_fraction", "BoSS-models/loss_fraction"))
surrogate.predict(record)                                    # loss fraction at s0 = 0.1

heat_flux = models.load(models.download("heat_flux", "BoSS-models/heat_flux"))
prediction = heat_flux.sample(record, s0=0.1)                # (zeta, theta, energy, loss_fraction)
bb.heat_flux_error(prediction, record, s0=0.1)

generator = models.load(models.download("inverse_design", "BoSS-models/inverse_design"))
conditions = {name: record[name] for name in models.inverse_design.CONDITIONS if name != "loss_fraction"}
boundary_xyz = generator.sample({**conditions, "loss_fraction": 0.01})   # laid out as record["boundary_xyz"]
```

### Downloading from Hugging Face directly

Each model is a folder of the Hugging Face repository, holding `config.json`, the PyTorch
weights `model.safetensors` and the ONNX files. The ONNX files run without PyTorch, for
example with ONNX Runtime; `examples/onnx_models.py` shows how. To download whole model
folders, use the Hugging Face command line (`pip install huggingface_hub`):

```bash
hf download mishapadidar/boss --include "heat_flux/*" --local-dir BoSS-models                # one model
hf download mishapadidar/boss --include "*/config.json" --include "*.onnx" --local-dir BoSS-models   # ONNX only
hf download mishapadidar/boss --local-dir BoSS-models                                         # everything
```

Each model then sits in its own folder, e.g. `BoSS-models/heat_flux`, which `models.load`
reads. Single files can also be downloaded from the **Files and versions** tab of the
[repository](https://huggingface.co/mishapadidar/boss/tree/main).

## Examples

The `examples/` folder has five short scripts. Download the data first, then run each
as `python <script> DATA_DIR`, where `DATA_DIR` is a directory containing the downloaded
files (it is searched recursively):

- `explore_configuration.py` loads a configuration and plots its shape, `|B|`, loss
  fraction curves, and true against predicted strike points and heat flux. `DATA_DIR`
  must contain at least one confinement data file from the test split.
- `heat_flux_benchmark.py` scores made-up heat-flux predictions on a small batch of test
  configurations. `DATA_DIR` must contain at least one confinement data file from the
  test split.
- `trajectory_rollouts.py` loads the orbits of one configuration, plots made-up
  predictions against them and scores them. `DATA_DIR` must contain at least one
  trajectory data file from the test split.
- `trained_models.py` runs the three trained models on test configurations: it
  scores the loss-fraction surrogate and the heat-flux model, and plots boundaries
  generated with a lower loss fraction than a test configuration's. `DATA_DIR` must
  contain at least one confinement data file from the test split. It takes the models
  folder as a second argument, by default `BoSS-models`.
- `onnx_models.py` runs the loss-fraction surrogate and the heat-flux model with ONNX
  Runtime, without PyTorch. It takes the same arguments, with the ONNX files downloaded
  into the models folder first (see above).

## Tests

The tests run on small synthetic files with known answers, so they need no download:

```bash
pip install ".[test]"
pytest
```

<!-- Restore once the paper is out.
## Citation

If you use the BoSS data or this package, please cite:

```bibtex
<BIBTEX>
```
-->

## License

Copyright 2026 The Simons Foundation.

The code is released under the
[Apache License 2.0](https://github.com/mishapadidar/boss_bench/blob/main/LICENSE).
The data license is listed on the [BoSS website](https://boss.flatironinstitute.org/).
