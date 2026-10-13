# boss-bench

[![PyPI](https://img.shields.io/pypi/v/boss-bench)](https://pypi.org/project/boss-bench/)
[![Python](https://img.shields.io/pypi/pyversions/boss-bench)](https://pypi.org/project/boss-bench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](<PAPER_URL>)
[![Data](https://img.shields.io/badge/Globus-data-2d6db5.svg)](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F)

**The BoSS** is a database of stellarator configurations paired with physics
simulations: ideal-MHD equilibria and the collisionless guiding-center orbits of
fusion-born alpha particles. This package accompanies the BoSS benchmark paper.

`boss-bench` loads the data and computes the benchmark metrics exactly as they are
defined in the paper.

[Documentation](<DOCUMENTATION_URL>) | [Paper](<PAPER_URL>) | [Data](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F)

## Installation

`boss-bench` needs Python `3.9` or newer.

```bash
pip install boss-bench
```

From source:

```bash
git clone <REPOSITORY_URL>
cd boss-bench
pip install .
```

## Downloading the data

The data is distributed through [Globus](https://app.globus.org/file-manager?origin_id=04779ab9-4a4e-4381-be6a-860f0433a4af&origin_path=%2F) as HDF5 files of two kinds, each
split into `train` and `test`:

- `confinement` files hold about `3000` configurations each, with `15000` traced alpha
  particles per configuration, `5000` born on each of the surfaces `s0 = 0.1, 0.25, 0.5`.
- `trajectory` files hold `40` configurations each, with `6000` full orbits per configuration.

Open the Globus link, log in with your institution, Google or ORCID account, select the
files and click **Download**. For large transfers, use **Transfer or Sync to...** to send
them to your own Globus endpoint, such as a cluster or a laptop running
[Globus Connect Personal](https://www.globus.org/globus-connect-personal).

## Loading the data

Row `i` of every per-configuration field belongs to configuration `i`.

```python
import h5py
import boss_bench as bb

files = bb.list_files("BoSS", "confinement", "test")   # files under the download folder
with h5py.File(files[0], "r") as f:
    record = bb.load_record(f, 0)                       # every field of configuration 0

data = bb.load(files, ["nfp", "aspect_ratio"])          # fields stacked over files
```

## Particle losses

A particle is lost if `confinement_time < t_max - 1e-14`. Each quantity is computed over
the particles born on one surface `s0`:

```python
ct, t_max, s_init = record["confinement_time"], record["t_max"], record["s_init"]
bb.loss_fraction(ct, t_max, s_init, s0=0.1)           # fraction lost
bb.energy_loss(ct, t_max, s_init, s0=0.1)             # mean of 3.5 MeV * exp(-2 ct / t_max)
bb.normalized_energy_loss(ct, t_max, s_init, s0=0.1)  # the same, rescaled to [0, 1]
```

## Benchmarks

**Surrogate models.** Predict a per-configuration quantity on the test split, such as
`bb.loss_fraction` or `bb.normalized_energy_loss` at `s0 = 0.1`.

**Heat flux.** For each test configuration and birth surface `s0`, predict the loss
fraction and the strike points of the lost particles on the plasma boundary: their Boozer
angles `zeta`, `theta` and energies. Predictions may use any field of the configuration
except the outcomes of the particle tracing (`confinement_time`, `s_end`, `theta_end`,
`zeta_end`, `vpar_end`). Read only the fields your model needs:

```python
paths = bb.list_files("BoSS", "confinement", "test")
predictions = {}
for path in paths:
    with h5py.File(path, "r") as f:
        for i in range(len(f["uuid"])):
            record = bb.load_record(f, i, ["uuid", "boundary_xyz", "iota", "nfp"])
            for s0 in (0.1, 0.25, 0.5):
                predictions[record["uuid"], s0] = my_model(record, s0)  # (zeta, theta, energy, loss_fraction)
bb.evaluate_heat_flux(paths, predictions)
```

**Trajectories.** Predict every orbit in the test files from its first sample, with
either a rollout model or a model of the state at any lead time:

```python
paths = bb.list_files("BoSS", "trajectory", "test")
bb.evaluate_trajectories(paths, rollout=my_rollout)
bb.evaluate_trajectories(paths, predict_at=my_predict_at)
```

**Inverse design.** Evaluate each design with VMEC and particle tracing at ARIES-CS scale,
then score the set:

```python
bb.design_metrics(loss_fraction, mirror_ratio, mirror_target, aspect_ratio, aspect_target)
```

## Plotting

```python
bb.plot_device(record)                          # plasma boundary colored by |B|
bb.plot_modB_boozer(record, s=0.5)              # |B| on a flux surface in Boozer coordinates
bb.plot_loss_fraction_curves(record)            # fraction lost vs time, per birth surface
bb.plot_strike_points(targets, predictions)     # heat-flux strike points and their density
bb.plot_rollouts(z_true, z_pred)                # trajectories vs lead time, per channel
```

## Tests

The tests run on small synthetic files with known answers, so they need no download:

```bash
pip install ".[test]"
pytest
```

## Citation

If you use the BoSS data or this package, please cite:

```bibtex
<BIBTEX>
```

## License

The code is released under the [MIT License](LICENSE). The data is released under the
<DATA_LICENSE>.

## Contact

<CONTACT>
