# Release checklist

Outstanding items before publishing `boss-bench` 1.0.0. Delete this file as the last step.

## Data

- [ ] Upload all release files to Globus: confinement and trajectory, train and test.
- [x] Check the field names of the release trajectory files. `read_trajectories` and
      `trajectory_metrics` read `uuid`, `traj_data`, `traj_offsets` and `config_offsets`.
- [ ] Rerun `verification/check_against_benchmark.py` on a release trajectory file. So far
      trajectories are checked only on an old-format local file.
- [ ] Rerun the verification on more of the release confinement data. So far it covers the
      first 20 configurations of `confinement_data_test_split_00000.hdf5`.
- [ ] Decide whether the Globus collection stays frozen at the paper version. If it will
      grow, mark which files are the paper release.
- [ ] Publish a manifest of the release files with checksums (sha256).
- [ ] Choose the data license.

## Pretrained models

- [ ] Upload the model folders (`loss_fraction`, `heat_flux`, `inverse_design`) to
      https://huggingface.co/mishapadidar/boss, with the model card, the conversion
      and verification scripts, and `PROVENANCE.md`.
- [ ] Choose the license of the weights; the model card says Apache 2.0 for now.
- [ ] Run `examples/pretrained_models.py` and `examples/onnx_models.py` against the live
      repository, with an empty models folder.
- [ ] Tag the Hugging Face repository at the paper version, and pin the default
      `revision` of `download_model` to the tag.

## Benchmark definitions to confirm

- [ ] Heat flux: the allowed model inputs. The docs say any configuration field except
      the tracing outcomes (`confinement_time`, `s_end`, `theta_end`, `zeta_end`,
      `vpar_end`). Are the initial particle states (`*_init`) allowed? Does the paper
      restrict the inputs further?
- [ ] Trajectories: the allowed model inputs, and that predictions follow the paper's
      conventions (loss at the first predicted s >= 0.98, state held at the loss state
      afterwards). The package now takes predictions, so it cannot enforce these.
- [ ] Trajectories: the package scores every orbit of the test files, while the paper's
      table used 500 particles per configuration. Regenerate the table or update its caption.
- [ ] Trajectories: a predicted crossing at the last sample (`t = 1 ms`) counts as a loss,
      while a true orbit is lost if `t_loss < 1 ms - 1e-14`. This follows `metrics.py`.

## Package

- [ ] After the paper is out, uncomment the paper and citation parts and fill in
      `<AUTHORS>`, `<PAPER_TITLE>`, `<PAPER_URL>` and `<BIBTEX>` (in `README.md`,
      `pyproject.toml` and `CITATION.cff`; on the website, the citation section of
      `docs/index.md`). The authors are "The BoSS Development Team" until then.
- [ ] Run the tests on Python 3.9 and on the newest Python.
- [ ] Optionally, add a GitHub Actions workflow that runs the tests.
- [ ] Publish 1.0.0 to PyPI.

## Website

- [ ] Update the data documentation to the release field names.
- [ ] Make the tutorials consistent with the package: loss fraction per birth surface,
      heat flux per birth surface.
- [ ] Update the package Overview page (`docs/package.md`) to the new heat-flux API
      (`heat_flux_metrics`, `binned_heat_flux`, `lost_particles`) and the trajectory API
      (`trajectory_metrics` takes predictions); its formulas now also live in the docstrings.
- [ ] Deploy after the PyPI release; the workflow installs `boss-bench==1.0.0`.
