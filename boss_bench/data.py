"""Find and read the BoSS HDF5 files, once they are downloaded from Globus."""

import glob
import os

import h5py
import numpy as np

TRAJECTORY_FIELDS = ("traj_data", "traj_offsets", "config_offsets")


def list_files(data_dir, kind="confinement", split="test"):
    """List the data files of one kind and split.

    Args:
        data_dir: the directory the data was downloaded to; it is searched recursively.
        kind: "confinement" or "trajectory".
        split: "train" or "test".

    Returns:
        Sorted list of file paths.
    """
    pattern = f"{kind}_data_{split}_split_*.hdf5"
    return sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))


def load(paths, fields):
    """Read fields from several files, stacking the configurations of all files.

    Args:
        paths: file paths.
        fields: names of the per-configuration fields to read.

    Returns:
        Dict mapping each field name to an array whose first axis runs over configurations.
    """
    out = {name: [] for name in fields}
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in fields:
                out[name].append(_read_rows(f[name], slice(None)))
    return {name: np.concatenate(arrays) for name, arrays in out.items()}


def load_record(f, i, fields=None):
    """Read the fields of one configuration.

    Only the requested fields are read from disk.

    Args:
        f: an open file, e.g. ``h5py.File(path, "r")``.
        i: index of the configuration in the file.
        fields: names of the fields to read. If None, every per-configuration field is
            read (the trajectory arrays, which are not indexed by configuration, are left out).

    Returns:
        Dict mapping each field name to its value for configuration i.
    """
    if fields is None:
        fields = [name for name in f if name not in TRAJECTORY_FIELDS]
    return {name: _read_rows(f[name], i) for name in fields}


def _read_rows(dataset, rows):
    """Read rows of an HDF5 dataset, decoding strings.

    Args:
        dataset: an h5py.Dataset.
        rows: an index or slice into the first axis.

    Returns:
        The selected rows, with byte strings decoded to str.
    """
    if dataset.dtype.kind == "O":
        return dataset.asstr()[rows]
    return dataset[rows]
