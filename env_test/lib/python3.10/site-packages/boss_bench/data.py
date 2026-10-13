"""Find and read the BoSS HDF5 files, once they are downloaded from Globus.

There are two kinds of files. Confinement files hold stellarator configurations and the
outcomes of 15000 traced alpha particles per configuration. Trajectory files hold
stellarator configurations and 6000 full particle orbits per configuration. In both,
row i of every per-configuration field belongs to configuration i.
"""

import glob
import os

import h5py
import numpy as np

TRAJECTORY_FIELDS = ("traj_data", "traj_offsets", "config_offsets")


def list_files(data_dir, kind="confinement", split="test"):
    """List the data files of one kind and split.

    Args:
        data_dir (str): the directory the data was downloaded to; it is searched recursively.
        kind (str): "confinement" or "trajectory".
        split (str): "train" or "test".

    Returns:
        list of str: the sorted file paths.
    """
    pattern = f"{kind}_data_{split}_split_*.hdf5"
    return sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))


def load(paths, fields):
    """Read fields of every stellarator configuration in several files.

    Works on confinement and trajectory files alike, for the per-configuration fields
    (not the orbit arrays ``traj_data``, ``traj_offsets`` and ``config_offsets``, see
    ``trajectory.read_trajectories``). Fields are returned as stored, with one row per
    configuration: gridded fields are flattened, e.g. ``modB`` has shape (n, 15 * 41 * 41).

    Args:
        paths (list of str): file paths.
        fields (list of str): names of the fields to read.

    Returns:
        dict: maps each field name to an np.ndarray of shape (n_configurations, ...),
        with the configurations of all files stacked in order.
    """
    out = {name: [] for name in fields}
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in fields:
                out[name].append(_read_rows(f[name], slice(None)))
    return {name: np.concatenate(arrays) for name, arrays in out.items()}


def load_record(f, i, fields=None):
    """Read the fields of one stellarator configuration.

    Only the requested fields are read from disk. Works on confinement and trajectory files.

    Args:
        f (h5py.File): an open file, e.g. ``h5py.File(path, "r")``.
        i (int): index of the configuration in the file.
        fields (list of str, optional): names of the fields to read. By default every
            per-configuration field is read.

    Returns:
        dict: maps each field name to its value for configuration i, as stored (gridded
        fields flattened).
    """
    if fields is None:
        fields = [name for name in f if name not in TRAJECTORY_FIELDS]
    return {name: _read_rows(f[name], i) for name in fields}


def _read_rows(dataset, rows):
    """Read rows of an HDF5 dataset, decoding strings.

    Args:
        dataset (h5py.Dataset): the dataset.
        rows (int or slice): an index or slice into the first axis.

    Returns:
        np.ndarray or str: the selected rows, with byte strings decoded to str.
    """
    if dataset.dtype.kind == "O":
        return dataset.asstr()[rows]
    return dataset[rows]
