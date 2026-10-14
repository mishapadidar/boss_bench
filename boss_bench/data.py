"""Find and read the BoSS data files."""

import glob
import os

import h5py
import numpy as np

TRAJECTORY_FIELDS = ("traj_data", "traj_offsets", "config_offsets")


def list_files(data_dir, kind="confinement", split="test"):
    """List the data files of one kind and split.

    Args:
        data_dir (str): the folder with the downloaded files; it is searched recursively.
        kind (str): ``"confinement"`` or ``"trajectory"``.
        split (str): ``"train"`` or ``"test"``.

    Returns:
        list of str: the file paths, sorted.
    """
    pattern = f"{kind}_data_{split}_split_*.hdf5"
    return sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))


def load(paths, fields):
    """Read chosen fields of every configuration in a list of files.

    Every field can be read except the trajectory fields themselves: ``traj_data``,
    ``traj_offsets`` and ``config_offsets``. To load trajectories, use
    ``trajectory.read_trajectories``. Fields are returned as stored, with one row per
    configuration and gridded fields flattened: ``modB``, for example, has shape
    ``(n, 15 * 41 * 41)``.

    Args:
        paths (list of str): the files.
        fields (list of str): the names of the fields to read.

    Returns:
        dict: maps each field name to an array with one row per configuration, with the
        configurations of all files stacked in order.
    """
    out = {name: [] for name in fields}
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in fields:
                out[name].append(_read_rows(f[name], slice(None)))
    return {name: np.concatenate(arrays) for name, arrays in out.items()}


def load_record(f, config, fields=None):
    """Read the fields of one configuration.

    Every field can be read except the trajectory fields themselves: ``traj_data``,
    ``traj_offsets`` and ``config_offsets``. To load trajectories, use
    ``trajectory.read_trajectories``. Only the requested fields are read from disk, and
    gridded fields are returned flattened, as stored.

    Args:
        f (h5py.File): an open file, e.g. ``h5py.File(path, "r")``.
        config (int or str): the row index in the file or the ``uuid`` of the
            configuration.
        fields (list of str, optional): the names of the fields to read. By default, every
            field of the configuration.

    Returns:
        dict: maps each field name to its value for the configuration.

    Raises:
        KeyError: if no configuration in the file has this ``uuid``.
    """
    i = _config_index(f, config)
    if fields is None:
        fields = [name for name in f if name not in TRAJECTORY_FIELDS]
    return {name: _read_rows(f[name], i) for name in fields}


def _config_index(f, config):
    """The index in a file of a stellarator configuration given by index or uuid.

    Args:
        f (h5py.File): an open file.
        config (int or str): the configuration, by its index in the file or its uuid.

    Returns:
        int: the index of the configuration in the file.

    Raises:
        KeyError: if a uuid is given that is not in the file.
    """
    if not isinstance(config, str):
        return int(config)
    matches = np.flatnonzero(_read_rows(f["uuid"], slice(None)) == config)
    if len(matches) == 0:
        raise KeyError(f"no configuration with uuid {config} in {f.filename}")
    return int(matches[0])


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
