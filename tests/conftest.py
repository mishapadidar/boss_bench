"""Small synthetic BoSS files with known answers, shared by the tests.

The configuration is a circular torus, major radius R0 and minor radius A, so its
boundary area 4 pi^2 R0 A is known exactly.
"""

import h5py
import numpy as np
import pytest

R0, A, NFP = 3.0, 1.0, 2
NS, N_ANGLE = 15, 41
T_MAX = 0.1

# Four particles per birth surface: lost at 0.05 s and 0.001 s, confined (t_max),
# and confined just inside the loss tolerance (t_max - 1e-15).
CONFINEMENT_TIMES = [T_MAX, 0.05, T_MAX - 1e-15, 0.001]


def torus_fields():
    """Grids and fields of the circular torus, flattened as in the data files.

    Returns:
        Dict of per-configuration fields: grids, boundary points, |B| and Boozer geometry.
    """
    s1d = np.linspace(1e-4, 1, NS)
    phi1d = np.linspace(0, 2 * np.pi / NFP, N_ANGLE, endpoint=False)
    theta1d = np.linspace(0, 2 * np.pi, N_ANGLE, endpoint=False)
    S, P, T = np.meshgrid(s1d, phi1d, theta1d, indexing="ij")
    R = R0 + A * np.sqrt(S) * np.cos(T)
    Z = A * np.sqrt(S) * np.sin(T)
    xyz = np.stack([R[-1] * np.cos(P[-1]), R[-1] * np.sin(P[-1]), Z[-1]], axis=-1)
    modB = 5.0 * R0 / R
    return {
        "nfp": NFP, "s1d": s1d, "phi1d": phi1d, "theta1d": theta1d,
        "iota": np.full(NS, 0.4), "boundary_xyz": xyz.ravel(), "modB": modB.ravel(),
        "modB_boozer": modB.ravel(), "R_boozer": R.ravel(), "Z_boozer": Z.ravel(),
        "nu_boozer": np.zeros(R.size),
    }


def write_configs(f, n_configs):
    """Write n_configs copies of the torus, with string uuids, to an open h5py file.

    Args:
        f: an h5py.File open for writing.
        n_configs: number of configurations.
    """
    for name, value in torus_fields().items():
        f[name] = np.stack([np.asarray(value)] * n_configs)
    f["uuid"] = np.array([f"config-{i}" for i in range(n_configs)], dtype=h5py.string_dtype())


@pytest.fixture
def confinement_file(tmp_path):
    """A confinement file with two configurations of four particles per birth surface.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        Path of the file.
    """
    path = tmp_path / "confinement_data_test_split_00000.hdf5"
    n_configs, n_per_surface = 2, len(CONFINEMENT_TIMES)
    with h5py.File(path, "w") as f:
        write_configs(f, n_configs)
        s_init = np.repeat([0.1, 0.25, 0.5], n_per_surface)
        f["s_init"] = np.tile(s_init, (n_configs, 1))
        f["confinement_time"] = np.tile(CONFINEMENT_TIMES * 3, (n_configs, 1))
        f["t_max"] = np.full(n_configs, T_MAX)
        f["zeta_end"] = np.tile(np.linspace(0, 6, s_init.size), (n_configs, 1))
        f["theta_end"] = np.tile(np.linspace(-3, 3, s_init.size), (n_configs, 1))
    return path


def orbit(n_samples, s, t_loss=None):
    """Rows (t, s, theta, phi, vpar) of an orbit sampled every 1 us from t = 1 us.

    Args:
        n_samples: number of on-grid samples.
        s: flux label of the on-grid samples (constant).
        t_loss: if given, a final off-grid sample at this time, at s = 0.98.

    Returns:
        (n_rows, 5) array.
    """
    k = np.arange(n_samples)
    rows = np.stack([(k + 1) * 1e-6, np.full(n_samples, s), 0.01 * k, 0.5 * k, np.full(n_samples, 1e6)], axis=1)
    if t_loss is not None:
        rows = np.vstack([rows, [t_loss, 0.98, 0.0, 0.0, 1e6]])
    return rows


@pytest.fixture
def trajectory_file(tmp_path):
    """A trajectory file with one configuration and three orbits.

    Orbit 0 is confined for the full 1 ms, orbit 1 is lost at 500.4 us, and orbit 2 is
    lost at 0.5 us, before its first sample, so it has no initial state.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        Path of the file.
    """
    path = tmp_path / "trajectory_data_test_split_00000.hdf5"
    orbits = [orbit(1000, 0.3), orbit(500, 0.6, t_loss=500.4e-6), orbit(0, 0.6, t_loss=0.5e-6)]
    with h5py.File(path, "w") as f:
        write_configs(f, 1)
        f["traj_data"] = np.vstack(orbits)
        f["traj_offsets"] = np.concatenate([[0], np.cumsum([len(o) for o in orbits])])
        f["config_offsets"] = np.array([0, len(orbits)])
    return path
