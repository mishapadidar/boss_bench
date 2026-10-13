"""Trajectory benchmark: predict guiding-center orbits from their initial state.

Orbits are sampled every DT = 1 us, from t = 1 us up to T_MAX = 1 ms. A lost orbit ends
with one extra, off-grid sample at its loss time. A model is given an orbit's first
sample z0, in the embedded form of ``embed_state``, and predicts the state at lead times
tau = k * DT after it, k = 1, ..., 999, and whether and when the particle is lost.
"""

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from ._checks import require_finite
from .confinement import LOSS_TOL
from .data import _config_index, _read_rows

DT = 1e-6
T_MAX = 1e-3
N_STEPS = 999
TABLE_STEPS = np.array([10, 100, 999])  # the lead times scored [us]
# Speed of a 3.5 MeV alpha particle, v = sqrt(2 E / m) (non-relativistic, as in the tracer) [m/s].
V_ALPHA = np.sqrt(2 * 3.5e6 * 1.602176634e-19 / 6.6446573450e-27)

# Channels of the embedded state in each group, and the group's frozen RMS over the training set.
CHANNEL_GROUPS = {
    "pol_x": (slice(0, 1), 0.3958040045685409),
    "pol_y": (slice(1, 2), 0.4034089137677494),
    "phi": (slice(2, 4), 1.0),
    "lambda": (slice(4, 5), 0.5778076495076678),
}


def embed_state(s, theta, phi, vpar):
    r"""The embedded form of a guiding-center state, in which models take and return states.

    .. math::

        z = \left(\sqrt{s}\cos\theta,\ \sqrt{s}\sin\theta,\ \cos\varphi,\ \sin\varphi,\
        v_\parallel / v_\alpha\right),

    with :math:`v_\alpha` the speed of a 3.5 MeV alpha particle.

    Args:
        s (np.ndarray): normalized toroidal flux, in [0, 1], shape (n,).
        theta (np.ndarray): Boozer poloidal angle [rad], shape (n,).
        phi (np.ndarray): Boozer toroidal angle [rad], shape (n,).
        vpar (np.ndarray): parallel velocity [m/s], shape (n,).

    Returns:
        np.ndarray: the embedded states, shape (n, 5). Inputs of any other shape also
        work; the states then have one more axis, of length 5.
    """
    r = np.sqrt(s)
    return np.stack([r * np.cos(theta), r * np.sin(theta), np.cos(phi), np.sin(phi), vpar / V_ALPHA], axis=-1)


def load_trajectory(f, config, particle):
    """Read one orbit of one stellarator configuration of a trajectory file, as stored.

    Args:
        f (h5py.File): an open trajectory file, e.g. ``h5py.File(path, "r")``.
        config (int or str): the configuration, by its index in the file or its uuid.
        particle (int): the orbit, by its index among the configuration's orbits.

    Returns:
        dict: the orbit's samples (np.ndarray, shape (n_samples,) each): "t", the time [s];
        "s", the normalized toroidal flux; "theta" and "zeta", the Boozer poloidal and
        toroidal angles [rad]; and "vpar", the parallel velocity [m/s]. Samples are every
        DT = 1 us from t = 1 us; a lost orbit ends with one extra sample at its loss time.

    Raises:
        KeyError: if a uuid is given that is not in the file.
        IndexError: if the configuration has no orbit with index particle.
    """
    i = _config_index(f, config)
    c0, c1 = f["config_offsets"][i], f["config_offsets"][i + 1]
    if not 0 <= particle < c1 - c0:
        raise IndexError(f"configuration {config} has {c1 - c0} orbits; there is no orbit {particle}")
    a, b = f["traj_offsets"][c0 + particle], f["traj_offsets"][c0 + particle + 1]
    t, s, theta, zeta, vpar = f["traj_data"][a:b].T
    return {"t": t, "s": s, "theta": theta, "zeta": zeta, "vpar": vpar}


def to_xyz(record, s, theta, zeta):
    r"""Cartesian positions of points given in Boozer coordinates, e.g. the samples of an orbit.

    The cylindrical coordinates :math:`R`, :math:`Z` and the angle difference :math:`\nu`
    are interpolated from the configuration's Boozer grid: spectrally in the two angles,
    in which the grid is periodic, and with a cubic spline in :math:`\sqrt{s}` across the
    flux surfaces. With the cylindrical toroidal angle :math:`\phi = \zeta - \nu`,

    .. math::

        (x, y, z) = (R \cos\phi,\ R \sin\phi,\ Z).

    Args:
        record (dict): the configuration's fields, from ``data.load_record``; at least
            ``R_boozer``, ``Z_boozer``, ``nu_boozer``, ``s1d`` and ``nfp``.
        s (np.ndarray): normalized toroidal flux, in [0, 1], shape (n,).
        theta (np.ndarray): Boozer poloidal angle [rad], shape (n,).
        zeta (np.ndarray): Boozer toroidal angle [rad], shape (n,).

    Returns:
        np.ndarray: the positions [m], shape (n, 3). Inputs of any other shape also
        work; the positions then have one more axis, of length 3.

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(s=s, theta=theta, zeta=zeta)
    s, theta, zeta = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (s, theta, zeta)))
    shape = s.shape
    s, theta, zeta = s.ravel(), theta.ravel(), zeta.ravel()
    s1d, nfp = np.asarray(record["s1d"], dtype=float), int(record["nfp"])
    # the cubic spline through the flux surfaces is linear in their values: weights per point
    weights = CubicSpline(np.sqrt(s1d), np.eye(len(s1d)))(np.sqrt(np.clip(s, 0, None)))
    R, Z, nu = (_interpolate_boozer(record[name], len(s1d), nfp, weights, theta, zeta)
                for name in ("R_boozer", "Z_boozer", "nu_boozer"))
    phi = zeta - nu
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1).reshape(shape + (3,))


def _interpolate_boozer(values, n_s, nfp, weights, theta, zeta, block=4096):
    """Interpolate a field on the Boozer grid at points, spectrally in the angles.

    Args:
        values (array_like): the field, flattened over (s, zeta, theta), on uniform angle grids
            that start at 0 and span one field period in zeta and 2 pi in theta.
        n_s (int): number of flux surfaces of the grid.
        nfp (int): number of field periods.
        weights (np.ndarray): radial interpolation weights of each point on each surface,
            shape (n, n_s).
        theta, zeta (np.ndarray): Boozer angles of the points [rad], shape (n,).
        block (int): points evaluated at once.

    Returns:
        np.ndarray: the field at the points, shape (n,).
    """
    n = int(round(np.sqrt(np.size(values) / n_s)))  # the zeta and theta grids have n points each
    grid = np.asarray(values, dtype=float).reshape(n_s, n, n)
    coef = np.fft.fft2(grid, axes=(1, 2)) / (grid.shape[1] * grid.shape[2])
    k_zeta = np.fft.fftfreq(grid.shape[1], 1 / grid.shape[1]) * nfp
    k_theta = np.fft.fftfreq(grid.shape[2], 1 / grid.shape[2])
    out = np.empty(len(theta))
    for a in range(0, len(theta), block):
        b = slice(a, a + block)
        c = (weights[b] @ coef.reshape(n_s, -1)).reshape(-1, n, n)  # coefficients at each point's s
        c = (c * np.exp(1j * np.outer(zeta[b], k_zeta))[:, :, None]).sum(axis=1)
        out[b] = (c * np.exp(1j * np.outer(theta[b], k_theta))).sum(axis=1).real
    return out


def read_trajectories(f, config):
    """Read the true orbits of one stellarator configuration of a trajectory file.

    Args:
        f (h5py.File): an open trajectory file, e.g. ``h5py.File(path, "r")``.
        config (int or str): the configuration, by its index in the file or its uuid.

    Returns:
        dict: "z" (np.ndarray), the embedded states (see ``embed_state``) on the 1 us
        grid, shape (n_orbits, 1000, 5), where row k is lead time k * DT after the first
        sample and rows after the last on-grid sample are NaN; "n_valid" (np.ndarray), the
        number of on-grid samples of each orbit, shape (n_orbits,); and "t_loss"
        (np.ndarray), the loss time of each orbit [s], NaN if confined, shape (n_orbits,).
        Orbits with n_valid = 0 were lost before their first sample.

    Raises:
        KeyError: if a uuid is given that is not in the file.
    """
    i = _config_index(f, config)
    c0, c1 = f["config_offsets"][i], f["config_offsets"][i + 1]
    offsets = f["traj_offsets"][c0:c1 + 1]
    block = f["traj_data"][offsets[0]:offsets[-1]]  # columns: t, s, theta, phi, vpar
    starts, lengths = offsets[:-1] - offsets[0], np.diff(offsets)
    t_last = block[starts + lengths - 1, 0]
    lost = t_last < T_MAX - LOSS_TOL
    n_valid = lengths - lost  # a lost orbit's final sample is off the grid

    k = np.arange(N_STEPS + 1)
    on_grid = k[None, :] < n_valid[:, None]
    rows = (starts[:, None] + k[None, :])[on_grid]
    z = np.full((len(lengths), N_STEPS + 1, 5), np.nan)
    z[on_grid] = embed_state(*block[rows, 1:].T)
    return {"z": z, "n_valid": n_valid, "t_loss": np.where(lost, t_last, np.nan)}


def trajectory_metrics(paths, predictions):
    """Evaluate trajectory predictions against the true orbits using the metrics of the paper.

    For each predicted stellarator configuration, the predicted states and loss times of
    its orbits are compared with the truth, and over all of them the function computes the
    nRMSE at lead times 10, 100 and 999 us (``nrmse``), the mean absolute error of the
    loss fraction (``loss_fraction_mae``), and the Wasserstein-1 distance of the
    confinement times (``confinement_time_w1``). Orbits lost before their first sample are
    not scored. The benchmark scores every configuration of the test files;
    ``n_configurations`` in the result says how many were scored.

    As in the paper, a predicted particle is lost at the first time its predicted s >= 0.98,
    and after a predicted loss its predicted state is held at the loss state.

    Args:
        paths (list of str): trajectory files holding the predicted configurations, usually
            ``data.list_files(data_dir, "trajectory", "test")``.
        predictions (dict): maps the uuid of each predicted configuration to a tuple
            (z, t_loss), with one row per orbit, in the order of ``read_trajectories``.
            z (np.ndarray) holds the predicted embedded states at lead times 10, 100 and
            999 us, shape (n_orbits, 3, 5). t_loss (np.ndarray) holds the predicted loss
            times [s], measured like the data from the start of the simulation (the first
            sample is at t = 1 us), NaN for particles predicted to stay confined, shape
            (n_orbits,). Rows of orbits lost before their first sample are ignored.

    Returns:
        dict: "nrmse_10_microsec", "nrmse_100_microsec", "nrmse_999_microsec" (the nRMSE at
        each lead time), "loss_fraction_mae", "confinement_time_w1_microsec" (float), and
        "n_configurations" (int).

    Raises:
        KeyError: if a predicted configuration is not in the files.
        ValueError: if a prediction has the wrong shape, or contains inf, or NaN states.
    """
    z_pred, z_true, t_pred, t_true, in_files = [], [], [], [], set()
    for path in paths:
        with h5py.File(path, "r") as f:
            uuids = _read_rows(f["uuid"], slice(None))
            in_files.update(uuids)
            for i, uuid in enumerate(uuids):
                if uuid in predictions:
                    truth = read_trajectories(f, i)
                    keep = truth["n_valid"] >= 1  # orbits lost before their first sample
                    z, t_loss = _checked_prediction(*predictions[uuid], len(keep))
                    z_pred.append(z[keep])
                    t_pred.append(t_loss[keep])
                    z_true.append(truth["z"][keep][:, TABLE_STEPS])
                    t_true.append(truth["t_loss"][keep])
    missing = set(predictions) - in_files
    if missing:
        raise KeyError(f"{len(missing)} predicted configurations are not in the files, e.g. {next(iter(missing))}")
    out = {f"nrmse_{k}_microsec": float(v) for k, v in zip(TABLE_STEPS, nrmse(z_pred, z_true))}
    out["loss_fraction_mae"] = loss_fraction_mae(t_pred, t_true)
    out["confinement_time_w1_microsec"] = confinement_time_w1(t_pred, t_true)
    out["n_configurations"] = len(z_pred)
    return out


def nrmse(z_pred, z_true):
    r"""Normalized root-mean-square error of predicted states, averaged over configurations.

    For the channel groups :math:`g` of the embedded state: :math:`x`, :math:`y`,
    :math:`(\cos\varphi, \sin\varphi)` and :math:`\lambda`, with frozen training-set RMS
    :math:`\sigma_g`,

    .. math::

        \mathrm{nRMSE}(\tau) = \frac{1}{4} \sum_g \frac{1}{\sigma_g} \frac{1}{D} \sum_d
        \left( \frac{1}{n_{d,\tau}} \sum_i \left\|\hat z_{i,g}(\tau) - z_{i,g}(\tau)\right\|^2 \right)^{1/2},

    where :math:`i` runs over the :math:`n_{d,\tau}` particles of configuration :math:`d`
    still confined in the truth at :math:`\tau`, and :math:`d` over the :math:`D`
    configurations with any.

    Args:
        z_pred (list of np.ndarray): per configuration, the predicted embedded states of its
            particles at the lead times, shape (n_d, n_tau, 5).
        z_true (list of np.ndarray): per configuration, the true states, of the same
            shapes; NaN where the particle is no longer confined.

    Returns:
        np.ndarray: the nRMSE at each lead time, shape (n_tau,).

    Raises:
        ValueError: if a predicted state contains NaN or inf.
    """
    rmse = {g: [] for g in CHANNEL_GROUPS}
    for zp, zt in zip(z_pred, z_true):
        require_finite(z_pred=zp)
        valid = np.isfinite(zt).all(axis=-1)
        n_valid = valid.sum(axis=0)
        for g, (channels, _) in CHANNEL_GROUPS.items():
            err2 = ((zp[..., channels] - zt[..., channels]) ** 2).sum(axis=-1)
            sse = np.where(valid, err2, 0).sum(axis=0)
            rmse[g].append(np.where(n_valid > 0, np.sqrt(sse / np.maximum(n_valid, 1)), np.nan))
    return np.mean([np.nanmean(rmse[g], axis=0) / sigma for g, (_, sigma) in CHANNEL_GROUPS.items()], axis=0)


def loss_fraction_mae(t_loss_pred, t_loss_true):
    r"""Mean absolute error of the fraction of particles lost within 1 ms, over configurations.

    With :math:`f_d` the fraction of the particles of configuration :math:`d` lost by
    :math:`t = 1` ms,

    .. math::

        \text{MAE} = \frac{1}{D} \sum_d \left|\hat f_d - f_d\right|.

    Args:
        t_loss_pred (list of np.ndarray): per configuration, the predicted loss times of its
            particles [s], NaN if confined, shape (n_d,).
        t_loss_true (list of np.ndarray): per configuration, the true loss times, of the same shapes.

    Returns:
        float: the MAE.

    Raises:
        ValueError: if a loss time is infinite.
    """
    errors = [abs(_lost(tp).mean() - _lost(tt).mean()) for tp, tt in zip(t_loss_pred, t_loss_true)]
    return float(np.mean(errors))


def confinement_time_w1(t_loss_pred, t_loss_true):
    r"""Wasserstein-1 distance of predicted and true confinement times, averaged over configurations.

    With :math:`T = \min(t_\text{loss}, 1\text{ ms})` the confinement time of a particle,
    and :math:`T_{d,(i)}` the :math:`i`-th smallest of the :math:`n_d` confinement times
    of configuration :math:`d`,

    .. math::

        W_1 = \frac{1}{D} \sum_d \frac{1}{n_d} \sum_i \left|\hat T_{d,(i)} - T_{d,(i)}\right|.

    Args:
        t_loss_pred (list of np.ndarray): per configuration, the predicted loss times of its
            particles [s], NaN if confined, shape (n_d,).
        t_loss_true (list of np.ndarray): per configuration, the true loss times, of the same shapes.

    Returns:
        float: W1 [us].

    Raises:
        ValueError: if a loss time is infinite.
    """
    distances = []
    for tp, tt in zip(t_loss_pred, t_loss_true):
        T_pred = np.where(_lost(tp), tp, T_MAX) * 1e6
        T_true = np.where(_lost(tt), tt, T_MAX) * 1e6
        distances.append(np.mean(np.abs(np.sort(T_pred) - np.sort(T_true))))
    return float(np.mean(distances))


def _lost(t_loss):
    """Flag the particles lost within the 1 ms window; a loss at the last sample, t = 1 ms, counts.

    Args:
        t_loss (np.ndarray): loss times [s], NaN if confined.

    Returns:
        np.ndarray: boolean, the shape of t_loss.

    Raises:
        ValueError: if a loss time is infinite.
    """
    t_loss = np.asarray(t_loss, dtype=float)
    require_finite(t_loss=t_loss[~np.isnan(t_loss)])
    return t_loss <= T_MAX + DT / 2


def _checked_prediction(z, t_loss, n_orbits):
    """Check the shapes of a configuration's prediction and convert it to arrays.

    Args:
        z (array_like): predicted states, expected shape (n_orbits, 3, 5).
        t_loss (array_like): predicted loss times [s], expected shape (n_orbits,).
        n_orbits (int): the number of orbits of the configuration.

    Returns:
        tuple: (z, t_loss) as float np.ndarray.

    Raises:
        ValueError: if a shape is wrong.
    """
    z, t_loss = np.asarray(z, dtype=float), np.asarray(t_loss, dtype=float)
    if z.shape != (n_orbits, len(TABLE_STEPS), 5) or t_loss.shape != (n_orbits,):
        raise ValueError(f"expected z of shape {(n_orbits, len(TABLE_STEPS), 5)} and t_loss of "
                         f"shape {(n_orbits,)}, got {z.shape} and {t_loss.shape}")
    return z, t_loss
