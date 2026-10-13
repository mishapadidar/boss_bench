"""Trajectory benchmark: predict guiding-center orbits from their initial state.

Orbits are sampled every DT = 1 us, from t = 1 us up to T_MAX = 1 ms. A lost orbit ends
with one extra, off-grid sample at its loss time. A model is given an orbit's first
sample z0, in the embedded form of ``embed_state``, and predicts the state at lead times
tau = k * DT after it, k = 1, ..., 999, and whether and when the particle is lost.
"""

import h5py
import numpy as np

from ._checks import require_finite
from .confinement import LOSS_TOL
from .data import _read_rows

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
        s (np.ndarray): normalized toroidal flux, in [0, 1].
        theta (np.ndarray): Boozer poloidal angle [rad], the shape of s.
        phi (np.ndarray): Boozer toroidal angle [rad], the shape of s.
        vpar (np.ndarray): parallel velocity [m/s], the shape of s.

    Returns:
        np.ndarray: the embedded states, shape s.shape + (5,).
    """
    r = np.sqrt(s)
    return np.stack([r * np.cos(theta), r * np.sin(theta), np.cos(phi), np.sin(phi), vpar / V_ALPHA], axis=-1)


def read_trajectories(f, i):
    """Read the true orbits of one stellarator configuration of a trajectory file.

    Args:
        f (h5py.File): an open trajectory file, e.g. ``h5py.File(path, "r")``.
        i (int): index of the configuration in the file.

    Returns:
        dict: "z" (np.ndarray), the embedded states (see ``embed_state``) on the 1 us
        grid, shape (n_orbits, 1000, 5), where row k is lead time k * DT after the first
        sample and rows after the last on-grid sample are NaN; "n_valid" (np.ndarray), the
        number of on-grid samples of each orbit, shape (n_orbits,); and "t_loss"
        (np.ndarray), the loss time of each orbit [s], NaN if confined, shape (n_orbits,).
        Orbits with n_valid = 0 were lost before their first sample.
    """
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
    r"""Score trajectory predictions.

    Following the paper, a predicted particle is lost at the first time its predicted
    :math:`s \ge 0.98`, and after a predicted loss its predicted state is held at the
    loss state. Orbits lost before their first sample are not scored. For the channel
    groups :math:`g`: :math:`x`, :math:`y`, :math:`(\cos\varphi, \sin\varphi)` and
    :math:`\lambda`, with frozen training-set RMS :math:`\sigma_g`,

    .. math::

        \mathrm{nRMSE}(\tau) = \frac{1}{4} \sum_g \frac{1}{\sigma_g} \frac{1}{D} \sum_d
        \left( \frac{1}{n_{d,\tau}} \sum_i \left\|\hat z_{i,g}(\tau) - z_{i,g}(\tau)\right\|^2 \right)^{1/2},

    where :math:`i` runs over the :math:`n_{d,\tau}` particles of configuration :math:`d`
    still confined in the truth at :math:`\tau`, and :math:`d` over the :math:`D`
    configurations with any. With :math:`f_d` the fraction of the particles of
    configuration :math:`d` lost within 1 ms, and :math:`T = \min(t_\text{loss}, 1\text{ ms})`,

    .. math::

        \text{loss-fraction MAE} = \frac{1}{D} \sum_d |\hat f_d - f_d|, \qquad
        W_1 = \frac{1}{D} \sum_d W_1\!\left(\hat T_d, T_d\right),

    where :math:`W_1(\hat T_d, T_d)` is the Wasserstein-1 distance between the predicted
    and true distributions of :math:`T` over the particles of configuration :math:`d`.
    The benchmark scores every configuration of the test files; ``n_configurations`` in
    the result says how many were scored.

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
        dict: "nrmse_tau10us", "nrmse_tau100us", "nrmse_tau999us", "loss_fraction_mae",
        "w1_us" (float), and "n_configurations" (int).

    Raises:
        KeyError: if a predicted configuration is not in the files.
        ValueError: if a prediction has the wrong shape, or contains inf, or NaN states.
    """
    scores, in_files = [], set()
    for path in paths:
        with h5py.File(path, "r") as f:
            uuids = _read_rows(f["uuid"], slice(None))
            in_files.update(uuids)
            for i, uuid in enumerate(uuids):
                if uuid in predictions:
                    truth = read_trajectories(f, i)
                    scores.append(_score_configuration(truth, *predictions[uuid]))
    missing = set(predictions) - in_files
    if missing:
        raise KeyError(f"{len(missing)} predicted configurations are not in the files, e.g. {next(iter(missing))}")

    nrmse = np.mean([np.nanmean([s["rmse"][g] for s in scores], axis=0) / rms
                     for g, (_, rms) in CHANNEL_GROUPS.items()], axis=0)
    out = {f"nrmse_tau{k}us": float(v) for k, v in zip(TABLE_STEPS, nrmse)}
    out["loss_fraction_mae"] = float(np.mean([abs(s["f_pred"] - s["f_true"]) for s in scores]))
    out["w1_us"] = float(np.mean([s["w1_us"] for s in scores]))
    out["n_configurations"] = len(scores)
    return out


def _score_configuration(truth, z_pred, t_pred):
    """Score the predicted orbits of one configuration.

    Args:
        truth (dict): the configuration's true orbits, from ``read_trajectories``.
        z_pred (np.ndarray): predicted states at TABLE_STEPS, shape (n_orbits, 3, 5).
        t_pred (np.ndarray): predicted loss times [s], NaN if confined, shape (n_orbits,).

    Returns:
        dict: "rmse", the RMSE of each channel group at each scored lead time over the
        particles still confined in the truth; the true and predicted loss fractions
        "f_true", "f_pred"; and "w1_us", the Wasserstein-1 distance between the true and
        predicted distributions of min(loss time, T_MAX) [us].

    Raises:
        ValueError: if a prediction has the wrong shape, or contains inf, or NaN states.
    """
    n_orbits = len(truth["n_valid"])
    z_pred, t_pred = np.asarray(z_pred, dtype=float), np.asarray(t_pred, dtype=float)
    if z_pred.shape != (n_orbits, len(TABLE_STEPS), 5) or t_pred.shape != (n_orbits,):
        raise ValueError(f"expected z of shape {(n_orbits, len(TABLE_STEPS), 5)} and t_loss of "
                         f"shape {(n_orbits,)}, got {z_pred.shape} and {t_pred.shape}")
    keep = truth["n_valid"] >= 1  # orbits lost before their first sample have no initial state
    z_pred, t_pred = z_pred[keep], t_pred[keep]
    require_finite(z=z_pred, t_loss=t_pred[~np.isnan(t_pred)])
    z_true, t_true = truth["z"][keep][:, TABLE_STEPS], truth["t_loss"][keep]

    valid = np.isfinite(z_true).all(-1)
    n_valid = valid.sum(axis=0)
    rmse = {}
    for g, (channels, _) in CHANNEL_GROUPS.items():
        err2 = ((z_pred[..., channels] - z_true[..., channels]) ** 2).sum(-1)
        sse = np.where(valid, err2, 0).sum(axis=0)
        rmse[g] = np.where(n_valid > 0, np.sqrt(sse / np.maximum(n_valid, 1)), np.nan)

    lost_true = np.isfinite(t_true)
    lost_pred = t_pred <= T_MAX + DT / 2  # a loss at the last sample, t = 1 ms, counts; NaN does not
    T_true = np.where(lost_true, t_true, T_MAX) * 1e6
    T_pred = np.where(lost_pred, t_pred, T_MAX) * 1e6
    return {
        "rmse": rmse,
        "f_true": lost_true.mean(),
        "f_pred": lost_pred.mean(),
        "w1_us": np.mean(np.abs(np.sort(T_true) - np.sort(T_pred))),
    }

