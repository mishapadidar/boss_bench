"""Trajectory benchmark: predict guiding-center orbits from their initial state.

Orbits are sampled every DT = 1 us, from t = 1 us up to T_MAX = 1 ms. A lost orbit ends
with one extra, off-grid sample at its loss time. A model is given an orbit's first
sample z0 and predicts the embedded state::

    z = (sqrt(s) cos(theta), sqrt(s) sin(theta), cos(phi), sin(phi), vpar / V_ALPHA)

at lead times tau = k * DT after it, k = 1, ..., 999. A predicted particle is lost when
its predicted s first reaches 0.98, and its prediction is held at that state afterwards.
Model outputs must be finite.
"""

import h5py
import numpy as np

from ._checks import require_finite
from .confinement import LOSS_TOL
from .data import load_record

DT = 1e-6
T_MAX = 1e-3
N_STEPS = 999
TABLE_STEPS = np.array([10, 100, 999])
S_LOSS = 0.98
N_BISECT = 8
V_ALPHA = np.sqrt(2 * 3.5e6 * 1.602176634e-19 / 6.6446573450e-27)  # 3.5 MeV alpha speed [m/s]

# Channels of z in each group, and the group's frozen RMS over the training set.
CHANNEL_GROUPS = {
    "pol_x": (slice(0, 1), 0.3958040045685409),
    "pol_y": (slice(1, 2), 0.4034089137677494),
    "phi": (slice(2, 4), 1.0),
    "lambda": (slice(4, 5), 0.5778076495076678),
}


def embed(s, theta, phi, vpar):
    r"""Embed a guiding-center state. Models take and return states in this form.

    .. math::

        z = \left(\sqrt{s}\cos\theta,\ \sqrt{s}\sin\theta,\ \cos\varphi,\ \sin\varphi,\
        v_\parallel / v_\alpha\right),

    with :math:`v_\alpha` the speed of a 3.5 MeV alpha particle.

    Args:
        s: normalized toroidal flux, in [0, 1].
        theta, phi: Boozer poloidal and toroidal angles [rad].
        vpar: parallel velocity [m/s].

    Returns:
        Array of shape s.shape + (5,), holding
        (sqrt(s) cos theta, sqrt(s) sin theta, cos phi, sin phi, vpar / V_ALPHA).
    """
    r = np.sqrt(s)
    return np.stack([r * np.cos(theta), r * np.sin(theta), np.cos(phi), np.sin(phi), vpar / V_ALPHA], axis=-1)


def read_device(f, i):
    """Read all orbits of one configuration of a trajectory file.

    Args:
        f: an open trajectory file, e.g. ``h5py.File(path, "r")``.
        i: index of the configuration in the file.

    Returns:
        Dict with "z", the (n, 1000, 5) embedded states on the 1 us grid (row k is lead
        time k * DT; NaN after the last on-grid sample), "n_valid", the number of on-grid
        samples of each orbit, and "t_loss", the loss time of each orbit [s] (NaN if confined).
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
    z[on_grid] = embed(*block[rows, 1:].T)
    return {"z": z, "n_valid": n_valid, "t_loss": np.where(lost, t_last, np.nan)}


def evaluate_trajectories(paths, rollout=None, predict_at=None):
    r"""Score a model on every orbit of the given trajectory files.

    Give exactly one of ``rollout`` and ``predict_at``. In both, z0 is (n, 5) and record is
    the configuration's fields (from ``data.load_record``).

    A predicted particle is lost at the first lead time at which its predicted
    :math:`s \ge 0.98`; for ``predict_at`` this time is refined by 8 bisection steps.
    After a predicted loss, the prediction is held at the loss state. Orbits lost before
    their first sample are not scored. For the channel groups :math:`g`: :math:`x`,
    :math:`y`, :math:`(\cos\varphi, \sin\varphi)` and :math:`\lambda`, with frozen
    training-set RMS :math:`\sigma_g`,

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

    Args:
        paths: trajectory file paths, usually ``data.list_files(data_dir, "trajectory", "test")``.
        rollout: function (z0, record, n_steps) -> (n, n_steps, 5), the predicted states
            at lead times 1, ..., n_steps us.
        predict_at: function (z0, tau, record) -> (n, 5), the predicted states at lead
            times tau (n,) [s]. The loss time is refined by bisection between grid points.

    Returns:
        Dict with the nRMSE at lead times 10, 100 and 999 us ("nrmse_tau10us", ...), the
        mean absolute error of the per-configuration loss fraction ("loss_fraction_mae"),
        and the confinement-time Wasserstein-1 distance [us] ("w1_us").

    Raises:
        ValueError: if not exactly one model is given, or if the model returns NaN or inf.
    """
    if (rollout is None) == (predict_at is None):
        raise ValueError("give exactly one of rollout and predict_at")
    scores = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for i in range(len(f["config_offsets"]) - 1):
                record = load_record(f, i)
                truth = read_device(f, i)
                keep = truth["n_valid"] >= 1  # orbits lost within 1 us have no initial state
                if not keep.any():
                    continue
                z0 = truth["z"][keep, 0]
                if rollout is not None:
                    z_pred, t_pred = _predict_rollout(rollout, z0, record)
                else:
                    z_pred, t_pred = _predict_direct(predict_at, z0, record)
                z_true = truth["z"][keep][:, TABLE_STEPS]
                scores.append(_score_device(z_true, truth["t_loss"][keep], z_pred, t_pred))

    nrmse = np.mean([np.nanmean([s["rmse"][g] for s in scores], axis=0) / rms
                     for g, (_, rms) in CHANNEL_GROUPS.items()], axis=0)
    out = {f"nrmse_tau{k}us": float(v) for k, v in zip(TABLE_STEPS, nrmse)}
    out["loss_fraction_mae"] = float(np.mean([abs(s["f_pred"] - s["f_true"]) for s in scores]))
    out["w1_us"] = float(np.mean([s["w1_us"] for s in scores]))
    return out


def _score_device(z_true, t_true, z_pred, t_pred):
    """Score the predicted orbits of one configuration.

    Args:
        z_true: true states at TABLE_STEPS, (n, len(TABLE_STEPS), 5), NaN once lost.
        t_true: true loss times [s], NaN if confined.
        z_pred: predicted states at TABLE_STEPS, same shape as z_true.
        t_pred: predicted loss times [s], NaN if confined.

    Returns:
        Dict with "rmse", the RMSE of each channel group at each table lead time over the
        particles still confined in the truth; the true and predicted loss fractions
        "f_true", "f_pred"; and "w1_us", the Wasserstein-1 distance between the true and
        predicted distributions of min(loss time, T_MAX) [us].
    """
    valid = np.isfinite(z_true).all(-1)
    n_valid = valid.sum(axis=0)
    rmse = {}
    for g, (channels, _) in CHANNEL_GROUPS.items():
        err2 = ((z_pred[..., channels] - z_true[..., channels]) ** 2).sum(-1)
        sse = np.where(valid, err2, 0).sum(axis=0)
        rmse[g] = np.where(n_valid > 0, np.sqrt(sse / np.maximum(n_valid, 1)), np.nan)

    lost_true = np.isfinite(t_true)
    lost_pred = t_pred <= T_MAX + DT / 2  # a crossing at the last sample, t = 1 ms, counts as lost
    T_true = np.where(lost_true, t_true, T_MAX) * 1e6
    T_pred = np.where(lost_pred, t_pred, T_MAX) * 1e6
    return {
        "rmse": rmse,
        "f_true": lost_true.mean(),
        "f_pred": lost_pred.mean(),
        "w1_us": np.mean(np.abs(np.sort(T_true) - np.sort(T_pred))),
    }


def _predict_rollout(rollout, z0, record):
    """Run a rollout model and read off its predictions.

    Args:
        rollout: the model, see ``evaluate_trajectories``.
        z0: initial states, (n, 5).
        record: the configuration's fields.

    Returns:
        (z, t_loss): the predicted states at TABLE_STEPS, (n, len(TABLE_STEPS), 5), held
        at the loss state after a predicted loss; and the predicted loss times [s], NaN if confined.
    """
    predicted = rollout(z0, record, N_STEPS)
    require_finite(rollout_output=predicted)
    traj = np.concatenate([z0[:, None], predicted], axis=1)  # row k is lead time k
    crossed = _flux_label(traj) >= S_LOSS
    first = np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1)
    last = np.where(first >= 0, first, N_STEPS)
    z = np.take_along_axis(traj, np.minimum(TABLE_STEPS[None, :], last[:, None])[..., None], axis=1)
    t_loss = np.where(first >= 0, DT + first * DT, np.nan)
    return z, t_loss


def _predict_direct(predict_at, z0, record):
    """Query a direct model on the 1 us grid and read off its predictions.

    The loss time is first located on the grid, then refined by N_BISECT bisection steps.
    Every model output is checked to be finite.

    Args:
        predict_at: the model, see ``evaluate_trajectories``.
        z0: initial states, (n, 5).
        record: the configuration's fields.

    Returns:
        (z, t_loss), as in ``_predict_rollout``.
    """
    n = len(z0)
    z = np.full((n, len(TABLE_STEPS), 5), np.nan)
    first = np.full(n, -1)
    z_first = np.full((n, 5), np.nan)
    for k in range(N_STEPS + 1):
        z_k = z0 if k == 0 else _query(predict_at, z0, np.full(n, k * DT), record)
        new = (first < 0) & (_flux_label(z_k) >= S_LOSS)
        first[new] = k
        z_first[new] = z_k[new]
        if k in TABLE_STEPS:
            z[:, TABLE_STEPS == k] = z_k[:, None]
    after_loss = (first[:, None] >= 0) & (TABLE_STEPS[None, :] > first[:, None])
    z = np.where(after_loss[..., None], z_first[:, None], z)

    t_hit = np.where(first >= 0, first * DT, np.nan)
    m = first > 0
    inside, outside = (first[m] - 1) * DT, first[m] * DT
    for _ in range(N_BISECT):
        mid = (inside + outside) / 2
        out = _flux_label(_query(predict_at, z0[m], mid, record)) >= S_LOSS
        inside, outside = np.where(out, inside, mid), np.where(out, mid, outside)
    t_hit[m] = outside
    return z, DT + t_hit


def _query(predict_at, z0, tau, record):
    """Call a direct model and check that its output is finite.

    Args:
        predict_at: the model, see ``evaluate_trajectories``.
        z0: initial states, (n, 5).
        tau: lead times [s], (n,).
        record: the configuration's fields.

    Returns:
        The predicted states, (n, 5).

    Raises:
        ValueError: if the model returns NaN or inf.
    """
    z = predict_at(z0, tau, record)
    require_finite(predict_at_output=z)
    return z


def _flux_label(z):
    """Recover s from embedded states.

    Args:
        z: embedded states, (..., 5).

    Returns:
        s = z[..., 0]^2 + z[..., 1]^2, of shape z.shape[:-1].
    """
    return z[..., 0] ** 2 + z[..., 1] ** 2
