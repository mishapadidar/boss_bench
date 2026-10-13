"""Heat-flux benchmark: predict where lost alpha particles strike the plasma boundary.

A sample is one configuration and one birth surface ``s0``. For each sample, a model
predicts the loss fraction and the strike points of the lost particles: the Boozer
angles (zeta, theta) where they reach the boundary, and their energies. A particle lost
at time tau carries energy ``3.5 MeV * exp(-2 tau / t_max)``.

Predictions may use any field of a configuration except the outcomes of the particle
tracing: ``confinement_time``, ``s_end``, ``theta_end``, ``zeta_end`` and ``vpar_end``.
"""

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist

from ._checks import require_finite
from .confinement import E_ALPHA_MEV, S_INITS, is_lost
from .data import _read_rows

HF_RESOLUTIONS = (16, 32)
MIN_LOST_ED = 10
BLOCK_SIZE = 64  # configurations read from disk at once
PARTICLE_FIELDS = ("uuid", "nfp", "t_max", "s_init", "confinement_time", "zeta_end", "theta_end")
BOOZER_FIELDS = {"R": "R_boozer", "Z": "Z_boozer", "nu": "nu_boozer"}  # boundary geometry
HEAT_FLUX_FIELDS = PARTICLE_FIELDS + tuple(BOOZER_FIELDS.values()) + ("s1d",)  # read by heat_flux_targets


def evaluate_heat_flux(paths, predictions):
    """Score heat-flux predictions on every sample of the given confinement files.

    Each sample is scored with ``heat_flux_metrics``. The reported numbers are the median
    energy distance (over samples with at least 10 lost particles), the median and mean
    heat-flux error on 16 x 16 and 32 x 32 grids, and the median and mean absolute error
    of the loss fraction. Only the fields needed are read, in blocks of configurations.

    Args:
        paths: confinement files, usually ``data.list_files(data_dir, "confinement", "test")``.
        predictions: dict mapping the (uuid, s0) of every sample to a tuple
            (zeta, theta, energy, loss_fraction): the Boozer angles [rad] and energies [MeV]
            of any number of predicted strike points, and the predicted loss fraction.

    Returns:
        Dict with "ED_median", "HF_16_median", "HF_16_mean", "HF_32_median", "HF_32_mean",
        "loss_fraction_median_abs_error" and "loss_fraction_mean_abs_error".

    Raises:
        KeyError: if a sample has no prediction.
        ValueError: if a prediction contains NaN or inf, or has no strike points for a
            sample with an energy distance.
    """
    results = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for record in _read_configurations(f):
                for target in _targets(record):
                    key = (record["uuid"], target["s_init"])
                    if key not in predictions:
                        raise KeyError(f"no prediction for sample {key}")
                    results.append(heat_flux_metrics(*predictions[key], target))
    return _summarize(results)


def heat_flux_targets(record):
    """Build the true samples of one configuration, one per birth surface.

    Args:
        record: the configuration's fields, from ``data.load_record``; at least those in
            ``HEAT_FLUX_FIELDS``.

    Returns:
        List of dicts, one per birth surface, with the birth surface "s_init", the number
        of particles launched "n_launched", the true "loss_fraction", the lost particles'
        strike points "zeta", "theta" [rad] and energies "energy" [MeV], and the boundary
        geometry "R", "Z", "nu" (on the (zeta, theta) grid of one field period) and "nfp".

    Raises:
        ValueError: if the confinement times or the lost particles' strike points contain NaN or inf.
    """
    boundary = {key: _boundary_slice(record, name) for key, name in BOOZER_FIELDS.items()}
    return _targets({**record, **boundary})


def heat_flux_metrics(zeta, theta, energy, loss_fraction, target):
    r"""Score one predicted sample against its target.

    The heat flux is binned on an :math:`m \times m` grid in :math:`(\zeta, \theta)` over
    one field period, with the strike points of all :math:`n_\text{fp}` field periods
    folded in. Bin :math:`b`, of boundary area :math:`A_b`, receives

    .. math::

        Q_b = \frac{1}{N\, n_\text{fp}\, A_b} \sum_{i \in b} E_i,

    where :math:`N` is the number of particles launched. The predicted strike points are
    scaled so that their total is the predicted loss fraction. Relative to a uniform loss
    of every particle at the birth energy :math:`E_\alpha = 3.5` MeV over the boundary
    area :math:`S`, the heat-flux error is

    .. math::

        \mathrm{HF}_m^2 = \frac{n_\text{fp} \sum_b A_b \left(\hat Q_b - Q_b\right)^2}{E_\alpha^2 / S}.

    With :math:`X` and :math:`Y` the predicted and true strike points as positions on the
    boundary, the energy distance is

    .. math::

        \mathrm{ED} = \left| \frac{2\,\mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\|
        - \mathbb{E}\|Y - Y'\|}{\mathbb{E}\|Y - Y'\|} \right|^{1/2},

    where the expectations of the self terms average over pairs of distinct points.

    Args:
        zeta, theta: Boozer angles of the predicted strike points [rad], any number of them.
        energy: energies of the predicted strike points [MeV].
        loss_fraction: the predicted loss fraction.
        target: the true sample, from ``heat_flux_targets``.

    Returns:
        Dict with "HF_16" and "HF_32", the heat-flux errors for m = 16 and 32; "ED", the
        energy distance (NaN if the target has fewer than 10 lost particles); and
        "loss_fraction_error", the absolute error of the loss fraction.

    Raises:
        ValueError: if a prediction contains NaN or inf, or if the target has an energy
            distance but no strike points are predicted.
    """
    require_finite(zeta=zeta, theta=theta, energy=energy, loss_fraction=loss_fraction)
    out = {"loss_fraction_error": abs(loss_fraction - target["loss_fraction"])}
    for m in HF_RESOLUTIONS:
        out[f"HF_{m}"] = _heat_flux_error(zeta, theta, energy, loss_fraction, target, m)
    out["ED"] = np.nan
    if len(target["zeta"]) >= MIN_LOST_ED:
        if len(zeta) == 0:
            raise ValueError(f"the target has {len(target['zeta'])} lost particles; predict at least one strike point")
        x_pred = _boundary_points(zeta, theta, target)
        x_true = _boundary_points(target["zeta"], target["theta"], target)
        out["ED"] = _energy_distance(x_pred, x_true)
    return out


def _read_configurations(f):
    """Read the fields needed for scoring, a block of configurations at a time.

    Of the Boozer fields only the boundary surface is read.

    Args:
        f: an open confinement file.

    Yields:
        Dict of one configuration's fields, with the boundary geometry as "R", "Z", "nu".
    """
    n_phi, n_theta = f["phi1d"].shape[1], f["theta1d"].shape[1]
    n_configs = len(f["uuid"])
    for start in range(0, n_configs, BLOCK_SIZE):
        rows = slice(start, min(start + BLOCK_SIZE, n_configs))
        block = {name: _read_rows(f[name], rows) for name in PARTICLE_FIELDS}
        for key, name in BOOZER_FIELDS.items():
            boundary = f[name][rows, -n_phi * n_theta:]  # the last surface, s = 1
            block[key] = boundary.reshape(-1, n_phi, n_theta)
        for j in range(rows.stop - rows.start):
            yield {name: values[j] for name, values in block.items()}


def _targets(record):
    """Build the true samples of one configuration whose boundary geometry is already sliced.

    Args:
        record: the fields in PARTICLE_FIELDS, and the boundary geometry "R", "Z", "nu".

    Returns:
        List of target dicts, as from ``heat_flux_targets``.

    Raises:
        ValueError: if the confinement times or the lost particles' strike points contain NaN or inf.
    """
    ct, t_max = record["confinement_time"], record["t_max"]
    geometry = {"R": record["R"], "Z": record["Z"], "nu": record["nu"], "nfp": int(record["nfp"])}
    samples = []
    for s0 in S_INITS:
        born = np.isclose(record["s_init"], s0)
        lost = born & is_lost(ct, t_max)
        require_finite(zeta_end=record["zeta_end"][lost], theta_end=record["theta_end"][lost])
        samples.append({
            "s_init": s0,
            "n_launched": int(born.sum()),
            "loss_fraction": lost.sum() / born.sum(),
            "zeta": np.mod(record["zeta_end"][lost], 2 * np.pi),
            "theta": np.mod(record["theta_end"][lost], 2 * np.pi),
            "energy": E_ALPHA_MEV * np.exp(-2 * ct[lost] / t_max),
            **geometry,
        })
    return samples


def _summarize(results):
    """Aggregate per-sample scores into the reported numbers. NaN entries are skipped.

    Args:
        results: list of ``heat_flux_metrics`` outputs, one per sample.

    Returns:
        Dict with the median ED, the median and mean HF at each resolution, and the
        median and mean absolute loss-fraction error (NaN where there are no values).
    """
    out = {"ED_median": _statistic(np.median, results, "ED")}
    for m in HF_RESOLUTIONS:
        out[f"HF_{m}_median"] = _statistic(np.median, results, f"HF_{m}")
        out[f"HF_{m}_mean"] = _statistic(np.mean, results, f"HF_{m}")
    out["loss_fraction_median_abs_error"] = _statistic(np.median, results, "loss_fraction_error")
    out["loss_fraction_mean_abs_error"] = _statistic(np.mean, results, "loss_fraction_error")
    return out


def _statistic(f, results, key):
    """Apply a statistic to the finite values of one score over all samples.

    Args:
        f: the statistic, e.g. np.median.
        results: list of ``heat_flux_metrics`` outputs.
        key: the score.

    Returns:
        The statistic, or NaN if no sample has a finite value.
    """
    a = np.array([r[key] for r in results], dtype=float)
    a = a[np.isfinite(a)]
    return float(f(a)) if a.size else np.nan


def _heat_flux_error(zeta, theta, energy, loss_fraction, target, m):
    """Relative L2 error of the heat flux Q on the boundary.

    HF^2 = int (Q_pred - Q_true)^2 dA / int Q_ref^2 dA, where Q_ref = 3.5 MeV / (boundary
    area) is the flux if every particle were lost at once and spread evenly. Q is binned
    on an m x m grid over one field period, with all field periods folded in, and the
    predicted strike points are scaled so their total matches the predicted loss fraction.

    Args:
        zeta, theta, energy: the predicted strike points and energies.
        loss_fraction: the predicted loss fraction.
        target: the true sample, from ``heat_flux_targets``.
        m: number of bins per angle.

    Returns:
        HF, a float.
    """
    nfp, n = target["nfp"], target["n_launched"]
    area = _period_bin_areas(target["R"], target["Z"], target["nu"], nfp, m)
    scale = loss_fraction * n / len(energy) if len(energy) else 0.0
    diff = (scale * _bin_energy(zeta, theta, energy, nfp, m)
            - _bin_energy(target["zeta"], target["theta"], target["energy"], nfp, m))
    err2 = (diff ** 2 / area).sum() / (n ** 2 * nfp)  # int (Q_pred - Q_true)^2 dA
    ref2 = E_ALPHA_MEV ** 2 / (nfp * area.sum())      # int Q_ref^2 dA
    return float(np.sqrt(err2 / ref2))


def _bin_energy(zeta, theta, energy, nfp, m):
    """Sum strike-point energies on an m x m (zeta, theta) grid over one field period.

    Args:
        zeta, theta: Boozer angles [rad], any range; zeta is folded into one field period.
        energy: energy of each strike point [MeV].
        nfp: number of field periods.
        m: number of bins per angle.

    Returns:
        (m, m) array of summed energies, indexed (zeta bin, theta bin).
    """
    period = 2 * np.pi / nfp
    return np.histogram2d(np.mod(zeta, period), np.mod(theta, 2 * np.pi), bins=m,
                          range=[[0, period], [0, 2 * np.pi]], weights=energy)[0]


def _energy_distance(x, y):
    """Normalized energy distance between two point clouds.

    ED = sqrt(|2 E|X - Y| - E|X - X'| - E|Y - Y'|| / E|Y - Y'|), where the self terms
    average over pairs of distinct points.

    Args:
        x: predicted points, (n, 3).
        y: true points, (k, 3).

    Returns:
        ED, a float.
    """
    n, k = len(x), len(y)
    exy = cdist(x, y).mean()
    exx = cdist(x, x).sum() / max(n * (n - 1), 1)
    eyy = cdist(y, y).sum() / max(k * (k - 1), 1)
    return float(np.sqrt(abs((2 * exy - exx - eyy) / eyy)))


def _boundary_points(zeta, theta, target):
    """Map Boozer angles to Cartesian positions on the boundary.

    R, Z and nu are interpolated with cubic splines, and the cylindrical angle is zeta - nu.

    Args:
        zeta, theta: Boozer angles [rad].
        target: a sample from ``heat_flux_targets``, for the boundary geometry.

    Returns:
        (n, 3) array of positions [m].
    """
    R, Z, nu, nfp = target["R"], target["Z"], target["nu"], target["nfp"]
    n_zeta, n_theta = R.shape
    period = 2 * np.pi / nfp
    coords = np.stack([np.mod(zeta, period) / period * n_zeta,
                       np.mod(theta, 2 * np.pi) / (2 * np.pi) * n_theta])
    R_i, Z_i, nu_i = (map_coordinates(a, coords, order=3, mode="grid-wrap") for a in (R, Z, nu))
    phi = np.asarray(zeta, dtype=float) - nu_i
    return np.stack([R_i * np.cos(phi), R_i * np.sin(phi), Z_i], axis=1)


def _period_bin_areas(R, Z, nu, nfp, m, sub=8):
    """Boundary area of each bin of an m x m (zeta, theta) grid over one field period.

    Uses a midpoint rule on sub x sub points per bin, interpolating the area element bilinearly.

    Args:
        R, Z, nu: boundary geometry on the (zeta, theta) grid of one field period.
        nfp: number of field periods.
        m: number of bins per angle.
        sub: midpoint-rule points per bin and angle.

    Returns:
        (m, m) array of bin areas [m^2].
    """
    dA = _area_element(R, Z, nu, nfp)
    fz = (np.arange(m * sub) + 0.5) / (m * sub) * dA.shape[0]
    ft = (np.arange(m * sub) + 0.5) / (m * sub) * dA.shape[1]
    vals = map_coordinates(dA, np.stack(np.meshgrid(fz, ft, indexing="ij")), order=1, mode="grid-wrap")
    cell = vals.reshape(m, sub, m, sub).mean(axis=(1, 3))
    return cell * (2 * np.pi / nfp / m) * (2 * np.pi / m)


def _area_element(R, Z, nu, nfp):
    """Area element |dr/dtheta x dr/dzeta| of the boundary, with cylindrical angle zeta - nu.

    Args:
        R, Z, nu: boundary geometry on the (zeta, theta) grid of one field period.
        nfp: number of field periods.

    Returns:
        Array of the same shape as R [m^2 / rad^2].
    """
    R_t, R_z = _d_dtheta(R), _d_dzeta(R, nfp)
    Z_t, Z_z = _d_dtheta(Z), _d_dzeta(Z, nfp)
    Rphi_t = -R * _d_dtheta(nu)
    Rphi_z = R * (1 - _d_dzeta(nu, nfp))
    c_R = Rphi_t * Z_z - Z_t * Rphi_z
    c_phi = Z_t * R_z - R_t * Z_z
    c_Z = R_t * Rphi_z - Rphi_t * R_z
    return np.sqrt(c_R ** 2 + c_phi ** 2 + c_Z ** 2)


def _d_dtheta(a):
    """Spectral derivative along theta, which has period 2 pi.

    Args:
        a: values on the (zeta, theta) grid.

    Returns:
        da/dtheta on the same grid.
    """
    k = np.fft.fftfreq(a.shape[1]) * a.shape[1]
    return np.real(np.fft.ifft(1j * k[None, :] * np.fft.fft(a, axis=1), axis=1))


def _d_dzeta(a, nfp):
    """Spectral derivative along zeta, which has period 2 pi / nfp on the grid.

    Args:
        a: values on the (zeta, theta) grid.
        nfp: number of field periods.

    Returns:
        da/dzeta on the same grid.
    """
    k = np.fft.fftfreq(a.shape[0]) * a.shape[0] * nfp
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(a, axis=0), axis=0))


def _boundary_slice(record, name):
    """Read the boundary (s = 1) slice of a Boozer field.

    Args:
        record: the configuration's fields, from ``data.load_record``.
        name: a field stored flattened over (s, zeta, theta), e.g. "R_boozer".

    Returns:
        (n_zeta, n_theta) array over one field period.
    """
    a = np.asarray(record[name], dtype=float)
    n = int(round(np.sqrt(a.size / len(record["s1d"]))))
    return a[-n * n:].reshape(n, n)
