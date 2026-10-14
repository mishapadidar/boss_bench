"""Heat-flux benchmark: predict where lost alpha particles strike the plasma boundary."""

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


def lost_particles(record, s0):
    r"""Compute the strike points ``(zeta, theta)`` and energies ``energy`` of the lost particles born on one surface.

    For the :math:`k` lost particles born on ``s0``, ``zeta`` and ``theta`` are the Boozer
    angles where they strike the boundary, in :math:`[0, 2\pi)` [rad], and ``energy`` is
    their energy [MeV], each of shape ``(k,)``. A particle lost at time :math:`t_\text{loss}`
    carries the energy :math:`E = 3.5\,\text{MeV}\; e^{-2 t_\text{loss} / t_\text{slowing}}`,
    where the slowing-down time is approximated by :math:`t_\text{slowing} \approx
    t_\text{max} = 0.1` s. ``n_launched`` is the number of particles born on ``s0``.

    Args:
        record (dict): the configuration's fields from ``data.load_record``, including
            ``confinement_time``, ``t_max``, ``s_init``, ``zeta_end`` and ``theta_end``.
        s0 (float): the birth surface.

    Returns:
        tuple: **zeta, theta, energy, n_launched**, the strike points, their energies, and
        the number of particles launched.

    Raises:
        ValueError: if the confinement times or the strike points contain NaN or inf.
    """
    ct, t_max = record["confinement_time"], record["t_max"]
    born = np.isclose(record["s_init"], s0)
    lost = born & is_lost(ct, t_max)
    require_finite(zeta_end=record["zeta_end"][lost], theta_end=record["theta_end"][lost])
    zeta = np.mod(record["zeta_end"][lost], 2 * np.pi)
    theta = np.mod(record["theta_end"][lost], 2 * np.pi)
    energy = E_ALPHA_MEV * np.exp(-2 * ct[lost] / t_max)
    return zeta, theta, energy, int(born.sum())


def binned_heat_flux(zeta, theta, energy, n_launched, record, resolution=32):
    r"""Compute the heat flux on the plasma boundary [MeV/m² per launched particle], binned over one field period.

    The strike points of all :math:`n_\text{fp}` field periods are first folded into one.
    With :math:`m` = ``resolution``, :math:`Q_{ij}` is the heat flux in the bin
    :math:`[\zeta_i, \zeta_{i+1}) \times [\theta_j, \theta_{j+1})`, where
    :math:`\zeta_i = 2\pi i / (m\, n_\text{fp})` and :math:`\theta_j = 2\pi j / m` for
    :math:`i, j = 0, \dots, m - 1`:

    .. math::

        Q_{ij} = \frac{1}{N\, n_\text{fp}\, A_{ij}} \sum_k E_k,

    where the sum runs over the strike points in the bin, with energies :math:`E_k`,
    :math:`A_{ij}` is the boundary area of the bin, and :math:`N` is the number of particles
    launched.

    To estimate the absolute heat flux per unit time [W/m²], multiply :math:`Q` by a birth
    rate [1/s] and by :math:`1.602 \times 10^{-13}` J/MeV.

    Args:
        zeta (np.ndarray): Boozer toroidal angles of the strike points [rad], shape ``(k,)``.
        theta (np.ndarray): Boozer poloidal angles of the strike points [rad], shape ``(k,)``.
        energy (np.ndarray): energies of the strike points [MeV], shape ``(k,)``.
        n_launched (int): the number of particles launched.
        record (dict): the configuration's fields from ``data.load_record``, including
            ``R_boozer``, ``Z_boozer``, ``nu_boozer``, ``s1d`` and ``nfp``.
        resolution (int): the number of bins per angle.

    Returns:
        np.ndarray: **Q**, the heat flux :math:`Q_{ij}` [MeV/m² per launched particle], shape
        ``(resolution, resolution)``.

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(zeta=zeta, theta=theta, energy=energy)
    nfp = int(record["nfp"])
    R, Z, nu = (_boundary_slice(record, name) for name in BOOZER_FIELDS.values())
    area = _period_bin_areas(R, Z, nu, nfp, resolution)
    return _heat_flux_map(zeta, theta, energy, n_launched, nfp, area)


def heat_flux_metrics(paths, predictions):
    """Score heat-flux predictions with the metrics of the paper.

    A sample is the point cloud of strike points of the lost particles born on one surface
    ``s0`` of one configuration, so each configuration gives three samples, one per birth
    surface. Each predicted sample is compared with the true one by the heat-flux error on
    16 × 16 and 32 × 32 grids (``heat_flux_error``), the energy distance between the two
    point clouds (``energy_distance``), and the absolute error of the loss fraction.

    Over all samples, the function reports the median energy distance ``ED_median`` (over
    samples with at least 10 lost particles); the median and mean heat-flux errors
    ``HF_16_median``, ``HF_16_mean``, ``HF_32_median`` and ``HF_32_mean``; the median and
    mean absolute error of the loss fraction ``loss_fraction_median_abs_error`` and
    ``loss_fraction_mean_abs_error``; and the number of samples scored ``n_samples``. Only
    the fields it needs are read from disk.

    Args:
        paths (list of str): the confinement files holding the predicted samples, usually
            ``data.list_files(data_dir, "confinement", "test")``.
        predictions (dict): maps ``(uuid, s0)`` of each sample to the tuple
            ``(zeta, theta, energy, loss_fraction)``: the Boozer angles [rad] and energies
            [MeV] of any number of predicted strike points, arrays of shape ``(k,)``, and the
            predicted loss fraction, a float.

    Returns:
        dict: **ED_median, HF_16_median, HF_16_mean, HF_32_median, HF_32_mean,
        loss_fraction_median_abs_error, loss_fraction_mean_abs_error, n_samples**, the
        metrics above and the number of samples scored.

    Raises:
        KeyError: if a predicted sample is not in the files.
        ValueError: if a prediction contains NaN or inf, or has no strike points for a
            sample with at least 10 lost particles.
    """
    results, scored = [], set()
    for path in paths:
        with h5py.File(path, "r") as f:
            for record in _read_configurations(f):
                for s0 in S_INITS:
                    key = (record["uuid"], s0)
                    if key in predictions:
                        results.append(_sample_metrics(*predictions[key], _target(record, s0)))
                        scored.add(key)
    missing = set(predictions) - scored
    if missing:
        raise KeyError(f"{len(missing)} predicted samples are not in the files, e.g. {next(iter(missing))}")
    return {**_summarize(results), "n_samples": len(results)}


def heat_flux_error(prediction, record, s0, resolution=32):
    r"""Compute the heat-flux error of a prediction for one sample.

    With :math:`Q` and :math:`\hat Q` the true and predicted heat flux from
    ``binned_heat_flux`` on :math:`m \times m` bins of areas :math:`A_b`, relative to a
    uniform loss of every particle at the birth energy :math:`E_\alpha = 3.5` MeV over the
    boundary area :math:`S`,

    .. math::

        \mathrm{HF}_m^2 = \frac{n_\text{fp} \sum_b A_b \left(\hat Q_b - Q_b\right)^2}{E_\alpha^2 / S}.

    The predicted strike points are scaled so that their total is the predicted loss
    fraction.

    Args:
        prediction (tuple): the tuple ``(zeta, theta, energy, loss_fraction)``: the Boozer
            angles [rad] and energies [MeV] of any number of predicted strike points, arrays
            of shape ``(k,)``, and the predicted loss fraction, a float.
        record (dict): the configuration's fields from ``data.load_record`` on a confinement
            file.
        s0 (float): the birth surface.
        resolution (int): the number of bins per angle, :math:`m`.

    Returns:
        float: :math:`\mathrm{HF}_m`.

    Raises:
        ValueError: if a prediction contains NaN or inf.
    """
    zeta, theta, energy, loss_fraction = prediction
    require_finite(zeta=zeta, theta=theta, energy=energy, loss_fraction=loss_fraction)
    return _heat_flux_error(zeta, theta, energy, loss_fraction, _target(_with_boundary(record), s0), resolution)


def energy_distance(prediction, record, s0):
    r"""Compute the energy distance between predicted and true strike points, for one sample.

    With :math:`X` and :math:`Y` the predicted and true strike points as positions on the
    boundary,

    .. math::

        \mathrm{ED} = \left| \frac{2\,\mathbb{E}\|X - Y\| - \mathbb{E}\|X - X'\|
        - \mathbb{E}\|Y - Y'\|}{\mathbb{E}\|Y - Y'\|} \right|^{1/2},

    where the self terms average over pairs of distinct points. It is defined for samples
    with at least 10 lost particles. The energy distance compares only where the particles
    strike, so it uses the predicted angles and ignores the energies and the loss fraction;
    ``prediction`` is the same tuple as for ``heat_flux_error``, so one prediction can be
    scored by both.

    Args:
        prediction (tuple): the tuple ``(zeta, theta, energy, loss_fraction)``: the Boozer
            angles [rad] and energies [MeV] of any number of predicted strike points, arrays
            of shape ``(k,)``, and the predicted loss fraction, a float.
        record (dict): the configuration's fields from ``data.load_record`` on a confinement
            file.
        s0 (float): the birth surface.

    Returns:
        float: :math:`\mathrm{ED}`, or NaN if fewer than 10 particles born on ``s0`` are lost.

    Raises:
        ValueError: if a prediction contains NaN or inf, or has no strike points while at
            least 10 particles are lost.
    """
    zeta, theta, _, _ = prediction
    require_finite(zeta=zeta, theta=theta)
    return _energy_distance_of_sample(zeta, theta, _target(_with_boundary(record), s0))


def _sample_metrics(zeta, theta, energy, loss_fraction, target):
    """Score one predicted sample against its target.

    Args:
        zeta (np.ndarray): Boozer toroidal angles of the predicted strike points [rad].
        theta (np.ndarray): Boozer poloidal angles of the predicted strike points [rad].
        energy (np.ndarray): energies of the predicted strike points [MeV].
        loss_fraction (float): the predicted loss fraction.
        target (dict): the true sample, from ``_target``.

    Returns:
        dict: "HF_16" and "HF_32"; "ED" (NaN with fewer than 10 true lost particles); and
        "loss_fraction_error".

    Raises:
        ValueError: if a prediction contains NaN or inf, or if the target has an energy
            distance but no strike points are predicted.
    """
    require_finite(zeta=zeta, theta=theta, energy=energy, loss_fraction=loss_fraction)
    out = {"loss_fraction_error": abs(loss_fraction - target["loss_fraction"])}
    for m in HF_RESOLUTIONS:
        out[f"HF_{m}"] = _heat_flux_error(zeta, theta, energy, loss_fraction, target, m)
    out["ED"] = _energy_distance_of_sample(zeta, theta, target)
    return out


def _energy_distance_of_sample(zeta, theta, target):
    """The energy distance of predicted strike points from a target's (see ``energy_distance``).

    Args:
        zeta, theta (np.ndarray): Boozer angles of the predicted strike points [rad].
        target (dict): the true sample, from ``_target``.

    Returns:
        float: ED, or NaN if the target has fewer than MIN_LOST_ED lost particles.

    Raises:
        ValueError: if the target has an energy distance but no strike points are predicted.
    """
    if len(target["zeta"]) < MIN_LOST_ED:
        return np.nan
    if len(zeta) == 0:
        raise ValueError(f"the target has {len(target['zeta'])} lost particles; predict at least one strike point")
    x_pred = _boundary_points(zeta, theta, target)
    x_true = _boundary_points(target["zeta"], target["theta"], target)
    return _energy_distance(x_pred, x_true)


def _read_configurations(f):
    """Read the fields needed for scoring, a block of configurations at a time.

    Of the Boozer fields, only the boundary surface is read.

    Args:
        f (h5py.File): an open confinement file.

    Yields:
        dict: one configuration's fields in PARTICLE_FIELDS, and its boundary geometry
        "R", "Z", "nu" on the (zeta, theta) grid of one field period.
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


def _with_boundary(record):
    """Add the boundary geometry "R", "Z", "nu" to a record read with ``data.load_record``.

    Args:
        record (dict): the configuration's fields, including the Boozer fields and ``s1d``.

    Returns:
        dict: the record, with the boundary geometry on the (zeta, theta) grid of one field period.
    """
    return {**record, **{key: _boundary_slice(record, name) for key, name in BOOZER_FIELDS.items()}}


def _target(record, s0):
    """The true sample of one configuration and birth surface, for scoring.

    Args:
        record (dict): the fields in PARTICLE_FIELDS, and the boundary geometry "R", "Z", "nu".
        s0 (float): the birth surface.

    Returns:
        dict: the "s_init", "n_launched", true "loss_fraction", the lost particles'
        "zeta", "theta" and "energy", and the geometry "R", "Z", "nu", "nfp".
    """
    zeta, theta, energy, n_launched = lost_particles(record, s0)
    return {"s_init": s0, "n_launched": n_launched, "loss_fraction": len(zeta) / n_launched,
            "zeta": zeta, "theta": theta, "energy": energy,
            "R": record["R"], "Z": record["Z"], "nu": record["nu"], "nfp": int(record["nfp"])}


def _summarize(results):
    """Aggregate per-sample scores into the reported numbers. NaN entries are skipped.

    Args:
        results (list of dict): ``_sample_metrics`` outputs, one per sample.

    Returns:
        dict: the median ED, the median and mean HF at each resolution, and the median and
        mean absolute loss-fraction error (NaN where there are no values).
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
        f (callable): the statistic, e.g. np.median.
        results (list of dict): ``_sample_metrics`` outputs.
        key (str): the score.

    Returns:
        float: the statistic, or NaN if no sample has a finite value.
    """
    a = np.array([r[key] for r in results], dtype=float)
    a = a[np.isfinite(a)]
    return float(f(a)) if a.size else np.nan


def _heat_flux_error(zeta, theta, energy, loss_fraction, target, m):
    """The heat-flux error HF_m of one predicted sample (see ``heat_flux_error``).

    Args:
        zeta, theta, energy (np.ndarray): the predicted strike points and energies.
        loss_fraction (float): the predicted loss fraction.
        target (dict): the true sample, from ``_target``.
        m (int): number of bins per angle.

    Returns:
        float: HF_m.
    """
    nfp, n = target["nfp"], target["n_launched"]
    area = _period_bin_areas(target["R"], target["Z"], target["nu"], nfp, m)
    q_true = _heat_flux_map(target["zeta"], target["theta"], target["energy"], n, nfp, area)
    scale = loss_fraction * n / len(energy) if len(energy) else 0.0  # total = predicted loss fraction
    q_pred = scale * _heat_flux_map(zeta, theta, energy, n, nfp, area)
    err2 = nfp * (area * (q_pred - q_true) ** 2).sum()  # int (Q_pred - Q_true)^2 dA
    ref2 = E_ALPHA_MEV ** 2 / (nfp * area.sum())        # int Q_ref^2 dA
    return float(np.sqrt(err2 / ref2))


def _heat_flux_map(zeta, theta, energy, n_launched, nfp, area):
    """Bin strike-point energies over one field period and divide by the bin areas.

    Args:
        zeta, theta (np.ndarray): Boozer angles [rad], any range; zeta is folded into one field period.
        energy (np.ndarray): energy of each strike point [MeV].
        n_launched (int): number of particles launched.
        nfp (int): number of field periods.
        area (np.ndarray): bin areas [m^2], shape (m, m), from ``_period_bin_areas``.

    Returns:
        np.ndarray: the heat flux [MeV / m^2 per launched particle], shape (m, m).
    """
    m = area.shape[0]
    period = 2 * np.pi / nfp
    energy_sum = np.histogram2d(np.mod(zeta, period), np.mod(theta, 2 * np.pi), bins=m,
                                range=[[0, period], [0, 2 * np.pi]], weights=energy)[0]
    return energy_sum / (n_launched * nfp * area)


def _energy_distance(x, y):
    """Normalized energy distance between two point clouds (see ``energy_distance``).

    Args:
        x (np.ndarray): predicted points, shape (n, 3).
        y (np.ndarray): true points, shape (k, 3).

    Returns:
        float: ED.
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
        zeta, theta (np.ndarray): Boozer angles [rad].
        target (dict): a sample from ``_target``, for the boundary geometry.

    Returns:
        np.ndarray: positions [m], shape (n, 3).
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
        R, Z, nu (np.ndarray): boundary geometry on the (zeta, theta) grid of one field period.
        nfp (int): number of field periods.
        m (int): number of bins per angle.
        sub (int): midpoint-rule points per bin and angle.

    Returns:
        np.ndarray: bin areas [m^2], shape (m, m).
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
        R, Z, nu (np.ndarray): boundary geometry on the (zeta, theta) grid of one field period.
        nfp (int): number of field periods.

    Returns:
        np.ndarray: the area element [m^2 / rad^2], the shape of R.
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
        a (np.ndarray): values on the (zeta, theta) grid.

    Returns:
        np.ndarray: da/dtheta on the same grid.
    """
    k = np.fft.fftfreq(a.shape[1]) * a.shape[1]
    return np.real(np.fft.ifft(1j * k[None, :] * np.fft.fft(a, axis=1), axis=1))


def _d_dzeta(a, nfp):
    """Spectral derivative along zeta, which has period 2 pi / nfp on the grid.

    Args:
        a (np.ndarray): values on the (zeta, theta) grid.
        nfp (int): number of field periods.

    Returns:
        np.ndarray: da/dzeta on the same grid.
    """
    k = np.fft.fftfreq(a.shape[0]) * a.shape[0] * nfp
    return np.real(np.fft.ifft(1j * k[:, None] * np.fft.fft(a, axis=0), axis=0))


def _boundary_slice(record, name):
    """Read the boundary (s = 1) slice of a Boozer field.

    Args:
        record (dict): the configuration's fields, from ``data.load_record``.
        name (str): a field stored flattened over (s, zeta, theta), e.g. "R_boozer".

    Returns:
        np.ndarray: the field on the boundary, shape (n_zeta, n_theta), over one field period.
    """
    a = np.asarray(record[name], dtype=float)
    n = int(round(np.sqrt(a.size / len(record["s1d"]))))
    return a[-n * n:].reshape(n, n)
