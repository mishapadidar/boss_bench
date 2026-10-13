"""Plots of configurations, particle losses, heat flux and trajectories."""

import matplotlib.pyplot as plt
import numpy as np

from .confinement import S_INITS, is_lost
from .heat_flux import binned_heat_flux, lost_particles

TRAJECTORY_CHANNELS = [  # (channel of z, title, toroidal)
    (0, r"$x = \sqrt{s}\,\cos\theta$", False),
    (1, r"$y = \sqrt{s}\,\sin\theta$", False),
    (2, r"$\cos\varphi$", True),
    (3, r"$\sin\varphi$", True),
    (4, r"$\lambda = v_\parallel / v$", False),
]


def plot_device(record, ax=None):
    """Draw the plasma boundary over the full torus, colored by ``|B|``, viewed from above.

    Args:
        record (dict): the configuration's fields, from ``data.load_record``; at least
            ``boundary_xyz``, ``modB``, ``phi1d``, ``theta1d`` and ``nfp``.
        ax (matplotlib.axes.Axes, optional): a 3D axis to draw on. By default a new figure is made.

    Returns:
        matplotlib.axes.Axes: the axis.
    """
    n_phi, n_theta = len(record["phi1d"]), len(record["theta1d"])
    xyz = np.asarray(record["boundary_xyz"]).reshape(n_phi, n_theta, 3)
    modB = np.asarray(record["modB"]).reshape(-1, n_phi, n_theta)[-1]  # s = 1
    xyz, modB = _full_torus(xyz, modB, int(record["nfp"]))
    if ax is None:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    colors = plt.cm.jet((modB - modB.min()) / (modB.max() - modB.min()))
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=colors,
                    rstride=1, cstride=1, linewidth=0, shade=False)
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect(np.ptp(xyz.reshape(-1, 3), axis=0))
    ax.set_axis_off()
    return ax


def plot_modB_boozer(record, s=0.5, ax=None):
    """Contours of ``|B|`` on a flux surface in Boozer coordinates, with field lines in black.

    Args:
        record (dict): the configuration's fields, from ``data.load_record``; at least
            ``modB_boozer``, ``iota``, ``s1d``, ``phi1d``, ``theta1d`` and ``nfp``.
        s (float): the flux surface; the nearest stored surface is used.
        ax (matplotlib.axes.Axes, optional): the axis to draw on. By default a new figure is made.

    Returns:
        matplotlib.axes.Axes: the axis.
    """
    s1d, phi, theta = (np.asarray(record[k]) for k in ("s1d", "phi1d", "theta1d"))
    js = int(np.argmin(np.abs(s1d - s)))
    B = np.asarray(record["modB_boozer"]).reshape(len(s1d), len(phi), len(theta))[js]
    # close the periodic grid so the contours reach the edges
    B = np.pad(B, ((0, 1), (0, 1)), mode="wrap")
    P, T = np.meshgrid(np.append(phi, 2 * np.pi / int(record["nfp"])), np.append(theta, 2 * np.pi), indexing="ij")
    if ax is None:
        ax = plt.figure(figsize=(5, 4)).add_subplot()
    filled = ax.contourf(P, T, B, levels=np.linspace(B.min(), B.max(), 25), cmap="jet")
    ax.contour(P, T, T - record["iota"][js] * P, levels=6, colors="k", linewidths=0.7, alpha=0.7)
    ax.set_xlabel(r"Boozer $\varphi$")
    ax.set_ylabel(r"Boozer $\theta$")
    ax.set_title(f"$s$ = {s1d[js]:.2f}")
    ax.figure.colorbar(filled, ax=ax, label="|B| [T]")
    return ax


def plot_loss_fraction_curves(record, ax=None):
    """Fraction of particles lost by time t, for each birth surface.

    Args:
        record (dict): the configuration's fields, from ``data.load_record`` on a
            confinement file; at least ``confinement_time``, ``s_init`` and ``t_max``.
        ax (matplotlib.axes.Axes, optional): the axis to draw on. By default a new figure is made.

    Returns:
        matplotlib.axes.Axes: the axis.
    """
    ct, s_init, t_max = record["confinement_time"], record["s_init"], record["t_max"]
    t = np.logspace(-6, np.log10(t_max), 400)
    lost = is_lost(ct, t_max)
    if ax is None:
        ax = plt.figure(figsize=(5, 4)).add_subplot()
    styles = zip(S_INITS, ["mediumseagreen", "orange", "lightskyblue"], ["-", "--", ":"])
    for s0, color, linestyle in styles:
        born = np.isclose(s_init, s0)
        lost_by_t = (ct[born][None, :] <= t[:, None]) & lost[born][None, :]
        ax.plot(t, lost_by_t.mean(axis=1), color=color, ls=linestyle, lw=3, label=f"$s_0$ = {s0}")
    ax.set_xscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loss fraction")
    ax.grid(True, color="lightgrey")
    ax.legend(loc="upper left")
    return ax


def plot_strike_points(record, predictions=None, bins=32):
    """Where the lost particles strike the boundary: as points colored by energy, and as a density.

    One row per birth surface. The density of lost particles is their histogram over the
    Boozer angles of the full torus, normalized to integrate to 1 [1 / rad^2]. Predictions,
    if given, are drawn next to the truth with the same color scales.

    Args:
        record (dict): the configuration's fields, from ``data.load_record`` on a
            confinement file; at least those read by ``heat_flux.lost_particles``.
        predictions (dict, optional): maps each birth surface s0 to a tuple (zeta, theta,
            energy, loss_fraction), as in ``heat_flux.heat_flux_metrics``.
        bins (int): number of density bins per angle.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    n_cols = 2 if predictions is None else 4
    fig, axs = plt.subplots(len(S_INITS), n_cols, figsize=(4.8 * n_cols, 3.6 * len(S_INITS)), squeeze=False)
    truths = {s0: lost_particles(record, s0)[:3] for s0 in S_INITS}
    energies = np.concatenate([energy for _, _, energy in truths.values()])
    e_min, e_max = (energies.min(), energies.max()) if len(energies) else (None, None)
    for i, s0 in enumerate(S_INITS):
        clouds = [("True", *truths[s0])]
        if predictions is not None:
            clouds.append(("Predicted", *predictions[s0][:3]))
        density_max = None
        for j, (name, zeta, theta, energy) in enumerate(clouds):
            zeta, theta = np.mod(zeta, 2 * np.pi), np.mod(theta, 2 * np.pi)
            ax = axs[i, 2 * j]
            points = ax.scatter(zeta, theta, c=energy, s=0.5, alpha=0.5, cmap="jet",
                                vmin=e_min, vmax=e_max, rasterized=True)
            fig.colorbar(points, ax=ax, label="Energy [MeV]", fraction=0.046)
            ax.set_title(f"{name} strike points")
            density = _density(zeta, theta, bins)
            density_max = density_max or density.max() or None
            ax = axs[i, 2 * j + 1]
            image = ax.imshow(density.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                              cmap="YlOrRd", vmin=0, vmax=density_max)
            fig.colorbar(image, ax=ax, label="[1 / rad$^2$]", fraction=0.046)
            ax.set_title(f"{name} density of lost particles")
        for ax in axs[i]:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)
            ax.set_aspect("equal")
            ax.set_xlabel(r"Boozer $\zeta$")
            ax.set_ylabel(r"Boozer $\theta$")
        _label_row(axs[i, 0], s0)
    fig.tight_layout()
    return fig


def plot_heat_flux(record, predictions=None, resolution=32):
    """The heat flux on the boundary over one field period, from ``heat_flux.binned_heat_flux``.

    One row per birth surface. Predictions, if given, are drawn next to the truth on the
    same color scale, scaled so that their total is the predicted loss fraction, as in
    ``heat_flux.heat_flux_metrics``.

    Args:
        record (dict): the configuration's fields, from ``data.load_record`` on a
            confinement file; at least those read by ``heat_flux.lost_particles`` and
            ``heat_flux.binned_heat_flux``.
        predictions (dict, optional): maps each birth surface s0 to a tuple (zeta, theta,
            energy, loss_fraction), as in ``heat_flux.heat_flux_metrics``.
        resolution (int): number of bins per angle.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    nfp = int(record["nfp"])
    n_cols = 1 if predictions is None else 2
    fig, axs = plt.subplots(len(S_INITS), n_cols, figsize=(5.5 * n_cols, 3.6 * len(S_INITS)), squeeze=False)
    zeta_edges = np.linspace(0, 2 * np.pi / nfp, resolution + 1)
    theta_edges = np.linspace(0, 2 * np.pi, resolution + 1)
    for i, s0 in enumerate(S_INITS):
        zeta, theta, energy, n_launched = lost_particles(record, s0)
        fluxes = [("True", binned_heat_flux(zeta, theta, energy, n_launched, record, resolution))]
        if predictions is not None:
            zeta, theta, energy, loss_fraction = predictions[s0]
            scale = loss_fraction * n_launched / len(energy) if len(energy) else 0.0
            fluxes.append(("Predicted", scale * binned_heat_flux(zeta, theta, energy, n_launched, record, resolution)))
        vmax = fluxes[0][1].max() or None
        for j, (name, q) in enumerate(fluxes):
            ax = axs[i, j]
            mesh = ax.pcolormesh(zeta_edges, theta_edges, q.T, cmap="YlOrRd", vmin=0, vmax=vmax)
            fig.colorbar(mesh, ax=ax, label="[MeV / m$^2$ per particle]")
            ax.set_title(f"{name} heat flux")
            ax.set_xlabel(r"Boozer $\zeta$ (one field period)")
            ax.set_ylabel(r"Boozer $\theta$")
        _label_row(axs[i, 0], s0)
    fig.tight_layout()
    return fig


def plot_rollouts(z_true, z_pred=None, window_us=350, toroidal_window_us=20):
    """Embedded trajectories against lead time: one row per particle, one column per channel.

    The toroidal channels complete about one transit per sample, so they are drawn over
    a shorter window, with markers.

    Args:
        z_true (np.ndarray): true embedded states, shape (n, T, 5), with row k at lead time
            k us, as from ``trajectory.read_trajectories``; NaN after a loss.
        z_pred (np.ndarray, optional): predicted states in the same layout.
        window_us (float): lead times shown for the x, y and lambda channels [us].
        toroidal_window_us (float): lead times shown for the toroidal channels [us].

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    z_true = np.atleast_3d(z_true)
    tau = np.arange(z_true.shape[1])
    n_rows, n_cols = len(z_true), len(TRAJECTORY_CHANNELS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.5 * n_rows), sharey=True, squeeze=False)
    for r in range(n_rows):
        for c, (channel, title, toroidal) in enumerate(TRAJECTORY_CHANNELS):
            ax = axs[r, c]
            window = toroidal_window_us if toroidal else window_us
            shown = tau <= window
            markers = {"marker": "o", "markersize": 2.6} if toroidal else {}
            ax.plot(tau[shown], z_true[r, shown, channel], color="black", lw=2, label="truth", **markers)
            if z_pred is not None:
                ax.plot(tau[shown], z_pred[r, shown, channel], color="#3C87C8", lw=2,
                        label="prediction", **markers)
            ax.set_xlim(0, window)
            ax.set_ylim(-1.08, 1.08)
            ax.grid(True, color="0.88", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(title + (f"\n(first {window:g} $\\mu$s)" if toroidal else ""))
            if r == n_rows - 1:
                ax.set_xlabel(r"Lead time $\tau$ [$\mu$s]")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def _full_torus(xyz, values, nfp):
    """Copy one field period around the torus, and close the grid in both angles.

    Args:
        xyz (np.ndarray): points over one field period, shape (n_phi, n_theta, 3).
        values (np.ndarray): a quantity at those points, shape (n_phi, n_theta).
        nfp (int): number of field periods.

    Returns:
        tuple: (xyz, values) over the full torus, of shapes (nfp * n_phi + 1, n_theta + 1, 3)
        and (nfp * n_phi + 1, n_theta + 1).
    """
    periods = []
    for k in range(nfp):
        a = 2 * np.pi * k / nfp
        rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        periods.append(xyz @ rotation.T)
    xyz, values = np.concatenate(periods), np.tile(values, (nfp, 1))
    xyz, values = np.concatenate([xyz, xyz[:1]]), np.concatenate([values, values[:1]])
    return np.concatenate([xyz, xyz[:, :1]], axis=1), np.concatenate([values, values[:, :1]], axis=1)


def _density(zeta, theta, bins):
    """Normalized 2D histogram of points on [0, 2 pi)^2.

    Args:
        zeta, theta (np.ndarray): angles of the points [rad].
        bins (int): number of bins per angle.

    Returns:
        np.ndarray: shape (bins, bins), integrating to 1 (all zeros if there are no points).
    """
    counts = np.histogram2d(zeta, theta, bins=bins, range=[[0, 2 * np.pi], [0, 2 * np.pi]])[0]
    if counts.sum() == 0:
        return counts
    return counts / (counts.sum() * (2 * np.pi / bins) ** 2)


def _label_row(ax, s0):
    """Label a row of panels with its birth surface, to the left of its first panel.

    Args:
        ax (matplotlib.axes.Axes): the first panel of the row.
        s0 (float): the birth surface.
    """
    ax.annotate(f"$s_0$ = {s0}", xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=12)
