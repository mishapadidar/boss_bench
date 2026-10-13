"""Plots of configurations, particle losses, heat flux and trajectories."""

import matplotlib.pyplot as plt
import numpy as np

from .confinement import S_INITS, is_lost

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
        record: the configuration's fields, from ``data.load_record``.
        ax: a 3D matplotlib axis. If None, a new figure is made.

    Returns:
        The axis.
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
        record: the configuration's fields, from ``data.load_record``.
        s: the flux surface; the nearest stored surface is used.
        ax: a matplotlib axis. If None, a new figure is made.

    Returns:
        The axis.
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
        record: the configuration's fields, from ``data.load_record`` on a confinement file.
        ax: a matplotlib axis. If None, a new figure is made.

    Returns:
        The axis.
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


def plot_strike_points(targets, predictions=None, bins=32):
    """Strike points of the lost particles on the boundary, as points colored by energy and as a density.

    Args:
        targets: true samples, from ``heat_flux.heat_flux_targets``; one row each.
        predictions: optional list of (zeta, theta, energy), one per target, drawn
            next to the truth with the same color scales.
        bins: number of density bins per angle.

    Returns:
        The figure.
    """
    n_cols = 2 if predictions is None else 4
    fig, axs = plt.subplots(len(targets), n_cols, figsize=(4 * n_cols, 3.6 * len(targets)), squeeze=False)
    energies = np.concatenate([t["energy"] for t in targets])
    e_min, e_max = (energies.min(), energies.max()) if len(energies) else (None, None)
    points = None
    for i, target in enumerate(targets):
        clouds = [("True", target["zeta"], target["theta"], target["energy"])]
        if predictions is not None:
            clouds.append(("Predicted", *predictions[i]))
        density_max = None
        for j, (name, zeta, theta, energy) in enumerate(clouds):
            zeta, theta = np.mod(zeta, 2 * np.pi), np.mod(theta, 2 * np.pi)
            ax = axs[i, 2 * j]
            points = ax.scatter(zeta, theta, c=energy, s=0.5, alpha=0.5, cmap="jet",
                                vmin=e_min, vmax=e_max, rasterized=True)
            ax.set_title(f"{name} strike points")
            density = _density(zeta, theta, bins)
            density_max = density_max or density.max() or None
            ax = axs[i, 2 * j + 1]
            ax.imshow(density.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                      cmap="YlOrRd", vmin=0, vmax=density_max)
            ax.set_title(f"{name} density")
        for ax in axs[i]:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)
            ax.set_aspect("equal")
            ax.set_xlabel(r"Boozer $\zeta$")
            ax.set_ylabel(r"Boozer $\theta$")
        axs[i, 0].annotate(f"$s_0$ = {target['s_init']}", xy=(-0.35, 0.5), xycoords="axes fraction",
                           rotation=90, va="center", fontsize=12)
    fig.tight_layout()
    fig.colorbar(points, ax=axs, orientation="horizontal", fraction=0.03, pad=0.08, label="Energy [MeV]")
    return fig


def plot_rollouts(z_true, z_pred=None, window_us=350, toroidal_window_us=20):
    """Embedded trajectories against lead time: one row per particle, one column per channel.

    The toroidal channels complete about one transit per sample, so they are drawn over
    a shorter window, with markers.

    Args:
        z_true: true embedded states, (n, T, 5), row k at lead time k us, as in
            ``trajectory.read_device``; NaN after a loss.
        z_pred: optional predicted states in the same layout.
        window_us: lead times shown for the x, y and lambda channels [us].
        toroidal_window_us: lead times shown for the toroidal channels [us].

    Returns:
        The figure.
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
        xyz: points over one field period, (n_phi, n_theta, 3).
        values: a quantity at those points, (n_phi, n_theta).
        nfp: number of field periods.

    Returns:
        (xyz, values) over the full torus, of shapes (nfp * n_phi + 1, n_theta + 1, 3)
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
        zeta, theta: angles of the points [rad].
        bins: number of bins per angle.

    Returns:
        (bins, bins) array integrating to 1 (all zeros if there are no points).
    """
    counts = np.histogram2d(zeta, theta, bins=bins, range=[[0, 2 * np.pi], [0, 2 * np.pi]])[0]
    if counts.sum() == 0:
        return counts
    return counts / (counts.sum() * (2 * np.pi / bins) ** 2)
