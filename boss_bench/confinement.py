"""Alpha-particle loss quantities.

Particle arrays have shape (n,) for the n particles of one stellarator configuration, or
(m, n) for m configurations, and ``t_max`` is then a scalar or has shape (m,). All inputs
must be finite.
"""

import numpy as np

from ._checks import require_finite

E_ALPHA_MEV = 3.5
LOSS_TOL = 1e-14
S_INITS = (0.1, 0.25, 0.5)  # birth surfaces of the traced particles


def is_lost(confinement_time, t_max):
    r"""Flag the particles that left the plasma before the end of the simulation.

    Particle :math:`i`, with confinement time :math:`t_i`, is lost if

    .. math::

        t_i < t_\text{max} - 10^{-14}.

    Args:
        confinement_time (np.ndarray): particle confinement times [s], shape (n,) or (m, n).
        t_max (float or np.ndarray): simulation time [s], a scalar or shape (m,).

    Returns:
        np.ndarray: boolean, the shape of confinement_time, True for lost particles.

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(confinement_time=confinement_time, t_max=t_max)
    return confinement_time < np.asarray(t_max)[..., None] - LOSS_TOL


def loss_fraction(confinement_time, t_max, s_init, s0=0.1):
    r"""Fraction of the particles born on one surface that are lost.

    With :math:`P` the set of the :math:`N` particles born on the surface :math:`s_0`,

    .. math::

        f = \frac{1}{N} \sum_{i \in P} \mathbb{1}\left[t_i < t_\text{max} - 10^{-14}\right].

    Args:
        confinement_time (np.ndarray): particle confinement times [s], shape (n,) or (m, n).
        t_max (float or np.ndarray): simulation time [s], a scalar or shape (m,).
        s_init (np.ndarray): particle birth surfaces, the shape of confinement_time.
        s0 (float): the birth surface to average over.

    Returns:
        float or np.ndarray: the loss fraction, a scalar or shape (m,).

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    return _mean_on_surface(is_lost(confinement_time, t_max), s_init, s0)


def energy_loss(confinement_time, t_max, s_init, s0=0.1):
    r"""Mean energy lost per particle born on one surface.

    With :math:`P` the set of the :math:`N` particles born on the surface :math:`s_0`,

    .. math::

        E = \frac{1}{N} \sum_{i \in P} 3.5\,\text{MeV}\; e^{-2 t_i / t_\text{max}},

    so a confined particle (:math:`t_i = t_\text{max}`) counts :math:`3.5\,e^{-2}` MeV.

    Args:
        confinement_time (np.ndarray): particle confinement times [s], shape (n,) or (m, n).
        t_max (float or np.ndarray): simulation time [s], a scalar or shape (m,).
        s_init (np.ndarray): particle birth surfaces, the shape of confinement_time.
        s0 (float): the birth surface to average over.

    Returns:
        float or np.ndarray: the energy loss [MeV], a scalar or shape (m,).

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(confinement_time=confinement_time, t_max=t_max)
    tau = confinement_time / np.asarray(t_max)[..., None]
    return _mean_on_surface(E_ALPHA_MEV * np.exp(-2 * tau), s_init, s0)


def normalized_energy_loss(confinement_time, t_max, s_init, s0=0.1):
    r"""``energy_loss`` rescaled to [0, 1].

    With :math:`P` the set of the :math:`N` particles born on the surface :math:`s_0`,

    .. math::

        \bar E = \frac{1}{N} \sum_{i \in P} \frac{e^{-2 t_i / t_\text{max}} - e^{-2}}{1 - e^{-2}},

    which is 0 if every particle is confined and 1 if every particle is lost at once.

    Args:
        confinement_time (np.ndarray): particle confinement times [s], shape (n,) or (m, n).
        t_max (float or np.ndarray): simulation time [s], a scalar or shape (m,).
        s_init (np.ndarray): particle birth surfaces, the shape of confinement_time.
        s0 (float): the birth surface to average over.

    Returns:
        float or np.ndarray: the normalized energy loss, a scalar or shape (m,).

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(confinement_time=confinement_time, t_max=t_max)
    tau = confinement_time / np.asarray(t_max)[..., None]
    e = (np.exp(-2 * tau) - np.exp(-2)) / (1 - np.exp(-2))
    return _mean_on_surface(e, s_init, s0)


def _mean_on_surface(values, s_init, s0):
    """Average a per-particle quantity over the particles born on one surface.

    Args:
        values (np.ndarray): per-particle values, shape (n,) or (m, n).
        s_init (np.ndarray): particle birth surfaces, the shape of values.
        s0 (float): the birth surface to average over.

    Returns:
        float or np.ndarray: the mean over the last axis of the particles with ``s_init == s0``.

    Raises:
        ValueError: if s_init contains NaN or inf.
    """
    require_finite(s_init=s_init)
    born = np.isclose(s_init, s0)
    return (values * born).sum(axis=-1) / born.sum(axis=-1)
