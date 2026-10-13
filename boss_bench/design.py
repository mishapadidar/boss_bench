"""Inverse-design benchmark: score a set of evaluated designs.

Each design is evaluated as in the paper: its boundary is rescaled to ARIES-CS size,
minor radius 1.7 m and volume-averaged field 5.86 T, and its VMEC equilibrium is
computed. Its mirror ratio is max ``|B|`` / min ``|B|`` over a 15 x 41 x 41 grid in
(s, theta, phi), with s from 1e-4 to 1. Its loss fraction is that of 5000 alpha
particles born on s = 0.1, traced for 0.1 s.
"""

import numpy as np

from ._checks import require_finite


def design_metrics(loss_fraction, mirror_ratio, mirror_target, aspect_ratio, aspect_target):
    r"""Score designs that have been evaluated with VMEC and particle tracing.

    For :math:`n` designs with loss fractions :math:`f_j`, mirror ratios :math:`R_j` and
    aspect ratios :math:`A_j`, and targets :math:`R^*_j` and :math:`A^*_j`, the scores in
    percent are

    .. math::

        \text{low-loss rate}_c = \frac{100}{n} \sum_j \mathbb{1}[f_j \le c], \quad c = 0.10, 0.15,

        \text{mirror constraint}_k = \frac{100}{n} \sum_j \mathbb{1}[R_j \le k R^*_j], \quad k = 1, 1.1,

        \text{aspect error} = 100\, \operatorname{median}_j \frac{|A_j - A^*_j|}{A^*_j}.

    Args:
        loss_fraction (array_like): traced loss fraction of the particles born on s = 0.1,
            one value per design, shape (n,).
        mirror_ratio (array_like): mirror ratio, one value per design, shape (n,).
        mirror_target (array_like): target mirror ratio, one value per design, shape (n,).
        aspect_ratio (array_like): aspect ratio, one value per design, shape (n,).
        aspect_target (array_like): target aspect ratio, one value per design, shape (n,).

    Returns:
        dict: percentages, holding the share of designs with loss fraction <= 10% and <= 15%
        ("low_loss_rate_10", "low_loss_rate_15"), with mirror ratio <= target and
        <= 1.1 * target ("mirror_constraint", "mirror_constraint_1.1"), and the median
        relative aspect-ratio error ("aspect_error_median").

    Raises:
        ValueError: if an input contains NaN or inf.
    """
    require_finite(loss_fraction=loss_fraction, mirror_ratio=mirror_ratio, mirror_target=mirror_target,
                   aspect_ratio=aspect_ratio, aspect_target=aspect_target)
    loss_fraction = np.asarray(loss_fraction, dtype=float)
    mirror_ratio = np.asarray(mirror_ratio, dtype=float)
    mirror_target = np.asarray(mirror_target, dtype=float)
    aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    aspect_target = np.asarray(aspect_target, dtype=float)
    return {
        "low_loss_rate_10": 100 * np.mean(loss_fraction <= 0.10),
        "low_loss_rate_15": 100 * np.mean(loss_fraction <= 0.15),
        "mirror_constraint": 100 * np.mean(mirror_ratio <= mirror_target),
        "mirror_constraint_1.1": 100 * np.mean(mirror_ratio <= 1.1 * mirror_target),
        "aspect_error_median": 100 * np.median(np.abs(aspect_ratio - aspect_target) / aspect_target),
    }
