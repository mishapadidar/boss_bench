"""Input checks shared by the modules."""

import numpy as np


def require_finite(**arrays):
    """Raise an error if any input contains NaN or inf.

    Args:
        **arrays: the inputs to check, by name.

    Raises:
        ValueError: naming the first input that is not finite.
    """
    for name, a in arrays.items():
        if not np.all(np.isfinite(np.asarray(a, dtype=float))):
            raise ValueError(f"{name} contains NaN or inf")
