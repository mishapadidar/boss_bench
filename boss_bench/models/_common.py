"""Helpers shared by the models: batching the inputs and integrating a flow."""

import numpy as np
import torch

from .._checks import require_finite


def as_batch(data, fields):
    """Read the model inputs from one configuration or a batch of them.

    Args:
        data (dict): the fields of one configuration, as from ``data.load_record``, or of
            several, with one row per configuration, as from ``data.load``.
        fields (list of str): the names of the fields to read.

    Returns:
        tuple: **batch, single**, the fields as float64 arrays with one row per
        configuration, and whether ``data`` held a single configuration.

    Raises:
        KeyError: if a field is missing from ``data``.
        ValueError: if a field contains NaN or inf.
    """
    missing = [name for name in fields if name not in data]
    if missing:
        raise KeyError(f"missing fields: {missing}")
    single = np.ndim(data["nfp"]) == 0
    batch = {name: np.asarray(data[name], dtype=float) for name in fields}
    if single:
        batch = {name: a[None] for name, a in batch.items()}
    require_finite(**batch)
    return batch, single


def integrate(velocity, x, t0, t1, n_steps, method):
    """Integrate the flow :math:`dx/dt = v(x, t)` from ``t0`` to ``t1`` with fixed steps.

    Args:
        velocity (callable): the velocity field ``v(x, t)``, with ``t`` a float.
        x (torch.Tensor): the initial points, at ``t0``.
        t0 (float): the initial time.
        t1 (float): the final time.
        n_steps (int): the number of steps.
        method (str): ``"euler"`` or ``"midpoint"``.

    Returns:
        torch.Tensor: the points at ``t1``.

    Raises:
        ValueError: if the method is unknown.
    """
    if method not in ("euler", "midpoint"):
        raise ValueError(f"unknown method {method!r}; use 'euler' or 'midpoint'")
    dt = (t1 - t0) / n_steps
    for i in range(n_steps):
        t = t0 + i * dt
        v = velocity(x, t)
        if method == "midpoint":
            v = velocity(x + 0.5 * dt * v, t + 0.5 * dt)
        x = x + dt * v
    return x


def generator(seed, device):
    """Make a torch random number generator, seeded if a seed is given.

    Args:
        seed (int or None): the seed. By default, a random one.
        device (torch.device): the device the random numbers are drawn on.

    Returns:
        torch.Generator: the generator.
    """
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)
    return gen
