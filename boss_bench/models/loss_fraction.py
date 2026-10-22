"""Loss-fraction surrogate: predict the loss fraction from the boundary and profiles, without VMEC."""

import numpy as np
import torch
from torch import nn

from ._common import as_batch

FIELDS = ("boundary_xyz", "pressure", "toroidal_current", "volume_avg_modB", "nfp")
N_GRID = 41  # boundary points per angle
N_PROFILE = 15  # radial points of the profiles


class LossFractionModel(nn.Module):
    r"""Predict the alpha-particle loss fraction of a configuration without computing its equilibrium.

    The model predicts the fraction of the 5,000 alpha particles born on :math:`s = 0.1`
    that are lost within :math:`t_\text{max} = 0.1` s, as ``confinement.loss_fraction``
    computes it from the traced particles. Its inputs need no VMEC run: the plasma
    boundary, the pressure and toroidal current profiles, the volume-averaged field
    strength and the number of field periods. This is the surrogate optimized in the
    paper's surrogate-based optimization.

    The boundary, :math:`R` and :math:`Z` on the 41 × 41 grid of ``boundary_xyz``, is
    encoded by a convolutional network, periodic in both angles; each other input by a
    linear layer or, for ``nfp``, an embedding. A multilayer perceptron maps the encodings
    to the loss fraction, through a sigmoid. Before encoding, the poloidal angle of the
    boundary is shifted so that :math:`\theta = 0` is the outboard point (largest
    :math:`R`) of the cross section :math:`\phi = 0`, and :math:`\theta` increases with
    :math:`Z` there, as in training.

    The model was trained on the train split. On the test split, its mean squared error
    is 0.0047 and its mean absolute error 0.035.

    Load a pretrained model with ``models.load``. To predict, use ``predict``. To
    differentiate the prediction, for example with respect to the boundary, call the
    model on tensors.

    Args:
        conv_channels (tuple of int): channels of the stride-2 convolutions over the boundary.
        surf_dim (int): width of the boundary encoding.
        profile_dim (int): width of each profile encoding.
        scalar_dim (int): width of the encoding of ``volume_avg_modB`` and of ``nfp``.
        n_nfp (int): the number of field periods the ``nfp`` embedding covers, from 1.
        hidden_dim (int): width of the hidden layers.
        n_layers (int): the number of hidden layers.
    """

    def __init__(self, conv_channels=(32, 64, 128, 256, 512), surf_dim=512, profile_dim=32, scalar_dim=16,
                 n_nfp=12, hidden_dim=2048, n_layers=3):
        super().__init__()
        self.config = {"conv_channels": list(conv_channels), "surf_dim": surf_dim, "profile_dim": profile_dim,
                       "scalar_dim": scalar_dim, "n_nfp": n_nfp, "hidden_dim": hidden_dim, "n_layers": n_layers}
        self.surface_encoder = _PeriodicConvEncoder(2, conv_channels, surf_dim)
        self.pressure_encoder = nn.Linear(N_PROFILE, profile_dim)
        self.current_encoder = nn.Linear(N_PROFILE, profile_dim)
        self.modB_encoder = nn.Linear(1, scalar_dim)
        self.nfp_embedding = nn.Embedding(n_nfp, scalar_dim)
        layers, width = [], surf_dim + 2 * profile_dim + 2 * scalar_dim
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden_dim), nn.GELU()]
            width = hidden_dim
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1), nn.Sigmoid())
        # each input is scaled by (x - low) / (high - low), with its range over the training set
        for name in ("R", "Z", "pressure", "current", "modB"):
            self.register_buffer(f"{name}_range", torch.tensor([0.0, 1.0], dtype=torch.float64))

    def forward(self, boundary_xyz, pressure, toroidal_current, volume_avg_modB, nfp):
        """Predict the loss fraction of a batch of configurations, differentiably.

        Args:
            boundary_xyz (torch.Tensor): boundary points [m], shape ``(n, 41 * 41 * 3)`` or
                ``(n, 41, 41, 3)``, as the field ``boundary_xyz``.
            pressure (torch.Tensor): pressure profiles [Pa], shape ``(n, 15)``.
            toroidal_current (torch.Tensor): toroidal current profiles [A], shape ``(n, 15)``.
            volume_avg_modB (torch.Tensor): volume-averaged field strengths [T], shape ``(n,)``.
            nfp (torch.Tensor): numbers of field periods, shape ``(n,)``.

        Returns:
            torch.Tensor: the loss fractions, shape ``(n,)``.
        """
        xyz = _canonicalize_theta(boundary_xyz.reshape(-1, N_GRID, N_GRID, 3))
        R = torch.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
        surface = torch.stack([_scale(R, self.R_range), _scale(xyz[..., 2], self.Z_range)], dim=1)
        encodings = [  # in the order of training
            self.current_encoder(_scale(toroidal_current, self.current_range).float()),
            self.nfp_embedding(nfp.long() - 1),
            self.pressure_encoder(_scale(pressure, self.pressure_range).float()),
            self.surface_encoder(surface.float()),
            self.modB_encoder(_scale(volume_avg_modB, self.modB_range).float()[:, None]),
        ]
        return self.mlp(torch.cat(encodings, dim=-1))[:, 0]

    @torch.no_grad()
    def predict(self, data, batch_size=256):
        """Predict the loss fraction of one configuration or a batch of them.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only the
                fields in ``FIELDS`` are read: ``boundary_xyz``, ``pressure``,
                ``toroidal_current``, ``volume_avg_modB`` and ``nfp``.
            batch_size (int): the number of configurations evaluated at once.

        Returns:
            float or np.ndarray: the loss fraction, shape ``(n,)`` for a batch.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf, or ``nfp`` is out of range.
        """
        batch, single = as_batch(data, FIELDS)
        n_nfp = self.config["n_nfp"]
        if np.any((batch["nfp"] < 1) | (batch["nfp"] > n_nfp)):
            raise ValueError(f"nfp must be between 1 and {n_nfp}")
        device = self.R_range.device
        out = []
        for a in range(0, len(batch["nfp"]), batch_size):
            rows = {name: torch.as_tensor(x[a:a + batch_size], device=device) for name, x in batch.items()}
            out.append(self(**rows).cpu().numpy())
        out = np.concatenate(out).astype(float)
        return float(out[0]) if single else out


class _PeriodicConvEncoder(nn.Module):
    """Encode an image periodic in both axes by stride-2 convolutions and a global average.

    Args:
        in_channels (int): channels of the image.
        conv_channels (tuple of int): output channels of each convolution.
        out_dim (int): width of the encoding.
    """

    def __init__(self, in_channels, conv_channels, out_dim):
        super().__init__()
        blocks = []
        for c in conv_channels:
            blocks += [nn.Conv2d(in_channels, c, 3, stride=2, padding=1, padding_mode="circular"),
                       nn.BatchNorm2d(c), nn.GELU()]
            in_channels = c
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        """Encode images of shape ``(n, in_channels, h, w)`` to shape ``(n, out_dim)``."""
        return self.proj(self.conv(x).mean(dim=(-2, -1)))


def _canonicalize_theta(xyz):
    r"""Shift the poloidal angle of boundaries to start outboard and increase with :math:`Z`.

    The same shift of the poloidal index is applied to every cross section. It puts the
    point of largest :math:`R` of the cross section :math:`\phi = 0` at index 0, then, if
    :math:`Z` decreases along :math:`\theta` there, reverses the direction of
    :math:`\theta`.

    Args:
        xyz (torch.Tensor): boundary points, shape ``(n, n_phi, n_theta, 3)``.

    Returns:
        torch.Tensor: the reordered points, the same shape.
    """
    n_theta = xyz.shape[2]
    j = torch.arange(n_theta, device=xyz.device)
    shift = torch.argmax(torch.sqrt(xyz[:, 0, :, 0] ** 2 + xyz[:, 0, :, 1] ** 2), dim=1)[:, None]
    z = xyz[:, 0, :, 2]
    after = torch.gather(z, 1, (shift + 1) % n_theta)[:, 0]
    before = torch.gather(z, 1, (shift - 1) % n_theta)[:, 0]
    index = torch.where((after < before)[:, None], -j, j)  # -j reverses the direction
    index = (index + shift) % n_theta
    return torch.gather(xyz, 2, index[:, None, :, None].expand_as(xyz))


def _scale(x, value_range):
    """Scale x by (x - low) / (high - low), for value_range = (low, high)."""
    return (x - value_range[0]) / (value_range[1] - value_range[0])
