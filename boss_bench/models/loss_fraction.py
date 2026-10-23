"""Loss-fraction surrogate: predict the loss fraction from the boundary and profiles, without VMEC."""

import numpy as np
import torch
from torch import nn

from ._common import as_batch, load_trained

FIELDS = ("boundary_xyz", "pressure", "toroidal_current", "volume_avg_modB", "nfp")
N_GRID = 41  # boundary points per angle
N_PROFILE = 15  # radial points of the profiles
N_NFP = 12  # rows of the nfp embedding, for nfp from 1
NFP_MAX = 6  # the training data covers nfp from 1 to 6, with one configuration of nfp 12
BATCH_SIZE = 256  # configurations evaluated at once


class LossFractionModel(nn.Module):
    r"""A trained neural loss-fraction surrogate: predict the alpha-particle loss fraction without VMEC.

    The model predicts the fraction of the alpha particles born on :math:`s = 0.1` that are
    lost within 0.1 s, as ``confinement.loss_fraction`` computes it from the traced
    particles. Its inputs are fields of the configuration that need no VMEC run:

    - ``boundary_xyz``: points on the plasma boundary over one field period [m], shape
      ``(41 * 41 * 3,)`` as stored, or ``(41, 41, 3)``. Reshaped to ``(41, 41, 3)``, it is
      indexed :math:`(\phi, \theta, xyz)`: Cartesian points of the boundary surface on 41
      toroidal angles :math:`\phi` uniformly spaced in :math:`[0, 2\pi / n_\text{fp})` and
      41 poloidal angles :math:`\theta` uniformly spaced in :math:`[0, 2\pi)`, endpoints
      excluded. This is the layout of ``gamma()`` of a simsopt ``SurfaceRZFourier`` with
      these quadrature points. The model uses the cylindrical coordinates
      :math:`R = \sqrt{x^2 + y^2}` and :math:`Z = z` of the points; the toroidal angle is
      set by their place in the grid.
    - ``pressure``: the pressure profile [Pa], shape ``(15,)``, on the radial grid ``s1d``,
      15 points from :math:`s = 10^{-4}` to 1.
    - ``toroidal_current``: the enclosed toroidal current profile [A], shape ``(15,)``, on
      the same grid.
    - ``volume_avg_modB``: the volume-averaged field strength [T].
    - ``nfp``: the number of field periods, from 1 to 6.

    Load the model with ``load`` and predict with ``predict``. Calling the model on PyTorch
    tensors gives the prediction with gradients, for example with respect to the boundary.
    Gradients are available in PyTorch only: the ONNX export gives the prediction without
    them.

    Example:
        Predict the loss fraction of a test configuration::

            import h5py
            import boss_bench as bb
            from boss_bench import models
            from boss_bench.models import LossFractionModel

            path = models.download("loss_fraction", "BoSS-models/loss_fraction")
            model = LossFractionModel.load(path)
            with h5py.File(bb.list_files("BoSS", "confinement", "test")[0], "r") as f:
                record = bb.load_record(f, 0)
            model.predict(record)
    """

    def __init__(self):
        super().__init__()
        conv_channels, surf_dim, profile_dim, scalar_dim, hidden_dim = (32, 64, 128, 256, 512), 512, 32, 16, 2048
        self.surface_encoder = _PeriodicConvEncoder(2, conv_channels, surf_dim)
        self.pressure_encoder = nn.Linear(N_PROFILE, profile_dim)
        self.current_encoder = nn.Linear(N_PROFILE, profile_dim)
        self.modB_encoder = nn.Linear(1, scalar_dim)
        self.nfp_embedding = nn.Embedding(N_NFP, scalar_dim)
        self.mlp = nn.Sequential(
            nn.Linear(surf_dim + 2 * profile_dim + 2 * scalar_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        # each input is scaled by (x - low) / (high - low), with its range over the training set
        for name in ("R", "Z", "pressure", "current", "modB"):
            self.register_buffer(f"{name}_range", torch.tensor([0.0, 1.0], dtype=torch.float64))

    @classmethod
    def load(cls, path, device="cpu"):
        """Load the trained model, from its PyTorch weights.

        This loads ``model.safetensors``. The ONNX export of the model, from the same
        Hugging Face folder, is not loaded here: run it with ONNX Runtime instead.

        Args:
            path (str): the model's folder, holding ``config.json`` and ``model.safetensors``,
                e.g. from ``models.download("loss_fraction", path)``.
            device (str or torch.device): the device to put the model on.

        Returns:
            LossFractionModel: the model, ready to predict.

        Raises:
            ValueError: if the folder holds another model.
        """
        return load_trained(cls(), "loss_fraction", path, device)

    @torch.no_grad()
    def predict(self, data):
        """Predict the loss fraction of one configuration or a batch of them.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only
                ``boundary_xyz``, ``pressure``, ``toroidal_current``, ``volume_avg_modB`` and
                ``nfp`` are read.

        Returns:
            float or np.ndarray: the loss fraction, shape ``(n,)`` for a batch.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf, or ``nfp`` is out of range.
        """
        batch, single = as_batch(data, FIELDS)
        if np.any((batch["nfp"] < 1) | (batch["nfp"] > NFP_MAX)):
            raise ValueError(f"nfp must be between 1 and {NFP_MAX}")
        device = self.R_range.device
        out = []
        for a in range(0, len(batch["nfp"]), BATCH_SIZE):
            rows = {name: torch.as_tensor(x[a:a + BATCH_SIZE], device=device) for name, x in batch.items()}
            out.append(self(**rows).cpu().numpy())
        out = np.concatenate(out).astype(float)
        return float(out[0]) if single else out

    def forward(self, boundary_xyz, pressure, toroidal_current, volume_avg_modB, nfp):
        """Predict the loss fraction of a batch of configurations, with gradients.

        This is ``predict`` on PyTorch tensors, without the input checks, and
        differentiable with respect to every float input. Gradients are available in
        PyTorch only, not in the ONNX export.

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
