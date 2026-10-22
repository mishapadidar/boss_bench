"""Heat-flux model: sample where the lost alpha particles strike the plasma boundary."""

import math

import numpy as np
import torch
from torch import nn

from ._common import as_batch, generator, integrate

FIELDS = ("boundary_xyz", "nfp", "d_pressure_d_s", "iota", "volume_avg_modB", "minor_radius")
SCALAR_FIELDS = ("nfp", "d_pressure_d_s", "iota", "volume_avg_modB", "minor_radius")  # in the order of training
N_GRID = 41  # boundary points per angle
X_DIM = 5  # a strike point is (cos zeta, sin zeta, cos theta, sin theta, log energy)


class HeatFluxModel(nn.Module):
    r"""Sample the strike points of the lost alpha particles of a configuration, as in the heat-flux benchmark.

    For the particles born on one surface :math:`s_0`, the model predicts the loss fraction
    and generates strike points on the boundary, each with Boozer angles
    :math:`(\zeta, \theta)` and an energy, as ``heat_flux.lost_particles`` computes them
    from the traced particles. Each strike point is drawn independently, by integrating a
    learned flow (conditional flow matching) from a Gaussian sample to the strike-point
    distribution of the configuration. A sample with as many strike points as the predicted
    number of lost particles is a prediction for ``heat_flux_metrics``.

    The model is conditioned on the boundary (``boundary_xyz``), encoded by a convolutional
    network, and on ``nfp``, ``d_pressure_d_s``, ``iota``, ``volume_avg_modB``,
    ``minor_radius`` and :math:`s_0`, encoded by a linear layer. ``iota`` comes from the
    VMEC equilibrium. The velocity of the flow is a residual network of the strike point,
    the time and the encoding. A second head maps the encoding to the loss fraction.

    The model was trained on the train split. On the test split, the paper reports a
    median energy distance of 0.031, median heat-flux errors of 0.34 (16 × 16) and 0.57
    (32 × 32), and a mean absolute loss-fraction error of 0.034, with 50 Euler steps.

    Load a pretrained model with ``models.load``. To predict, use ``sample`` and
    ``predict_loss_fraction``.

    Args:
        embed_dim (int): width of the encoding of the configuration and of the time.
        hidden_dim (int): width of the residual blocks of the velocity.
        n_blocks (int): the number of residual blocks of the velocity.
    """

    def __init__(self, embed_dim=256, hidden_dim=512, n_blocks=4):
        super().__init__()
        self.config = {"embed_dim": embed_dim, "hidden_dim": hidden_dim, "n_blocks": n_blocks}
        d_cond = 1 + 2 * 15 + 2 + 1  # SCALAR_FIELDS and s0
        self.cond_encoder = _ConditioningEncoder(d_cond, embed_dim)
        self.time_embedding = _TimeEmbedding(embed_dim)
        self.velocity_net = _ResNetVelocity(X_DIM, embed_dim, hidden_dim, n_blocks)
        self.loss_frac_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
        # the conditioning and the log energy are standardized with the training-set statistics
        self.register_buffer("scalar_mean", torch.zeros(d_cond - 1))
        self.register_buffer("scalar_std", torch.ones(d_cond - 1))
        self.register_buffer("s0_mean_std", torch.tensor([0.0, 1.0]))
        self.register_buffer("xyz_mean", torch.zeros(3))
        self.register_buffer("xyz_std", torch.ones(3))
        self.register_buffer("log_energy_mean_std", torch.tensor([0.0, 1.0]))

    def encode(self, boundary_xyz, nfp, d_pressure_d_s, iota, volume_avg_modB, minor_radius, s0):
        """Encode a batch of configurations and birth surfaces, and predict their loss fractions.

        Args:
            boundary_xyz (torch.Tensor): boundary points [m], shape ``(n, 41 * 41 * 3)`` or
                ``(n, 41, 41, 3)``, as the field ``boundary_xyz``.
            nfp (torch.Tensor): numbers of field periods, shape ``(n,)``.
            d_pressure_d_s (torch.Tensor): pressure derivatives [Pa], shape ``(n, 15)``.
            iota (torch.Tensor): rotational transforms, shape ``(n, 15)``.
            volume_avg_modB (torch.Tensor): volume-averaged field strengths [T], shape ``(n,)``.
            minor_radius (torch.Tensor): minor radii [m], shape ``(n,)``.
            s0 (torch.Tensor): birth surfaces, shape ``(n,)``.

        Returns:
            tuple: **embedding, loss_fraction**, the encodings, shape ``(n, embed_dim)``, and
            the predicted loss fractions, shape ``(n,)``.
        """
        n = nfp.shape[0]
        scalars = torch.cat([nfp.float()[:, None], d_pressure_d_s.float().reshape(n, -1),
                             iota.float().reshape(n, -1), volume_avg_modB.float()[:, None],
                             minor_radius.float()[:, None]], dim=1)
        scalars = (scalars - self.scalar_mean) / self.scalar_std
        s0 = (s0.float()[:, None] - self.s0_mean_std[0]) / self.s0_mean_std[1]
        xyz = boundary_xyz.float().reshape(n, N_GRID, N_GRID, 3)
        xyz = ((xyz - self.xyz_mean) / self.xyz_std).permute(0, 3, 1, 2)
        embedding = self.cond_encoder(torch.cat([scalars, s0], dim=1), xyz)
        return embedding, self.loss_frac_head(embedding)[:, 0]

    def forward(self, x, t, embedding):
        """Evaluate the velocity of the flow.

        Args:
            x (torch.Tensor): points of the flow, shape ``(k, 5)``.
            t (torch.Tensor): the time, in :math:`[0, 1]`, one value.
            embedding (torch.Tensor): the encoding of the configuration of each point, from
                ``encode``, shape ``(k, embed_dim)``.

        Returns:
            torch.Tensor: the velocity, shape ``(k, 5)``.
        """
        t_emb = self.time_embedding(t.reshape(1, 1).float()).expand(x.shape[0], -1)
        return self.velocity_net(x, t_emb, embedding)

    def integrate(self, x, embedding, n_steps=50, method="euler"):
        """Map Gaussian samples to strike points by integrating the flow from time 0 to 1.

        Args:
            x (torch.Tensor): standard normal samples, shape ``(k, 5)``.
            embedding (torch.Tensor): the encoding of the configuration of each sample, from
                ``encode``, shape ``(k, embed_dim)``.
            n_steps (int): the number of steps.
            method (str): ``"euler"`` or ``"midpoint"``.

        Returns:
            torch.Tensor: the strike points, as ``(cos zeta, sin zeta, cos theta, sin theta,
            standardized log energy)``, shape ``(k, 5)``.
        """
        return integrate(lambda x, t: self(x, torch.tensor(t, device=x.device), embedding),
                         x, 0.0, 1.0, n_steps, method)

    @torch.no_grad()
    def predict_loss_fraction(self, data, s0=0.1, batch_size=512):
        """Predict the loss fraction of the particles born on one surface, for one configuration or a batch of them.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only the
                fields in ``FIELDS`` are read: ``boundary_xyz``, ``nfp``, ``d_pressure_d_s``,
                ``iota``, ``volume_avg_modB`` and ``minor_radius``.
            s0 (float): the birth surface, 0.1, 0.25 or 0.5.
            batch_size (int): the number of configurations encoded at once.

        Returns:
            float or np.ndarray: the loss fraction, shape ``(n,)`` for a batch.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf.
        """
        batch, single = as_batch(data, FIELDS)
        loss_fraction = self._encode(batch, s0, batch_size)[1].cpu().numpy().astype(float)
        return float(loss_fraction[0]) if single else loss_fraction

    @torch.no_grad()
    def sample(self, data, s0=0.1, n_launched=5000, n_steps=50, method="euler", seed=None, max_points=200_000,
               batch_size=512):
        r"""Sample the strike points of the lost particles born on one surface, for one or several configurations.

        For each configuration, the number of strike points is the predicted number of lost
        particles, :math:`\max(1, \operatorname{round}(\hat f N))`, with :math:`\hat f` the
        predicted loss fraction and :math:`N` = ``n_launched``.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only the
                fields in ``FIELDS`` are read: ``boundary_xyz``, ``nfp``, ``d_pressure_d_s``,
                ``iota``, ``volume_avg_modB`` and ``minor_radius``.
            s0 (float): the birth surface, 0.1, 0.25 or 0.5.
            n_launched (int): the number of particles launched, 5,000 in the BoSS data.
            n_steps (int): the number of integration steps.
            method (str): ``"euler"`` or ``"midpoint"``.
            seed (int, optional): the random seed. By default, a random one.
            max_points (int): the number of strike points integrated at once.
            batch_size (int): the number of configurations encoded at once.

        Returns:
            tuple or list of tuple: **zeta, theta, energy, loss_fraction**, the Boozer angles
            [rad] and energies [MeV] of the strike points, arrays of shape ``(k,)``, and the
            predicted loss fraction, a float; one tuple per configuration for a batch. Each
            tuple is a prediction for ``heat_flux_metrics``.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf.
        """
        batch, single = as_batch(data, FIELDS)
        embedding, loss_fraction = self._encode(batch, s0, batch_size)
        loss_fraction = loss_fraction.cpu().numpy()
        counts = np.maximum(1, np.round(loss_fraction * n_launched)).astype(int)
        gen = generator(seed, embedding.device)
        out, a = [], 0
        while a < len(counts):  # configurations a to b - 1 are integrated together
            b = a + 1
            while b < len(counts) and counts[a:b + 1].sum() <= max_points:
                b += 1
            x = torch.randn(int(counts[a:b].sum()), X_DIM, generator=gen, device=embedding.device)
            rows = torch.as_tensor(np.repeat(np.arange(a, b), counts[a:b]), device=embedding.device)
            x = self.integrate(x, embedding[rows], n_steps, method)
            for i, points in zip(range(a, b), torch.split(x, counts[a:b].tolist())):
                strike_points = tuple(p.cpu().numpy().astype(float) for p in self.strike_points(points))
                out.append((*strike_points, float(loss_fraction[i])))
            a = b
        return out[0] if single else out

    def strike_points(self, x):
        r"""Convert points of the flow, from ``integrate``, to strike points.

        Args:
            x (torch.Tensor): points of the flow at time 1, shape ``(k, 5)``.

        Returns:
            tuple: **zeta, theta, energy**, the Boozer angles in :math:`[0, 2\pi)` [rad] and
            energies [MeV] of the strike points, each of shape ``(k,)``.
        """
        zeta = torch.atan2(x[:, 1], x[:, 0]) % (2 * math.pi)
        theta = torch.atan2(x[:, 3], x[:, 2]) % (2 * math.pi)
        energy = torch.exp(x[:, 4] * self.log_energy_mean_std[1] + self.log_energy_mean_std[0])
        return zeta, theta, energy

    def _encode(self, batch, s0, batch_size):
        """Encode the configurations of a batch, batch_size at a time; see encode."""
        device = self.scalar_mean.device
        n = len(batch["nfp"])
        out = []
        for a in range(0, n, batch_size):
            rows = {name: torch.as_tensor(x[a:a + batch_size], device=device) for name, x in batch.items()}
            out.append(self.encode(**rows, s0=torch.full((len(rows["nfp"]),), s0, device=device)))
        return torch.cat([e for e, _ in out]), torch.cat([f for _, f in out])


class _ConditioningEncoder(nn.Module):
    """Encode the boundary by a convolutional network and the scalars by a linear layer, then combine them.

    Args:
        d_cond (int): the number of scalars.
        embed_dim (int): width of the encoding.
    """

    def __init__(self, d_cond, embed_dim):
        super().__init__()
        self.surf_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.scalar_net = nn.Sequential(nn.Linear(d_cond, 128), nn.SiLU())
        self.combine = nn.Sequential(nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
                                     nn.Linear(embed_dim, embed_dim))

    def forward(self, cond, surf):
        """Encode scalars of shape ``(n, d_cond)`` and boundaries of shape ``(n, 3, 41, 41)``."""
        return self.combine(torch.cat([self.surf_cnn(surf), self.scalar_net(cond)], dim=-1))


class _TimeEmbedding(nn.Module):
    """Embed the time by sinusoids of 32 log-spaced frequencies and a two-layer network.

    Args:
        embed_dim (int): width of the embedding.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(0, 3, 32))
        self.mlp = nn.Sequential(nn.Linear(64, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, t):
        """Embed times of shape ``(n, 1)`` to shape ``(n, embed_dim)``."""
        angles = 2 * math.pi * t * self.freqs
        return self.mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))


class _ResBlock(nn.Module):
    """A pre-activation residual block, x + MLP(norm(x)).

    Args:
        dim (int): width of the block.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        """Apply the block to x of shape ``(n, dim)``."""
        return x + self.net(x)


class _ResNetVelocity(nn.Module):
    """The velocity of the flow: a residual network of the point, the time and the encoding.

    Args:
        x_dim (int): dimension of a point.
        embed_dim (int): width of the time and conditioning encodings.
        hidden (int): width of the residual blocks.
        n_blocks (int): the number of residual blocks.
    """

    def __init__(self, x_dim, embed_dim, hidden, n_blocks):
        super().__init__()
        self.proj_in = nn.Linear(x_dim + 2 * embed_dim, hidden)
        self.blocks = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, x_dim))

    def forward(self, x, t_emb, c_emb):
        """The velocity at points x, shape ``(n, x_dim)``, given their time and conditioning encodings."""
        return self.proj_out(self.blocks(self.proj_in(torch.cat([x, t_emb, c_emb], dim=-1))))
