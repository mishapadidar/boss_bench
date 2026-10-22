"""Heat-flux model: sample where the lost alpha particles strike the plasma boundary."""

import math

import numpy as np
import torch
from torch import nn

from ._common import as_batch, generator, integrate, load_trained

FIELDS = ("boundary_xyz", "nfp", "d_pressure_d_s", "iota", "volume_avg_modB", "minor_radius")
N_GRID = 41  # boundary points per angle
X_DIM = 5  # a strike point is (cos zeta, sin zeta, cos theta, sin theta, log energy)
N_LAUNCHED = 5000  # particles launched per birth surface in the BoSS data
N_STEPS = 50  # Euler steps of the flow, as in the paper
MAX_POINTS = 200_000  # strike points integrated at once
BATCH_SIZE = 512  # configurations encoded at once


class HeatFluxModel(nn.Module):
    r"""The IID baseline of the paper's heat-flux task: sample where the lost alpha particles strike the boundary.

    For the alpha particles born on one surface :math:`s_0` = 0.1, 0.25 or 0.5, the model
    predicts the loss fraction, and samples the strike points of the lost particles on the
    boundary: their Boozer angles :math:`(\zeta, \theta)` and energies, as
    ``heat_flux.lost_particles`` computes them from the traced particles. Each strike point
    is drawn independently by a conditional flow-matching model, as many as the predicted
    number of lost particles. A sample is a prediction for ``heat_flux.heat_flux_metrics``.

    The inputs are the birth surface :math:`s_0` and these fields of the configuration:

    - ``boundary_xyz``: the plasma boundary over one field period [m], shape
      ``(41 * 41 * 3,)`` as stored, or ``(41, 41, 3)``. Reshaped to ``(41, 41, 3)``, it is
      indexed :math:`(\phi, \theta, xyz)`: Cartesian points of the boundary surface on 41
      toroidal angles :math:`\phi` uniformly spaced in :math:`[0, 2\pi / n_\text{fp})` and
      41 poloidal angles :math:`\theta` uniformly spaced in :math:`[0, 2\pi)`, endpoints
      excluded. This is the layout of ``gamma()`` of a simsopt ``SurfaceRZFourier`` with
      these quadrature points.
    - ``nfp``: the number of field periods.
    - ``d_pressure_d_s``: the radial derivative of the pressure [Pa], shape ``(15,)``,
      on the radial grid ``s1d``, 15 points from :math:`s = 10^{-4}` to 1.
    - ``iota``: the rotational transform of the VMEC equilibrium, shape ``(15,)``, on the
      same grid.
    - ``volume_avg_modB``: the volume-averaged field strength [T].
    - ``minor_radius``: the effective minor radius of the boundary [m].

    Load the model with ``load``, and predict with ``sample`` and ``predict_loss_fraction``.
    The predictions are not differentiable.

    Example:
        Sample the strike points of the particles born on :math:`s_0 = 0.1` in a test
        configuration, and compare their heat flux with the truth::

            import h5py
            import boss_bench as bb
            from boss_bench import models
            from boss_bench.models import HeatFluxModel

            path = models.download("heat_flux", "BoSS-models/heat_flux")
            model = HeatFluxModel.load(path)
            with h5py.File(bb.list_files("BoSS", "confinement", "test")[0], "r") as f:
                record = bb.load_record(f, 0)
            prediction = model.sample(record, s0=0.1, seed=0)
            bb.heat_flux_error(prediction, record, s0=0.1)
    """

    def __init__(self):
        super().__init__()
        embed_dim, d_cond = 256, 1 + 2 * 15 + 2 + 1  # nfp, d_pressure_d_s, iota, modB, minor radius and s0
        self.cond_encoder = _ConditioningEncoder(d_cond, embed_dim)
        self.time_embedding = _TimeEmbedding(embed_dim)
        self.velocity_net = _ResNetVelocity(X_DIM, embed_dim, hidden=512, n_blocks=4)
        self.loss_frac_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
        # the conditioning and the log energy are standardized with the training-set statistics
        self.register_buffer("scalar_mean", torch.zeros(d_cond - 1))
        self.register_buffer("scalar_std", torch.ones(d_cond - 1))
        self.register_buffer("s0_mean_std", torch.tensor([0.0, 1.0]))
        self.register_buffer("xyz_mean", torch.zeros(3))
        self.register_buffer("xyz_std", torch.ones(3))
        self.register_buffer("log_energy_mean_std", torch.tensor([0.0, 1.0]))

    @classmethod
    def load(cls, path, device="cpu"):
        """Load the trained model.

        Args:
            path (str): the model's folder, holding ``config.json`` and ``model.safetensors``,
                e.g. from ``models.download("heat_flux", path)``.
            device (str or torch.device): the device to put the model on.

        Returns:
            HeatFluxModel: the model, ready to predict.

        Raises:
            ValueError: if the folder holds another model.
        """
        return load_trained(cls(), "heat_flux", path, device)

    @torch.no_grad()
    def predict_loss_fraction(self, data, s0=0.1):
        """Predict the loss fraction of the particles born on one surface, for one or several configurations.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only
                ``boundary_xyz``, ``nfp``, ``d_pressure_d_s``, ``iota``, ``volume_avg_modB``
                and ``minor_radius`` are read.
            s0 (float): the birth surface, 0.1, 0.25 or 0.5.

        Returns:
            float or np.ndarray: the loss fraction, shape ``(n,)`` for a batch.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf.
        """
        batch, single = as_batch(data, FIELDS)
        loss_fraction = self._embed_batch(batch, s0)[1].cpu().numpy().astype(float)
        return float(loss_fraction[0]) if single else loss_fraction

    @torch.no_grad()
    def sample(self, data, s0=0.1, seed=None):
        r"""Sample the strike points of the lost particles born on one surface, for one or several configurations.

        For each configuration, the number of strike points is the predicted number of lost
        particles of the 5,000 launched, :math:`\max(1, \operatorname{round}(5000 \hat f))`,
        with :math:`\hat f` the predicted loss fraction.

        Args:
            data (dict): the fields of one configuration, as from ``data.load_record``, or of
                several, with one row per configuration, as from ``data.load``. Only
                ``boundary_xyz``, ``nfp``, ``d_pressure_d_s``, ``iota``, ``volume_avg_modB``
                and ``minor_radius`` are read.
            s0 (float): the birth surface, 0.1, 0.25 or 0.5.
            seed (int, optional): the random seed. By default, a random one.

        Returns:
            tuple or list of tuple: **zeta, theta, energy, loss_fraction**, the Boozer angles
            [rad] and energies [MeV] of the strike points, arrays of shape ``(k,)``, and the
            predicted loss fraction, a float; one tuple per configuration for a batch.

        Raises:
            KeyError: if a field is missing from ``data``.
            ValueError: if a field contains NaN or inf.
        """
        batch, single = as_batch(data, FIELDS)
        embedding, loss_fraction = self._embed_batch(batch, s0)
        loss_fraction = loss_fraction.cpu().numpy()
        counts = np.maximum(1, np.round(loss_fraction * N_LAUNCHED)).astype(int)
        gen = generator(seed, embedding.device)
        out, a = [], 0
        while a < len(counts):  # configurations a to b - 1 are integrated together
            b = a + 1
            while b < len(counts) and counts[a:b + 1].sum() <= MAX_POINTS:
                b += 1
            x = torch.randn(int(counts[a:b].sum()), X_DIM, generator=gen, device=embedding.device)
            rows = torch.as_tensor(np.repeat(np.arange(a, b), counts[a:b]), device=embedding.device)
            x = self._integrate(x, embedding[rows])
            for i, points in zip(range(a, b), torch.split(x, counts[a:b].tolist())):
                strike_points = tuple(p.cpu().numpy().astype(float) for p in self._strike_points(points))
                out.append((*strike_points, float(loss_fraction[i])))
            a = b
        return out[0] if single else out

    def _embed(self, boundary_xyz, nfp, d_pressure_d_s, iota, volume_avg_modB, minor_radius, s0):
        """Encode a batch of configurations and birth surfaces, all tensors with one row each.

        Returns:
            tuple: **embedding, loss_fraction**, the encodings, shape ``(n, 256)``, and the
            predicted loss fractions, shape ``(n,)``.
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

    def _embed_batch(self, batch, s0):
        """Encode the configurations of a batch of numpy fields, BATCH_SIZE at a time; see _embed."""
        device = self.scalar_mean.device
        n = len(batch["nfp"])
        out = []
        for a in range(0, n, BATCH_SIZE):
            rows = {name: torch.as_tensor(x[a:a + BATCH_SIZE], device=device) for name, x in batch.items()}
            out.append(self._embed(**rows, s0=torch.full((len(rows["nfp"]),), s0, device=device)))
        return torch.cat([e for e, _ in out]), torch.cat([f for _, f in out])

    def _velocity(self, x, t, embedding):
        """The velocity of the flow at points x, shape ``(k, 5)``, at time t (one value), given their encodings."""
        t_emb = self.time_embedding(t.reshape(1, 1).float()).expand(x.shape[0], -1)
        return self.velocity_net(x, t_emb, embedding)

    def _integrate(self, x, embedding):
        """Map standard normal samples x, shape ``(k, 5)``, to points of the flow at time 1, by N_STEPS Euler steps."""
        return integrate(lambda x, t: self._velocity(x, torch.tensor(t, device=x.device), embedding),
                         x, 0.0, 1.0, N_STEPS, "euler")

    def _strike_points(self, x):
        """Convert points of the flow at time 1 to strike points: Boozer angles [rad] and energies [MeV]."""
        zeta = torch.atan2(x[:, 1], x[:, 0]) % (2 * math.pi)
        theta = torch.atan2(x[:, 3], x[:, 2]) % (2 * math.pi)
        energy = torch.exp(x[:, 4] * self.log_energy_mean_std[1] + self.log_energy_mean_std[0])
        return zeta, theta, energy


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
