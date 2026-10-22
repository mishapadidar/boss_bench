"""Inverse-design model: generate plasma boundaries with chosen properties."""

import math

import numpy as np
import torch
from torch import nn

from ._common import as_batch, generator, integrate, load_trained

CONDITIONS = ("nfp", "loss_fraction", "toroidal_flux_edge", "minor_radius", "aspect_ratio", "min_modB",
              "mirror_ratio", "pressure", "iota", "vmec_ncurr")
SCALAR_CONDITIONS = CONDITIONS[1:7]
N_GRID = 41  # boundary points per angle
N_PROFILE = 15  # radial points of the profiles
N_NFP = 6  # the model covers nfp from 1 to N_NFP
D_DESIGN = N_GRID * N_GRID * 2  # a design is (R, Z) on the boundary grid
D_PROPS = 1 + len(SCALAR_CONDITIONS) + 2 * N_PROFILE + 1  # the conditioning vector
T_MIN, T_MAX = 1e-4, 1 - 1e-4  # the time interval of the flow
N_STEPS = 200  # midpoint steps of the flow, as in the paper
BATCH_SIZE = 64  # designs generated at once
# the UNet: channels of the first level, channel multiplier of each level, residual blocks
# per level, the levels with self-attention, attention heads
BASE_CH, CH_MULT, NUM_RES_BLOCKS, ATTN_LEVELS, NUM_HEADS = 128, (1, 2, 2), 4, (2,), 4


class InverseDesignModel(nn.Module):
    """The trained inverse-design model: generate plasma boundaries with a desired loss fraction, shape and field.

    The inputs are the properties the configuration should have, under the names of the
    BoSS fields and in their units:

    - ``nfp``, the number of field periods, from 1 to 6;
    - ``loss_fraction``, the fraction of the alpha particles born on :math:`s = 0.1` that
      are lost, as ``confinement.loss_fraction`` computes it;
    - ``toroidal_flux_edge``, ``minor_radius``, ``aspect_ratio``, ``min_modB`` and
      ``mirror_ratio``;
    - the ``pressure`` and ``iota`` profiles, of 15 points each;
    - ``vmec_ncurr``, 0 if the equilibrium is computed with the ``iota`` profile, 1 if it
      is current-free.

    The output is a plasma boundary, laid out as the field ``boundary_xyz``. Each call
    draws new boundaries.

    Load the model with ``InverseDesignModel.load``, and generate with ``sample``. The
    boundaries are not differentiable with respect to the properties.
    """

    def __init__(self):
        super().__init__()
        n_levels = len(CH_MULT)
        pad = (-N_GRID) % 2 ** (n_levels - 1)  # the grid is padded to a multiple of 2**(n_levels - 1)
        self._pad = (pad // 2, pad - pad // 2)
        time_dim, nfp_embed_dim, cond_dim = 128, 8, BASE_CH * 4
        self.time_embed = _SinusoidalTimeEmbedding(time_dim)
        self.nfp_embed = nn.Embedding(N_NFP + 1, nfp_embed_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(time_dim + nfp_embed_dim + D_PROPS - 1, cond_dim), nn.SiLU(),
                                      nn.Linear(cond_dim, cond_dim))
        self.input_conv = nn.Conv2d(2, BASE_CH, 3, padding=1)

        self.encoders, self.downsamples = nn.ModuleList(), nn.ModuleList()
        ch = BASE_CH
        for level in range(n_levels):
            blocks = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS):
                blocks.append(_ResBlock(ch, BASE_CH * CH_MULT[level], cond_dim))
                ch = BASE_CH * CH_MULT[level]
                if level in ATTN_LEVELS:
                    blocks.append(_SelfAttention(ch, NUM_HEADS))
            self.encoders.append(blocks)
            if level < n_levels - 1:
                self.downsamples.append(_Downsample(ch))

        self.mid_block1 = _ResBlock(ch, ch, cond_dim)
        self.mid_attn = _SelfAttention(ch, NUM_HEADS)
        self.mid_block2 = _ResBlock(ch, ch, cond_dim)

        self.decoders, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for level in reversed(range(n_levels)):
            out_ch = BASE_CH * CH_MULT[level]
            blocks = nn.ModuleList()
            for i in range(NUM_RES_BLOCKS):  # the first block also takes the skip connection
                blocks.append(_ResBlock(ch + out_ch if i == 0 else out_ch, out_ch, cond_dim))
                ch = out_ch
                if level in ATTN_LEVELS:
                    blocks.append(_SelfAttention(ch, NUM_HEADS))
            self.decoders.append(blocks)
            if level > 0:
                self.upsamples.append(_Upsample(ch))

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, 2, 3, padding=1)

        # designs and properties are standardized with the training-set statistics; the
        # profiles first by asinh(x / scale)
        self.register_buffer("design_mean", torch.zeros(D_DESIGN))
        self.register_buffer("design_std", torch.ones(D_DESIGN))
        self.register_buffer("scalar_mean", torch.zeros(len(SCALAR_CONDITIONS)))
        self.register_buffer("scalar_std", torch.ones(len(SCALAR_CONDITIONS)))
        for name in ("pressure", "iota"):
            self.register_buffer(f"{name}_scale", torch.tensor(1.0))
            self.register_buffer(f"{name}_mean", torch.zeros(N_PROFILE))
            self.register_buffer(f"{name}_std", torch.ones(N_PROFILE))

    @classmethod
    def load(cls, path, device="cpu"):
        """Load the trained model.

        Args:
            path (str): the model's folder, holding ``config.json`` and ``model.safetensors``,
                e.g. from ``models.download("inverse_design", path)``.
            device (str or torch.device): the device to put the model on.

        Returns:
            InverseDesignModel: the model, ready to generate.

        Raises:
            ValueError: if the folder holds another model.
        """
        return load_trained(cls(), "inverse_design", path, device)

    @torch.no_grad()
    def sample(self, conditions, seed=None):
        """Generate one boundary per set of properties.

        Args:
            conditions (dict): the properties of one design, or of several, with one row per
                design: ``nfp``, ``loss_fraction``, ``toroidal_flux_edge``, ``minor_radius``,
                ``aspect_ratio``, ``min_modB``, ``mirror_ratio``, ``pressure``, ``iota`` and
                ``vmec_ncurr``. To ask for several designs with the same properties, repeat
                the rows.
            seed (int, optional): the random seed. By default, a random one.

        Returns:
            np.ndarray: the boundary points [m], laid out as the field ``boundary_xyz``, shape
            ``(41 * 41 * 3,)``, or ``(n, 41 * 41 * 3)`` for several designs.

        Raises:
            KeyError: if a property is missing.
            ValueError: if a property contains NaN or inf, or ``nfp`` is not between 1 and 6.
        """
        batch, single = as_batch(conditions, CONDITIONS)
        nfp = batch["nfp"]
        if np.any((nfp < 1) | (nfp > N_NFP) | (nfp != np.round(nfp))):
            raise ValueError(f"nfp must be an integer between 1 and {N_NFP}")
        device = self.design_mean.device
        gen = generator(seed, device)
        out = []
        for a in range(0, len(nfp), BATCH_SIZE):
            rows = {name: torch.as_tensor(x[a:a + BATCH_SIZE], device=device) for name, x in batch.items()}
            x = torch.randn(len(rows["nfp"]), D_DESIGN, generator=gen, device=device)
            x = self._integrate(x, self._condition(**rows))
            out.append(self._boundary(x, rows["nfp"]).cpu().numpy())
        out = np.concatenate(out)
        return out[0] if single else out

    def _condition(self, nfp, loss_fraction, toroidal_flux_edge, minor_radius, aspect_ratio, min_modB, mirror_ratio,
                   pressure, iota, vmec_ncurr):
        """Build the standardized conditioning vectors, shape ``(n, 38)``, of a batch of properties (tensors)."""
        scalars = torch.stack([loss_fraction, toroidal_flux_edge, minor_radius, aspect_ratio, min_modB,
                               mirror_ratio], dim=1).float()
        scalars = (scalars - self.scalar_mean) / self.scalar_std
        pressure = (torch.asinh(pressure.float() / self.pressure_scale) - self.pressure_mean) / self.pressure_std
        iota = (torch.asinh(iota.float() / self.iota_scale) - self.iota_mean) / self.iota_std
        return torch.cat([nfp.float()[:, None], scalars, pressure, iota, vmec_ncurr.float()[:, None]], dim=1)

    def _velocity(self, x, t, props):
        """The velocity of the flow at standardized designs x, shape ``(n, 3362)``, at times t, shape ``(n,)``."""
        n = x.shape[0]
        h = x.view(n, N_GRID, N_GRID, 2).permute(0, 3, 1, 2)
        h = nn.functional.pad(h, (*self._pad, *self._pad), mode="reflect")
        nfp = self.nfp_embed(props[:, 0].round().long().clamp(0, N_NFP))
        cond = self.cond_mlp(torch.cat([self.time_embed(t), nfp, props[:, 1:]], dim=-1))

        h = self.input_conv(h)
        skips = []
        for level, blocks in enumerate(self.encoders):
            for block in blocks:
                h = block(h, cond) if isinstance(block, _ResBlock) else block(h)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h, cond)), cond)
        for level, blocks in enumerate(self.decoders):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, cond) if isinstance(block, _ResBlock) else block(h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)
        h = self.out_conv(nn.functional.silu(self.out_norm(h)))
        h = h[:, :, self._pad[0]:self._pad[0] + N_GRID, self._pad[0]:self._pad[0] + N_GRID]
        return h.permute(0, 2, 3, 1).reshape(n, -1)

    def _integrate(self, x, props):
        """Map standard normal samples x to standardized designs, by N_STEPS midpoint steps over [T_MIN, T_MAX]."""
        return integrate(lambda x, t: self._velocity(x, torch.full((x.shape[0],), t, device=x.device), props),
                         x, T_MIN, T_MAX, N_STEPS, "midpoint")

    def _boundary(self, x, nfp):
        """Convert standardized designs, (R, Z) on the boundary grid, to boundary points [m] laid out as boundary_xyz.

        The toroidal angles of the grid are phi_i = 2 pi i / (41 nfp), i = 0, ..., 40. The
        points are float64.
        """
        rz = (x * self.design_std + self.design_mean).double().reshape(-1, N_GRID, N_GRID, 2)
        i = torch.arange(N_GRID, device=x.device, dtype=torch.float64)
        phi = 2 * math.pi * i[None, :, None] / (N_GRID * nfp.double()[:, None, None])
        xyz = torch.stack([rz[..., 0] * torch.cos(phi), rz[..., 0] * torch.sin(phi), rz[..., 1]], dim=-1)
        return xyz.reshape(rz.shape[0], -1)


class _SinusoidalTimeEmbedding(nn.Module):
    """Embed the time by sinusoids of log-spaced frequencies.

    Args:
        dim (int): width of the embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """Embed times of shape ``(n,)`` to shape ``(n, dim)``."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _ResBlock(nn.Module):
    """A residual block whose second normalization is scaled and shifted by the conditioning (FiLM).

    Args:
        in_ch (int): input channels.
        out_ch (int): output channels.
        cond_dim (int): width of the conditioning.
    """

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        """Apply the block to feature maps x of shape ``(n, in_ch, h, w)``, given the conditioning."""
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.conv2(self.act(self.norm2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class _SelfAttention(nn.Module):
    """Multi-head self-attention over the positions of a feature map, with a residual connection.

    Args:
        channels (int): channels of the feature map.
        num_heads (int): the number of heads.
    """

    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Apply attention to feature maps x of shape ``(n, channels, h, w)``."""
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, self.num_heads, c // self.num_heads, h * w).unbind(1)
        out = nn.functional.scaled_dot_product_attention(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3))
        return x + self.proj(out.transpose(2, 3).reshape(n, c, h, w))


class _Downsample(nn.Module):
    """Halve the resolution of a feature map by a stride-2 convolution.

    Args:
        channels (int): channels of the feature map.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        """Downsample x of shape ``(n, channels, h, w)``."""
        return self.conv(x)


class _Upsample(nn.Module):
    """Double the resolution of a feature map by nearest-neighbor interpolation and a convolution.

    Args:
        channels (int): channels of the feature map.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        """Upsample x of shape ``(n, channels, h, w)``."""
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))
