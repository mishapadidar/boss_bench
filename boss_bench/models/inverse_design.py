"""Inverse-design model: generate plasma boundaries with chosen properties."""

import math

import numpy as np
import torch
from torch import nn

from ._common import as_batch, generator, integrate

CONDITIONS = ("nfp", "loss_fraction", "toroidal_flux_edge", "minor_radius", "aspect_ratio", "min_modB",
              "mirror_ratio", "pressure", "iota", "vmec_ncurr")
SCALAR_CONDITIONS = CONDITIONS[1:7]
N_GRID = 41  # boundary points per angle
N_PROFILE = 15  # radial points of the profiles
D_DESIGN = N_GRID * N_GRID * 2  # a design is (R, Z) on the boundary grid
D_PROPS = 1 + len(SCALAR_CONDITIONS) + 2 * N_PROFILE + 1  # the conditioning vector
T_MIN, T_MAX = 1e-4, 1 - 1e-4  # the time interval of the flow


class InverseDesignModel(nn.Module):
    r"""Generate plasma boundaries with a chosen loss fraction, shape, field and profiles.

    The model generates the boundary of a configuration, :math:`R` and :math:`Z` on the
    41 × 41 grid of ``boundary_xyz``, from properties the configuration should have:

    - ``nfp``, the number of field periods, from 1 to 6;
    - ``loss_fraction``, the fraction of the alpha particles born on :math:`s = 0.1` that
      are lost, as ``confinement.loss_fraction`` computes it;
    - ``toroidal_flux_edge``, ``minor_radius``, ``aspect_ratio``, ``min_modB`` and
      ``mirror_ratio``;
    - the ``pressure`` and ``iota`` profiles, of 15 points each;
    - ``vmec_ncurr``, 0 if the equilibrium is computed with the ``iota`` profile, 1 if it
      is current-free.

    Each boundary is drawn by integrating a learned flow (conditional flow matching) from a
    Gaussian sample. The velocity of the flow is a UNet over the boundary grid,
    conditioned on the time and on the properties by FiLM layers.

    The model was trained on the train split. In the paper, its designs are fitted by a
    Fourier surface (``mpol = ntor = 10``), rescaled to ARIES-CS size, and evaluated with
    VMEC and particle tracing (see ``design.design_metrics``).

    Load a pretrained model with ``models.load``. To generate, use ``sample``.

    Args:
        base_ch (int): channels of the first UNet level.
        ch_mult (tuple of int): channel multiplier of each UNet level.
        num_res_blocks (int): the number of residual blocks per level.
        attn_levels (tuple of int): the levels with self-attention.
        num_heads (int): the number of attention heads.
        time_dim (int): width of the time embedding.
        n_nfp (int): the number of rows of the ``nfp`` embedding, for ``nfp`` from 0.
        nfp_embed_dim (int): width of the ``nfp`` embedding.
    """

    def __init__(self, base_ch=128, ch_mult=(1, 2, 2), num_res_blocks=4, attn_levels=(2,), num_heads=4,
                 time_dim=128, n_nfp=7, nfp_embed_dim=8):
        super().__init__()
        self.config = {"base_ch": base_ch, "ch_mult": list(ch_mult), "num_res_blocks": num_res_blocks,
                       "attn_levels": list(attn_levels), "num_heads": num_heads, "time_dim": time_dim,
                       "n_nfp": n_nfp, "nfp_embed_dim": nfp_embed_dim}
        n_levels = len(ch_mult)
        pad = (-N_GRID) % 2 ** (n_levels - 1)  # the grid is padded to a multiple of 2**(n_levels - 1)
        self._pad = (pad // 2, pad - pad // 2)
        self.time_embed = _SinusoidalTimeEmbedding(time_dim)
        self.nfp_embed = nn.Embedding(n_nfp, nfp_embed_dim)
        cond_dim = base_ch * 4
        self.cond_mlp = nn.Sequential(nn.Linear(time_dim + nfp_embed_dim + D_PROPS - 1, cond_dim), nn.SiLU(),
                                      nn.Linear(cond_dim, cond_dim))
        self.input_conv = nn.Conv2d(2, base_ch, 3, padding=1)

        self.encoders, self.downsamples = nn.ModuleList(), nn.ModuleList()
        ch = base_ch
        for level in range(n_levels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock(ch, base_ch * ch_mult[level], cond_dim))
                ch = base_ch * ch_mult[level]
                if level in attn_levels:
                    blocks.append(_SelfAttention(ch, num_heads))
            self.encoders.append(blocks)
            if level < n_levels - 1:
                self.downsamples.append(_Downsample(ch))

        self.mid_block1 = _ResBlock(ch, ch, cond_dim)
        self.mid_attn = _SelfAttention(ch, num_heads)
        self.mid_block2 = _ResBlock(ch, ch, cond_dim)

        self.decoders, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for level in reversed(range(n_levels)):
            out_ch = base_ch * ch_mult[level]
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):  # the first block also takes the skip connection
                blocks.append(_ResBlock(ch + out_ch if i == 0 else out_ch, out_ch, cond_dim))
                ch = out_ch
                if level in attn_levels:
                    blocks.append(_SelfAttention(ch, num_heads))
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

    def condition(self, nfp, loss_fraction, toroidal_flux_edge, minor_radius, aspect_ratio, min_modB, mirror_ratio,
                  pressure, iota, vmec_ncurr):
        """Build the standardized conditioning vectors of a batch of properties.

        Args:
            nfp (torch.Tensor): numbers of field periods, shape ``(n,)``.
            loss_fraction (torch.Tensor): loss fractions, shape ``(n,)``.
            toroidal_flux_edge (torch.Tensor): toroidal fluxes at the boundary [Wb], shape ``(n,)``.
            minor_radius (torch.Tensor): minor radii [m], shape ``(n,)``.
            aspect_ratio (torch.Tensor): aspect ratios, shape ``(n,)``.
            min_modB (torch.Tensor): minimum field strengths [T], shape ``(n,)``.
            mirror_ratio (torch.Tensor): mirror ratios, shape ``(n,)``.
            pressure (torch.Tensor): pressure profiles [Pa], shape ``(n, 15)``.
            iota (torch.Tensor): rotational transforms, shape ``(n, 15)``.
            vmec_ncurr (torch.Tensor): 0 for an ``iota`` constraint, 1 for current-free, shape ``(n,)``.

        Returns:
            torch.Tensor: the conditioning vectors, shape ``(n, 38)``.
        """
        scalars = torch.stack([loss_fraction, toroidal_flux_edge, minor_radius, aspect_ratio, min_modB,
                               mirror_ratio], dim=1).float()
        scalars = (scalars - self.scalar_mean) / self.scalar_std
        pressure = (torch.asinh(pressure.float() / self.pressure_scale) - self.pressure_mean) / self.pressure_std
        iota = (torch.asinh(iota.float() / self.iota_scale) - self.iota_mean) / self.iota_std
        return torch.cat([nfp.float()[:, None], scalars, pressure, iota, vmec_ncurr.float()[:, None]], dim=1)

    def forward(self, x, t, props):
        """Evaluate the velocity of the flow.

        Args:
            x (torch.Tensor): standardized designs, shape ``(n, 41 * 41 * 2)``.
            t (torch.Tensor): the times, shape ``(n,)``.
            props (torch.Tensor): the conditioning vectors, from ``condition``, shape ``(n, 38)``.

        Returns:
            torch.Tensor: the velocity, shape ``(n, 41 * 41 * 2)``.
        """
        n = x.shape[0]
        h = x.view(n, N_GRID, N_GRID, 2).permute(0, 3, 1, 2)
        h = nn.functional.pad(h, (*self._pad, *self._pad), mode="reflect")
        nfp = self.nfp_embed(props[:, 0].round().long().clamp(0, self.nfp_embed.num_embeddings - 1))
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

    def integrate(self, x, props, n_steps=200, method="midpoint"):
        r"""Map Gaussian samples to standardized designs by integrating the flow over :math:`[10^{-4}, 1 - 10^{-4}]`.

        Args:
            x (torch.Tensor): standard normal samples, shape ``(n, 41 * 41 * 2)``.
            props (torch.Tensor): the conditioning vectors, from ``condition``, shape ``(n, 38)``.
            n_steps (int): the number of steps.
            method (str): ``"euler"`` or ``"midpoint"``.

        Returns:
            torch.Tensor: the standardized designs, shape ``(n, 41 * 41 * 2)``.
        """
        return integrate(lambda x, t: self(x, torch.full((len(x),), t, device=x.device), props),
                         x, T_MIN, T_MAX, n_steps, method)

    @torch.no_grad()
    def sample(self, conditions, n_steps=200, method="midpoint", seed=None, batch_size=64):
        """Generate one boundary per set of properties.

        Args:
            conditions (dict): the properties of one design, or of several, with one row per
                design: ``nfp``, ``loss_fraction``, ``toroidal_flux_edge``, ``minor_radius``,
                ``aspect_ratio``, ``min_modB``, ``mirror_ratio``, ``pressure``, ``iota`` and
                ``vmec_ncurr``, in the units of the BoSS fields of these names. To ask for
                several designs with the same properties, repeat the rows.
            n_steps (int): the number of integration steps.
            method (str): ``"euler"`` or ``"midpoint"``.
            seed (int, optional): the random seed. By default, a random one.
            batch_size (int): the number of designs generated at once.

        Returns:
            np.ndarray: the boundary points [m], laid out as the field ``boundary_xyz``, shape
            ``(41 * 41 * 3,)``, or ``(n, 41 * 41 * 3)`` for several designs.

        Raises:
            KeyError: if a property is missing.
            ValueError: if a property contains NaN or inf, or ``nfp`` is not between 1 and 6.
        """
        batch, single = as_batch(conditions, CONDITIONS)
        nfp = batch["nfp"]
        if np.any((nfp < 1) | (nfp > self.nfp_embed.num_embeddings - 1) | (nfp != np.round(nfp))):
            raise ValueError(f"nfp must be an integer between 1 and {self.nfp_embed.num_embeddings - 1}")
        device = self.design_mean.device
        gen = generator(seed, device)
        out = []
        for a in range(0, len(nfp), batch_size):
            rows = {name: torch.as_tensor(x[a:a + batch_size], device=device) for name, x in batch.items()}
            x = torch.randn(len(rows["nfp"]), D_DESIGN, generator=gen, device=device)
            design = self.integrate(x, self.condition(**rows), n_steps, method) * self.design_std + self.design_mean
            out.append(_boundary_xyz(design.cpu().numpy().astype(float), nfp[a:a + batch_size]))
        out = np.concatenate(out)
        return out[0] if single else out


def _boundary_xyz(design, nfp):
    """Convert designs, (R, Z) on the boundary grid, to boundary points laid out as ``boundary_xyz``.

    Args:
        design (np.ndarray): the designs, shape ``(n, 41 * 41 * 2)``.
        nfp (np.ndarray): their numbers of field periods, shape ``(n,)``.

    Returns:
        np.ndarray: the boundary points [m], shape ``(n, 41 * 41 * 3)``.
    """
    rz = design.reshape(-1, N_GRID, N_GRID, 2)
    phi = 2 * np.pi * np.arange(N_GRID)[None, :, None] / (N_GRID * np.asarray(nfp)[:, None, None])
    xyz = np.stack([rz[..., 0] * np.cos(phi), rz[..., 0] * np.sin(phi), rz[..., 1]], axis=-1)
    return xyz.reshape(len(rz), -1)


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
