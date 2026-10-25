"""Small DiT backbone shared by Drifting and denoising diffusion.

The input is fresh Gaussian noise, not a noisy training image. ``cfg_scale`` is
embedded as a condition, as in the Drift generator; it is neither a diffusion
timestep nor an instruction to blend conditional/unconditional predictions.
``forward_denoise`` reuses exactly the same parameters with a noise-level
condition and optional zero label embedding for conventional diffusion CFG.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _sincos(values: Tensor, width: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(width // 2, device=values.device, dtype=torch.float32)
        / (width // 2)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _position_embedding(grid_size: int, hidden_size: int) -> Tensor:
    row, column = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    return torch.cat(
        (_sincos(row.flatten(), hidden_size // 2),
         _sincos(column.flatten(), hidden_size // 2)),
        dim=-1,
    ).unsqueeze(0)


def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GuidanceEmbedder(nn.Module):
    """Sinusoidal encoding followed by a learned scalar-condition projection."""

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, scale: Tensor) -> Tensor:
        encoded = _sincos(scale, self.frequency_size)
        return self.mlp(encoded.to(dtype=self.mlp[0].weight.dtype))


class DiTBlock(nn.Module):
    """Self-attention and MLP with adaLN-Zero residual gates."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        normalized = _modulate(self.norm_attention(tokens), shift_a, scale_a)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + gate_a.unsqueeze(1) * attended
        normalized = _modulate(self.norm_mlp(tokens), shift_m, scale_m)
        return tokens + gate_m.unsqueeze(1) * self.mlp(normalized)


class DiT(nn.Module):
    """Patch-based DiT mapping ``(noise, digit labels, cfg_scale)`` to images.

    Output pixels are unconstrained; the data adapter determines training
    normalization and the preview writer handles display clipping. No dataset,
    loss, optimizer, or sampling schedule is coupled to this architecture.

    The transformer gates use the usual adaLN-Zero initialization. Unlike a
    diffusion noise predictor, the final projection starts with small random
    weights: a zero projection would give all generated particles the same
    image and remove the initial diversity used by the Drift objective.
    """

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 4,
        hidden_size: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        dimensions = {
            "image_size": image_size, "in_channels": in_channels,
            "num_classes": num_classes, "patch_size": patch_size,
            "hidden_size": hidden_size, "depth": depth, "num_heads": num_heads,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if hidden_size % 4 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by 4 and num_heads")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_size * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and give at least one hidden unit")

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.register_buffer("position_embedding", _position_embedding(self.grid_size, hidden_size))
        self.label_embed = nn.Embedding(num_classes, hidden_size)
        self.guidance_embed = GuidanceEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final_projection = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        for layer in (self.guidance_embed.mlp[0], self.guidance_embed.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attention.in_proj_weight)
            nn.init.zeros_(block.attention.in_proj_bias)
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.normal_(self.final_projection.weight, std=0.02)

    def _validate_inputs(
        self, noise: Tensor, labels: Tensor, cfg_scale: float | Tensor,
        *, allow_signed: bool = False,
    ) -> Tensor:
        expected = (self.in_channels, self.image_size, self.image_size)
        if noise.ndim != 4 or tuple(noise.shape[1:]) != expected or noise.shape[0] == 0:
            raise ValueError(f"noise must have nonempty shape [B, {expected[0]}, {expected[1]}, {expected[2]}]")
        if not noise.is_floating_point():
            raise TypeError("noise must be a floating-point tensor")
        if labels.shape != (noise.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be a torch.long tensor of shape [B]")
        if labels.device != noise.device:
            raise ValueError("noise and labels must be on the same device")
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"labels must be in [0, {self.num_classes - 1}]")
        scales = torch.as_tensor(cfg_scale, dtype=torch.float32, device=noise.device)
        if scales.ndim == 0:
            scales = scales.expand(noise.shape[0])
        if scales.shape != (noise.shape[0],):
            raise ValueError("cfg_scale must be a scalar or tensor of shape [B]")
        if not torch.isfinite(scales).all() or (not allow_signed and torch.any(scales < 0)):
            raise ValueError("Condition must be finite (and nonnegative for Drift guidance)")
        return scales

    def unpatchify(self, patches: Tensor) -> Tensor:
        batch = patches.shape[0]
        patches = patches.reshape(
            batch, self.grid_size, self.grid_size,
            self.patch_size, self.patch_size, self.in_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self.in_channels, self.image_size, self.image_size
        )

    def forward(
        self, noise: Tensor, labels: Tensor, cfg_scale: float | Tensor = 1.0
    ) -> Tensor:
        scales = self._validate_inputs(noise, labels, cfg_scale)
        condition = self.label_embed(labels) + self.guidance_embed(scales)
        return self._forward_conditioned(noise, condition)

    def forward_denoise(
        self, images: Tensor, labels: Tensor, noise_condition: float | Tensor,
        label_mask: Tensor | None = None,
    ) -> Tensor:
        """Use the same DiT as an EDM residual predictor.

        ``noise_condition`` is log(sigma)/4, which can be negative. A false
        ``label_mask`` entry zeros the entire class embedding, leaving the
        noise-level embedding active. Digit zero remains an ordinary class.
        No new parameter, null class, or change to Drift checkpoints is needed.
        """
        levels = self._validate_inputs(images, labels, noise_condition, allow_signed=True)
        class_condition = self.label_embed(labels)
        if label_mask is not None:
            if (label_mask.shape != labels.shape or label_mask.dtype != torch.bool
                    or label_mask.device != images.device):
                raise ValueError("label_mask must be a bool tensor of shape [B] on the image device")
            class_condition = class_condition * label_mask[:, None]
        condition = class_condition + self.guidance_embed(levels)
        return self._forward_conditioned(images, condition)

    def _forward_conditioned(self, noise: Tensor, condition: Tensor) -> Tensor:
        tokens = self.patch_embed(noise).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        patches = self.final_projection(_modulate(self.final_norm(tokens), shift, scale))
        return self.unpatchify(patches)
