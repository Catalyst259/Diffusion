"""Pixel-space Drifting with learned, single-forward classifier-free guidance.

The field follows ``Drift/drifting/drift_loss.py`` and CFG follows its
``train.py``. The original MIT license is included alongside this module.
Unlike diffusion/flow matching, a generated image is regressed towards a
detached drifting target; sampling needs neither a time step nor an ODE solver.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _temperatures(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("temperatures must contain finite, positive values")
    return result


def _normalized_affinity(distances: Tensor, temperature: float, self_mask: Tensor) -> Tensor:
    """Symmetric row/column normalization with exactly zero self affinity."""
    logits = (-distances / temperature).masked_fill(self_mask, -torch.inf)
    affinity = torch.sqrt(
        (logits.softmax(dim=-1) * logits.softmax(dim=-2)).clamp_min(1e-6)
    )
    # The reference clips affinities away from zero. Explicitly remove the
    # diagonal afterwards so this numerical floor cannot reintroduce self force.
    return affinity.masked_fill(self_mask, 0.0)


def drift_loss(
    gen: Tensor,
    positives: Tensor,
    negatives: Tensor | None = None,
    *,
    negative_weight: float | Tensor = 1.0,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
) -> Tensor:
    """Return normalized Drift loss for grouped ``[B, samples, features]`` data.

    Each group must have at least two generated samples and one positive.
    Generated samples also act as negatives, excluding their own diagonal.
    Optional fixed negatives have one nonnegative weight per group. Only
    ``gen`` receives gradients: real samples, coordinate scale and the whole
    drifting target are detached.

    Each temperature's field is RMS-normalized, as in the reference. Therefore
    this objective is not an image reconstruction error and need not decrease
    monotonically as samples improve.
    """
    radii = _temperatures(temperatures)
    if gen.ndim != 3 or positives.ndim != 3:
        raise ValueError("gen and positives must have shape [groups, samples, features]")
    batch, generated_count, features = gen.shape
    if batch < 1 or generated_count < 2 or features < 1:
        raise ValueError("Drift needs nonempty groups/features and at least two generated samples")
    if positives.shape[0] != batch or positives.shape[2] != features or positives.shape[1] < 1:
        raise ValueError("positives must match groups/features and contain at least one sample")
    if positives.device != gen.device:
        raise ValueError("all Drift samples must be on the same device")
    if not gen.is_floating_point() or not positives.is_floating_point():
        raise ValueError("Drift samples must be floating point tensors")
    if negatives is None:
        negatives = gen.new_empty(batch, 0, features)
    if (
        negatives.ndim != 3
        or negatives.shape[0] != batch
        or negatives.shape[2] != features
        or negatives.device != gen.device
        or not negatives.is_floating_point()
    ):
        raise ValueError("negatives must match groups/features, floating dtype and device")
    weights = torch.as_tensor(negative_weight, dtype=torch.float32, device=gen.device)
    if weights.ndim == 0:
        weights = weights.expand(batch)
    if weights.shape != (batch,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("negative_weight must be finite, nonnegative and scalar or [groups]")

    # Float32 fields avoid half precision overflow in distances and softmax.
    # Keeping this block outside autograd is essential: without stopgrad, a
    # normalized field can have constant loss and almost no learning signal.
    with torch.no_grad(), torch.autocast(device_type=gen.device.type, enabled=False):
        old_gen = gen.detach().float()
        targets = torch.cat((old_gen, negatives.detach().float(), positives.detach().float()), dim=1)
        negative_count = negatives.shape[1]
        target_weights = torch.cat(
            (
                torch.ones(batch, generated_count, device=gen.device),
                weights.detach()[:, None].expand(batch, negative_count),
                torch.ones(batch, positives.shape[1], device=gen.device),
            ),
            dim=1,
        )
        distances = torch.cdist(old_gen, targets).clamp_min(1e-4)
        scale = (distances * target_weights[:, None, :]).mean() / target_weights.mean()
        coordinate_scale = (scale / math.sqrt(features)).clamp_min(1e-3)
        old_gen_scaled = old_gen / coordinate_scale
        targets_scaled = targets / coordinate_scale
        normalized_distances = distances / scale.clamp_min(1e-3)
        self_mask = torch.zeros(
            generated_count, targets.shape[1], dtype=torch.bool, device=gen.device
        )
        self_mask[:, :generated_count] = torch.eye(generated_count, dtype=torch.bool, device=gen.device)
        negative_end = generated_count + negative_count
        total_force = torch.zeros_like(old_gen_scaled)
        for temperature in radii:
            affinity = _normalized_affinity(normalized_distances, temperature, self_mask)
            affinity = affinity * target_weights[:, None, :]
            negative_affinity = affinity[:, :, :negative_end]
            positive_affinity = affinity[:, :, negative_end:]
            coefficients = torch.cat(
                (
                    -negative_affinity * positive_affinity.sum(dim=-1, keepdim=True),
                    positive_affinity * negative_affinity.sum(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            force = coefficients @ targets_scaled
            force = force - coefficients.sum(dim=-1, keepdim=True) * old_gen_scaled
            force_rms = force.square().mean().clamp_min(1e-8).sqrt()
            total_force = total_force + force / force_rms
        goal = old_gen_scaled + total_force

    return (gen.float() / coordinate_scale - goal).square().mean()


class DriftAlgorithm:
    """Class-conditional Drift objective and amortized 1-NFE CFG sampling.

    The model contract is ``model(noise, labels, cfg_scale) -> images``. During
    training, real images are grouped by their labels. Each label gets a fixed
    number of generated samples, its own positive samples, and the complete
    current batch as unconditional negative samples. Group losses are weighted
    by their positive counts, so the returned scalar follows the real batch's
    class frequencies.
    """

    def __init__(
        self,
        temperatures: Sequence[float] = (0.02, 0.05, 0.2),
        generated_per_class: int = 8,
        cfg_min: float = 1.0,
        cfg_max: float = 4.0,
        cfg_scale: float = 2.0,
    ) -> None:
        self.temperatures = _temperatures(temperatures)
        if isinstance(generated_per_class, bool) or int(generated_per_class) != generated_per_class or generated_per_class < 2:
            raise ValueError("generated_per_class must be an integer of at least 2")
        self.generated_per_class = int(generated_per_class)
        self.cfg_min = float(cfg_min)
        self.cfg_max = float(cfg_max)
        self.cfg_scale = float(cfg_scale)
        if not all(math.isfinite(value) for value in (self.cfg_min, self.cfg_max, self.cfg_scale)):
            raise ValueError("CFG values must be finite")
        if self.cfg_min < 1 or self.cfg_max < self.cfg_min or self.cfg_scale < 1:
            raise ValueError("CFG requires 1 <= cfg_min <= cfg_max and cfg_scale >= 1")

    def loss(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        *,
        generator: torch.Generator | None = None,
        training: bool = True,
    ) -> Tensor:
        """Evaluate the same objective for train/test; never change model mode.

        ``training`` is accepted for interchangeable algorithm implementations.
        Both phases sample the same CFG distribution. Supply an independently
        seeded device-matched generator for reproducible held-out evaluation.
        """
        del training
        if images.ndim < 2 or images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError("images must be a nonempty floating point batch")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0] or labels.device != images.device:
            raise ValueError("labels must be a device-matched vector with one label per image")
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("labels must contain integer class IDs")
        if (labels < 0).any():
            raise ValueError("labels must be nonnegative class IDs")
        classes, counts = labels.unique(sorted=True, return_counts=True)
        cfg_fraction = torch.rand(classes.numel(), device=images.device, generator=generator)
        cfg = (math.log(self.cfg_min) + cfg_fraction * math.log(self.cfg_max / self.cfg_min)).exp()
        generated_labels = classes.repeat_interleave(self.generated_per_class)
        generated_cfg = cfg.repeat_interleave(self.generated_per_class)
        noise = torch.randn(
            (generated_labels.numel(), *images.shape[1:]),
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        )
        generated = model(noise, generated_labels, generated_cfg)
        if generated.shape != noise.shape:
            raise ValueError("model output must have the same image shape as its noise input")
        generated = generated.reshape(classes.numel(), self.generated_per_class, -1)
        unconditional = images.flatten(start_dim=1).unsqueeze(0)
        losses = []
        for index, label in enumerate(classes):
            positive_images = images[labels == label].flatten(start_dim=1).unsqueeze(0)
            weight = (cfg[index] - 1) * (self.generated_per_class - 1) / images.shape[0]
            group_loss = drift_loss(
                generated[index : index + 1],
                positive_images,
                unconditional,
                negative_weight=weight,
                temperatures=self.temperatures,
            )
            losses.append(group_loss * (counts[index] / images.shape[0]))
        return torch.stack(losses).sum()

    def sample(
        self,
        model: nn.Module,
        noise: Tensor,
        labels: Tensor,
        *,
        cfg_scale: float | Tensor | None = None,
    ) -> Tensor:
        """Generate directly in one model call, restoring the caller's mode."""
        scale = self.cfg_scale if cfg_scale is None else cfg_scale
        scale_tensor = torch.as_tensor(scale)
        if not torch.isfinite(scale_tensor).all() or (scale_tensor < 1).any():
            raise ValueError("cfg_scale must be finite and at least 1")
        prior_modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                return model(noise, labels, scale)
        finally:
            for module, was_training in prior_modes:
                module.training = was_training
