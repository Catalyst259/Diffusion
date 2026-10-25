"""Minimal contracts consumed by the trainer, independent of DiT or Drift."""
from typing import Protocol
import torch
from torch import nn


class TrainingAlgorithm(Protocol):
    def loss(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
             *, generator: torch.Generator, training: bool = True) -> torch.Tensor: ...


class Sampler(Protocol):
    def sample(self, model: nn.Module, noise: torch.Tensor, labels: torch.Tensor,
               *, cfg_scale: float | None = None) -> torch.Tensor: ...
