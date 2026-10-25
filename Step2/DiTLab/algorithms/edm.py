"""Pixel-space EDM denoising and deterministic, second-order CFG sampling.

Implements the EDM preconditioning, log-normal training distribution, and
Karras/Heun sampler from Karras et al., "Elucidating the Design Space of
Diffusion-Based Generative Models" (2022), https://arxiv.org/abs/2206.00364.
The network uses the same DiT parameters as Drift through ``forward_denoise``.
No autoencoder, latent representation, stochastic sampler churn, or EMA is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


class EDMAlgorithm:
    """EDM weighted denoising loss and explicit classifier-free guidance.

    ``num_steps`` counts the nonzero noise levels, including ``sigma_max``
    and ``sigma_min``; a final Euler step reaches zero. Sampling therefore
    makes ``2 * num_steps - 1`` denoiser calls. CFG scales other than one use
    two network branches in each call (batched unconditional/conditional
    inputs), doubling the number of per-image network evaluations.

    The network contract is ``model.forward_denoise(images, labels,
    noise_condition, label_mask=None)``. ``noise_condition`` is log(sigma)/4
    times ``noise_condition_scale`` (one by default, retained for ablations);
    a false boolean label mask selects the zero class embedding.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        label_dropout: float = 0.1,
        num_steps: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        cfg_scale: float = 2.0,
        noise_condition_scale: float = 1.0,
    ) -> None:
        self.sigma_data = _finite_float("sigma_data", sigma_data)
        self.p_mean = _finite_float("p_mean", p_mean)
        self.p_std = _finite_float("p_std", p_std)
        self.label_dropout = _finite_float("label_dropout", label_dropout)
        self.sigma_min = _finite_float("sigma_min", sigma_min)
        self.sigma_max = _finite_float("sigma_max", sigma_max)
        self.rho = _finite_float("rho", rho)
        self.cfg_scale = _finite_float("cfg_scale", cfg_scale)
        self.noise_condition_scale = _finite_float("noise_condition_scale", noise_condition_scale)
        if self.noise_condition_scale <= 0:
            raise ValueError("noise_condition_scale must be positive")
        if self.sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        if self.p_std < 0:
            raise ValueError("p_std must be nonnegative")
        if not 0 <= self.label_dropout <= 1:
            raise ValueError("label_dropout must be in [0, 1]")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 2:
            raise ValueError("num_steps must be an integer of at least 2")
        self.num_steps = num_steps
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("noise levels must satisfy 0 < sigma_min < sigma_max")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.cfg_scale < 0:
            raise ValueError("cfg_scale must be nonnegative")
        # Training and network inputs use float32, even with a half-precision
        # autocast network. Reject scales that cannot be represented there.
        float_info = torch.finfo(torch.float32)
        if not float_info.tiny <= self.sigma_data <= math.sqrt(float_info.max):
            raise ValueError("sigma_data must be representable for float32 preconditioning")
        if max(abs(self.p_mean), self.p_std) > float_info.max / 16:
            raise ValueError("log-normal parameters must be representable in float32")
        if not math.isfinite((math.log(self.sigma_max) - math.log(self.sigma_min)) / self.rho):
            raise ValueError("rho and noise levels must permit a finite sampling schedule")

    @staticmethod
    def _validate_batch(images: Tensor, labels: Tensor) -> None:
        if images.ndim != 4 or images.shape[0] < 1 or any(size < 1 for size in images.shape[1:]):
            raise ValueError("images must have nonempty shape [B, C, H, W]")
        if not images.is_floating_point():
            raise ValueError("images must be floating point tensors")
        if labels.shape != (images.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be a torch.long vector with one label per image")
        if labels.device != images.device:
            raise ValueError("images and labels must be on the same device")

    @staticmethod
    def _model_dtype(model: nn.Module) -> torch.dtype:
        parameter = next(model.parameters(), None)
        return torch.float32 if parameter is None else parameter.dtype

    @staticmethod
    def _check_prediction(prediction: Tensor, images: Tensor) -> None:
        if prediction.shape != images.shape or not prediction.is_floating_point():
            raise ValueError("denoising model must return floating point images matching its input shape")
        if prediction.device != images.device:
            raise ValueError("denoising model output must remain on the input device")

    def loss(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        *,
        generator: torch.Generator,
        training: bool = True,
    ) -> Tensor:
        """Weighted EDM MSE, with independently sampled noise for each image.

        Algebraically this is ``mean(w * (D(y, sigma) - images)**2)``, where
        ``w = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2``.
        We compute the equivalent residual-network target directly, avoiding
        cancellation at tiny sigma and overflow from constructing huge noisy
        images. Label dropout applies only to training. All random draws use
        the supplied device-matched generator; model mode is unchanged.
        """
        self._validate_batch(images, labels)
        if not isinstance(generator, torch.Generator):
            raise ValueError("loss requires an explicit torch.Generator")
        batch = images.shape[0]
        log_sigma = self.p_mean + self.p_std * torch.randn(
            batch, device=images.device, dtype=torch.float32, generator=generator,
        )
        noise = torch.randn(
            images.shape, device=images.device, dtype=torch.float32, generator=generator,
        )
        label_mask = None
        if training and self.label_dropout > 0:
            label_mask = torch.rand(batch, device=images.device, generator=generator) >= self.label_dropout

        # a = sigma / hypot(sigma, sigma_data), b = sigma_data / hypot(...).
        # The log-space form preserves the full log-normal distribution without
        # clipping sigma; both coefficients remain bounded for its long tails.
        log_ratio = log_sigma - math.log(self.sigma_data)
        a = torch.sigmoid(2 * log_ratio).sqrt().reshape(batch, 1, 1, 1)
        b = torch.sigmoid(-2 * log_ratio).sqrt().reshape(batch, 1, 1, 1)
        clean = images.float()
        network_input = (b / self.sigma_data) * clean + a * noise
        target = (a / self.sigma_data) * clean - b * noise
        prediction = model.forward_denoise(
            network_input.to(dtype=self._model_dtype(model)), labels,
            log_sigma / 4 * self.noise_condition_scale,
            label_mask=label_mask,
        )
        self._check_prediction(prediction, images)
        return (prediction.float() - target).square().mean()

    def _denoise(
        self,
        model: nn.Module,
        noisy: Tensor,
        labels: Tensor,
        sigma: Tensor,
        label_mask: Tensor | None = None,
    ) -> Tensor:
        """EDM D(x, sigma) = c_skip*x + c_out*F(c_in*x, c_noise)."""
        sigma = torch.as_tensor(sigma, dtype=noisy.dtype, device=noisy.device)
        if sigma.ndim == 0:
            sigma = sigma.expand(noisy.shape[0])
        if sigma.shape != (noisy.shape[0],):
            raise ValueError("sigma must be a scalar or vector with one value per image")
        magnitude = torch.hypot(sigma, sigma.new_tensor(self.sigma_data))
        shape = (noisy.shape[0], 1, 1, 1)
        c_in = magnitude.reciprocal().reshape(shape)
        c_skip = (self.sigma_data / magnitude).square().reshape(shape)
        c_out = (self.sigma_data * (sigma / magnitude)).reshape(shape)
        prediction = model.forward_denoise(
            (c_in * noisy).to(dtype=self._model_dtype(model)), labels,
            (sigma.log() / 4 * self.noise_condition_scale).float(), label_mask=label_mask,
        )
        self._check_prediction(prediction, noisy)
        return c_skip * noisy + c_out * prediction.to(dtype=noisy.dtype)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: Tensor,
        labels: Tensor,
        *,
        cfg_scale: float | None = None,
    ) -> Tensor:
        """Integrate from sigma_max*noise to zero; never clip intermediate images.

        The supplied noise is the only random input. Float64 integration state
        limits accumulation/cancellation error; network calls retain the model's
        parameter dtype. The returned images use the input noise dtype.
        """
        self._validate_batch(noise, labels)
        scale = self.cfg_scale if cfg_scale is None else _finite_float("cfg_scale", cfg_scale)
        if scale < 0:
            raise ValueError("cfg_scale must be nonnegative")
        previous_modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            ramp = torch.linspace(0, 1, self.num_steps, dtype=torch.float64, device=noise.device)
            # Normalize root-space levels by sigma_max to avoid exponentiating
            # a large positive number for otherwise valid small rho values.
            relative_delta = math.expm1((math.log(self.sigma_min) - math.log(self.sigma_max)) / self.rho)
            sigmas = self.sigma_max * torch.exp(self.rho * torch.log1p(ramp * relative_delta))
            sigmas[0], sigmas[-1] = self.sigma_max, self.sigma_min
            sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
            state = noise.to(dtype=torch.float64) * sigmas[0]
            batch = noise.shape[0]
            paired_labels = torch.cat((labels, labels)) if scale != 1 else labels
            paired_mask = torch.cat((
                torch.zeros(batch, device=noise.device, dtype=torch.bool),
                torch.ones(batch, device=noise.device, dtype=torch.bool),
            )) if scale != 1 else None

            def denoise(current: Tensor, sigma: Tensor) -> Tensor:
                if scale == 1:
                    return self._denoise(model, current, labels, sigma)
                unconditional, conditional = self._denoise(
                    model, torch.cat((current, current)), paired_labels, sigma,
                    label_mask=paired_mask,
                ).chunk(2)
                return unconditional + scale * (conditional - unconditional)

            for index in range(self.num_steps):
                sigma, next_sigma = sigmas[index], sigmas[index + 1]
                derivative = (state - denoise(state, sigma)) / sigma
                step_size = next_sigma - sigma
                proposal = state + step_size * derivative
                if index < self.num_steps - 1:
                    next_derivative = (proposal - denoise(proposal, next_sigma)) / next_sigma
                    state = state + step_size * (derivative + next_derivative) / 2
                else:
                    state = proposal
            result = state.to(dtype=noise.dtype)
            if not torch.isfinite(result).all():
                raise FloatingPointError("EDM sampler produced non-finite images")
            return result
        finally:
            for module, was_training in previous_modes:
                module.training = was_training
