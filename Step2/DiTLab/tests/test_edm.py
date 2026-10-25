"""CPU mathematical and lifecycle checks for the denoising comparison."""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from DiTLab.algorithms.edm import EDMAlgorithm
from DiTLab.artifacts import load_checkpoint
from DiTLab.config import ComponentConfig, ExperimentConfig, TrainConfig
from DiTLab.data import DataBundle, DataSpec
from DiTLab.engine import Trainer
from DiTLab.models import DiT


class RecordingResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(0.2)
        self.calls = []

    def forward_denoise(self, images, labels, noise_condition, label_mask=None):
        self.calls.append((images.detach().clone(), labels.detach().clone(),
                           noise_condition.detach().clone(),
                           None if label_mask is None else label_mask.clone(),
                           self.training, torch.is_grad_enabled()))
        return torch.ones_like(images) * self.value


class KnownCleanImage(nn.Module):
    """Exact EDM residual for a point mass at the label (or -0.3 if null)."""

    def forward_denoise(self, images, labels, noise_condition, label_mask=None):
        sigma = (4 * noise_condition).exp().reshape(-1, 1, 1, 1)
        keep = torch.ones_like(labels, dtype=torch.bool) if label_mask is None else label_mask
        clean = torch.where(keep, labels.float(), -0.3).reshape(-1, 1, 1, 1)
        # Invert D(x,sigma) = c_skip*x + c_out*F(c_in*x,c_noise).
        sigma_data = 0.5
        norm = (sigma.square() + sigma_data**2).sqrt()
        return (clean * norm - sigma_data**2 * images) / (sigma * sigma_data)


class TinyDenoiser(nn.Module):
    """Both explicit and global RNG must survive interrupted training."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.2))
        self.bias = nn.Parameter(torch.linspace(-0.1, 0.1, 10))
        self.dropout = nn.Dropout(0.25)

    def forward_denoise(self, images, labels, noise_condition, label_mask=None):
        keep = torch.ones_like(labels, dtype=torch.bool) if label_mask is None else label_mask
        bias = (self.bias[labels] * keep + 0.02 * noise_condition).reshape(-1, 1, 1, 1)
        return self.gain * self.dropout(images) + bias + 0.01 * torch.randn_like(images)


class EDMTests(unittest.TestCase):
    def test_loss_matches_weighted_denoised_mse_and_noise_preconditioning(self):
        model = RecordingResidual()
        algorithm = EDMAlgorithm(p_mean=-0.5, p_std=0.7, label_dropout=0.0)
        images = torch.linspace(-1, 1, 5 * 16).reshape(5, 1, 4, 4)
        labels = torch.tensor([0, 3, 4, 7, 9])
        rng = torch.Generator().manual_seed(91)
        sigma = (-0.5 + 0.7 * torch.randn(5, generator=rng)).exp().reshape(-1, 1, 1, 1)
        noisy = images + sigma * torch.randn(images.shape, generator=rng)

        loss = algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(91))
        # Independently use the weighted denoised-image objective from EDM,
        # rather than reproducing its numerically stable residual target.
        sd = 0.5
        variance = sigma.square() + sd**2
        estimate = sd**2 / variance * noisy + sigma * sd / variance.sqrt() * model.value
        weight = variance / (sigma * sd).square()
        expected = (weight * (estimate - images).square()).mean()
        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
        recorded_input, recorded_labels, condition, _, _, _ = model.calls[0]
        torch.testing.assert_close(recorded_input, noisy / variance.sqrt())
        torch.testing.assert_close(condition, sigma.flatten().log() / 4)
        torch.testing.assert_close(recorded_labels, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(model.value.grad))
        self.assertGreater(model.value.grad.abs().item(), 0.0)

    def test_label_dropout_applies_only_to_training_and_keeps_noise_condition(self):
        model = RecordingResidual()
        algorithm = EDMAlgorithm(label_dropout=1.0)
        images, labels = torch.zeros(3, 1, 4, 4), torch.tensor([0, 4, 9])
        first = algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(6))
        repeated = algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(6))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(model.calls[0][3].any())
        algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(6), training=False)
        evaluation_mask = model.calls[-1][3]
        self.assertTrue(evaluation_mask is None or evaluation_mask.all())
        torch.testing.assert_close(model.calls[0][2], model.calls[-1][2], rtol=0, atol=0)
        self.assertGreater(model.calls[0][2].abs().sum().item(), 0.0)

    def test_cfg_blends_conditional_and_null_predictions(self):
        model = KnownCleanImage()
        algorithm = EDMAlgorithm(num_steps=4, sigma_min=0.1, sigma_max=2.0)
        noise = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(17))
        labels = torch.tensor([1, 3])
        for cfg in (0.0, 1.0, 2.0):
            with self.subTest(cfg=cfg):
                actual = algorithm.sample(model, noise, labels, cfg_scale=cfg)
                expected = (-0.3 + cfg * (labels.float() + 0.3)).reshape(-1, 1, 1, 1)
                torch.testing.assert_close(actual, expected.expand_as(actual), rtol=2e-5, atol=2e-5)

    def test_heun_integrates_linear_time_field_and_uses_final_euler_step(self):
        algorithm = EDMAlgorithm(num_steps=3, sigma_min=0.5, sigma_max=2.0, rho=1.0, cfg_scale=1.0)
        model = RecordingResidual()
        noise = torch.zeros(2, 1, 4, 4)
        labels = torch.tensor([1, 7])

        def denoiser(_model, noisy, _labels, sigma, label_mask=None):
            del label_mask
            sigma = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
            if sigma.ndim == 1:
                sigma = sigma.reshape(-1, 1, 1, 1)
            # D=x-sigma^2 gives dx/dsigma=sigma. Heun is exact on every
            # nonzero interval; the last Euler interval contributes -min^2.
            return noisy - sigma.square()

        with patch.object(algorithm, "_denoise", side_effect=denoiser) as calls:
            actual = algorithm.sample(model, noise, labels)
        expected = torch.full_like(noise, -(2.0**2 + 0.5**2) / 2)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(calls.call_count, 2 * algorithm.num_steps - 1)

    def test_sampling_is_deterministic_no_grad_and_restores_mixed_modes(self):
        model = RecordingResidual().train()
        model.dropout.eval()
        algorithm = EDMAlgorithm(num_steps=3, sigma_min=0.1, sigma_max=2.0, cfg_scale=1.0)
        noise = torch.randn(2, 1, 4, 4, requires_grad=True)
        labels = torch.tensor([1, 7])
        before = torch.get_rng_state().clone()
        actual = algorithm.sample(model, noise, labels)
        self.assertEqual(len(model.calls), 5)
        self.assertTrue(all(not training and not grad for *_, training, grad in model.calls))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, algorithm.sample(model, noise, labels), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        with patch.object(model, "forward_denoise", side_effect=RuntimeError("sample failure")):
            with self.assertRaisesRegex(RuntimeError, "sample failure"):
                algorithm.sample(model, noise, labels)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    def test_invalid_schedule_and_loss_settings_are_rejected(self):
        for options in ({"num_steps": 1}, {"num_steps": 2.5}, {"sigma_data": 0},
                        {"sigma_min": 0}, {"sigma_min": 2, "sigma_max": 1},
                        {"rho": 0}, {"p_std": -1}, {"label_dropout": 1.1},
                        {"cfg_scale": -1}, {"p_mean": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EDMAlgorithm(**options)


class DiTDenoisingTests(unittest.TestCase):
    def test_null_class_drops_label_information_but_keeps_signed_noise_condition(self):
        torch.manual_seed(32)
        model = DiT(image_size=4, patch_size=2, hidden_size=16, depth=1, num_heads=2)
        nn.init.normal_(model.final_modulation[-1].weight, std=0.2)
        images = torch.randn(2, 1, 4, 4)
        labels = torch.tensor([0, 9])
        mask = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            null = model.forward_denoise(images, labels, -1.0, mask)
            other_labels = model.forward_denoise(images, labels.flip(0), -1.0, mask)
            other_noise = model.forward_denoise(images, labels, 1.0, mask)
            conditional = model.forward_denoise(images, labels, -1.0)
        torch.testing.assert_close(null, other_labels, rtol=0, atol=0)
        self.assertGreater((null - other_noise).abs().max().item(), 1e-7)
        self.assertGreater((null - conditional).abs().max().item(), 1e-7)
        model.zero_grad()
        model.forward_denoise(images, labels, -1.0, mask).square().mean().backward()
        torch.testing.assert_close(model.label_embed.weight.grad, torch.zeros_like(model.label_embed.weight))
        self.assertGreater(model.guidance_embed.mlp[0].weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_denoising_reuses_original_dit_parameters_and_drift_forward(self):
        model = DiT(image_size=4, patch_size=2, hidden_size=16, depth=1, num_heads=2)
        # Snapshot of the original Drift architecture's parameter count and
        # state schema: adding a learned null class would break this contract.
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10084)
        original_keys = {
            "position_embedding", "patch_embed.weight", "patch_embed.bias", "label_embed.weight",
            "guidance_embed.mlp.0.weight", "guidance_embed.mlp.0.bias",
            "guidance_embed.mlp.2.weight", "guidance_embed.mlp.2.bias",
            "blocks.0.attention.in_proj_weight", "blocks.0.attention.in_proj_bias",
            "blocks.0.attention.out_proj.weight", "blocks.0.attention.out_proj.bias",
            "blocks.0.mlp.0.weight", "blocks.0.mlp.0.bias", "blocks.0.mlp.2.weight", "blocks.0.mlp.2.bias",
            "blocks.0.modulation.1.weight", "blocks.0.modulation.1.bias",
            "final_modulation.1.weight", "final_modulation.1.bias",
            "final_projection.weight", "final_projection.bias",
        }
        state = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertEqual(set(state), original_keys)
        images, labels = torch.randn(2, 1, 4, 4), torch.tensor([0, 8])
        torch.testing.assert_close(model(images, labels, 1.0), model.forward_denoise(images, labels, 1.0))
        model.forward_denoise(images, labels, -1.0, torch.zeros(2, dtype=torch.bool))
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


class EDMTrainingLifecycleTests(unittest.TestCase):
    def test_mid_epoch_checkpoint_resume_matches_complete_run_exactly(self):
        data = DataBundle(
            TensorDataset(torch.linspace(-0.9, 0.9, 9 * 16).reshape(9, 1, 4, 4), torch.arange(9)),
            TensorDataset(torch.linspace(0.9, -0.9, 10 * 16).reshape(10, 1, 4, 4), torch.arange(10)),
            DataSpec(image_size=4, in_channels=1, num_classes=10),
        )

        def trainer(path, checkpoint=None):
            config = ExperimentConfig()
            options = {"num_steps": 3, "sigma_min": 0.1, "sigma_max": 2.0, "label_dropout": 0.3}
            config.model = ComponentConfig("tiny-denoiser")
            config.algorithm = ComponentConfig("edm", options)
            config.sampler = ComponentConfig("edm", options)
            config.training = TrainConfig(
                epochs=2, batch_size=4, preview_every=2, save_every=2, preview_count=10,
                seed=23, device="cpu", log_every=1000, output_dir=str(path),
            )
            return Trainer(config, data, TinyDenoiser(), EDMAlgorithm(**options),
                           EDMAlgorithm(**options), checkpoint=checkpoint)

        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            torch.manual_seed(991)
            complete = trainer(root / "complete")
            expected_history = complete.run()
            expected_global_rng = torch.get_rng_state().clone()
            expected_algorithm_rng = complete.generator.get_state().clone()
            torch.manual_seed(991)
            interrupted = trainer(root / "resumed")
            self.assertEqual(interrupted.run(max_steps=2), [])
            saved = load_checkpoint(interrupted.checkpoint_path)
            self.assertEqual(saved["progress"]["batch_in_epoch"], 2)
            torch.randn(117)
            resumed = trainer(root / "resumed", checkpoint=saved)
            self.assertEqual(resumed.run(), expected_history)
            for key, value in complete.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(resumed.generator.get_state(), expected_algorithm_rng, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_global_rng, rtol=0, atol=0)
            self.assertEqual(resumed.global_step, 6)


if __name__ == "__main__":
    unittest.main()
