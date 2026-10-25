"""CPU-only architecture checks; no dataset download is needed."""

import unittest

import torch

from DiTLab.models import DiT


class DiTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = DiT(hidden_size=32, depth=2, num_heads=4)
        self.noise = torch.randn(3, 1, 28, 28)
        self.labels = torch.tensor([0, 4, 9])

    def test_image_shape_finite_outputs_and_initial_noise_diversity(self):
        output = self.model(self.noise, self.labels)
        self.assertEqual(output.shape, self.noise.shape)
        self.assertTrue(torch.isfinite(output).all())
        same_label_output = self.model(self.noise, torch.zeros(3, dtype=torch.long))
        self.assertGreater((same_label_output[0] - same_label_output[1]).abs().mean().item(), 1e-4)

    def test_optimization_reaches_noise_and_condition_paths(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        target = torch.randn_like(self.noise)
        for _ in range(2):
            optimizer.zero_grad()
            output = self.model(self.noise, self.labels, torch.tensor([1.0, 2.0, 3.0]))
            (output - target).square().mean().backward()
            optimizer.step()
        for name in (
            "patch_embed.weight", "final_projection.weight",
            "label_embed.weight", "guidance_embed.mlp.0.weight",
            "blocks.0.attention.in_proj_weight",
        ):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)
        with torch.no_grad():
            base = self.model(self.noise, self.labels, 1.0)
            guided = self.model(self.noise, self.labels, 3.0)
            other_labels = self.model(self.noise, (self.labels + 1) % 10, 1.0)
        self.assertGreater((base - guided).abs().max().item(), 1e-8)
        self.assertGreater((base - other_labels).abs().max().item(), 1e-8)

    def test_scalar_and_batch_guidance_are_equivalent(self):
        # Exercise active conditioning instead of relying on zero initial modulation.
        torch.nn.init.normal_(self.model.final_modulation[-1].weight, std=0.02)
        with torch.no_grad():
            scalar = self.model(self.noise, self.labels, 2.5)
            tensor_scalar = self.model(self.noise, self.labels, torch.tensor(2.5))
            batched = self.model(self.noise, self.labels, torch.full((3,), 2.5))
        torch.testing.assert_close(scalar, tensor_scalar)
        torch.testing.assert_close(scalar, batched)

    def test_unpatchify_preserves_pixel_and_channel_order(self):
        model = DiT(image_size=4, patch_size=2, in_channels=2, hidden_size=16, depth=1, num_heads=2)
        expected = torch.arange(32).reshape(1, 2, 4, 4)
        patches = expected.reshape(1, 2, 2, 2, 2, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 4, 8)
        torch.testing.assert_close(model.unpatchify(patches), expected)

    def test_rejects_invalid_dimensions_and_conditions(self):
        for options in (
            {"patch_size": 3}, {"hidden_size": 30}, {"depth": 0},
            {"num_heads": 5}, {"mlp_ratio": float("nan")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DiT(**options)
        for labels, scale in (
            (torch.tensor([0, 1, 10]), 1.0),
            (self.labels.float(), 1.0),
            (self.labels, torch.ones(3, 1)),
            (self.labels, float("nan")),
            (self.labels, -1.0),
        ):
            with self.subTest(labels=labels, scale=scale), self.assertRaises(ValueError):
                self.model(self.noise, labels, scale)
        with self.assertRaises(ValueError):
            self.model(torch.randn(3, 1, 32, 32), self.labels)


if __name__ == "__main__":
    unittest.main()
