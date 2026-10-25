"""Small deterministic checks of the Drift learning/sampling contract."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from DiTLab.algorithms.drift import DriftAlgorithm, _normalized_affinity, drift_loss


class RecordingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(0.1)
        self.calls = []
        self.raise_error = False

    def forward(self, noise, labels, cfg_scale):
        self.calls.append((noise.detach().clone(), labels.detach().clone(), cfg_scale, self.training, torch.is_grad_enabled()))
        if self.raise_error:
            raise RuntimeError("intentional sample failure")
        return noise * self.gain


class DriftLossTests(unittest.TestCase):
    def test_detached_target_produces_learning_signal_and_detaches_real_data(self):
        generated = torch.zeros(1, 2, 3, requires_grad=True)
        positive = torch.ones(1, 3, 3, requires_grad=True)
        negative = torch.zeros(1, 2, 3, requires_grad=True)
        loss = drift_loss(generated, positive, negative, temperatures=(0.2,))
        self.assertAlmostEqual(loss.item(), 1.0, places=5)
        loss.backward()
        # The normalized scalar is one, but stopgrad still creates an update
        # towards the positive data. Differentiating the field itself loses it.
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertTrue((generated.grad < 0).all())
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)

    def test_collapsed_equal_samples_have_zero_finite_loss_and_gradient(self):
        generated = torch.zeros(2, 3, 4, requires_grad=True)
        real = torch.zeros(2, 4, 4)
        loss = drift_loss(generated, real, real, negative_weight=torch.tensor([0.0, 2.0]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertTrue(torch.equal(generated.grad, torch.zeros_like(generated)))

    def test_self_affinity_is_exactly_zero_after_numerical_floor(self):
        distances = torch.zeros(1, 3, 5)
        mask = torch.zeros(3, 5, dtype=torch.bool)
        mask[:, :3] = torch.eye(3, dtype=torch.bool)
        affinity = _normalized_affinity(distances, 0.2, mask)
        self.assertTrue(torch.equal(affinity[0][mask], torch.zeros(3)))
        self.assertTrue((affinity[0][~mask] > 0).all())
        self.assertTrue(torch.isfinite(affinity).all())

    def test_generated_permutation_does_not_change_loss(self):
        rng = torch.Generator().manual_seed(11)
        generated = torch.randn(2, 4, 7, generator=rng)
        real = torch.randn(2, 5, 7, generator=rng)
        baseline = drift_loss(generated, real)
        reordered = drift_loss(generated[:, [2, 0, 3, 1]], real[:, [4, 1, 0, 3, 2]])
        torch.testing.assert_close(baseline, reordered, atol=2e-6, rtol=2e-6)

    def test_invalid_group_and_negative_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            drift_loss(torch.zeros(1, 1, 3), torch.zeros(1, 2, 3))
        with self.assertRaises(ValueError):
            drift_loss(torch.zeros(1, 2, 3), torch.zeros(1, 0, 3))
        for weight in (-1.0, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                drift_loss(torch.zeros(1, 2, 3), torch.ones(1, 2, 3), negative_weight=weight)


class DriftAlgorithmTests(unittest.TestCase):
    def test_loss_generates_all_classes_once_with_matching_cfg_and_reproducible_rng(self):
        model = RecordingGenerator().eval()
        algorithm = DriftAlgorithm(generated_per_class=3)
        images = torch.randn(5, 1, 4, 4, generator=torch.Generator().manual_seed(4))
        labels = torch.tensor([7, 2, 7, 2, 2])
        loss = algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(10), training=False)
        self.assertEqual(len(model.calls), 1)
        noise, conditioned_labels, cfg, mode, gradients = model.calls[0]
        self.assertEqual(tuple(noise.shape), (6, 1, 4, 4))
        self.assertEqual(conditioned_labels.tolist(), [2, 2, 2, 7, 7, 7])
        self.assertTrue(((cfg >= 1) & (cfg <= 4)).all())
        self.assertTrue(torch.equal(cfg[0].expand(3), cfg[:3]))
        self.assertFalse(mode)
        self.assertTrue(gradients)
        repeat = algorithm.loss(model, images, labels, generator=torch.Generator().manual_seed(10), training=False)
        torch.testing.assert_close(loss, repeat, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.gain.grad))

    def test_class_objectives_are_weighted_by_actual_positive_counts(self):
        model = RecordingGenerator()
        algorithm = DriftAlgorithm(generated_per_class=2, cfg_min=2, cfg_max=2)
        images = torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(4, 1, 1, 1)
        labels = torch.tensor([0, 0, 0, 1])
        recorded = []

        def fake_field(gen, positives, negatives, *, negative_weight, temperatures):
            recorded.append((positives.shape[1], negatives.shape[1], negative_weight.item()))
            return positives.mean() + gen.sum() * 0

        with patch("DiTLab.algorithms.drift.drift_loss", side_effect=fake_field):
            loss = algorithm.loss(model, images, labels)
        self.assertAlmostEqual(loss.item(), 0.25)
        self.assertEqual(recorded, [(3, 4, 0.25), (1, 4, 0.25)])

    def test_sampling_uses_one_forward_and_restores_mixed_module_modes(self):
        model = RecordingGenerator().train()
        model.dropout.eval()
        algorithm = DriftAlgorithm(cfg_scale=2.5)
        noise = torch.randn(3, 1, 4, 4, requires_grad=True)
        labels = torch.tensor([0, 4, 9])
        images = algorithm.sample(model, noise, labels)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][2], 2.5)
        self.assertFalse(model.calls[0][3])
        self.assertFalse(model.calls[0][4])
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertFalse(images.requires_grad)
        torch.testing.assert_close(images, noise.detach() * model.gain.detach())

    def test_failed_sampling_restores_model_mode(self):
        model = RecordingGenerator().train()
        model.raise_error = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            DriftAlgorithm().sample(model, torch.zeros(1, 1, 2, 2), torch.tensor([1]))
        self.assertTrue(model.training)

    def test_invalid_cfg_and_generated_counts_are_rejected(self):
        for options in ({"generated_per_class": 1}, {"cfg_min": 0}, {"cfg_max": 0}, {"cfg_scale": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DriftAlgorithm(**options)


if __name__ == "__main__":
    unittest.main()
