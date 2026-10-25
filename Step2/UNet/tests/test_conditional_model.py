"""CPU checks for class conditioning and legacy checkpoint compatibility."""

import unittest

import torch

from UNet.src.model import UNet


class ConditionalUNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, num_classes=None, class_dropout_prob=0.1):
        return UNet(1, 1, time_emb_dim=16, base_channels=8,
                    num_classes=num_classes, class_dropout_prob=class_dropout_prob)

    def test_legacy_state_keys_and_two_argument_forward_remain_compatible(self):
        original = self.model()
        restored = self.model()
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(any("label_embed" in key for key in original.state_dict()))
        images, timesteps = torch.randn(2, 1, 28, 28), torch.tensor([1, 999])
        torch.testing.assert_close(original(images, timesteps), restored(images, timesteps))
        self.assertEqual(original(images, timesteps).shape, images.shape)

    def test_zero_class_embedding_reproduces_legacy_network_and_only_adds_embedding(self):
        legacy, conditioned = self.model(), self.model(num_classes=10, class_dropout_prob=0)
        state = legacy.state_dict()
        self.assertEqual(set(conditioned.state_dict()) - set(state), {"label_embed.weight"})
        state["label_embed.weight"] = torch.zeros(10, 16)
        conditioned.load_state_dict(state, strict=True)
        images, timesteps = torch.randn(2, 1, 28, 28), torch.tensor([2, 500])
        torch.testing.assert_close(legacy(images, timesteps), conditioned(images, timesteps, torch.tensor([0, 9])))

    def test_label_changes_prediction_and_receives_gradients(self):
        model = self.model(num_classes=10).eval()
        image, timestep = torch.randn(1, 1, 28, 28), torch.tensor([250])
        first = model(image, timestep, torch.tensor([1]))
        second = model(image, timestep, torch.tensor([8]))
        self.assertFalse(torch.allclose(first, second))
        first.square().mean().backward()
        self.assertGreater(model.label_embed.weight.grad[1].abs().sum().item(), 0)
        self.assertEqual(model.label_embed.weight.grad[8].abs().sum().item(), 0)

    def test_time_and_class_embeddings_are_summed_before_residual_injection(self):
        model = self.model(num_classes=10).eval()
        images, timesteps, labels = torch.randn(2, 1, 28, 28), torch.tensor([1, 700]), torch.tensor([3, 4])
        injected = []
        hook = model.encoder1.register_forward_pre_hook(lambda module, args: injected.append(args[1].detach()))
        try:
            model(images, timesteps, labels)
        finally:
            hook.remove()
        torch.testing.assert_close(injected[0], model.time_mlp(timesteps) + model.label_embed(labels))

    def test_conditional_labels_are_required_and_validated(self):
        model = self.model(num_classes=10)
        image, timestep = torch.randn(1, 1, 28, 28), torch.tensor([1])
        for labels in (None, torch.tensor([1.0]), torch.tensor([1, 2])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                model(image, timestep, labels)
        with self.assertRaises(ValueError):
            self.model()(image, timestep, torch.tensor([1]))

    def test_default_dropout_uses_independent_examples_and_preserves_labels(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = self.model(num_classes=10).train()
            self.assertEqual(model.class_dropout_prob, 0.1)
            self.assertEqual(model.null_label, 10)
            self.assertEqual(model.label_embed.weight.shape, (11, 16))
            labels = torch.arange(4096) % 10
            original = labels.clone()
            embeddings = model.label_embed(labels)
            dropped = (embeddings == model.label_embed.weight[10]).all(dim=1)
            # A whole-batch dropout or a shared mask cannot pass this check.
            self.assertGreater(dropped.float().mean().item(), 0.07)
            self.assertLess(dropped.float().mean().item(), 0.13)
            torch.testing.assert_close(labels, original)
            torch.testing.assert_close(embeddings[~dropped], model.label_embed.weight[labels[~dropped]])

    def test_all_dropped_labels_train_only_the_null_embedding(self):
        model = self.model(num_classes=10, class_dropout_prob=1).train()
        images = torch.randn(1, 1, 28, 28).repeat(2, 1, 1, 1)
        timesteps = torch.tensor([250, 250])
        labels = torch.tensor([1, 8])
        prediction = model(images, timesteps, labels)
        torch.testing.assert_close(prediction[0], prediction[1])
        prediction.square().mean().backward()
        gradient = model.label_embed.weight.grad
        self.assertGreater(gradient[10].abs().sum().item(), 0)
        self.assertEqual(gradient[:10].abs().sum().item(), 0)
        torch.testing.assert_close(labels, torch.tensor([1, 8]))

    def test_eval_keeps_labels_and_does_not_consume_rng(self):
        model = self.model(num_classes=10, class_dropout_prob=1).eval()
        image, timestep, label = torch.randn(1, 1, 28, 28), torch.tensor([250]), torch.tensor([3])
        before = torch.get_rng_state().clone()
        actual = model(image, timestep, label)
        expected = model(image, timestep, label, force_drop_ids=torch.tensor([0]))
        null_prediction = model(image, timestep, torch.tensor([10]))
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, null_prediction))
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_forced_mask_overrides_random_dropout_in_both_modes(self):
        model = self.model(num_classes=10, class_dropout_prob=1)
        images, timesteps = torch.randn(2, 1, 28, 28), torch.tensor([1, 700])
        labels, mask = torch.tensor([3, 4]), torch.tensor([False, True])
        expected = model.eval()(images, timesteps, torch.tensor([3, 10]))
        before = torch.get_rng_state().clone()
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                actual = model(images, timesteps, labels, force_drop_ids=mask)
                torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(labels, torch.tensor([3, 4]))
        torch.testing.assert_close(mask, torch.tensor([False, True]))
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_zero_dropout_preserves_old_conditional_checkpoints_and_rng(self):
        original = self.model(num_classes=10, class_dropout_prob=0)
        restored = self.model(num_classes=10, class_dropout_prob=0)
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(original.label_embed.weight.shape, (10, 16))
        self.assertIsNone(original.null_label)
        images, timesteps, labels = torch.randn(2, 1, 28, 28), torch.tensor([2, 500]), torch.tensor([0, 9])
        before = torch.get_rng_state().clone()
        torch.testing.assert_close(original(images, timesteps, labels), restored(images, timesteps, labels))
        torch.testing.assert_close(torch.get_rng_state(), before)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.model(num_classes=10).load_state_dict(original.state_dict(), strict=True)
        with self.assertRaisesRegex(ValueError, "null embedding"):
            original(images, timesteps, labels, force_drop_ids=torch.ones(2))

    def test_dropout_probability_is_validated(self):
        for probability in (-0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.model(num_classes=10, class_dropout_prob=probability)

    def test_forced_masks_are_validated(self):
        model = self.model(num_classes=10)
        image, timestep, label = torch.randn(1, 1, 28, 28), torch.tensor([1]), torch.tensor([3])
        for mask in ([1], torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([0.5])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                model(image, timestep, label, force_drop_ids=mask)
        with self.assertRaisesRegex(ValueError, "class-conditional"):
            self.model()(image, timestep, force_drop_ids=torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
