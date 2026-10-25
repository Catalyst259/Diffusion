"""Sampling regression tests; no MNIST download or trained checkpoint required."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sample import (  # noqa: E402
    NUM_TIMESTEPS,
    eps_sample,
    inference,
    q_posterior_mean,
    q_posterior_variance,
    q_sample,
    save_inference_gif,
)


class ConstantImageOracle(torch.nn.Module):
    """Ideal epsilon predictor for a known constant clean image."""

    def __init__(self, value=0.375, *, record_images=False):
        super().__init__()
        self.value = value
        # Independent float64 schedule avoids relying on the sampler's tables.
        beta = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float64)
        self.register_buffer("alpha_bar", torch.cat((torch.ones(1), (1 - beta).cumprod(0))))
        self.calls = []
        self.inputs_by_t = {} if record_images else None

    def forward(self, images, timesteps):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Inference must use eval mode with gradients disabled")
        self.calls.append(timesteps.detach().cpu().tolist())
        if self.inputs_by_t is not None:
            if len(timesteps) != 1:
                raise AssertionError("Image recording expects a single preview image")
            self.inputs_by_t[int(timesteps.item())] = images.detach().clone()
        a = self.alpha_bar[timesteps].to(images.dtype).view(-1, 1, 1, 1)
        return (images - a.sqrt() * self.value) / (1 - a).sqrt()


class ConditionalImageOracle(ConstantImageOracle):
    """The clean pixel value depends on each sample's digit label."""

    def __init__(self):
        super().__init__()
        self.label_calls = []

    def forward(self, images, timesteps, labels):
        self.label_calls.append(labels.detach().cpu().tolist())
        self.value = labels.to(images.dtype).view(-1, 1, 1, 1) / 10
        return super().forward(images, timesteps)


class NonOracle(torch.nn.Module):
    def forward(self, images, timesteps):
        return 0.05 * images + timesteps.view(-1, 1, 1, 1) / 10000


class CFGPredictor(torch.nn.Module):
    num_classes = 10
    null_label = 10

    def __init__(self):
        super().__init__()
        self.calls = []

    @staticmethod
    def prediction(images, timesteps, labels):
        class_offset = torch.where(labels == 10, -0.3, labels / 10)
        return (0.05 * images + timesteps.view(-1, 1, 1, 1) / 10000
                + class_offset.view(-1, 1, 1, 1))

    def forward(self, images, timesteps, labels):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("CFG must use eval mode with gradients disabled")
        self.calls.append((images.clone(), timesteps.clone(), labels.clone()))
        return self.prediction(images, timesteps, labels)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def check_forward_noise(self, device):
        x0 = torch.linspace(0, 1, 32, device=device).reshape(2, 1, 4, 4)
        noise = torch.randn_like(x0)
        t = torch.tensor([1, 1000], device=device)
        xt = q_sample(x0, t, noise=noise)
        recovered = eps_sample(x0, xt, t)
        torch.testing.assert_close(recovered, noise, rtol=2e-5, atol=1e-5)
        clean_t = torch.zeros(2, dtype=torch.long, device=device)
        torch.testing.assert_close(q_sample(x0, clean_t, noise=noise), x0, rtol=0, atol=0)
        self.assertEqual(xt.device, x0.device)

        # The final posterior is a deterministic clean-image prediction.
        last_t = torch.ones(2, dtype=torch.long, device=device)
        torch.testing.assert_close(q_posterior_mean(x0, xt, last_t), x0, rtol=0, atol=0)
        torch.testing.assert_close(q_posterior_variance(last_t), torch.zeros(2, 1, 1, 1, device=device))

    def test_forward_noise_and_clean_boundary_cpu(self):
        self.check_forward_noise(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_forward_noise_and_clean_boundary_cuda(self):
        self.check_forward_noise(torch.device("cuda"))

    def test_full_1000_step_trajectory_recovers_oracle_image(self):
        oracle = ConstantImageOracle()
        x0 = torch.full((1, 1, 4, 4), oracle.value)
        xt = q_sample(x0, torch.tensor([NUM_TIMESTEPS]))
        original_xt = xt.clone()
        seen_times = []

        def record(images, timesteps):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(images.requires_grad)
            seen_times.append(int(timesteps.item()))

        result = inference(xt.requires_grad_(), NUM_TIMESTEPS, oracle, on_step=record)
        torch.testing.assert_close(result, x0, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(xt.detach(), original_xt, rtol=0, atol=0)
        self.assertEqual(seen_times, list(range(NUM_TIMESTEPS, -1, -1)))
        self.assertEqual(oracle.calls, [[t] for t in range(NUM_TIMESTEPS, 0, -1)])
        self.assertFalse(result.requires_grad)
        self.assertTrue(oracle.training)

    def check_mixed_timesteps(self, device):
        oracle = ConstantImageOracle().to(device)
        oracle.eval()
        x0 = torch.full((3, 1, 4, 4), oracle.value, device=device)
        # A clean sample is already finished and must remain untouched.
        x0[0].fill_(0.75)
        t = torch.tensor([0, 1, 7], device=device)
        xt = q_sample(x0, t)
        original_xt = xt.clone()
        frames = []
        result = inference(xt, t, oracle, on_step=lambda images, times: frames.append((images.clone(), times.clone())))
        torch.testing.assert_close(result, x0, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(xt, original_xt, rtol=0, atol=0)
        torch.testing.assert_close(t, torch.tensor([0, 1, 7], device=device), rtol=0, atol=0)
        self.assertEqual(oracle.calls, [[1, 7], [6], [5], [4], [3], [2], [1]])
        self.assertEqual(len(frames), 8)
        for index, (images, times) in enumerate(frames):
            torch.testing.assert_close(images[0], original_xt[0], rtol=0, atol=0)
            self.assertEqual(times.tolist(), [0, max(1 - index, 0), max(7 - index, 0)])
        self.assertFalse(oracle.training)

    def test_mixed_timesteps_cpu(self):
        self.check_mixed_timesteps(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_mixed_timesteps_cuda(self):
        self.check_mixed_timesteps(torch.device("cuda"))

    def test_mode_restored_when_model_raises(self):
        class BrokenModel(torch.nn.Module):
            def forward(self, images, timesteps):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("Inference mode was not enabled")
                raise RuntimeError("prediction failed")

        for training in (True, False):
            with self.subTest(training=training):
                model = BrokenModel().train(training)
                with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                    inference(torch.zeros(1, 1, 4, 4, requires_grad=True), 1, model)
                self.assertEqual(model.training, training)
                self.assertTrue(torch.is_grad_enabled())

    def test_mode_restored_when_callback_raises(self):
        def broken_callback(images, timesteps):
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("frame failed")

        model = ConstantImageOracle()
        with self.assertRaisesRegex(RuntimeError, "frame failed"):
            inference(torch.zeros(1, 1, 4, 4), 1, model, on_step=broken_callback)
        self.assertTrue(model.training)
        self.assertEqual(model.calls, [])

    def test_invalid_timesteps_rejected(self):
        model = ConstantImageOracle()
        for t in (-1, 1001, [1, 2], 1.5, float("nan"), float("inf")):
            with self.subTest(t=t), self.assertRaises(ValueError):
                inference(torch.zeros(1, 1, 4, 4), t, model)
        self.assertTrue(model.training)
        self.assertEqual(model.calls, [])

    def test_ddim_skip_steps_recovers_oracle_without_clipping(self):
        for start_t, num_steps in ((1000, 50), (993, 31), (7, 50), (1, 1), (0, 50)):
            with self.subTest(start_t=start_t, num_steps=num_steps):
                oracle = ConstantImageOracle(value=1.375)
                clean = torch.full((1, 1, 4, 4), oracle.value)
                xt = q_sample(clean, torch.tensor([start_t])).requires_grad_()
                original = xt.detach().clone()
                frames = []
                result = inference(xt, start_t, oracle, sampler="ddim", num_steps=num_steps,
                                   on_step=lambda images, times: frames.append((images.clone(), times.clone())))
                torch.testing.assert_close(result, clean, atol=3e-5, rtol=3e-5)
                torch.testing.assert_close(xt.detach(), original, atol=0, rtol=0)
                self.assertEqual(len(oracle.calls), min(num_steps, start_t))
                self.assertLessEqual(len(oracle.calls), num_steps)
                times = [int(t.item()) for _, t in frames]
                self.assertEqual(times[0], start_t)
                self.assertEqual(times[-1], 0)
                self.assertEqual(times, sorted(set(times), reverse=True))
                self.assertFalse(result.requires_grad)
                self.assertTrue(oracle.training)

    def test_ddim_eta_zero_deterministic_and_does_not_consume_rng(self):
        model = NonOracle().eval()
        xt = torch.randn(2, 1, 4, 4)
        generator = torch.Generator().manual_seed(432)
        generator_before = generator.get_state().clone()
        torch.manual_seed(1)
        global_before = torch.get_rng_state().clone()
        first = inference(xt, 1000, model, sampler="ddim", num_steps=23, generator=generator)
        torch.testing.assert_close(torch.get_rng_state(), global_before, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), generator_before, rtol=0, atol=0)
        torch.manual_seed(999)
        second = inference(xt, 1000, model, sampler="ddim", num_steps=23,
                           generator=torch.Generator().manual_seed(999))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(model.training)

    def test_ddim_full_steps_eta_one_matches_ddpm_exactly(self):
        model = NonOracle()
        xt = torch.randn(3, 1, 2, 2)
        times = torch.tensor([0, 7, 1000])
        ddpm = inference(xt, times, model, generator=torch.Generator().manual_seed(521))
        ddim = inference(xt, times, model, sampler="ddim", num_steps=1000, eta=1,
                         generator=torch.Generator().manual_seed(521))
        torch.testing.assert_close(ddim, ddpm, rtol=0, atol=0)

    def test_ddim_eta_one_skipped_step_matches_independent_formula(self):
        model = NonOracle()
        xt = torch.randn(1, 1, 2, 2, dtype=torch.float64)
        frames = []
        generator = torch.Generator().manual_seed(731)
        inference(xt, 1000, model, sampler="ddim", num_steps=2, eta=1, generator=generator,
                  on_step=lambda images, times: frames.append((images.clone(), times.clone())))
        # Independent schedule and formula, first jump is from 1000 to 500.
        beta = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
        bars = (1 - beta).cumprod(0)
        a, previous = bars[999], bars[499]
        eps = model(xt, torch.tensor([1000]))
        x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        variance = (1 - previous) / (1 - a) * (1 - a / previous)
        noise = torch.randn(xt.shape, dtype=xt.dtype, generator=torch.Generator().manual_seed(731))
        expected = previous.sqrt() * x0 + (1 - previous - variance).sqrt() * eps + variance.sqrt() * noise
        torch.testing.assert_close(frames[1][0], expected, rtol=3e-6, atol=3e-6)
        self.assertEqual([int(t.item()) for _, t in frames], [1000, 500, 0])

    def test_condition_labels_follow_active_mixed_timesteps(self):
        for sampler in ("ddpm", "ddim"):
            with self.subTest(sampler=sampler):
                oracle = ConditionalImageOracle()
                labels = torch.tensor([2, 4, 9])
                times = torch.tensor([0, 1, 7])
                clean = (labels.float() / 10).view(3, 1, 1, 1).expand(3, 1, 4, 4).clone()
                xt = q_sample(clean, times)
                result = inference(xt, times, oracle, c=labels, sampler=sampler, num_steps=3)
                torch.testing.assert_close(result, clean, rtol=2e-5, atol=2e-5)
                expected_calls = [[4, 9]] + [[9]] * (6 if sampler == "ddpm" else 2)
                self.assertEqual(oracle.label_calls, expected_calls)
                torch.testing.assert_close(times, torch.tensor([0, 1, 7]), rtol=0, atol=0)
                torch.testing.assert_close(labels, torch.tensor([2, 4, 9]), rtol=0, atol=0)
                torch.testing.assert_close(result[0], xt[0], rtol=0, atol=0)

    def test_scalar_condition_broadcasts(self):
        oracle = ConditionalImageOracle()
        result = inference(torch.zeros(2, 1, 2, 2), [2, 8], oracle, c=6,
                           sampler="ddim", num_steps=2)
        torch.testing.assert_close(result, torch.full_like(result, 0.6), rtol=2e-5, atol=2e-5)
        self.assertEqual(oracle.label_calls, [[6, 6], [6, 6]])

    def test_cfg_prediction_mixture_and_active_label_mask(self):
        class GuidedReference(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, images, timesteps, labels):
                conditional = CFGPredictor.prediction(images, timesteps, labels)
                unconditional = CFGPredictor.prediction(images, timesteps, torch.full_like(labels, 10))
                # Independently expressed affine combination for all image pixels.
                return self.scale * conditional + (1 - self.scale) * unconditional

        xt = torch.randn(3, 1, 2, 2)
        original = xt.clone()
        times = torch.tensor([0, 1, 7])
        labels = torch.tensor([2, 4, 9])
        for sampler in ("ddpm", "ddim"):
            for scale in (0.0, 1.0, 2.5):
                for training in (False, True):
                    with self.subTest(sampler=sampler, scale=scale, training=training):
                        model = CFGPredictor().train(training)
                        kwargs = {"c": labels, "sampler": sampler, "num_steps": 3}
                        actual_frames, expected_frames = [], []
                        actual = inference(
                            xt, times, model, cfg_scale=scale, **kwargs,
                            generator=torch.Generator().manual_seed(181),
                            on_step=lambda images, _: actual_frames.append(images.clone()),
                        )
                        expected = inference(
                            xt, times, GuidedReference(scale), **kwargs,
                            generator=torch.Generator().manual_seed(181),
                            on_step=lambda images, _: expected_frames.append(images.clone()),
                        )
                        for actual_frame, expected_frame in zip(actual_frames, expected_frames):
                            torch.testing.assert_close(actual_frame, expected_frame, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(actual[0], xt[0], rtol=0, atol=0)
                        expected_labels = [[4, 9]] + [[9]] * (6 if sampler == "ddpm" else 2)
                        if scale == 1:
                            self.assertEqual([c.tolist() for _, _, c in model.calls], expected_labels)
                        else:
                            self.assertEqual(len(model.calls), 2 * len(expected_labels))
                            for index, expected_label in enumerate(expected_labels):
                                cond_x, cond_t, cond_c = model.calls[2 * index]
                                null_x, null_t, null_c = model.calls[2 * index + 1]
                                self.assertEqual(cond_c.tolist(), expected_label)
                                self.assertEqual(null_c.tolist(), [10] * len(expected_label))
                                torch.testing.assert_close(cond_x, null_x, rtol=0, atol=0)
                                torch.testing.assert_close(cond_t, null_t, rtol=0, atol=0)
                        self.assertEqual(model.training, training)
                        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(xt, original, rtol=0, atol=0)
        self.assertEqual(times.tolist(), [0, 1, 7])
        self.assertEqual(labels.tolist(), [2, 4, 9])

    def test_cfg_scale_one_preserves_original_sampling_exactly(self):
        xt = torch.randn(2, 1, 3, 3)
        for sampler in ("ddpm", "ddim"):
            with self.subTest(sampler=sampler):
                original_model, explicit_model = ConditionalImageOracle(), ConditionalImageOracle()
                kwargs = {"c": [3, 7], "sampler": sampler, "num_steps": 4}
                original = inference(xt, [4, 9], original_model, **kwargs,
                                     generator=torch.Generator().manual_seed(94))
                explicit = inference(xt, [4, 9], explicit_model, cfg_scale=1.0, **kwargs,
                                     generator=torch.Generator().manual_seed(94))
                torch.testing.assert_close(explicit, original, rtol=0, atol=0)
                self.assertEqual(explicit_model.label_calls, original_model.label_calls)

    def test_cfg_rejects_missing_labels_or_unsupported_model(self):
        xt = torch.zeros(1, 1, 2, 2)
        for scale in (0.0, 2.0):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "explicit class labels"):
                    inference(xt, 1, CFGPredictor(), cfg_scale=scale)
                legacy_models = [ConditionalImageOracle(), CFGPredictor(), CFGPredictor()]
                legacy_models[1].null_label = None
                legacy_models[2].num_classes = None
                for model in legacy_models:
                    with self.assertRaisesRegex(ValueError, "does not support CFG"):
                        inference(xt, 1, model, c=2, cfg_scale=scale)
                    self.assertTrue(model.training)
                    self.assertEqual(model.calls, [])
                for label in (-1, 10):
                    with self.assertRaisesRegex(ValueError, "between 0 and 9"):
                        inference(xt, 1, CFGPredictor(), c=label, cfg_scale=scale)

    def test_cfg_mode_restored_when_null_prediction_raises(self):
        class BrokenNullPredictor(CFGPredictor):
            def forward(self, images, timesteps, labels):
                prediction = super().forward(images, timesteps, labels)
                if (labels == self.null_label).any():
                    raise RuntimeError("null prediction failed")
                return prediction

        for sampler in ("ddpm", "ddim"):
            for training in (True, False):
                with self.subTest(sampler=sampler, training=training):
                    model = BrokenNullPredictor().train(training)
                    with self.assertRaisesRegex(RuntimeError, "null prediction failed"):
                        inference(torch.zeros(1, 1, 2, 2), 4, model, c=2,
                                  sampler=sampler, cfg_scale=2)
                    self.assertEqual(model.training, training)
                    self.assertTrue(torch.is_grad_enabled())
                    self.assertEqual(len(model.calls), 2)

    def test_invalid_sampler_arguments_rejected(self):
        model = ConstantImageOracle()
        arguments = [
            {"sampler": "unknown"}, {"sampler": None},
            *({"num_steps": value} for value in (0, -1, 1001, 2.5, True)),
            *({"eta": value} for value in (-0.1, 1.1, float("nan"), float("inf"), True)),
            *({"cfg_scale": value} for value in (-0.1, float("nan"), float("inf"), True, "2")),
            {"c": [1, 2]}, {"c": 1.2}, {"c": float("nan")},
        ]
        for kwargs in arguments:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inference(torch.zeros(1, 1, 4, 4), 10, model, **kwargs)
        self.assertTrue(model.training)
        self.assertEqual(model.calls, [])

    def test_ddim_mode_restored_when_model_or_callback_raises(self):
        class BrokenModel(torch.nn.Module):
            def forward(self, images, timesteps):
                self.assertions = (self.training, torch.is_grad_enabled())
                raise RuntimeError("prediction failed")

        def broken_callback(images, timesteps):
            raise RuntimeError("frame failed")

        for training in (True, False):
            with self.subTest(training=training):
                model = BrokenModel().train(training)
                with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                    inference(torch.zeros(1, 1, 4, 4), 10, model, sampler="ddim")
                self.assertEqual(model.assertions, (False, False))
                self.assertEqual(model.training, training)
                with self.assertRaisesRegex(RuntimeError, "frame failed"):
                    inference(torch.zeros(1, 1, 4, 4), 10, model, sampler="ddim", on_step=broken_callback)
                self.assertEqual(model.training, training)

    def test_saved_ddim_gif_keeps_initial_and_final_frames(self):
        model = ConditionalImageOracle()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ddim.gif"
            result = save_inference_gif(torch.zeros(1, 1, 4, 4), 1000, model, path,
                                        sampler="ddim", num_steps=5, frame_every=2, c=3)
            torch.testing.assert_close(result, torch.full_like(result, 0.3), rtol=2e-5, atol=2e-5)
            self.assertEqual(model.calls, [[1000], [800], [600], [400], [200]])
            self.assertEqual(model.label_calls, [[3]] * 5)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 4)  # initial, updates 2/4, final update 5
                gif.seek(3)
                self.assertEqual(gif.info["duration"], 800)

    def test_saved_gif_forwards_cfg_scale(self):
        xt = torch.randn(1, 1, 2, 2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cfg.gif"
            result = save_inference_gif(xt, 7, CFGPredictor(), path, c=5,
                                        sampler="ddim", num_steps=3, cfg_scale=2)
            expected = inference(xt, 7, CFGPredictor(), c=5,
                                 sampler="ddim", num_steps=3, cfg_scale=2)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertTrue(path.is_file())

    def test_saved_gif_frames_pixels_and_durations(self):
        cases = (
            (7, 3, [7, 4, 1, 0]),  # Non-divisible final interval.
            (6, 3, [6, 3, 0]),  # No duplicate final frame.
            (2, 10, [2, 0]),  # Stride larger than the trajectory.
            (0, 10, [0]),  # An already clean image.
        )
        with TemporaryDirectory() as directory:
            for start_t, stride, expected_times in cases:
                with self.subTest(start_t=start_t, stride=stride):
                    model = ConstantImageOracle(record_images=True)
                    clean = torch.full((1, 1, 4, 4), model.value)
                    xt = q_sample(clean, torch.tensor([start_t]))
                    path = Path(directory) / str(start_t) / "preview.gif"
                    result = save_inference_gif(xt, start_t, model, path, frame_every=stride, duration_ms=80)
                    torch.testing.assert_close(result, clean, rtol=2e-5, atol=2e-5)
                    self.assertEqual(len(model.calls), start_t)
                    self.assertTrue(model.training)

                    with Image.open(path) as gif:
                        self.assertEqual(gif.format, "GIF")
                        self.assertEqual(gif.n_frames, len(expected_times))
                        self.assertEqual(gif.info.get("loop"), 0)
                        self.assertEqual(gif.size, (224, 64))
                        for frame_index, timestep in enumerate(expected_times):
                            gif.seek(frame_index)
                            self.assertEqual(gif.info["duration"], 800 if timestep == 0 else 80)
                            expected = result if timestep == 0 else model.inputs_by_t[timestep]
                            expected_pixels = expected[0, 0].clamp(0, 1).mul(255).round().byte().flatten().tolist()
                            # Decode the image area, excluding the caption and white margin.
                            actual = gif.convert("L").crop((96, 32, 128, 64)).resize((4, 4), Image.Resampling.NEAREST)
                            self.assertEqual(list(actual.tobytes()), expected_pixels)


if __name__ == "__main__":
    unittest.main()
