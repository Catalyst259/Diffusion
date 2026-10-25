"""Training lifecycle integration tests using tiny in-memory image datasets."""

import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import TensorDataset

from DiTLab.algorithms.drift import DriftAlgorithm
from DiTLab.artifacts import atomic_checkpoint, load_checkpoint
from DiTLab.config import ComponentConfig, ExperimentConfig, TrainConfig
from DiTLab.data import DataBundle, DataSpec
from DiTLab.engine import Trainer
from DiTLab.preview import save_preview


class TinyConditionalModel(nn.Module):
    """Uses global RNG, including training dropout, to expose resume mistakes."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.35))
        self.class_bias = nn.Parameter(torch.linspace(-0.1, 0.1, 10))
        self.dropout = nn.Dropout(0.25)

    def forward(self, noise, labels, cfg_scale):
        cfg = torch.as_tensor(cfg_scale, device=noise.device, dtype=noise.dtype)
        if cfg.ndim == 0:
            cfg = cfg.expand(noise.shape[0])
        bias = (self.class_bias[labels] * cfg).reshape(-1, 1, 1, 1)
        # Intentionally stochastic even in eval mode; evaluation and previews
        # must isolate this RNG from training as well as their explicit RNG.
        jitter = torch.randn_like(noise) * 0.01
        return self.dropout(noise) * self.scale + bias + jitter


class KnownExampleLoss:
    """A known per-example objective detects unweighted averaging of batches."""

    def __init__(self):
        self.observations = []

    def loss(self, model, images, labels, *, generator, training):
        del labels, generator
        loss = images[:, 0, 0, 0].square().mean() + model.scale * 0
        self.observations.append((training, len(images), float(loss.detach())))
        return loss


def tiny_data():
    train_images = torch.linspace(-0.9, 0.9, 9 * 16).reshape(9, 1, 4, 4)
    test_images = torch.linspace(0.85, -0.85, 10 * 16).reshape(10, 1, 4, 4)
    return DataBundle(
        TensorDataset(train_images, torch.tensor([2, 7, 2, 7, 2, 7, 2, 7, 2])),
        TensorDataset(test_images, torch.tensor([7, 2, 7, 2, 7, 2, 7, 2, 7, 2])),
        DataSpec(image_size=4, in_channels=1, num_classes=10),
    )


def tiny_config(output_dir):
    config = ExperimentConfig()
    config.model = ComponentConfig("tiny-test-model")
    config.algorithm = ComponentConfig("drift", {
        "temperatures": [0.05, 0.2], "generated_per_class": 3,
        "cfg_min": 1.0, "cfg_max": 2.0,
    })
    config.training = TrainConfig(
        epochs=2, batch_size=4, preview_every=2, save_every=2,
        preview_count=10, seed=23, device="cpu", log_every=1000,
        output_dir=str(output_dir),
    )
    return config


def tiny_trainer(output_dir, *, checkpoint=None, data=None, algorithm=None):
    config = tiny_config(output_dir)
    return Trainer(
        config, tiny_data() if data is None else data, TinyConditionalModel(),
        DriftAlgorithm(**config.algorithm.options) if algorithm is None else algorithm,
        DriftAlgorithm(cfg_scale=2.0), checkpoint=checkpoint,
    )


def run_quietly(trainer, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return trainer.run(**kwargs)


class TestTrainingLifecycle(unittest.TestCase):
    def test_periodic_preview_latest_checkpoint_and_epoch_artifacts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = tiny_trainer(output)
            saved_progress = []

            def record_save(payload, path):
                atomic_checkpoint(payload, path)
                saved_progress.append(dict(payload["progress"]))
                self.assertEqual(Path(path), output / "weights" / "latest.pt")
                self.assertEqual([item.name for item in Path(path).parent.iterdir()], ["latest.pt"])

            with (
                patch("DiTLab.engine.atomic_checkpoint", side_effect=record_save),
                patch("DiTLab.engine.save_preview", wraps=save_preview) as render,
            ):
                history = run_quietly(trainer)

            self.assertEqual([row["global_step"] for row in history], [3, 6])
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            self.assertEqual({row["global_step"] for row in saved_progress}, {2, 3, 4, 6})
            self.assertEqual(render.call_count, 3)
            preview_paths = sorted((output / "previews").glob("*.png"))
            self.assertEqual(
                [path.name for path in preview_paths],
                [f"step_{step:08d}.png" for step in (2, 4, 6)],
            )
            for path, call in zip(preview_paths, render.call_args_list):
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                indices = metadata["test_indices"]
                self.assertEqual(len(set(indices)), 10)
                self.assertEqual(metadata["step"], int(path.stem.split("_")[1]))
                expected_labels = trainer.data.test_dataset.tensors[1][indices]
                self.assertEqual(metadata["labels"], expected_labels.tolist())
                reference, generated, labels = call.args[:3]
                self.assertEqual(reference.shape, (10, 1, 4, 4))
                self.assertEqual(generated.shape, reference.shape)
                torch.testing.assert_close(reference, trainer.data.test_dataset.tensors[0][indices])
                torch.testing.assert_close(labels, expected_labels)
                with Image.open(path) as panel:
                    self.assertEqual(panel.format, "PNG")
                    self.assertGreater(panel.width, 10 * 4)
                    self.assertGreater(panel.height, 2 * 4)

            checkpoint = load_checkpoint(output / "weights" / "latest.pt")
            self.assertEqual(checkpoint["history"], history)
            self.assertEqual(checkpoint["progress"]["global_step"], 6)
            self.assertEqual(checkpoint["progress"]["epoch"], 2)
            self.assertEqual(checkpoint["progress"]["batch_in_epoch"], 0)
            self.assertEqual(checkpoint["progress"]["last_preview_step"], 6)
            self.assertEqual(list((output / "weights").glob("*.pt")), [output / "weights" / "latest.pt"])
            with (output / "losses.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            for stored, expected in zip(rows, history):
                for key in ("epoch", "global_step"):
                    self.assertEqual(int(stored[key]), expected[key])
                for key in ("train_loss", "test_loss"):
                    self.assertEqual(float(stored[key]), expected[key])
            with Image.open(output / "graph" / "loss_curve.png") as plot:
                self.assertEqual(plot.format, "PNG")
                self.assertGreater(plot.width, 100)

    def test_mid_epoch_resume_exactly_matches_uninterrupted_drift_training(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            torch.manual_seed(991)
            full = tiny_trainer(output / "full")
            expected_history = run_quietly(full)
            expected_rng = torch.get_rng_state().clone()
            expected_algorithm_rng = full.generator.get_state().clone()

            torch.manual_seed(991)
            first = tiny_trainer(output / "resumed")
            self.assertEqual(run_quietly(first, max_steps=2), [])
            checkpoint = load_checkpoint(first.checkpoint_path)
            progress = checkpoint["progress"]
            self.assertEqual(progress["epoch"], 0)
            self.assertEqual(progress["global_step"], 2)
            self.assertEqual(progress["batch_in_epoch"], 2)
            self.assertEqual(progress["train_sample_count"], 8)
            self.assertEqual(progress["last_preview_step"], 2)

            # A new process/model may consume arbitrary randomness before load.
            torch.randn(117)
            resumed = tiny_trainer(output / "resumed", checkpoint=checkpoint)
            actual_history = run_quietly(resumed)
            self.assertEqual(actual_history, expected_history)
            for name, expected in full.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(resumed.generator.get_state(), expected_algorithm_rng, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            self.assertEqual(resumed.global_step, full.global_step)

    def test_evaluation_preview_preserve_rng_and_caller_mode(self):
        with TemporaryDirectory() as directory:
            trainer = tiny_trainer(directory)
            for training_mode in (True, False):
                with self.subTest(training_mode=training_mode):
                    trainer.model.train(training_mode)
                    global_rng = torch.get_rng_state().clone()
                    algorithm_rng = trainer.generator.get_state().clone()
                    first_loss = trainer.evaluate()
                    self.assertEqual(trainer.model.training, training_mode)
                    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
                    torch.testing.assert_close(trainer.generator.get_state(), algorithm_rng, rtol=0, atol=0)
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.preview()
                    self.assertEqual(trainer.model.training, training_mode)
                    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
                    torch.testing.assert_close(trainer.generator.get_state(), algorithm_rng, rtol=0, atol=0)
                    self.assertEqual(trainer.evaluate(), first_loss)

    def test_epoch_losses_weight_uneven_batches_by_sample_count(self):
        train_values = torch.arange(1, 10, dtype=torch.float32) / 10
        test_values = torch.arange(1, 11, dtype=torch.float32) / 10
        data = DataBundle(
            TensorDataset(train_values[:, None, None, None].expand(-1, 1, 4, 4), torch.zeros(9, dtype=torch.long)),
            TensorDataset(test_values[:, None, None, None].expand(-1, 1, 4, 4), torch.ones(10, dtype=torch.long)),
            DataSpec(image_size=4, in_channels=1, num_classes=10),
        )
        algorithm = KnownExampleLoss()
        with TemporaryDirectory() as directory:
            trainer = tiny_trainer(directory, data=data, algorithm=algorithm)
            history = run_quietly(trainer)
        self.assertEqual([n for training, n, _ in algorithm.observations if training], [4, 4, 1] * 2)
        self.assertEqual([n for training, n, _ in algorithm.observations if not training], [4, 4, 2] * 2)
        for row in history:
            self.assertAlmostEqual(row["train_loss"], train_values.square().mean().item(), places=7)
            self.assertAlmostEqual(row["test_loss"], test_values.square().mean().item(), places=7)


if __name__ == "__main__":
    unittest.main()
