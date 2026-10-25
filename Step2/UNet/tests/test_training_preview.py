"""Verify preview timing and source without MNIST downloads or full training."""
import contextlib
import io
import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main
from sample import q_sample, save_inference_gif


class TinyEpsilonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_labels = []

    def forward(self, images, timesteps, c):
        self.seen_labels.append(c.detach().cpu().tolist())
        return images * self.scale


class TrainingPreviewTest(unittest.TestCase):
    def test_every_100_updates_across_epochs_uses_training_image(self):
        # 21 batches * 10 epochs: the previews occur inside epochs 5 and 10.
        train_images = torch.zeros(1, 1, 8, 8)
        test_images = torch.ones_like(train_images)
        data = types.ModuleType("data")
        data.DATA_ROOT = Path("fake-mnist")
        data.train_loader = [(train_images, torch.tensor([2]))] * 21
        data.test_loader = [(test_images, torch.tensor([9]))]
        model = TinyEpsilonModel()
        calls = []

        def fixed_t(low, high, size, *, device):
            return torch.full(size, 7, dtype=torch.long, device=device)

        def save_preview(xt, t, model, path, **kwargs):
            calls.append(Path(path).name)
            torch.testing.assert_close(kwargs["c"], torch.tensor([2]))
            expected = q_sample(train_images, t, noise=torch.full_like(xt, 0.25))
            torch.testing.assert_close(xt, expected)
            self.assertTrue(model.training)
            result = save_inference_gif(xt, t, model, path, **kwargs)
            self.assertTrue(model.training)
            self.assertFalse(result.requires_grad)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 2)  # Initial t=7 and final t=0.
            return result

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(sys.modules, {"data": data}),
                patch.object(main, "UNet", return_value=model) as make_model,
                patch.object(main, "OUTPUT_DIR", Path(directory)),
                patch.object(main, "NUM_EPOCHS", 10),
                patch.object(main, "save_loss_graph"),
                patch.object(main, "save_inference_gif", side_effect=save_preview),
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(torch, "randint", side_effect=fixed_t),
                patch.object(torch, "randn_like", side_effect=lambda x: torch.full_like(x, 0.25)),
                patch.object(torch, "save") as checkpoint,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main.train()
            self.assertEqual(make_model.call_args.kwargs["num_classes"], 10)
            self.assertEqual(make_model.call_args.kwargs["class_dropout_prob"], 0.1)
            self.assertEqual(checkpoint.call_count, 3)  # Two previews plus final weights.
            self.assertEqual(Path(checkpoint.call_args.args[1]).name, "model_final.pth")
            with (Path(directory) / "loss.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[-1]["epoch"], "10")
            config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["num_classes"], 10)
            self.assertEqual(config["class_dropout_prob"], 0.1)
            self.assertEqual(config["null_label"], 10)
            self.assertEqual(config["preview_sampler"], "ddpm")
            self.assertEqual(config["preview_cfg_scale"], 1.0)
            completed = json.loads((Path(directory) / "completed.json").read_text(encoding="utf-8"))
            self.assertEqual(completed["global_step"], 210)
        self.assertEqual(calls, ["denoise_epoch5_step100.gif", "denoise_epoch10_step200.gif"])
        self.assertEqual(model.seen_labels.count([9]), 10)  # Validation also receives c.


if __name__ == "__main__":
    unittest.main()
