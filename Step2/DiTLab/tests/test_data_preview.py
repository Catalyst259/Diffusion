"""Held-out selection and preview checks without downloads or training."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import TensorDataset

from DiTLab.data import build_mnist, select_test_samples
from DiTLab.preview import save_preview


class TestSampleSelection(unittest.TestCase):
    def setUp(self):
        self.images = torch.arange(17 * 16, dtype=torch.float32).reshape(17, 1, 4, 4)
        # Deliberately exclude some digits; selection must retain these labels.
        self.labels = torch.tensor([3, 7, 3, 8, 7, 3, 8, 8, 3, 7, 3, 7, 8, 8, 7, 3, 7])
        self.dataset = TensorDataset(self.images, self.labels)

    def test_deterministic_distinct_selection_preserves_labels_and_global_rng(self):
        rng_state = torch.get_rng_state().clone()
        first = select_test_samples(self.dataset, seed=19)
        second = select_test_samples(self.dataset, seed=19)
        images, labels, indices = first
        self.assertEqual(len(set(indices)), 10)
        self.assertTrue(all(isinstance(index, int) for index in indices))
        self.assertEqual(indices, second[2])
        torch.testing.assert_close(images, second[0])
        torch.testing.assert_close(images, self.images[indices])
        torch.testing.assert_close(labels, self.labels[indices])
        self.assertEqual(labels.dtype, torch.long)
        torch.testing.assert_close(torch.get_rng_state(), rng_state)

    def test_selection_validates_available_count(self):
        for count in (0, -1, 18):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_test_samples(self.dataset, count=count, seed=0)

    def test_mnist_adapter_uses_official_splits_and_normalizes_images(self):
        with patch("torchvision.datasets.MNIST", side_effect=["train", "test"]) as mnist:
            bundle = build_mnist("unused-test-root", download=False)
        self.assertEqual(bundle.train_dataset, "train")
        self.assertEqual(bundle.test_dataset, "test")
        self.assertEqual(bundle.spec.num_classes, 10)
        self.assertEqual(bundle.spec.image_size, 28)
        self.assertEqual(bundle.spec.in_channels, 1)
        self.assertEqual([call.kwargs["train"] for call in mnist.call_args_list], [True, False])
        self.assertTrue(all(not call.kwargs["download"] for call in mnist.call_args_list))
        transform = mnist.call_args_list[0].kwargs["transform"]
        black = transform(Image.new("L", (28, 28), color=0))
        white = transform(Image.new("L", (28, 28), color=255))
        torch.testing.assert_close(black, torch.full((1, 28, 28), -1.0))
        torch.testing.assert_close(white, torch.ones(1, 28, 28))


class TestPreview(unittest.TestCase):
    def test_rgb_panel_preserves_images_in_two_rows(self):
        # Exact red and blue pixels let us locate the two rows independently
        # of font size, margins, and PNG compression.
        references = torch.zeros(2, 3, 8, 8)
        references[:, 0] = 1
        generations = torch.zeros_like(references)
        generations[:, 2] = 1
        original_references = references.clone()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "previews" / "test.png"
            result = save_preview(
                references, generations, torch.tensor([7, 3]), path,
                value_min=0, value_max=1, title="Step 10000",
            )
            self.assertEqual(result, path)
            with Image.open(path) as panel:
                self.assertEqual(panel.format, "PNG")
                self.assertEqual(panel.mode, "RGB")
                red = []
                blue = []
                for y in range(panel.height):
                    for x in range(panel.width):
                        pixel = panel.getpixel((x, y))
                        if pixel == (255, 0, 0):
                            red.append((x, y))
                        elif pixel == (0, 0, 255):
                            blue.append((x, y))
                self.assertGreater(len(red), 2 * 8 * 8)
                self.assertEqual(len(red), len(blue))
                self.assertLess(max(y for _, y in red), min(y for _, y in blue))
                self.assertEqual({x for x, _ in red}, {x for x, _ in blue})
                self.assertGreater(panel.width, 2 * 8)
                self.assertGreater(panel.height, 2 * 8)
        torch.testing.assert_close(references, original_references)

    def test_grayscale_and_invalid_shapes(self):
        references = torch.full((10, 1, 28, 28), -1.0)
        generations = torch.ones_like(references)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "grayscale.png"
            save_preview(references, generations, torch.arange(10), path)
            with Image.open(path) as panel:
                colors = dict((color, amount) for amount, color in panel.getcolors())
                self.assertGreaterEqual(colors[(0, 0, 0)], 10 * 28 * 28)
                self.assertEqual(colors[(255, 255, 255)], 10 * 112 * 112)
            with self.assertRaises(ValueError):
                save_preview(references, generations[:1], torch.arange(10), path)
            with self.assertRaises(ValueError):
                save_preview(references, generations, torch.arange(9), path)
            with self.assertRaises(ValueError):
                save_preview(references, generations, torch.arange(10), path, value_min=1)


if __name__ == "__main__":
    unittest.main()
