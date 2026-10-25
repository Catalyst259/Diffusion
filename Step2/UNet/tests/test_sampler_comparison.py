"""Check paired inputs and exported comparison artifacts without a checkpoint."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image

from UNet.src.compare_samplers import noise_for_groups, save_comparison_gif, save_overview


class ComparisonTests(unittest.TestCase):
    def test_group_noise_is_independent_of_batching(self):
        full = noise_for_groups(0, 7, 42)
        split = torch.cat((noise_for_groups(0, 3, 42), noise_for_groups(3, 4, 42)))
        torch.testing.assert_close(full, split, rtol=0, atol=0)
        self.assertFalse(torch.equal(full[0], full[1]))

    def test_gif_has_shared_start_and_distinct_final_panels(self):
        initial = torch.zeros(28, 28)
        times = [1000, 500, 0]
        left = {t: torch.full((28, 28), v) for t, v in zip(times, [0.0, 0.5, 0.25])}
        right = {t: torch.full((28, 28), v) for t, v in zip(times, [0.0, 0.4, 0.75])}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pair.gif"
            save_comparison_gif(path, initial, left, right, times, 1, 42)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, 3)
                self.assertEqual(gif.info["loop"], 0)
                gif.seek(2)
                self.assertEqual(gif.info["duration"], 1200)
                pixels = gif.convert("RGB")
                self.assertEqual(pixels.getpixel((20, 90)), (0, 0, 0))
                self.assertEqual(pixels.getpixel((204, 90)), (64, 64, 64))
                self.assertEqual(pixels.getpixel((388, 90)), (191, 191, 191))
            overview = Path(directory) / "overview.png"
            save_overview(overview, [left[0]] * 7, [right[0]] * 7)
            with Image.open(overview) as picture:
                self.assertEqual(picture.size, (640, 206))


if __name__ == "__main__":
    unittest.main()
