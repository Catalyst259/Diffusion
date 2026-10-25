"""Protect existing experiments before constructing models or loading datasets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from DiTLab.config import ExperimentConfig
from DiTLab.train import main


class CommandLineTests(unittest.TestCase):
    def test_default_schedule_matches_requested_experiment(self):
        settings = ExperimentConfig().training
        self.assertGreater(settings.epochs, 0)
        self.assertEqual((settings.preview_every, settings.save_every,
                          settings.preview_count), (500, 500, 10))

    def test_resume_cannot_overwrite_another_existing_run(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "run_b"
            latest = destination / "weights/latest.pt"
            latest.parent.mkdir(parents=True)
            latest.write_bytes(b"existing experiment")
            checkpoint = {"config": ExperimentConfig().to_dict()}
            with patch("DiTLab.train.load_checkpoint", return_value=checkpoint):
                with self.assertRaises(FileExistsError):
                    main(["--resume", str(Path(directory) / "run_a/weights/latest.pt"),
                          "--output-dir", str(destination)])
            self.assertEqual(latest.read_bytes(), b"existing experiment")

    def test_resume_rejects_changes_that_invalidate_saved_progress(self):
        checkpoint = {"config": ExperimentConfig().to_dict()}
        for flag, value in (("--batch-size", "32"), ("--seed", "99"), ("--learning-rate", "0.01")):
            with self.subTest(flag=flag), patch("DiTLab.train.load_checkpoint", return_value=checkpoint):
                with self.assertRaises(ValueError):
                    main(["--resume", "latest.pt", flag, value])


if __name__ == "__main__":
    unittest.main()
