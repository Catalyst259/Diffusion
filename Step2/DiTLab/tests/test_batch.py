"""Batch independence, checkpoint-aware continuation and existing-output protection."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from DiTLab.batch import create_plan, load_plan, run_batch, _execute_training
from DiTLab.config import ExperimentConfig


def save_fake_checkpoint(run, epoch):
    path = Path(run["output_dir"]) / "weights/latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": run["config"],
                "progress": {"epoch": epoch}}, path)


class BatchTests(unittest.TestCase):
    def test_twenty_plans_change_only_seed_and_output_directory(self):
        config = ExperimentConfig()
        config.training.epochs = 20
        config.training.learning_rate = 2e-5
        original = deepcopy(config.to_dict())
        with TemporaryDirectory() as directory:
            manifest = create_plan(config, directory)
            self.assertEqual([run["seed"] for run in manifest["runs"]], list(range(42, 62)))
            self.assertEqual(len({run["output_dir"] for run in manifest["runs"]}), 20)
            self.assertEqual(load_plan(directory), manifest)
            for run in manifest["runs"]:
                stored = json.loads(Path(run["config_path"]).read_text(encoding="utf-8"))
                self.assertEqual(stored, run["config"])
                expected = deepcopy(manifest["base_config"])
                expected["training"].update(seed=run["seed"], output_dir=run["output_dir"])
                self.assertEqual(stored, expected)
            self.assertEqual(config.to_dict(), original)

    def test_nonempty_directory_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                create_plan(ExperimentConfig(), directory)
            self.assertEqual(marker.read_text(), "keep")

    def test_edited_manifest_cannot_redirect_training_elsewhere(self):
        with TemporaryDirectory() as directory:
            manifest = create_plan(ExperimentConfig(), directory, runs=1)
            manifest["runs"][0]["config"]["training"]["output_dir"] = str(Path(directory).parent / "unrelated")
            (Path(directory) / "batch.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_plan(directory)

    def test_sequential_runs_use_current_python_and_resume_skips_completed(self):
        with TemporaryDirectory() as directory:
            manifest = create_plan(ExperimentConfig(), directory, runs=2)
            commands = []

            def execute(command, log_path):
                commands.append(command)
                run = next(run for run in manifest["runs"] if Path(run["output_dir"]) == Path(log_path).parent)
                save_fake_checkpoint(run, run["epochs"])
                return 0

            with contextlib.redirect_stdout(io.StringIO()), patch("DiTLab.batch._execute_training", side_effect=execute), patch("DiTLab.batch.summarize_runs"):
                run_batch(manifest)
                run_batch(load_plan(directory))
            self.assertEqual(len(commands), 2)
            self.assertTrue(all(command[0] == sys.executable for command in commands))
            self.assertTrue(all("--config" in command for command in commands))
            self.assertEqual([run["status"] for run in load_plan(directory)["runs"]], ["completed", "completed"])

    def test_partial_run_uses_its_checkpoint_and_preserves_frozen_settings(self):
        with TemporaryDirectory() as directory:
            manifest = create_plan(ExperimentConfig(), directory, runs=1)
            run = manifest["runs"][0]
            save_fake_checkpoint(run, 1)
            Path(run["config_path"]).write_text("unrelated edit", encoding="utf-8")

            def execute(command, log_path):
                self.assertIn("--resume", command)
                self.assertEqual(command[-1], str(Path(run["output_dir"]) / "weights/latest.pt"))
                save_fake_checkpoint(run, run["epochs"])
                return 0

            with contextlib.redirect_stdout(io.StringIO()), patch("DiTLab.batch._execute_training", side_effect=execute), patch("DiTLab.batch.summarize_runs"):
                run_batch(manifest)
            self.assertEqual(json.loads(Path(run["config_path"]).read_text(encoding="utf-8")), run["config"])

    def test_failed_run_stops_batch_and_records_failure(self):
        with TemporaryDirectory() as directory:
            manifest = create_plan(ExperimentConfig(), directory, runs=2)
            with contextlib.redirect_stdout(io.StringIO()), patch("DiTLab.batch._execute_training", return_value=7) as execute, patch("DiTLab.batch.summarize_runs"):
                with self.assertRaisesRegex(RuntimeError, "code 7"):
                    run_batch(manifest)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual([run["status"] for run in load_plan(directory)["runs"]], ["failed", "pending"])

    def test_checkpoint_with_different_settings_is_rejected(self):
        with TemporaryDirectory() as directory:
            manifest = create_plan(ExperimentConfig(), directory, runs=1)
            wrong_run = deepcopy(manifest["runs"][0])
            wrong_run["config"]["training"]["learning_rate"] *= 2
            save_fake_checkpoint(wrong_run, 1)
            with patch("DiTLab.batch._execute_training") as execute, patch("DiTLab.batch.summarize_runs"):
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    run_batch(manifest)
            execute.assert_not_called()

    def test_child_output_is_streamed_to_log_without_shell(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as display:
            log_path = Path(directory) / "train.log"
            code = _execute_training([sys.executable, "-c", "print('child test output')"], log_path)
            self.assertEqual(code, 0)
            self.assertIn("child test output", log_path.read_text(encoding="utf-8"))
            self.assertIn("child test output", display.getvalue())


if __name__ == "__main__":
    unittest.main()
