"""Loss comparison checks using recorded histories, without model training."""

import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from DiTLab.comparison import LOSS_FIELDS, summarize_runs


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_fixture(self, number, losses=None, *, epochs=3, status=None):
        output = self.root / f"run_{number:03d}"
        run = {"run_id": output.name, "seed": 40 + number, "output_dir": str(output), "epochs": epochs}
        if status is not None:
            run["status"] = status
        if losses is not None:
            output.mkdir(parents=True, exist_ok=True)
            with (output / "losses.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(LOSS_FIELDS)
                for epoch, train, test in losses:
                    writer.writerow((epoch, epoch * 10, train, test))
        return run

    @staticmethod
    def read_csv(path):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_partial_runs_mean_sample_std_and_png(self):
        runs = [
            self.run_fixture(1, [(1, 10, 8), (2, 6, 4), (3, 2, 1)]),
            self.run_fixture(2, [(1, 14, 12), (2, 10, 8)]),
            self.run_fixture(3),
        ]
        summary = summarize_runs(runs, self.root)
        self.assertEqual(summary["completed_runs"], 1)
        self.assertEqual(summary["incomplete_runs"], 1)
        self.assertEqual(summary["pending_runs"], 1)
        self.assertEqual(summary["runs_with_history"], 2)
        epochs = self.read_csv(summary["artifacts"]["epoch_summary"])
        self.assertEqual([row["n"] for row in epochs], ["2", "2", "1"])
        self.assertEqual(float(epochs[0]["train_mean"]), 12)
        self.assertAlmostEqual(float(epochs[0]["train_std"]), math.sqrt(8))
        self.assertEqual(float(epochs[1]["test_mean"]), 6)
        self.assertAlmostEqual(float(epochs[1]["test_std"]), math.sqrt(8))
        self.assertEqual(float(epochs[2]["train_std"]), 0)
        self.assertEqual(float(epochs[2]["test_mean"]), 1)
        combined = self.read_csv(summary["artifacts"]["combined_losses"])
        self.assertEqual(len(combined), 5)
        self.assertEqual(combined[0]["run_id"], "run_001")
        self.assertEqual(combined[0]["seed"], "41")
        summaries = self.read_csv(summary["artifacts"]["run_summary"])
        self.assertEqual(float(summaries[0]["train_change"]), -8)
        self.assertEqual(float(summaries[0]["test_change"]), -7)
        self.assertEqual(float(summaries[0]["min_test_loss"]), 1)
        self.assertEqual(summaries[2]["final_test_loss"], "")
        with Image.open(summary["artifacts"]["comparison"]) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 1000)
            self.assertGreater(image.height, 700)
            self.assertIsNotNone(image.convert("RGB").getbbox())

    def test_missing_empty_and_header_only_files_are_pending(self):
        runs = [self.run_fixture(1), self.run_fixture(2, []), self.run_fixture(3, [])]
        (Path(runs[2]["output_dir"]) / "losses.csv").write_text("", encoding="utf-8")
        summary = summarize_runs(runs, self.root)
        self.assertEqual(summary["pending_runs"], 3)
        self.assertEqual(summary["completed_runs"], 0)
        self.assertIsNone(summary["artifacts"]["comparison"])
        self.assertFalse((self.root / "comparison.png").exists())
        self.assertEqual(self.read_csv(summary["artifacts"]["epoch_summary"]), [])
        self.assertEqual(self.read_csv(summary["artifacts"]["combined_losses"]), [])

    def test_failed_and_interrupted_runs_keep_actual_history(self):
        runs = [
            self.run_fixture(1, [(1, 4, 5)], status="failed"),
            self.run_fixture(2, status="failed"),
            self.run_fixture(3, status="interrupted"),
        ]
        summary = summarize_runs(runs, self.root)
        self.assertEqual(summary["failed_runs"], 2)
        self.assertEqual(summary["incomplete_runs"], 1)
        self.assertEqual(summary["runs_with_history"], 1)
        self.assertEqual(self.read_csv(summary["artifacts"]["epoch_summary"])[0]["n"], "1")

    def test_invalid_losses_report_source_path(self):
        invalid_values = ("nan", "inf", "-inf", "not-a-number", "")
        for field in ("train_loss", "test_loss"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    losses = [(1, value, 1)] if field == "train_loss" else [(1, 1, value)]
                    run = self.run_fixture(1, losses)
                    with self.assertRaisesRegex(ValueError, rf"losses\.csv: row 2: {field} must be finite"):
                        summarize_runs([run], self.root)

    def test_invalid_epoch_order_and_values_rejected(self):
        for epochs in ([1, 1], [2, 1], [0], [-1], ["1.5"], ["nan"]):
            with self.subTest(epochs=epochs):
                run = self.run_fixture(1, [(epoch, 1, 2) for epoch in epochs])
                with self.assertRaisesRegex(ValueError, r"losses\.csv"):
                    summarize_runs([run], self.root)

    def test_missing_columns_rejected_before_writing_artifacts(self):
        run = self.run_fixture(1, [])
        path = Path(run["output_dir"]) / "losses.csv"
        path.write_text("epoch,train_loss\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"losses\.csv: missing required columns"):
            summarize_runs([run], self.root)
        self.assertFalse((self.root / "run_summary.csv").exists())

    def test_no_data_removes_stale_plot(self):
        (self.root / "comparison.png").write_bytes(b"stale image")
        summarize_runs([self.run_fixture(1)], self.root)
        self.assertFalse((self.root / "comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
