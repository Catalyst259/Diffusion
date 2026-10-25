"""Validate completed matched runs, then export their loss and sample comparison.

Run from Step2 with ``python -m DiTLab.compare_methods``. This command only reads
existing training artifacts and uses CPU tensors; it never runs either model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "DiTLab"

from PIL import Image, ImageDraw, ImageFont
import torch

from .comparison import _read_history
from .config import ROOT
from .models.dit import DiT


EXPECTED_PARAMETERS = 57_656_848
EXPECTED_EPOCHS = 5
EXPECTED_STEPS = 4_690
MATCHED_TRAINING_FIELDS = (
    "epochs", "batch_size", "learning_rate", "weight_decay", "max_grad_norm",
    "seed", "preview_count",
)
LOSS_NOTE = "Drifting and EDM optimize different objectives; loss magnitudes are not directly comparable."


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_history(history: list[dict], path: Path) -> None:
    if len(history) != EXPECTED_EPOCHS:
        raise ValueError(f"{path}: expected exactly five completed epochs")
    for epoch, row in enumerate(history, start=1):
        if row["epoch"] != epoch or row["global_step"] != epoch * 938:
            raise ValueError(f"{path}: incomplete or unexpected epoch/update sequence")
        if any(not math.isfinite(row[field]) or row[field] < 0
               for field in ("train_loss", "test_loss")):
            raise ValueError(f"{path}: losses must be finite and nonnegative")


def _validate_checkpoint(path: Path, config: dict, history: list[dict],
                         expected_parameters: int) -> dict:
    # mmap avoids copying the two large Adam states into RAM just to inspect
    # completion metadata and parameter shapes. All tensor storage stays on CPU.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported checkpoint format")
    if checkpoint.get("config") != config:
        raise ValueError(f"{path}: checkpoint config differs from config.json")
    progress = checkpoint.get("progress", {})
    for field, expected in (
        ("epoch", EXPECTED_EPOCHS), ("global_step", EXPECTED_STEPS),
        ("batch_in_epoch", 0), ("train_sample_count", 0),
        ("train_loss_sum", 0), ("last_preview_step", EXPECTED_STEPS),
    ):
        if progress.get(field) != expected:
            raise ValueError(f"{path}: partial/incomplete checkpoint: {field}={progress.get(field)!r}")
    if checkpoint.get("dataset_sizes") != [60_000, 10_000]:
        raise ValueError(f"{path}: expected the official MNIST train/test sizes")
    spec = checkpoint.get("data_spec", {})
    expected_spec = {"image_size": 28, "in_channels": 1, "num_classes": 10,
                     "value_min": -1.0, "value_max": 1.0}
    if spec != expected_spec:
        raise ValueError(f"{path}: expected MNIST pixels normalized to [-1, 1]")
    recorded_history = checkpoint.get("history", [])
    if len(recorded_history) != len(history):
        raise ValueError(f"{path}: checkpoint history differs from losses.csv")
    for stored, exported in zip(recorded_history, history):
        for field, value in exported.items():
            actual = stored.get(field)
            if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{path}: checkpoint history differs from losses.csv ({field})")
    if config["model"].get("name") != "dit":
        raise ValueError(f"{path}: expected a DiT model")
    with torch.device("meta"):
        model = DiT(image_size=28, in_channels=1, num_classes=10,
                    **config["model"].get("options", {}))
    parameter_names = dict(model.named_parameters())
    expected_state = model.state_dict()
    state = checkpoint.get("model", {})
    if set(state) != set(expected_state):
        raise ValueError(f"{path}: saved model keys do not match its configured architecture")
    for name, expected in expected_state.items():
        if not isinstance(state[name], torch.Tensor) or state[name].shape != expected.shape:
            raise ValueError(f"{path}: saved tensor shape differs from the configured model ({name})")
        tensor = state[name]
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"{path}: model tensor contains nonfinite values ({name})")
    parameters = sum(state[name].numel() for name in parameter_names)
    if parameters != expected_parameters:
        raise ValueError(f"{path}: expected {expected_parameters:,} parameters, found {parameters:,}")
    return {"parameters": parameters, "progress": dict(progress), "data_spec": spec,
            "dataset_sizes": checkpoint["dataset_sizes"], "all_floating_model_tensors_finite": True}


def _validate_runtime(root: Path, method: str, expected_parameters: int) -> tuple[dict, dict]:
    runtime_path = root / method / "runtime.json"
    preflight_path = root / f"preflight_{method}.json"
    runtime, preflight = _json(runtime_path), _json(preflight_path)
    for field, expected in (("method", method), ("status", "completed"),
                            ("epochs", EXPECTED_EPOCHS), ("global_step", EXPECTED_STEPS),
                            ("parameters", expected_parameters)):
        if runtime.get(field) != expected:
            raise ValueError(f"{runtime_path}: expected {field}={expected!r}, found {runtime.get(field)!r}")
    for field, expected in (("method", method), ("passed", True),
                            ("parameters", expected_parameters), ("real_batch_size", 64),
                            ("sampling_verified", True), ("preflight_weights_discarded", True)):
        if preflight.get(field) != expected:
            raise ValueError(f"{preflight_path}: expected {field}={expected!r}")
    digest = runtime.get("initial_weights_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{runtime_path}: invalid initial_weights_sha256")
    if preflight.get("initial_weights_sha256") != digest:
        raise ValueError(f"{runtime_path}: initial weights hash differs from its preflight record")
    if runtime.get("precision") != preflight.get("precision"):
        raise ValueError(f"{runtime_path}: precision differs from its preflight record")
    for field in ("wall_seconds", "peak_allocated_MiB", "peak_reserved_MiB"):
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{runtime_path}: {field} must be a finite positive number")
    return runtime, preflight


def _load_run(root: Path, method: str, expected_parameters: int) -> dict:
    run_dir = root / method
    config = _json(run_dir / "config.json")
    training = config["training"]
    if training.get("epochs") != EXPECTED_EPOCHS or training.get("batch_size") != 64:
        raise ValueError(f"{run_dir}: expected five epochs with a batch size of 64")
    if config.get("algorithm", {}).get("name") != method or config.get("sampler", {}).get("name") != method:
        raise ValueError(f"{run_dir}: algorithm and sampler must both be {method}")
    if config.get("dataset", {}).get("name") != "mnist":
        raise ValueError(f"{run_dir}: expected the MNIST dataset")
    sampler = config["sampler"]
    if sampler.get("options", {}).get("cfg_scale", 2.0) != 2.0:
        raise ValueError(f"{run_dir}: this comparison expects CFG scale 2")
    if method == "edm" and sampler.get("options", {}).get("num_steps", 32) != 32:
        raise ValueError(f"{run_dir}: this comparison expects 32-step Heun sampling")
    if method == "edm" and config["algorithm"].get("options", {}).get("p_mean") != 0.0:
        raise ValueError(f"{run_dir}: this comparison requires the verified EDM p_mean=0 recipe")
    runtime, preflight = _validate_runtime(root, method, expected_parameters)
    history_path = run_dir / "losses.csv"
    history = _read_history(history_path)
    _check_history(history, history_path)
    checkpoint = _validate_checkpoint(run_dir / "weights/latest.pt", config, history, expected_parameters)
    preview_path = run_dir / "previews" / f"step_{EXPECTED_STEPS:08d}.png"
    preview_metadata = _json(preview_path.with_suffix(".json"))
    if preview_metadata.get("step") != EXPECTED_STEPS:
        raise ValueError(f"{preview_path}: expected the final-step preview")
    if preview_metadata.get("seed") != training["seed"] + 2_000_000 + EXPECTED_STEPS:
        raise ValueError(f"{preview_path}: preview seed differs from the training convention")
    if preview_metadata.get("sampler") != sampler:
        raise ValueError(f"{preview_path}: preview sampler differs from config.json")
    labels, indices = preview_metadata.get("labels"), preview_metadata.get("test_indices")
    count = training.get("preview_count")
    if not isinstance(count, int) or count < 1 or not isinstance(labels, list) or not isinstance(indices, list):
        raise ValueError(f"{preview_path}: missing preview labels or test indices")
    if len(labels) != count or len(indices) != count or len(set(indices)) != count:
        raise ValueError(f"{preview_path}: inconsistent preview sample counts")
    if any(type(value) is not int or not 0 <= value < 10 for value in labels):
        raise ValueError(f"{preview_path}: invalid MNIST class labels")
    if any(type(value) is not int or not 0 <= value < 10_000 for value in indices):
        raise ValueError(f"{preview_path}: invalid MNIST test indices")
    # Exact geometry of save_preview for titled 28x28 MNIST previews. Reject
    # changed render layouts instead of silently cropping the wrong image rows.
    expected_size = (100 + count * 112 + (count - 1) * 10 + 12, 306)
    with Image.open(preview_path) as source:
        panel = source.convert("RGB")
    if panel.size != expected_size:
        raise ValueError(f"{preview_path}: unsupported preview layout {panel.size}, expected {expected_size}")
    references, generations = [], []
    for column in range(count):
        left = 100 + column * 122
        references.append(panel.crop((left, 60, left + 112, 172)))
        generations.append(panel.crop((left, 182, left + 112, 294)))
    return {"method": method, "directory": run_dir, "config": config,
            "history": history, "checkpoint": checkpoint, "metadata": preview_metadata,
            "runtime": runtime, "preflight": preflight,
            "references": references, "generations": generations,
            "source_hashes": {name: _sha256(run_dir / name) for name in (
                "config.json", "losses.csv", "runtime.json", f"../preflight_{method}.json",
                f"previews/step_{EXPECTED_STEPS:08d}.json",
                f"previews/step_{EXPECTED_STEPS:08d}.png")}}


def _validate_pair(drift: dict, edm: dict) -> None:
    first, second = drift["config"], edm["config"]
    if first["model"] != second["model"]:
        raise ValueError("The two runs have different model configurations")
    if first["dataset"] != second["dataset"]:
        raise ValueError("The two runs have different dataset configurations")
    if drift["runtime"]["initial_weights_sha256"] != edm["runtime"]["initial_weights_sha256"]:
        raise ValueError("The two runs have different initial weights hashes")
    if drift["runtime"].get("precision") != edm["runtime"].get("precision"):
        raise ValueError("The two runs have different recorded training precision")
    for field in MATCHED_TRAINING_FIELDS:
        if first["training"].get(field) != second["training"].get(field):
            raise ValueError(f"The two runs have different training.{field}")
    # Optional numerical settings must also agree when the trainer records them.
    for field in ("precision", "amp_dtype", "mixed_precision", "gradient_checkpointing"):
        if first["training"].get(field) != second["training"].get(field):
            raise ValueError(f"The two runs have different training.{field}")
    for field in ("step", "seed", "test_indices", "labels"):
        if drift["metadata"][field] != edm["metadata"][field]:
            raise ValueError(f"Final preview metadata mismatch: {field}")
    for index, (left, right) in enumerate(zip(drift["references"], edm["references"])):
        if left.tobytes() != right.tobytes():
            raise ValueError(f"Final preview reference pixels differ in column {index}")


def _plot_loss(run: dict, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    method = "Drifting" if run["method"] == "drift" else "EDM"
    title_method = method if run["method"] == "drift" else "EDM (p_mean=0)"
    figure = Figure(figsize=(9, 5.4))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    axes.plot(epochs, [row["train_loss"] for row in history], "o-", label="Train loss", color="#2456A6")
    axes.plot(epochs, [row["test_loss"] for row in history], "s-", label="Test loss", color="#D86A29")
    axes.set(xlabel="Epoch", ylabel=f"{method} objective", xticks=epochs,
             title=f"{title_method} | DiT {run['checkpoint']['parameters'] / 1e6:.2f}M | 5 epochs")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.19)
    figure.text(0.5, 0.045, LOSS_NOTE, ha="center", fontsize=9, color="#444444")
    try:
        figure.savefig(path, dpi=160)
    finally:
        figure.clear()


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _plot_samples(drift: dict, edm: dict, path: Path) -> None:
    labels = drift["metadata"]["labels"]
    width = 100 + len(labels) * 112 + (len(labels) - 1) * 10 + 12
    panel = Image.new("RGB", (width, 478), color=(235, 235, 235))
    draw = ImageDraw.Draw(panel)
    font = _font(14)
    draw.text((12, 12), f"MNIST | DiT {drift['checkpoint']['parameters'] / 1e6:.2f}M | "
              "5 epochs / 4690 updates | Same labels and initial noise", fill="black", font=font)
    for column, label in enumerate(labels):
        draw.text((100 + column * 122, 36), f"Label {label}", fill="black", font=font)
    for row, (name, images) in enumerate((("Test", drift["references"]),
                                          ("Drifting", drift["generations"]),
                                          ("EDM", edm["generations"]))):
        top = 60 + row * 122
        draw.text((12, top + 49), name, fill="black", font=font)
        for column, image in enumerate(images):
            panel.paste(image, (100 + column * 122, top))
    draw.text((12, 432), "CFG 2 | Drifting: 1 NFE | EDM (p_mean=0): 32-step Heun, "
              "63 denoiser calls / 126 branch evaluations", fill="#333333", font=_font(13))
    draw.text((12, 453), "Class-conditioned generation; test references supply labels only.",
              fill="#333333", font=_font(13))
    panel.save(path, format="PNG")


def compare_methods(root: str | Path, *, expected_parameters: int = EXPECTED_PARAMETERS) -> dict:
    """Export only after both complete runs and their shared preview are verified.

    The parameter override supports CPU fixtures using genuine smaller runs;
    the CLI always enforces the requested 57,656,848-parameter experiment.
    """
    root = Path(root).resolve()
    drift = _load_run(root, "drift", expected_parameters)
    edm = _load_run(root, "edm", expected_parameters)
    _validate_pair(drift, edm)
    for run in (drift, edm):
        _plot_loss(run, root / f"{run['method']}_loss.png")
    _plot_samples(drift, edm, root / "drift_vs_edm.png")
    summary = {
        "status": "completed_and_verified", "parameters_per_model": expected_parameters,
        "epochs": EXPECTED_EPOCHS, "optimizer_updates_per_model": EXPECTED_STEPS,
        "model": drift["config"]["model"],
        "initial_weights_sha256": drift["runtime"]["initial_weights_sha256"],
        "initial_weights_verified_against_both_preflights": True,
        "matched_training": {field: drift["config"]["training"][field] for field in MATCHED_TRAINING_FIELDS},
        "preview": {field: drift["metadata"][field] for field in ("step", "seed", "test_indices", "labels")},
        "reference_pixels_identical": True,
        "sampling": {"cfg_scale": 2.0, "drift_network_evaluations_per_image": 1,
                     "edm_heun_steps": 32, "edm_denoiser_calls": 63,
                     "edm_branch_evaluations_per_image": 126},
        "interpretation": [LOSS_NOTE,
                           "Equal parameter counts and epochs do not imply equal compute or sampling cost.",
                           "Reference images provide class labels; generated images start from Gaussian noise.",
                           "All loss points are recorded epoch means; no smoothing or interpolation is used."],
        "runs": {run["method"]: {"directory": str(run["directory"]),
                                  "checkpoint": run["checkpoint"], "history": run["history"],
                                  "runtime": run["runtime"], "preflight": run["preflight"],
                                  "algorithm": run["config"]["algorithm"],
                                  "sampler": run["config"]["sampler"],
                                  "source_sha256": run["source_hashes"]} for run in (drift, edm)},
        "artifacts": {name: str(root / name) for name in (
            "drift_loss.png", "edm_loss.png", "drift_vs_edm.png", "comparison.json")},
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "dit57m_compare")
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    summary = compare_methods(args.root)
    print(json.dumps({"status": summary["status"], "artifacts": summary["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
