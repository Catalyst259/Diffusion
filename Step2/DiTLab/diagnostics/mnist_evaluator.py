"""Reproducible CPU classifier for MNIST generation diagnostics.

This measures agreement between generated-image predictions and requested digit
labels. It is a diagnostic proxy, NOT FID, a perceptual-quality measure, or proof
that a generated image is a valid digit. The classifier can be confidently wrong
on out-of-distribution images. Only the official MNIST training split is read:
a fixed permutation assigns 50,000 examples to training and 10,000 to validation.
The official test split is never loaded or used to choose hyperparameters.

Run once from any directory:
    python -m DiTLab.diagnostics.mnist_evaluator

The two-epoch final checkpoint is retained without validation-based model or
hyperparameter selection. Existing checkpoints are protected unless --overwrite
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data"
DEFAULT_CHECKPOINT = WORKSPACE / "DiTLab" / "outputs" / "edm_diagnosis" / "evaluator.pt"
ARCHITECTURE = "mnist_cnn_32_64_fc128_v1"


class MNISTEvaluator(nn.Module):
    """Small digit classifier accepting float images in the [0, 1] range."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features((images - 0.1307) / 0.3081))


def _summary(predictions: Tensor, labels: Tensor, confidence: Tensor) -> dict:
    predictions, labels, confidence = predictions.cpu(), labels.cpu(), confidence.cpu()
    confusion = torch.bincount(10 * labels + predictions, minlength=100).reshape(10, 10)
    counts = confusion.sum(dim=1)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "count": labels.numel(),
        "mean_confidence": float(confidence.mean()),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=requested/true digit; columns=predicted digit",
        "per_class_accuracy": {
            str(digit): float(confusion[digit, digit] / counts[digit]) if counts[digit] else None
            for digit in range(10)
        },
        "predicted_class_counts": confusion.sum(dim=0).tolist(),
    }


@torch.no_grad()
def predict_accuracy(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    *,
    batch_size: int = 256,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> dict:
    """Return JSON-serializable label agreement, predictions, and confidence.

    ``images`` has shape [B, 1, 28, 28]; ``labels`` contains the requested class
    for each generated image. Map the declared value range to [0, 1] and clamp,
    matching displayed previews. The fraction of clamped pixels is reported.
    This inference routine does not sample randomness or alter model modes.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or not len(images):
        raise ValueError("images must have nonempty shape [B, 1, 28, 28]")
    if labels.shape != (len(images),) or labels.dtype != torch.long:
        raise ValueError("labels must be torch.long with shape [B]")
    if (labels < 0).any() or (labels > 9).any():
        raise ValueError("labels must be digit IDs in [0, 9]")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min >= value_max:
        raise ValueError("value_min and value_max must be finite and strictly increasing")
    if not torch.isfinite(images).all():
        raise ValueError("images contain non-finite values")
    previous_modes = [(module, module.training) for module in model.modules()]
    model_device = next(model.parameters()).device
    predictions, confidence = [], []
    try:
        model.eval()
        for batch in images.split(batch_size):
            unit_images = ((batch.detach().float().to(model_device) - value_min) / (value_max - value_min)).clamp(0, 1)
            probabilities = model(unit_images).softmax(dim=1)
            batch_confidence, batch_predictions = probabilities.max(dim=1)
            predictions.append(batch_predictions.cpu())
            confidence.append(batch_confidence.cpu())
    finally:
        for module, mode in previous_modes:
            module.training = mode
    predictions_tensor, confidence_tensor = torch.cat(predictions), torch.cat(confidence)
    result = _summary(predictions_tensor, labels, confidence_tensor)
    result.update({
        "metric": "mnist_classifier_label_consistency",
        "is_fid": False,
        "predicted_labels": predictions_tensor.tolist(),
        "confidence": confidence_tensor.tolist(),
        "input_value_range": [value_min, value_max],
        "observed_value_range": [float(images.min()), float(images.max())],
        "clipped_pixel_fraction": float(((images < value_min) | (images > value_max)).float().mean()),
    })
    return result


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    predictions, labels, confidence = [], [], []
    loss_sum = 0.0
    for image_batch, label_batch in loader:
        logits = model(image_batch.float() / 255)
        loss_sum += float(F.cross_entropy(logits, label_batch, reduction="sum"))
        batch_confidence, batch_predictions = logits.softmax(dim=1).max(dim=1)
        predictions.append(batch_predictions)
        labels.append(label_batch)
        confidence.append(batch_confidence)
    result = _summary(torch.cat(predictions), torch.cat(labels), torch.cat(confidence))
    result["cross_entropy"] = loss_sum / result["count"]
    return result


def load_evaluator(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device = "cpu",
) -> MNISTEvaluator:
    """Load the fixed evaluator; attach saved metadata as ``evaluator_metadata``."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported MNIST evaluator checkpoint architecture")
    # Initial layer allocations must not consume the caller's sampling RNG.
    with torch.random.fork_rng(devices=[]):
        model = MNISTEvaluator()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    model.evaluator_metadata = checkpoint["metadata"]
    return model


def train_evaluator(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    epochs: int = 2,
    seed: int = 1729,
    batch_size: int = 128,
    overwrite: bool = False,
) -> tuple[MNISTEvaluator, dict]:
    """Train once on CPU, save final weights plus ``evaluator_validation.json``.

    No downloads, GPU operations, test-set reads, or adaptive hyperparameter
    search occur. Fixed Adam settings: learning_rate=0.001, weight_decay=0.
    """
    from torchvision.datasets import MNIST

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluator already exists: {checkpoint_path}; load it or explicitly request overwrite")
    torch.set_num_threads(4)
    dataset = MNIST(str(data_root), train=True, download=False)
    if len(dataset) != 60000:
        raise ValueError("Expected the original 60,000-example MNIST training split")
    tensors = TensorDataset(dataset.data.unsqueeze(1), dataset.targets.long())
    permutation = torch.randperm(len(tensors), generator=torch.Generator().manual_seed(seed))
    train_indices, validation_indices = permutation[:50000], permutation[50000:]
    train_loader = DataLoader(
        Subset(tensors, train_indices.tolist()), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1), num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(tensors, validation_indices.tolist()), batch_size=256, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2), num_workers=0,
    )
    started = time.monotonic()
    history = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = MNISTEvaluator()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum, train_correct, train_count = 0.0, 0, 0
            epoch_started = time.monotonic()
            for batch_index, (images, labels) in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.float() / 255)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(labels)
                train_correct += int((logits.detach().argmax(dim=1) == labels).sum())
                train_count += len(labels)
                if batch_index % 100 == 0:
                    print(f"Evaluator epoch {epoch}/{epochs} batch {batch_index}/{len(train_loader)} "
                          f"train_accuracy={train_correct / train_count:.4f}", flush=True)
            validation = _validate(model, validation_loader)
            record = {
                "epoch": epoch,
                "training_cross_entropy": train_loss_sum / train_count,
                "training_accuracy": train_correct / train_count,
                "validation": validation,
                "elapsed_seconds": time.monotonic() - epoch_started,
            }
            history.append(record)
            print(f"Evaluator epoch {epoch}/{epochs}: validation_accuracy={validation['accuracy']:.4%}, "
                  f"validation_cross_entropy={validation['cross_entropy']:.6f}", flush=True)

    metadata = {
        "purpose": "MNIST generated-label consistency diagnostic proxy; not FID or a perceptual-quality metric",
        "architecture": ARCHITECTURE,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "data_root": str(Path(data_root).resolve()),
        "source_split": "official MNIST training split only",
        "training_count": 50000,
        "validation_count": 10000,
        "official_test_used": False,
        "split_method": "torch.randperm(60000, generator=CPU seed): first 50000 train, last 10000 validation",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0},
        "normalization": {"input_range": [0.0, 1.0], "mean": 0.1307, "std": 0.3081},
        "device": "cpu",
        "cpu_threads": 4,
        "torch_version": str(torch.__version__),
        "checkpoint_selection": "final epoch, no validation-based model/hyperparameter selection",
        "target_validation_accuracy": 0.98,
        "target_met": history[-1]["validation"]["accuracy"] >= 0.98,
        "history": history,
        "final_validation": history[-1]["validation"],
        "elapsed_seconds": time.monotonic() - started,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "model": model.state_dict(),
        "metadata": metadata,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }, checkpoint_path)
    validation_path = checkpoint_path.with_name("evaluator_validation.json")
    validation_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model.eval()
    model.evaluator_metadata = metadata
    print(f"Saved evaluator: {checkpoint_path}\nValidation report: {validation_path}", flush=True)
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_evaluator(args.data_root, args.checkpoint, epochs=args.epochs, seed=args.seed,
                    batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
