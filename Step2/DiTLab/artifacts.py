"""Local experiment artifacts, with a single atomically replaced checkpoint."""
import csv
import json
import os
from pathlib import Path
import torch


def atomic_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    # Checkpoints contain only tensors and primitive containers, no pickled model objects.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported DiTLab checkpoint format")
    return payload


def save_json(values, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_history(history, output_dir):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "losses.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "global_step", "train_loss", "test_loss"))
        writer.writeheader()
        writer.writerows(history)
    figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    epochs = [entry["epoch"] for entry in history]
    axes.plot(epochs, [entry["train_loss"] for entry in history], "o-", label="Train loss")
    axes.plot(epochs, [entry["test_loss"] for entry in history], "s-", label="Test loss")
    axes.set(xlabel="Epoch", ylabel="Training objective", title="Train and Test Loss")
    axes.set_xticks(epochs)
    axes.grid(alpha=0.3)
    axes.legend()
    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(graph_dir / "loss_curve.png", dpi=150)
    figure.clear()
