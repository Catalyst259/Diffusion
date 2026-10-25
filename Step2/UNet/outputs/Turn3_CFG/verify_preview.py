"""Verify the completed run and compare CFG with shared noise and labels."""
from pathlib import Path
import csv
import json
import sys

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from UNet.src.model import UNet
from UNet.src.sample import inference


def main():
    output = Path(__file__).resolve().parent
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    with (output / "loss.csv").open(newline="", encoding="utf-8") as handle:
        losses = list(csv.DictReader(handle))
    assert completed["epochs"] == config["num_epochs"] == len(losses) == 5
    assert completed["global_step"] == 4690
    assert config["class_dropout_prob"] == 0.1

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        config["in_channels"], config["out_channels"],
        time_emb_dim=config["time_emb_dim"], base_channels=config["base_channels"],
        num_classes=config["num_classes"], class_dropout_prob=config["class_dropout_prob"],
    )
    state = torch.load(output / "weights" / "model_final.pth", map_location="cpu", weights_only=True)
    assert all(torch.isfinite(value).all().item() for value in state.values())
    assert tuple(state["label_embed.weight"].shape) == (11, 256)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device).eval()

    seed = 20261025
    scales = [1.0, 2.0, 3.0]
    generator = torch.Generator(device=device).manual_seed(seed)
    base_noise = torch.randn(2, 1, 28, 28, generator=generator, device=device)
    noise = base_noise.repeat_interleave(10, dim=0)
    labels = torch.arange(10, device=device).repeat(2)
    results = []
    for scale in scales:
        samples = inference(noise, 1000, model, c=labels, sampler="ddim",
                            num_steps=50, eta=0.0, cfg_scale=scale)
        assert torch.isfinite(samples).all().item()
        results.append(samples.cpu().reshape(2, 10, 1, 28, 28))
        print(f"CFG {scale:g}: complete", flush=True)
    samples = torch.stack(results)
    torch.save({"images": samples, "base_noise": base_noise.cpu(), "labels": labels.cpu(),
                "cfg_scales": scales, "seed": seed, "sampler": "ddim", "num_steps": 50,
                "eta": 0.0}, output / "cfg_samples.pt")

    figure = Figure(figsize=(13, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(6, 10)
    for base in range(2):
        for scale_index, scale in enumerate(scales):
            row = base * len(scales) + scale_index
            for label in range(10):
                ax = axes[row, label]
                ax.imshow(samples[scale_index, base, label, 0].clamp(0, 1).numpy(),
                          cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"c = {label}")
                if label == 0:
                    ax.set_ylabel(f"Noise {base + 1}\nCFG {scale:g}", rotation=0,
                                  ha="right", va="center", labelpad=10)
    figure.suptitle("Turn3_CFG | DDIM 50, eta=0 | shared noise across labels and CFG scales")
    figure.savefig(output / "cfg_preview.png", dpi=150)
    figure.clear()

    verification = {
        "checkpoint": "weights/model_final.pth", "strict_load": True,
        "all_weights_finite": True, "label_embedding_shape": [11, 256],
        "null_label": model.null_label, "class_dropout_prob": model.class_dropout_prob,
        "epochs": completed["epochs"], "global_step": completed["global_step"],
        "final_train_loss": float(losses[-1]["train_loss"]),
        "final_test_loss": float(losses[-1]["test_loss"]),
        "preview": {"file": "cfg_preview.png", "seed": seed, "sampler": "ddim",
                    "steps": 50, "eta": 0.0, "cfg_scales": scales,
                    "base_noise_count": 2, "labels_per_noise": list(range(10)),
                    "all_samples_finite": True, "display_range": [0, 1],
                    "raw_min": samples.min().item(), "raw_max": samples.max().item()},
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
