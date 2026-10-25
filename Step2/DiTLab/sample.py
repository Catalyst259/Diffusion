"""Load the latest checkpoint and render a fresh held-out preview."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "DiTLab"

import torch
from .artifacts import load_checkpoint, save_json
from .config import ExperimentConfig, ROOT
from .data import DataSpec, select_test_samples
from .engine import resolve_device
from .preview import save_preview
from .registry import build_dataset, build_model, build_sampler


def load_generator(checkpoint_path, device="auto"):
    """Rebuild architecture from saved config; no dataset download is needed."""
    checkpoint = load_checkpoint(checkpoint_path)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    device = resolve_device(device)
    spec = DataSpec(**checkpoint["data_spec"])
    model = build_model(config, spec).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, build_sampler(config), config, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description="Class-conditioned preview from saved weights")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/default/weights/latest.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args(argv)
    model, sampler, config, spec = load_generator(args.checkpoint, args.device)
    if args.data_dir:
        config.dataset.options["root"] = str(args.data_dir.resolve())
    if args.no_download:
        config.dataset.options["download"] = False
    data = build_dataset(config)
    if data.spec != spec:
        raise ValueError("Dataset specification differs from checkpoint")
    reference, labels, indices = select_test_samples(data.test_dataset, args.count, seed=args.seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(reference.shape, device=device, generator=generator)
    generated = sampler.sample(model, noise, labels.to(device), cfg_scale=args.cfg_scale)
    output = args.output or args.checkpoint.resolve().parent.parent / "previews/inference.png"
    save_preview(reference, generated, labels, output, value_min=spec.value_min,
                 value_max=spec.value_max, title=f"{config.sampler.name.upper()} | class-conditioned generation")
    save_json({"seed": args.seed, "test_indices": indices, "labels": labels.tolist(),
               "cfg_scale": args.cfg_scale if args.cfg_scale is not None else
               config.sampler.options.get("cfg_scale"), "sampler": config.to_dict()["sampler"],
               "checkpoint": str(args.checkpoint.resolve())},
              output.with_suffix(".json"))
    print(f"Preview: {output}")


if __name__ == "__main__":
    main()
