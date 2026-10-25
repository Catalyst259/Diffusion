"""Run with: python -m DiTLab.train (or python train.py inside DiTLab)."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "DiTLab"

import torch
from .artifacts import load_checkpoint
from .config import ExperimentConfig, ROOT
from .engine import Trainer
from .registry import build_algorithm, build_dataset, build_model, build_sampler


def build_parser():
    parser = argparse.ArgumentParser(description="DiT + Drift / EDM on MNIST")
    parser.add_argument("--config", type=Path, help="JSON configuration (optional)")
    parser.add_argument("--resume", nargs="?", const=str(ROOT / "outputs/default/weights/latest.pt"),
                        help="Resume latest.pt, including optimizer and partial epoch")
    for flag in ("epochs", "batch-size", "preview-every", "save-every", "seed", "num-workers"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cfg-scale", type=float, help="Guidance scale for previews")
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Optional absolute update cap for a short smoke run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.resume and args.config:
        raise ValueError("--resume restores its own configuration; omit --config")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = (ExperimentConfig.from_dict(checkpoint["config"]) if checkpoint else
              ExperimentConfig.load(args.config) if args.config else ExperimentConfig())
    if checkpoint:
        # Changing these would invalidate the saved batch cursor and accumulated loss.
        for field in ("batch_size", "seed", "learning_rate"):
            value = getattr(args, field)
            if value is not None and value != getattr(config.training, field):
                raise ValueError(f"Cannot change {field} while resuming; start a new output directory")
    for field in ("epochs", "batch_size", "preview_every", "save_every", "seed", "num_workers",
                  "learning_rate", "device", "output_dir"):
        value = getattr(args, field)
        if value is not None:
            setattr(config.training, field, str(value) if isinstance(value, Path) else value)
    if args.data_dir:
        config.dataset.options["root"] = str(args.data_dir.resolve())
    if args.no_download:
        config.dataset.options["download"] = False
    if args.cfg_scale is not None:
        config.sampler.options["cfg_scale"] = args.cfg_scale
    config.validate()
    latest = Path(config.training.output_dir) / "weights/latest.pt"
    if latest.exists() and (checkpoint is None or latest.resolve() != Path(args.resume).resolve()):
        raise FileExistsError(f"Existing run: {latest}. Resume that checkpoint or use a new --output-dir.")
    torch.manual_seed(config.training.seed)
    data = build_dataset(config)
    trainer = Trainer(config, data, build_model(config, data.spec),
                      build_algorithm(config), build_sampler(config), checkpoint=checkpoint)
    trainer.run(max_steps=args.max_steps)
    print(f"Latest checkpoint: {trainer.checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
