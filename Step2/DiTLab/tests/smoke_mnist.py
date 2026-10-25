"""Small real-MNIST/CUDA integration run; explicitly not a full experiment.

From Step2: python -m DiTLab.tests.smoke_mnist --data-dir ../data
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import Subset

from DiTLab.config import ExperimentConfig, ROOT
from DiTLab.data import DataBundle
from DiTLab.engine import Trainer
from DiTLab.registry import build_algorithm, build_dataset, build_model, build_sampler
from DiTLab.sample import load_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke_mnist")
    args = parser.parse_args()
    if (args.output_dir / "weights/latest.pt").exists():
        raise FileExistsError("Choose a new --output-dir for this smoke test")
    config = ExperimentConfig()
    config.dataset.options.update(root=str(args.data_dir.resolve()), download=False)
    config.training.output_dir = str(args.output_dir.resolve())
    config.training.epochs = 2
    config.training.preview_every = config.training.save_every = 2
    config.training.log_every = 1
    torch.manual_seed(config.training.seed)
    full_data = build_dataset(config)
    data = DataBundle(Subset(full_data.train_dataset, range(128)),
                      Subset(full_data.test_dataset, range(40)), full_data.spec)
    trainer = Trainer(config, data, build_model(config, data.spec),
                      build_algorithm(config), build_sampler(config))
    trainer.run()
    model, sampler, _, spec = load_generator(trainer.checkpoint_path)
    device = next(model.parameters()).device
    labels = torch.arange(10, device=device)
    with torch.no_grad():
        images = sampler.sample(model, torch.randn(10, spec.in_channels, spec.image_size,
                                                  spec.image_size, device=device), labels)
    assert images.shape == (10, 1, 28, 28) and torch.isfinite(images).all()
    assert len(list((args.output_dir / "weights").glob("*.pt"))) == 1
    assert len(trainer.history) == 2
    print("Smoke passed: 128 real train / 40 real test images, 4 updates, default DiT, reload and sampling.")


if __name__ == "__main__":
    main()
