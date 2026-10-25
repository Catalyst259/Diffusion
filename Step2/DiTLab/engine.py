"""Training lifecycle; knows the component contracts, not their algorithms."""
from dataclasses import asdict
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from .artifacts import atomic_checkpoint, save_history, save_json
from .data import select_test_samples
from .preview import save_preview


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


class Trainer:
    def __init__(self, config, data, model, algorithm, sampler, *, checkpoint=None):
        config.validate()
        self.config, self.data = config, data
        self.settings = config.training
        self.device = resolve_device(self.settings.device)
        self.model = model.to(self.device)
        self.algorithm, self.sampler = algorithm, sampler
        if not len(data.train_dataset) or not len(data.test_dataset):
            raise ValueError("Training and test datasets must be nonempty")
        if len(data.test_dataset) < self.settings.preview_count:
            raise ValueError("Test dataset is smaller than preview_count")
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        self.output_dir = Path(self.settings.output_dir).resolve()
        self.checkpoint_path = self.output_dir / "weights" / "latest.pt"
        self.generator = torch.Generator(device=self.device).manual_seed(self.settings.seed)
        self.epoch = 0  # Number of fully evaluated epochs; current epoch is zero-based.
        self.batch_in_epoch = 0
        self.global_step = 0
        self.train_loss_sum = 0.0
        self.train_sample_count = 0
        self.history = []
        self.last_preview_step = -1
        if checkpoint is not None:
            self._restore(checkpoint)

    def _restore(self, checkpoint):
        if checkpoint["rng"]["device_type"] != self.device.type:
            raise ValueError("Resume training on the original device type (cpu/cuda) to restore RNG; "
                             "sample.py supports loading weights on either device")
        if checkpoint["data_spec"] != asdict(self.data.spec):
            raise ValueError("Checkpoint dataset shape/classes do not match")
        if checkpoint["dataset_sizes"] != [len(self.data.train_dataset), len(self.data.test_dataset)]:
            raise ValueError("Checkpoint dataset sizes do not match")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for name, value in checkpoint["progress"].items():
            setattr(self, name, value)
        self.history = checkpoint["history"]
        self.generator.set_state(checkpoint["rng"]["algorithm"])
        torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
        cuda_states = checkpoint["rng"].get("torch_cuda")
        if self.device.type == "cuda" and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)

    def save(self):
        payload = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "data_spec": asdict(self.data.spec),
            "dataset_sizes": [len(self.data.train_dataset), len(self.data.test_dataset)],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "progress": {name: getattr(self, name) for name in (
                "epoch", "batch_in_epoch", "global_step", "train_loss_sum",
                "train_sample_count", "last_preview_step",
            )},
            "history": self.history,
            "rng": {
                "algorithm": self.generator.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                "device_type": self.device.type,
            },
        }
        atomic_checkpoint(payload, self.checkpoint_path)

    def _loader(self, dataset, seed):
        return DataLoader(
            dataset, batch_size=self.settings.batch_size, shuffle=False,
            num_workers=self.settings.num_workers, pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
        )

    def _training_loader(self):
        # Reconstruct the exact epoch permutation without advancing the model RNG.
        order = torch.randperm(
            len(self.data.train_dataset),
            generator=torch.Generator().manual_seed(self.settings.seed + self.epoch),
        ).tolist()
        offset = self.batch_in_epoch * self.settings.batch_size
        return self._loader(Subset(self.data.train_dataset, order[offset:]), self.settings.seed + self.epoch)

    @torch.no_grad()
    def evaluate(self):
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed + 1000000)
        total, count = 0.0, 0
        # Also isolate global RNG for future models with stochastic evaluation code.
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                for images, labels in self._loader(self.data.test_dataset, self.settings.seed):
                    images, labels = images.to(self.device), labels.to(self.device)
                    loss = self.algorithm.loss(self.model, images, labels, generator=generator, training=False)
                    value = float(loss.item())
                    if not math.isfinite(value):
                        raise FloatingPointError("Non-finite test loss")
                    total += value * len(images)
                    count += len(images)
        finally:
            self.model.train(was_training)
        return total / count

    def preview(self):
        seed = self.settings.seed + 2000000 + self.global_step
        references, labels, indices = select_test_samples(
            self.data.test_dataset, self.settings.preview_count, seed=seed,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(references.shape, device=self.device, generator=generator)
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            generated = self.sampler.sample(self.model, noise, labels.to(self.device))
        path = self.output_dir / "previews" / f"step_{self.global_step:08d}.png"
        save_preview(
            references, generated, labels, path,
            value_min=self.data.spec.value_min, value_max=self.data.spec.value_max,
            title=f"Step {self.global_step} | class-conditioned generation",
        )
        save_json({"step": self.global_step, "seed": seed, "test_indices": indices,
                   "labels": labels.tolist(), "sampler": self.config.to_dict()["sampler"]},
                  path.with_suffix(".json"))
        self.last_preview_step = self.global_step
        print(f"Preview: {path}", flush=True)

    def run(self, *, max_steps=None):
        """Run to the configured total epochs. max_steps is an absolute smoke-test cap."""
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.config.to_dict(), self.output_dir / "config.json")
        if self.history:
            # A periodic checkpoint is authoritative after an interrupted epoch.
            save_history(self.history, self.output_dir)
        print(f"Device: {self.device}; parameters: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)
        batches_per_epoch = math.ceil(len(self.data.train_dataset) / self.settings.batch_size)
        self.model.train()
        while self.epoch < self.settings.epochs:
            if max_steps is not None and self.global_step >= max_steps:
                break
            for images, labels in self._training_loader():
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.algorithm.loss(self.model, images, labels, generator=self.generator, training=True)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("Training algorithm must return a finite scalar loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_grad_norm,
                                              error_if_nonfinite=True)
                self.optimizer.step()
                self.global_step += 1
                self.batch_in_epoch += 1
                value = float(loss.item())
                self.train_loss_sum += value * len(images)
                self.train_sample_count += len(images)
                if self.global_step % self.settings.log_every == 0 or self.global_step == 1:
                    print(f"Epoch {self.epoch + 1}/{self.settings.epochs} | "
                          f"batch {self.batch_in_epoch}/{batches_per_epoch} | "
                          f"step {self.global_step} | loss {value:.6f}", flush=True)
                if self.global_step % self.settings.preview_every == 0:
                    self.preview()
                if self.global_step % self.settings.save_every == 0:
                    self.save()
                if max_steps is not None and self.global_step >= max_steps:
                    break
            if self.batch_in_epoch == batches_per_epoch:
                test_loss = self.evaluate()
                train_loss = self.train_loss_sum / self.train_sample_count
                self.epoch += 1
                self.history.append({"epoch": self.epoch, "global_step": self.global_step,
                                     "train_loss": train_loss, "test_loss": test_loss})
                self.batch_in_epoch, self.train_loss_sum, self.train_sample_count = 0, 0.0, 0
                save_history(self.history, self.output_dir)
                self.save()
                print(f"Epoch {self.epoch}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}", flush=True)
            if max_steps is not None and self.global_step >= max_steps:
                break
        # Even a short run gets a usable checkpoint and a visible result.
        if self.global_step and self.last_preview_step != self.global_step:
            self.preview()
        self.save()
        return self.history
