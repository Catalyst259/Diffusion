"""Serializable experiment settings; paths are independent of the launch directory."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


ROOT = Path(__file__).resolve().parent


@dataclass
class ComponentConfig:
    name: str
    options: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 2.0
    preview_every: int = 500
    save_every: int = 500
    preview_count: int = 10
    log_every: int = 50
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    output_dir: str = str(ROOT / "outputs" / "default")


@dataclass
class ExperimentConfig:
    model: ComponentConfig = field(default_factory=lambda: ComponentConfig("dit", {
        "patch_size": 4, "hidden_size": 192, "depth": 4, "num_heads": 6,
    }))
    algorithm: ComponentConfig = field(default_factory=lambda: ComponentConfig("drift", {
        "temperatures": [0.02, 0.05, 0.2], "generated_per_class": 8,
        "cfg_min": 1.0, "cfg_max": 4.0,
    }))
    sampler: ComponentConfig = field(default_factory=lambda: ComponentConfig("drift", {
        "cfg_scale": 2.0,
    }))
    dataset: ComponentConfig = field(default_factory=lambda: ComponentConfig("mnist", {
        "root": str(ROOT / "data"), "download": True,
    }))
    training: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        t = self.training
        for name in ("epochs", "batch_size", "preview_every", "save_every", "preview_count", "log_every"):
            value = getattr(t, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if t.num_workers < 0 or t.seed < 0:
            raise ValueError("num_workers and seed must be nonnegative")
        if not math.isfinite(t.learning_rate) or t.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(t.max_grad_norm) or t.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(t.weight_decay) or t.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        config = cls()
        unknown = set(values) - {"model", "algorithm", "sampler", "dataset", "training"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        for name in ("model", "algorithm", "sampler", "dataset"):
            if name in values:
                section = values[name]
                previous = getattr(config, name)
                component_name = section.get("name", previous.name)
                options = dict(previous.options) if component_name == previous.name else {}
                options.update(section.get("options", {}))
                setattr(config, name, ComponentConfig(component_name, options))
        if "training" in values:
            config.training = TrainConfig(**{**asdict(config.training), **values["training"]})
        config.validate()
        return config

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
