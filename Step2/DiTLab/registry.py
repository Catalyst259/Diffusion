"""The only wiring point for model, objective, sampler and dataset implementations."""
from .models import DiT
from .algorithms import DriftAlgorithm
from .algorithms.edm import EDMAlgorithm
from .data import build_mnist


MODEL_BUILDERS = {
    "dit": lambda spec, **options: DiT(
        image_size=spec.image_size, in_channels=spec.in_channels,
        num_classes=spec.num_classes, **options,
    ),
}
ALGORITHM_BUILDERS = {"drift": DriftAlgorithm, "edm": EDMAlgorithm}
SAMPLER_BUILDERS = {"drift": DriftAlgorithm, "edm": EDMAlgorithm}
DATASET_BUILDERS = {"mnist": build_mnist}


def _factory(registry, component):
    if component.name not in registry:
        raise ValueError(f"Unknown component {component.name!r}; available: {', '.join(registry)}")
    return registry[component.name]


def build_dataset(config):
    return _factory(DATASET_BUILDERS, config.dataset)(**config.dataset.options)


def build_model(config, spec):
    return _factory(MODEL_BUILDERS, config.model)(spec, **config.model.options)


def build_algorithm(config):
    return _factory(ALGORITHM_BUILDERS, config.algorithm)(**config.algorithm.options)


def build_sampler(config):
    return _factory(SAMPLER_BUILDERS, config.sampler)(**config.sampler.options)
