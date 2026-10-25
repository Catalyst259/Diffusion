"""Measure completed Drift/EDM sampling latency on the same CUDA device."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import torch

from .artifacts import save_json
from .config import ROOT
from .sample import load_generator


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(values, batch):
    median = statistics.median(values)
    mean = statistics.mean(values)
    return {
        "batch_size": batch,
        "repeats": len(values),
        "median_batch_ms": median,
        "mean_batch_ms": mean,
        "p10_batch_ms": percentile(values, 0.1),
        "p90_batch_ms": percentile(values, 0.9),
        "amortized_median_ms_per_image": median / batch,
        "images_per_second": batch * 1000 / mean,
        "samples_ms": values,
    }


def benchmark(root, warmup=5, repeats=30):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if warmup < 1 or repeats < 2:
        raise ValueError("Use at least one warmup and two measured repetitions")
    torch.set_num_threads(4)
    root = Path(root).resolve()
    components, provenance = {}, {}
    for method in ("drift", "edm"):
        path = root / method / "weights/latest.pt"
        model, sampler, config, spec = load_generator(path, device="cuda")
        if config.model.name != "dit" or config.sampler.name != method:
            raise ValueError("Expected matched DiT Drift/EDM checkpoints")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != 57_656_848 or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Expected the completed 57.66M FP32 experiment")
        if sampler.cfg_scale != 2.0 or (method == "edm" and sampler.num_steps != 32):
            raise ValueError("Expected CFG 2 and the original 32-step EDM sampler")
        components[method] = model, sampler
        provenance[method] = {
            "checkpoint": str(path), "parameters": parameters,
            "model": config.to_dict()["model"], "sampler": config.to_dict()["sampler"],
        }
    if provenance["drift"]["model"] != provenance["edm"]["model"]:
        raise ValueError("Model configurations differ")
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    noise_pool = torch.randn(10, spec.in_channels, spec.image_size, spec.image_size,
                             generator=generator, device="cuda")
    label_pool = torch.arange(10, device="cuda", dtype=torch.long) % spec.num_classes
    results = {method: {} for method in components}
    with torch.inference_mode():
        for batch in (1, 10):
            noise, labels = noise_pool[:batch], label_pool[:batch]
            for method, (model, sampler) in components.items():
                for _ in range(warmup):
                    generated = sampler.sample(model, noise, labels)
                torch.cuda.synchronize()
                if generated.shape != noise.shape or not torch.isfinite(generated).all():
                    raise RuntimeError(f"Invalid {method} warmup output")
            timings = {method: [] for method in components}
            # Alternate order each repetition, keeping both models resident so
            # model loading and changing GPU allocator state are outside timing.
            for repeat in range(repeats):
                order = ("drift", "edm") if repeat % 2 == 0 else ("edm", "drift")
                for method in order:
                    model, sampler = components[method]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    generated = sampler.sample(model, noise, labels)
                    torch.cuda.synchronize()
                    timings[method].append((time.perf_counter() - start) * 1000)
                    if generated.shape != noise.shape or not torch.isfinite(generated).all():
                        raise RuntimeError(f"Invalid {method} measured output")
            for method in components:
                results[method][str(batch)] = summarize(timings[method], batch)
                print(json.dumps({"method": method, **{
                    key: value for key, value in results[method][str(batch)].items()
                    if key != "samples_ms"
                }}), flush=True)
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "model_precision": "float32",
        "edm_integration_precision": "float64",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(), "warmup_per_method_and_batch": warmup,
        "repeats_per_method_and_batch": repeats, "input_seed": 20261026,
        "inference_mode": True, "torch_compile": False, "autocast": False,
        "timing": "perf_counter around sampler.sample with CUDA synchronization before and after",
        "included": ["complete sampler", "model forwards", "CPU dispatch", "device synchronization"],
        "excluded": ["model loading", "input generation/transfers", "image rendering/saving", "warmup"],
        "measurement_order": "Both models resident; alternate Drift/EDM order every repetition",
        "network_evaluations": {"drift_calls": 1, "edm_batched_calls": 63,
                                "edm_branch_evaluations_per_image": 126},
        "checkpoints": provenance, "results": results,
        "drift_speedup_by_median_latency": {
            batch: results["edm"][batch]["median_batch_ms"] / results["drift"][batch]["median_batch_ms"]
            for batch in ("1", "10")
        },
    }
    save_json(report, root / "inference_benchmark.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/dit57m_compare")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    report = benchmark(args.root, args.warmup, args.repeats)
    print(json.dumps({"speedup": report["drift_speedup_by_median_latency"],
                      "output": str(args.root / "inference_benchmark.json")}, indent=2))


if __name__ == "__main__":
    main()
