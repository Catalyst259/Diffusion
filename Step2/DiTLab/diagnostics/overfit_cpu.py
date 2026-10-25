"""Small CPU-only fixed-noise capacity probe; never alters production modules."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from DiTLab.data import build_mnist
from DiTLab.models.dit import DiT
from DiTLab.preview import save_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resample-noise", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("DiTLab/outputs/edm_diagnosis/overfit_cpu/fixed_noise"))
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4:
        raise ValueError("Keep this CPU diagnosis batch at most four")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = DiT()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    data = build_mnist("../data", download=False).train_dataset
    clean = torch.stack([data[index][0] for index in range(args.batch_size)])
    labels = torch.tensor([data[index][1] for index in range(args.batch_size)])
    generator = torch.Generator().manual_seed(731)
    noise = torch.randn(clean.shape, generator=generator)
    fixed_noise = noise.clone()
    sigma_data = 0.5
    sigma = args.sigma
    magnitude = (sigma ** 2 + sigma_data ** 2) ** 0.5
    a, b = sigma / magnitude, sigma_data / magnitude
    levels = torch.full((args.batch_size,), sigma).log() / 4
    args.output.mkdir(parents=True, exist_ok=True)
    trace = []
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if args.resample_noise and step:
            noise = torch.randn(clean.shape, generator=generator)
        network_input = clean * (b / sigma_data) + noise * a
        target = clean * (a / sigma_data) - noise * b
        prediction = model.forward_denoise(network_input, labels, levels)
        loss = (prediction - target).square().mean()
        with torch.no_grad():
            denoised = b ** 2 * (clean + sigma * noise) + sigma_data * a * prediction
            reconstruction = float((denoised - clean).square().mean())
        if step % 10 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss.detach()), "reconstruction_mse": reconstruction}
            trace.append(row)
            print(json.dumps(row), flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        reference_input = clean * (b / sigma_data) + fixed_noise * a
        prediction = model.forward_denoise(reference_input, labels, levels)
        denoised = b ** 2 * (clean + sigma * fixed_noise) + sigma_data * a * prediction
    save_preview(clean, denoised, labels, args.output / "reconstruction.png", value_min=-1, value_max=1,
                 title=f"CPU capacity test | sigma {sigma:g} | {args.steps} updates | lr {args.lr:g}")
    summary = {"steps": args.steps, "sigma": sigma, "lr": args.lr,
               "batch_size": args.batch_size, "resample_noise": args.resample_noise,
               "labels": labels.tolist(), "elapsed_seconds": time.perf_counter() - start,
               "initial_reconstruction_mse": trace[0]["reconstruction_mse"],
               "final_reconstruction_mse": float((denoised-clean).square().mean()),
               "final_per_image_mse": (denoised-clean).square().mean((1,2,3)).tolist()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    torch.save({"clean": clean, "denoised": denoised, "labels": labels, "fixed_noise": fixed_noise,
                "model": model.state_dict(), "summary": summary}, args.output / "probe.pt")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
