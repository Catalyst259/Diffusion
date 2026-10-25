"""Compare pretrained DDPM/DDIM on 100 shared Gaussian-noise inputs.

Run from Step2: python -m UNet.src.compare_samplers
This experiment does not train or modify the checkpoint.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

if __package__:
    from .model import UNet
    from .sample import NUM_TIMESTEPS, inference
else:
    from model import UNet
    from sample import NUM_TIMESTEPS, inference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/Turn2/weights/model_epoch10_step9300.pth"
DEFAULT_OUTPUT = ROOT / "outputs/DDIMContrast"


def noise_for_groups(first, count, seed):
    """Each group's starting noise is independent of batching and global RNG."""
    return torch.stack([
        torch.randn((1, 28, 28), generator=torch.Generator().manual_seed(seed + index))
        for index in range(first, first + count)
    ])


def image_tile(pixels, scale):
    pixels = pixels.detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()
    picture = Image.fromarray(pixels)
    return picture.resize((picture.width * scale, picture.height * scale), Image.Resampling.NEAREST)


def save_comparison_gif(path, initial, ddpm, ddim, times, group, seed, duration=80):
    """Align panels by diffusion time, not by number of network evaluations."""
    scale, width, top = 6, 184, 70
    frames = []
    for index, timestep in enumerate(times):
        frame = Image.new("RGB", (width * 3, 254), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((8, 7), f"Group {group:03d} | initial seed {seed} | diffusion t={timestep}", fill="black")
        panels = (
            ("Shared initial noise", initial),
            (f"DDPM | NFE {NUM_TIMESTEPS - timestep}", ddpm[timestep]),
            (f"DDIM | NFE {index}", ddim[timestep]),
        )
        for column, (caption, pixels) in enumerate(panels):
            draw.text((column * width + 8, 38), caption, fill="black")
            frame.paste(image_tile(pixels, scale), (column * width + 8, top))
        frames.append(frame)
    durations = [duration] * len(frames)
    durations[-1] = 1200
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    return frames[-1]


def save_overview(path, ddpm, ddim):
    groups_per_row, cell_width, cell_height = 5, 128, 86
    rows = (len(ddpm) + groups_per_row - 1) // groups_per_row
    canvas = Image.new("RGB", (cell_width * groups_per_row, 34 + rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Shared noise per pair | left: DDPM | right: DDIM | display [0,1]", fill="black")
    for index, (left, right) in enumerate(zip(ddpm, ddim)):
        x, y = (index % groups_per_row) * cell_width, 34 + (index // groups_per_row) * cell_height
        draw.text((x + 4, y + 2), f"Group {index + 1:03d}", fill="black")
        canvas.paste(image_tile(left, 2), (x + 4, y + 22))
        canvas.paste(image_tile(right, 2), (x + 64, y + 22))
    canvas.save(path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    if args.groups < 1 or args.batch_size < 1:
        raise ValueError("groups and batch-size must be positive")
    if not 1 <= args.ddim_steps <= NUM_TIMESTEPS:
        raise ValueError("ddim-steps must be between 1 and 1000")
    if not 0 <= args.eta <= 1:
        raise ValueError("eta must be between 0 and 1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"An experiment already exists at {output}; choose --output")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "label_embed.weight" in state:
        raise ValueError("This comparison expects the original unconditional Turn2 checkpoint")
    base_channels = state["conv0.weight"].shape[0]
    time_dim = state["time_mlp.1.weight"].shape[1]
    model = UNet(1, 1, time_dim, base_channels=base_channels).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state
    times = torch.linspace(NUM_TIMESTEPS, 0, args.ddim_steps + 1, dtype=torch.float64).round().long().tolist()
    frame_times = set(times)
    with args.checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    summary = {
        "status": "running", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "groups_requested": args.groups, "groups_completed": 0, "seed": args.seed,
        "batch_size": args.batch_size, "device": str(device), "torch": torch.__version__,
        "ddpm_steps": NUM_TIMESTEPS, "ddim_steps": args.ddim_steps, "ddim_eta": args.eta,
        "start": "pure Gaussian noise; same input tensor for both samplers",
        "frame_alignment": "same diffusion timesteps; frames are not wall-clock animation",
        "display": "fixed [0,1] clipping, matching original ToTensor training; no sampling clipping",
        "timing": "synchronized GPU sampling including trajectory capture; excludes GIF encoding; batch-amortized seconds per image",
        "base_channels": base_channels, "time_emb_dim": time_dim,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    summary_path = output / "summary.json"

    def write_summary():
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(summary_path)

    write_summary()
    print(f"Loaded {args.checkpoint} on {device}; groups={args.groups}, DDPM=1000, DDIM={args.ddim_steps}, eta={args.eta}", flush=True)
    # Exclude initial CUDA/kernel setup from timing both algorithms.
    warmup = noise_for_groups(0, min(args.batch_size, args.groups), args.seed).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(warmup, torch.full((len(warmup),), NUM_TIMESTEPS, device=device, dtype=torch.long))
    synchronize(device)
    del warmup
    all_ddpm, all_ddim = [], []
    total_ddpm, total_ddim = 0.0, 0.0
    fields = ["group", "initial_seed", "ddpm_rng_seed", "ddim_rng_seed", "ddpm_nfe", "ddim_nfe",
              "ddpm_batch_seconds", "ddim_batch_seconds", "batch_size", "gif"]
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for first in range(0, args.groups, args.batch_size):
            count = min(args.batch_size, args.groups - first)
            initial = noise_for_groups(first, count, args.seed)
            xt = initial.to(device)
            trajectories, seconds = {}, {}
            for sampler in ("ddpm", "ddim"):
                captured = {}

                def capture(images, timesteps):
                    timestep = int(timesteps[0].item())
                    if timestep in frame_times:
                        captured[timestep] = images.detach().cpu().clone()

                rng_seed = args.seed + (100000 if sampler == "ddpm" else 200000) + first
                generator = torch.Generator(device=device).manual_seed(rng_seed)
                synchronize(device)
                start = time.perf_counter()
                result = inference(xt, NUM_TIMESTEPS, model, on_step=capture, sampler=sampler,
                                   num_steps=args.ddim_steps, eta=args.eta, generator=generator)
                synchronize(device)
                seconds[sampler] = time.perf_counter() - start
                if not torch.isfinite(result).all():
                    raise RuntimeError(f"Nonfinite {sampler} result in group {first + 1}")
                if set(captured) != frame_times:
                    raise RuntimeError(f"Unexpected {sampler} frame schedule: {sorted(captured)}")
                trajectories[sampler] = captured
                print(f"Groups {first + 1:03d}-{first + count:03d} {sampler.upper()}: {seconds[sampler]:.3f}s", flush=True)
            total_ddpm += seconds["ddpm"]
            total_ddim += seconds["ddim"]
            for offset in range(count):
                group, seed = first + offset + 1, args.seed + first + offset
                filename = f"group_{group:03d}_seed_{seed}.gif"
                ddpm = {t: v[offset, 0] for t, v in trajectories["ddpm"].items()}
                ddim = {t: v[offset, 0] for t, v in trajectories["ddim"].items()}
                save_comparison_gif(output / filename, initial[offset, 0], ddpm, ddim, times, group, seed)
                all_ddpm.append(ddpm[0])
                all_ddim.append(ddim[0])
                writer.writerow(dict(zip(fields, [group, seed, args.seed + 100000 + first,
                    args.seed + 200000 + first, NUM_TIMESTEPS, args.ddim_steps,
                    seconds["ddpm"], seconds["ddim"], count, filename])))
            handle.flush()
            summary.update(groups_completed=first + count, ddpm_seconds=total_ddpm, ddim_seconds=total_ddim)
            write_summary()
            print(f"Saved {first + count}/{args.groups} comparison GIFs", flush=True)
    save_overview(output / "final_comparison.png", all_ddpm, all_ddim)
    torch.save({"ddpm": torch.stack(all_ddpm), "ddim": torch.stack(all_ddim)}, output / "final_samples.pt")
    summary.update(status="complete", speedup=total_ddpm / total_ddim,
                   ddpm_seconds_per_image=total_ddpm / args.groups,
                   ddim_seconds_per_image=total_ddim / args.groups)
    write_summary()
    print(f"Complete: {args.groups} GIFs, speedup {summary['speedup']:.2f}x; {output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--groups", type=int, default=100)
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--ddim-steps", type=int, default=50)
    result.add_argument("--eta", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default=None)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
