"""Sequential independent training runs, resumable as a batch, with comparison plots."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "DiTLab"

from .artifacts import load_checkpoint, save_json
from .comparison import summarize_runs
from .config import ExperimentConfig, ROOT


def _save_manifest(manifest, root):
    path = Path(root) / "batch.json"
    temporary = path.with_suffix(".json.tmp")
    save_json(manifest, temporary)
    os.replace(temporary, path)


def create_plan(config, output_dir, *, runs=20, seed_start=None):
    """Freeze settings for all runs before training; never overwrite an existing batch."""
    config.validate()
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    seed_start = config.training.seed if seed_start is None else seed_start
    if not isinstance(seed_start, int) or seed_start < 0 or seed_start + runs > 2**63:
        raise ValueError("Seeds must be nonnegative integers below 2**63")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty batch directory: {root}. Use --resume or a new --output-dir.")
    base = config.to_dict()
    if "root" in base["dataset"]["options"]:
        base["dataset"]["options"]["root"] = str(Path(base["dataset"]["options"]["root"]).resolve())
    manifest = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(root), "base_config": base, "runs": []}
    for index in range(runs):
        seed = seed_start + index
        run_id = f"run_{index + 1:03d}_seed{seed}"
        run_config = deepcopy(base)
        run_dir = root / run_id
        run_config["training"].update(seed=seed, output_dir=str(run_dir))
        config_path = root / "configs" / f"{run_id}.json"
        manifest["runs"].append({
            "run_id": run_id, "seed": seed, "epochs": config.training.epochs,
            "output_dir": str(run_dir), "config_path": str(config_path),
            "config": run_config, "status": "pending",
        })
    root.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        save_json(run["config"], run["config_path"])
    _save_manifest(manifest, root)
    return manifest


def load_plan(output_dir):
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "batch.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not manifest.get("runs"):
        raise ValueError("Invalid batch manifest")
    if Path(manifest["output_dir"]).resolve() != root:
        raise ValueError("Resume a batch from its original output directory")
    # Guard against accidental hand-edited paths pointing at another experiment.
    for run in manifest["runs"]:
        if not isinstance(run["run_id"], str) or Path(run["run_id"]).name != run["run_id"] or run["run_id"] in (".", ".."):
            raise ValueError("run_id must be a simple directory name")
        if Path(run["output_dir"]).resolve() != root / run["run_id"]:
            raise ValueError("Run directory does not match the batch manifest")
        if Path(run["config_path"]).resolve() != root / "configs" / f"{run['run_id']}.json":
            raise ValueError("Run configuration path does not match the batch manifest")
        settings = ExperimentConfig.from_dict(run["config"]).training
        if (Path(settings.output_dir).resolve() != Path(run["output_dir"]).resolve()
                or settings.seed != run["seed"] or settings.epochs != run["epochs"]):
            raise ValueError("Run settings disagree with the batch manifest")
    return manifest


def _checkpoint_progress(run):
    checkpoint_path = Path(run["output_dir"]) / "weights/latest.pt"
    if not checkpoint_path.exists():
        return None
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["config"] != run["config"]:
        raise ValueError(f"{run['run_id']}: checkpoint settings differ from the frozen batch plan")
    return checkpoint["progress"]


def _execute_training(command, log_path):
    """One child at a time, using the currently activated Python environment."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- {datetime.now(timezone.utc).isoformat()} ---\n")
        log.write(json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT.parent, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            # Stop only the child created by this batch runner, never other jobs.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def run_batch(manifest, *, max_steps=None):
    """Resume pending runs, skip completed checkpoints, stop visibly on any failure."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    root = Path(manifest["output_dir"])
    for index, run in enumerate(manifest["runs"], start=1):
        try:
            progress = _checkpoint_progress(run)
            if progress is not None and progress["epoch"] >= run["epochs"]:
                run["status"] = "completed"
                run.pop("error", None)
                _save_manifest(manifest, root)
                print(f"[{index}/{len(manifest['runs'])}] {run['run_id']}: already completed", flush=True)
                continue
            if progress is None and run["status"] == "completed":
                raise FileNotFoundError(f"Completed run lost its checkpoint: {run['run_id']}")
            # Regenerate snapshots from the frozen manifest; later config.py edits
            # must not silently change the remaining runs in the comparison.
            save_json(run["config"], run["config_path"])
            command = [sys.executable, "-u", "-m", "DiTLab.train"]
            if progress is None:
                command += ["--config", run["config_path"]]
            else:
                command += ["--resume", str(Path(run["output_dir"]) / "weights/latest.pt")]
            if max_steps is not None:
                command += ["--max-steps", str(max_steps)]
            run["status"] = "running"
            run.pop("error", None)
            _save_manifest(manifest, root)
            print(f"[{index}/{len(manifest['runs'])}] {run['run_id']} | "
                  f"epochs={run['epochs']} | lr={run['config']['training']['learning_rate']}", flush=True)
            code = _execute_training(command, Path(run["output_dir"]) / "train.log")
            if code:
                raise RuntimeError(f"Training exited with code {code}; see {run['output_dir']}/train.log")
            progress = _checkpoint_progress(run)
            if progress is None:
                raise RuntimeError("Training returned without a checkpoint")
            run["status"] = "completed" if progress["epoch"] >= run["epochs"] else "incomplete"
            if run["status"] != "completed" and max_steps is None:
                raise RuntimeError("Training returned before the configured epochs were completed")
        except KeyboardInterrupt:
            run["status"] = "interrupted"
            _save_manifest(manifest, root)
            summarize_runs(manifest["runs"], root)
            print("Batch interrupted. Repeat with --resume to continue from the latest checkpoint.", flush=True)
            raise
        except Exception as error:
            run["status"] = "failed"
            run["error"] = str(error)
            _save_manifest(manifest, root)
            summarize_runs(manifest["runs"], root)
            raise
        _save_manifest(manifest, root)
        summarize_runs(manifest["runs"], root)
    return summarize_runs(manifest["runs"], root)


def build_parser():
    parser = argparse.ArgumentParser(description="Train independent seeds sequentially and compare loss curves")
    parser.add_argument("--runs", type=int, help="Independent runs (default: 20)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/batch20")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed-start", type=int, help="First seed; subsequent runs use consecutive seeds")
    for flag in ("epochs", "batch-size", "num-workers", "preview-every", "save-every"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue the saved batch plan, skipping completed runs")
    parser.add_argument("--summary-only", action="store_true", help="Refresh comparisons without starting training")
    parser.add_argument("--plan-only", action="store_true", help="Save/inspect the batch plan without training")
    parser.add_argument("--max-steps", type=int, help="Smoke test only: absolute update cap per run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summary_only and args.plan_only:
        raise ValueError("Choose --summary-only or --plan-only")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.resume or args.summary_only:
        changes = ("runs", "config", "seed_start", "epochs", "batch_size", "num_workers",
                   "preview_every", "save_every", "learning_rate", "device", "data_dir")
        if any(getattr(args, field) is not None for field in changes) or args.no_download:
            raise ValueError("An existing batch uses frozen settings; omit training overrides or choose a new output directory")
        manifest = load_plan(args.output_dir)
    else:
        config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        for field in ("epochs", "batch_size", "num_workers", "preview_every", "save_every", "learning_rate", "device"):
            value = getattr(args, field)
            if value is not None:
                setattr(config.training, field, value)
        if args.data_dir:
            config.dataset.options["root"] = str(args.data_dir.resolve())
        if args.no_download:
            config.dataset.options["download"] = False
        manifest = create_plan(config, args.output_dir, runs=20 if args.runs is None else args.runs,
                               seed_start=args.seed_start)
    settings = manifest["base_config"]["training"]
    print(f"Batch: {len(manifest['runs'])} independent runs x {settings['epochs']} epochs | "
          f"learning_rate={settings['learning_rate']} | "
          f"seeds={manifest['runs'][0]['seed']}..{manifest['runs'][-1]['seed']}", flush=True)
    print(f"Output: {manifest['output_dir']}", flush=True)
    if args.plan_only:
        print("Plan saved. Run again with --resume to start.")
        return
    summary = summarize_runs(manifest["runs"], manifest["output_dir"]) if args.summary_only else run_batch(
        manifest, max_steps=args.max_steps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
