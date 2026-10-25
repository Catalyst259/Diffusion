"""Compare recorded epoch losses across independent runs without loading models."""

import csv
import math
import os
from pathlib import Path
from statistics import mean, stdev


LOSS_FIELDS = ("epoch", "global_step", "train_loss", "test_loss")
COMBINED_FIELDS = ("run_id", "seed", *LOSS_FIELDS)
EPOCH_FIELDS = ("epoch", "n", "train_mean", "train_std", "test_mean", "test_std")
RUN_FIELDS = (
    "run_id", "seed", "target_epochs", "completed_epochs", "status",
    "final_train_loss", "final_test_loss", "min_test_loss", "train_change", "test_change",
)


def _integer(value, *, field, path, row, positive=True):
    try:
        # Decimal strings such as 1.0 are not integer epoch/step records.
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: row {row}: {field} must be an integer") from error
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{path}: row {row}: {field} must be {qualifier}")
    return result


def _read_history(path):
    """Missing files and header-only/empty files represent no completed epochs."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return []
        missing = set(LOSS_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        history = []
        previous_epoch = 0
        for row_number, raw in enumerate(reader, start=2):
            epoch = _integer(raw["epoch"], field="epoch", path=path, row=row_number)
            if epoch <= previous_epoch:
                raise ValueError(f"{path}: row {row_number}: epochs must be strictly increasing and unique")
            record = {
                "epoch": epoch,
                "global_step": _integer(
                    raw["global_step"], field="global_step", path=path,
                    row=row_number, positive=False,
                ),
            }
            for field in ("train_loss", "test_loss"):
                try:
                    value = float(raw[field])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: row {row_number}: {field} must be finite") from error
                if not math.isfinite(value):
                    raise ValueError(f"{path}: row {row_number}: {field} must be finite")
                record[field] = value
            history.append(record)
            previous_epoch = epoch
    return history


def _write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plot(histories, epoch_summary, path):
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(14, 7.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=True)
    colors = colormaps["tab20"]
    for index, (run, history) in enumerate(histories):
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        for axis, field in zip(axes, ("train_loss", "test_loss")):
            axis.plot(
                epochs, [row[field] for row in history], marker=".",
                linewidth=1.1, alpha=0.55, color=colors(index % 20),
                label=f"{run['run_id']} (seed {run['seed']})",
            )
    epochs = [row["epoch"] for row in epoch_summary]
    for axis, prefix, title in zip(axes, ("train", "test"), ("Train loss", "Test loss")):
        means = [row[f"{prefix}_mean"] for row in epoch_summary]
        deviations = [row[f"{prefix}_std"] for row in epoch_summary]
        axis.fill_between(
            epochs, [m - s for m, s in zip(means, deviations)],
            [m + s for m, s in zip(means, deviations)],
            color="black", alpha=0.12, label="Mean +/- sample std",
        )
        axis.plot(epochs, means, color="black", linewidth=2.3, marker="o", markersize=3, label="Mean")
        axis.set(xlabel="Epoch", ylabel="Training objective", title=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
    counts = [row["n"] for row in epoch_summary]
    figure.suptitle(
        "Independent runs: recorded epoch losses\n"
        f"Available runs per epoch: n={min(counts)}-{max(counts)} "
        "(see epoch_summary.csv); std=0 when n=1",
        fontsize=12,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncols=4, fontsize=8,
        frameon=False,
    )
    try:
        figure.savefig(path, dpi=160)
    finally:
        figure.clear()


def summarize_runs(runs: list[dict], output_dir: str | Path) -> dict:
    """Write comparison CSVs and curves from actual per-run ``losses.csv`` files.

    ``epochs`` in each run is its target; ``completed_epochs`` is the highest
    recorded epoch. Aggregate statistics use all available histories at each
    epoch, including partial/failed runs. No losses are interpolated or filled.
    The standard deviation is the sample standard deviation (zero for n=1).
    """
    histories = []
    combined = []
    run_summary = []
    grouped = {}
    identifiers = set()
    for run in runs:
        run_id = str(run["run_id"])
        if run_id in identifiers:
            raise ValueError(f"Duplicate run_id: {run_id}")
        identifiers.add(run_id)
        loss_path = Path(run["output_dir"]) / "losses.csv"
        target = _integer(str(run["epochs"]), field="target epochs", path=loss_path, row=0)
        history = _read_history(loss_path)
        histories.append((run, history))
        completed = history[-1]["epoch"] if history else 0
        if run.get("status") == "failed":
            status = "failed"
        elif completed >= target:
            status = "completed"
        elif history or run.get("status") in {"incomplete", "interrupted", "running"}:
            status = "incomplete"
        else:
            status = "pending"
        summary = dict.fromkeys(RUN_FIELDS, "")
        summary.update(
            run_id=run_id, seed=run["seed"], target_epochs=target,
            completed_epochs=completed, status=status,
        )
        if history:
            summary.update(
                final_train_loss=history[-1]["train_loss"],
                final_test_loss=history[-1]["test_loss"],
                min_test_loss=min(row["test_loss"] for row in history),
                train_change=history[-1]["train_loss"] - history[0]["train_loss"],
                test_change=history[-1]["test_loss"] - history[0]["test_loss"],
            )
        run_summary.append(summary)
        for row in history:
            combined.append({"run_id": run_id, "seed": run["seed"], **row})
            grouped.setdefault(row["epoch"], []).append(row)

    epoch_summary = []
    for epoch, rows in sorted(grouped.items()):
        record = {"epoch": epoch, "n": len(rows)}
        for prefix in ("train", "test"):
            values = [row[f"{prefix}_loss"] for row in rows]
            record[f"{prefix}_mean"] = mean(values)
            record[f"{prefix}_std"] = stdev(values) if len(values) > 1 else 0.0
        epoch_summary.append(record)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined_losses": output_dir / "combined_losses.csv",
        "epoch_summary": output_dir / "epoch_summary.csv",
        "run_summary": output_dir / "run_summary.csv",
        "comparison": output_dir / "comparison.png",
    }
    _write_csv(paths["combined_losses"], COMBINED_FIELDS, combined)
    _write_csv(paths["epoch_summary"], EPOCH_FIELDS, epoch_summary)
    _write_csv(paths["run_summary"], RUN_FIELDS, run_summary)
    if epoch_summary:
        _plot(histories, epoch_summary, paths["comparison"])
    elif paths["comparison"].exists():
        # A prior plot must not imply data exists after the source is removed.
        paths["comparison"].unlink()

    counts = {status: sum(row["status"] == status for row in run_summary)
              for status in ("completed", "incomplete", "failed", "pending")}
    return {
        "total_runs": len(runs),
        **{f"{status}_runs": count for status, count in counts.items()},
        "runs_with_history": sum(bool(history) for _, history in histories),
        "epoch_count": len(epoch_summary),
        "artifacts": {
            key: str(path) if key != "comparison" or epoch_summary else None
            for key, path in paths.items()
        },
    }
