"""Export the completed seed-42 rank study from JSON reports only.

No torch, model weights, datasets, training, or inference are needed. Exported
report copies preserve values/text and normalize artifact paths for sharing.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent
RANKS = (4, 8, 16, 32)
REPORT_FILES = ("config.json", "metrics.json", "evaluation.json", "evaluation_answer_only.json")
LIMITATIONS = [
    "One seed (42), 2000 training and 200 validation samples, one epoch.",
    "All adapters were trained with full-sequence labels; answer-only is an evaluation mask.",
    "Compare loss within the same label policy; the two masks have different target sets.",
    "Reference-answer loss is not a free-generation quality score; six prompts are illustrative.",
    "Timing is a single observation per rank, with uncontrolled machine-load variation.",
    "Training time excludes model loading, validation, and checkpoint serialization.",
    "Base training time and memory were not measured and are blank in the CSV.",
    "Source model/data revisions were not pinned. No weights or dataset are included.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def portable_paths(value, run_name, key=None):
    """Only rewrite known path fields, never prompts or generated text."""
    if isinstance(value, dict):
        return {k: portable_paths(v, run_name, k) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_paths(v, run_name) for v in value]
    if isinstance(value, str):
        name = PureWindowsPath(value).name
        if key == "data_dir":
            return f"data/{name}"
        if key in ("adapter_path", "checkpoint_path"):
            return f"checkpoint/{run_name}/{name}"
    return value


def check_metric(loss, perplexity, context):
    for value in (loss, perplexity):
        require(type(value) in (float, int) and math.isfinite(value), f"{context}: nonfinite/non-numeric metric")
    require(loss >= 0 and perplexity >= 1, f"{context}: invalid loss/perplexity")
    require(math.isclose(math.log(perplexity), loss, rel_tol=1e-9, abs_tol=1e-9), f"{context}: perplexity does not match loss")


def validate_run(reports, rank):
    config, metrics, full, answer = [reports[name] for name in REPORT_FILES]
    fixed = {
        "model_name": "Qwen/Qwen3-0.6B-Base", "seed": 42,
        "train_samples": 2000, "validation_samples": 200, "num_epochs": 1,
        "label_policy": "full_sequence", "lora_rank": rank, "lora_alpha": rank * 2,
        "trainable_parameters": rank * 143360,
    }
    for key, expected in fixed.items():
        require(config[key] == expected, f"r={rank}: unexpected {key}")
    require(metrics["status"] == "completed", f"r={rank}: training is incomplete")
    require(len(metrics["epochs"]) == 1, f"r={rank}: expected one completed epoch")
    epoch = metrics["epochs"][0]
    require(epoch["epoch"] == 1, f"r={rank}: wrong epoch")
    require(epoch["global_step"] == epoch["optimizer_updates"] == config["num_training_steps"], f"r={rank}: update count mismatch")
    for key in ("train_loss", "train_seconds", "peak_allocated_gib", "peak_reserved_gib"):
        value = epoch[key]
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0, f"r={rank}: invalid {key}")
    require(epoch["peak_reserved_gib"] >= epoch["peak_allocated_gib"], f"r={rank}: inconsistent GPU peaks")
    for report, policy in ((full, "full_sequence"), (answer, "answer_only")):
        require(report["label_policy"] == policy, f"r={rank}: wrong label policy")
        for key in ("model_name", "data_dir", "validation_samples"):
            require(report[key] == config[key], f"r={rank}: evaluation {key} mismatch")
        require(report["validation_max_length"] == config["max_length"], f"r={rank}: length mismatch")
        require(report["adapter_path"] == epoch["adapter_path"], f"r={rank}: adapter path mismatch")
        require(report["adapter_metadata"] == {
            "base_model_name": config["model_name"], "rank": rank,
            "alpha": config["lora_alpha"], "target_names": config["lora_target_names"],
        }, f"r={rank}: adapter metadata mismatch")
        ids = [prompt["id"] for prompt in report["prompts"]]
        require(len(ids) == len(set(ids)) == 6, f"r={rank}: expected six unique prompts")
        for model in ("base", "lora"):
            result = report[model]
            check_metric(result["loss"], result["perplexity"], f"r={rank}/{policy}/{model}")
            require([sample["id"] for sample in result["samples"]] == ids, f"r={rank}: sample IDs/order mismatch")
            require(all(isinstance(s["response"], str) and s["response"].strip() for s in result["samples"]), f"r={rank}: empty response")
    for key in ("generation", "prompts"):
        require(full[key] == answer[key], f"r={rank}: evaluation {key} differs by policy")
    for model in ("base", "lora"):
        require(full[model]["samples"] == answer[model]["samples"], f"r={rank}: responses differ by policy")
    require(metrics["initial_validation"] == {k: full["base"][k] for k in ("loss", "perplexity")}, f"r={rank}: initial metrics mismatch")
    require(epoch["validation_loss"] == full["lora"]["loss"] and epoch["validation_perplexity"] == full["lora"]["perplexity"], f"r={rank}: reloaded evaluation mismatch")


def collect_results(source_dir):
    """Validate the whole study before creating any output files."""
    runs, sources, rows = {}, [], []
    reference = None
    for rank in RANKS:
        run_name = f"lora_r{rank}_seed42_baseline_v1"
        reports = {}
        for name in REPORT_FILES:
            path = Path(source_dir) / run_name / name
            raw = path.read_bytes()
            report = json.loads(raw)
            require(isinstance(report, dict), f"{path}: expected JSON object")
            reports[name] = report
            sources.append({"source": f"{run_name}/{name}", "sha256": hashlib.sha256(raw).hexdigest()})
        try:
            validate_run(reports, rank)
            reports = {name: portable_paths(report, run_name) for name, report in reports.items()}
            config = reports["config.json"]
            full, answer = reports["evaluation.json"], reports["evaluation_answer_only.json"]
            controls = {k: v for k, v in config.items() if k not in ("lora_rank", "lora_alpha", "trainable_parameters")}
            shared = (controls, full["generation"], full["prompts"], full["base"], answer["base"])
            require(reference is None or shared == reference, f"r={rank}: study controls, Base results, or prompts differ")
            reference = shared
            epoch = reports["metrics.json"]["epochs"][0]
            if not rows:
                rows.append(make_row("Base", 0, None, 0, full["base"], answer["base"], {}))
            rows.append(make_row(f"LoRA r={rank}", rank, config["lora_alpha"], config["trainable_parameters"], full["lora"], answer["lora"], epoch))
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f"{run_name}: invalid report structure: {error}") from error
        runs[run_name] = reports
    return rows, runs, sources


def make_row(model, rank, alpha, count, full, answer, epoch):
    return {
        "model": model, "rank": rank, "alpha": alpha, "seed": 42,
        "trainable_parameters": count, "full_sequence_loss": full["loss"],
        "full_sequence_perplexity": full["perplexity"],
        "answer_only_loss": answer["loss"], "answer_only_perplexity": answer["perplexity"],
        **{key: epoch.get(key) for key in (
            "train_loss", "optimizer_updates", "train_seconds", "peak_allocated_gib", "peak_reserved_gib",
        )},
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.hashsalt": "mini-llm-rank-study", "font.size": 10})
    labels = [row["model"].replace("LoRA ", "") for row in rows]
    colors = ["#9aa5b1"] + ["#2878a5"] * 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, key, title in zip(axes, ("full_sequence_loss", "answer_only_loss"), ("Full-sequence validation loss", "Answer-only validation loss")):
        bars = ax.bar(labels, [row[key] for row in rows], color=colors)
        ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=9)
        ax.set_ylim(0, max(row[key] for row in rows) * 1.16)
        ax.set_title(title)
        ax.set_ylabel("Loss (lower is better)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Seed 42 | 200 validation samples | Compare ranks within each panel", fontsize=12)
    save_figure(fig, output_dir / "validation_loss")
    plt.close(fig)

    adapters = rows[1:]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, key, title, unit in (
        (axes[0], "trainable_parameters", "Trainable adapter parameters", "Millions"),
        (axes[1], "peak_allocated_gib", "Peak GPU memory", "GiB"),
        (axes[2], "train_seconds", "Observed training time", "Seconds"),
    ):
        values = [row[key] / (1e6 if key == "trainable_parameters" else 1) for row in adapters]
        if key == "peak_allocated_gib":
            positions = list(range(4))
            for offset, metric, color, legend in ((-0.2, key, "#2878a5", "Allocated"), (0.2, "peak_reserved_gib", "#e5a04b", "Reserved")):
                bars = ax.bar([p + offset for p in positions], [row[metric] for row in adapters], width=0.4, color=color, label=legend)
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.set_xticks(positions, labels[1:])
            ax.set_ylim(0, max(row["peak_reserved_gib"] for row in adapters) * 1.25)
            ax.legend(loc="upper left", fontsize=8, ncols=2)
        else:
            bars = ax.bar(labels[1:], values, color="#2878a5")
            ax.bar_label(bars, fmt="%.3f" if key == "trainable_parameters" else "%.1f", padding=3)
            ax.set_ylim(0, max(values) * 1.16)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("One run per rank | Timing differences do not establish a rank speed advantage", fontsize=12)
    save_figure(fig, output_dir / "training_resources")
    plt.close(fig)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=160)


def export_results(source_dir, output_dir, plots=True):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    require(source_dir != output_dir and source_dir not in output_dir.parents and output_dir not in source_dir.parents, "Source and output directories must not overlap")
    rows, runs, sources = collect_results(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_name, reports in runs.items():
        for name, report in reports.items():
            write_json(output_dir / "runs" / run_name / name, report)
    with (output_dir / "rank_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for source in sources:
        exported = output_dir / "runs" / source["source"]
        source["exported_sha256"] = hashlib.sha256(exported.read_bytes()).hexdigest()
    write_json(output_dir / "manifest.json", {
        "schema_version": 1, "source_files_relative_to_source_dir": sources,
        "path_normalization": "data_dir, adapter_path, checkpoint_path are project-relative; all metrics and text are preserved",
        "limitations": LIMITATIONS,
    })
    if plots:
        plot_results(rows, output_dir)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "checkpoint")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "results")
    parser.add_argument("--no-plots", action="store_true", help="Export JSON/CSV only; matplotlib is not required.")
    args = parser.parse_args(argv)
    try:
        rows = export_results(args.source_dir, args.output_dir, plots=not args.no_plots)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported {len(rows)} comparison rows and 16 report copies to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
