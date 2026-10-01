"""Standalone scientific figures from recorded metrics; no tensor loading."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("SLURM required for report rendering")
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    args = parser.parse_args()
    root = Path(args.trajectory)
    summary = json.loads(Path("cluster/full_rank_512step_summary.json").read_text())
    assert root.parent == Path("/home/prignano/qnormuon-runs/full-rank-512step")
    qso = [json.loads(s) for s in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(qso) == summary["completed_steps"] == 512
    assert qso[-1]["tokens"] == summary["tokens"] == 1048576
    reference = Path(json.loads((root / "provenance.json").read_text())["adamw_reference"])
    adam = [json.loads(s) for s in (reference / "metrics.jsonl").read_text().splitlines()]
    pairs = [json.loads(s) for s in (root / "solves.jsonl").read_text().splitlines()]
    output = Path("docs/figures/full_rank_epsilon_512")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    colors = {"qso": "#146b8c", "adamw": "#b14630"}

    def figure(name, title, ylabel, series, *, logarithmic=False, reference_line=None):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        for label, x, y, color, points in series:
            ax.plot(x, y, color=color, label=label, linewidth=1.1,
                    marker="o" if points else None, markersize=3)
        if reference_line is not None:
            ax.axhline(reference_line, color="0.45", linestyle="--", linewidth=1,
                       label="Production-v0 rcond guard")
        if logarithmic:
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Training tokens (equal-token axis)", ylabel=ylabel)
        ax.set_xlim(0, 1048576)
        ax.set_xticks([0, 262144, 524288, 786432, 1048576],
                      ["0", "262,144", "524,288", "786,432", "1,048,576"])
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
        for extension in ("svg", "pdf"):
            fig.savefig(output / f"{name}.{extension}")
        plt.close(fig)

    validation = summary["validation"]
    figure("validation", "Single-seed validation quality", "Validation loss", [
        ("Research full-rank QSO", [r["tokens"] for r in validation],
         [r["qso"] for r in validation], colors["qso"], True),
        ("Matched tuned AdamW", [r["tokens"] for r in validation],
         [r["adamw"] for r in validation], colors["adamw"], True)])
    for name, key, title, ylabel in [
        ("training", "loss", "Recorded training losses (unsmoothed)", "Training loss"),
        ("update_ratio", "update_parameter_ratio", "Raw update scale (not a gauge certificate)",
         "Update RMS / parameter RMS")]:
        figure(name, title, ylabel, [
            (label, [r["tokens"] for r in records], [r[key] for r in records], color, False)
            for label, records, color in [("Research full-rank QSO", qso, colors["qso"]),
                                         ("Matched tuned AdamW", adam, colors["adamw"])]])
    tokens = [(step + 1) * 2048 for step in range(512)]
    for name, key, title, ylabel in [
        ("rcond", "rcond_sides", "Minimum final residual conditioning across six pairs", "Residual rcond"),
        ("alpha_lower", "alpha_lower", "Minimum final numerical rank lower bound across six pairs", "Alpha lower")]:
        values = [min(min(r[key]) for r in pairs if r["step"] == step) for step in range(512)]
        figure(name, title, ylabel, [("Research full-rank QSO", tokens, values, colors["qso"], False)],
               logarithmic=True, reference_line=1e-4 if name == "rcond" else None)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    counts = summary["newton_bins"]
    bars = ax.bar(list(counts), list(counts.values()), color=colors["qso"])
    ax.bar_label(bars)
    ax.set(xlabel="Newton iterations at certification", ylabel="Pair solves",
           title="Research QSO: 3,072 certified pair solves")
    for extension in ("svg", "pdf"):
        fig.savefig(output / f"newton.{extension}")
    plt.close(fig)
    print("FIGURES", output)


if __name__ == "__main__":
    main()
