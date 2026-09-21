"""
Usage:
    python visualize_results.py --path results_judge.json --out_dir plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FEATURES = [
    "sex", "city_country", "relationship_status", "age",
    "education", "occupation", "birth_city_country", "income_level",
]
ABBR = {
    "sex": "SEX", "city_country": "LOC", "relationship_status": "MAR",
    "age": "AGE", "education": "SCH", "occupation": "OCC",
    "birth_city_country": "POB", "income_level": "INC",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_overall(data, out_dir):
    overall = data["overall"]
    ranked = sorted(overall.items(), key=lambda kv: -kv[1]["top1"])
    models = [m for m, _ in ranked]
    top1 = [v["top1"] * 100 for _, v in ranked]
    top3 = [v["top3"] * 100 for _, v in ranked]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, top1, width, label="Top-1")
    ax.bar(x + width / 2, top3, width, label="Top-3")

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall accuracy by model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "overall_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_attribute(data, out_dir, metric_index=0, label="top-1"):
    """metric_index: 0 for top-1 sum, 1 for top-3 sum (see per_feature schema:
    per_feature[model][feature] = [top1_sum, top3_sum, total])."""
    per_feature = data["per_feature"]
    overall = data["overall"]

    # keep the same model order as the overall top-1 ranking, for readability
    models = [m for m, _ in sorted(overall.items(), key=lambda kv: -kv[1]["top1"])]

    x = np.arange(len(models))
    n_feat = len(FEATURES)
    width = 0.8 / n_feat

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, feat in enumerate(FEATURES):
        vals = []
        for m in models:
            sum1, sum3, total = per_feature[m][feat]
            metric_sum = [sum1, sum3][metric_index]
            vals.append(metric_sum / total * 100 if total else 0.0)
        offset = (i - n_feat / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=ABBR[feat])

    ax.set_ylabel(f"{label} accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{label.capitalize()} accuracy by attribute and model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fname = f"per_attribute_accuracy_{label.replace('-', '')}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_stacked_top1(data, out_dir):
    per_feature = data["per_feature"]
    overall = data["overall"]
    models = [m for m, _ in sorted(overall.items(), key=lambda kv: -kv[1]["top1"])]

    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(FEATURES)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for feat, color in zip(FEATURES, colors):
        contrib = []
        for m in models:
            sum1, sum3, _ = per_feature[m][feat]
            grand_total = overall[m]["total"]  # 525 -- NOT the per-feature count
            contrib.append(sum1 / grand_total * 100)
        contrib = np.array(contrib)
        ax.bar(x, contrib, bottom=bottoms, label=ABBR[feat], color=color)
        bottoms += contrib

    # sanity check: each stack should reproduce the model's overall top-1 %
    for m, total_h in zip(models, bottoms):
        expected = overall[m]["top1"] * 100
        assert abs(total_h - expected) < 0.5, f"{m}: stack={total_h:.1f} overall={expected:.1f}"

    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Top-1 accuracy by model, stacked by attribute contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "top1_stacked_by_attribute.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_radar_per_model(data, out_dir):
    per_feature = data["per_feature"]
    overall = data["overall"]
    models = [m for m, _ in sorted(overall.items(), key=lambda kv: -kv[1]["top1"])]

    n_axes = len(FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    labels = [ABBR[f] for f in FEATURES]

    n_models = len(models)
    ncols = 5
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows),
                              subplot_kw=dict(polar=True))
    axes = np.array(axes).reshape(-1)

    for ax, m in zip(axes, models):
        values = [per_feature[m][f][0] / per_feature[m][f][2] * 100 for f in FEATURES]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, color="tab:blue")
        ax.fill(angles, values, alpha=0.25, color="tab:blue")
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(m, fontsize=10, pad=12)

    for ax in axes[len(models):]:
        ax.axis("off")

    fig.suptitle("Per-attribute top-1 accuracy profile by model", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "radar_per_model.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results_judge.json")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--top3", action="store_true",
                     help="Also render the per-attribute breakdown using top-3 scores.")
    args = ap.parse_args()

    data = load_results(args.path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall(data, out_dir)
    plot_stacked_top1(data, out_dir)
    plot_per_attribute(data, out_dir, metric_index=0, label="top-1")
    plot_radar_per_model(data, out_dir)
    if args.top3:
        plot_per_attribute(data, out_dir, metric_index=1, label="top-3")


if __name__ == "__main__":
    main()