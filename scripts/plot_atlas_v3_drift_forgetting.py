"""Create publication figures for ATLAS-v3 drift and transport correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FEATURE_ROOT = REPO_ROOT / "results/diagnostics/atlas_v3_drift_forgetting/feature_drift"
PROTOTYPE_ROOT = REPO_ROOT / "results/diagnostics/atlas_v3_transport_correction"
OUTPUT_ROOT = REPO_ROOT / "results/diagnostics/atlas_v3_drift_forgetting/figures_drift_transport"
METHODS = {
    "static": {
        "setting": "atlasv3_acl_normalized_oas_static",
        "title": "ACL + static OAS",
        "color": "#D55E00",
        "marker": "o",
    },
    "ungated": {
        "setting": "atlasv3_acl_transport_normalized_oas",
        "title": "ACL + full transport",
        "color": "#009E73",
        "marker": "s",
    },
    "gated": {
        "setting": "atlasv3_acl_gated_transport_normalized_oas_no_histneg",
        "title": "ACL + gated transport",
        "color": "#0072B2",
        "marker": "^",
    },
}


def _read_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("fold_*/task_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold_*/task_*.csv files under {root}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def _fold_curve(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    per_fold = (
        frame.groupby(["fold", "after_task"], as_index=False)[value]
        .mean()
        .rename(columns={value: "fold_mean"})
    )
    grouped = per_fold.groupby("after_task")["fold_mean"]
    output = grouped.agg(["mean", "std", "count"]).reset_index()
    output["ci95"] = 1.96 * output["std"].fillna(0.0) / np.sqrt(output["count"])
    return output


def _prototype_curves(prototype: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for role in METHODS:
        value = f"{role}_prototype_cosine_distance"
        if value not in prototype:
            raise ValueError(f"Missing oracle diagnostic column {value}")
        fold_curve = (
            prototype.groupby(["fold", "after_task"], as_index=False)[value]
            .mean()
            .rename(columns={value: "fold_mean"})
        )
        summary = fold_curve.groupby("after_task")["fold_mean"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        summary["ci95"] = (
            1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
        )
        summary["method"] = role
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def _plot(
    feature_curve: pd.DataFrame,
    prototype_curve: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    x = feature_curve["after_task"].to_numpy() + 1
    y = feature_curve["mean"].to_numpy()
    ci = feature_curve["ci95"].to_numpy()
    ax.plot(x, np.zeros_like(x), "--", color="#666666", label="Frozen OAS-LDA")
    ax.plot(x, y, marker="o", color="#CC79A7", label="ACL encoder")
    ax.fill_between(x, y - ci, y + ci, color="#CC79A7", alpha=0.18)
    ax.set_title("(a) Representation drift")
    ax.set_xlabel("Number of learned tasks")
    ax.set_ylabel("Cosine feature drift ↓")
    ax.legend(frameon=False)

    ax = axes[1]
    for role in METHODS:
        values = prototype_curve[prototype_curve["method"] == role]
        if values.empty:
            continue
        spec = METHODS[role]
        x = values["after_task"].to_numpy() + 1
        y = values["mean"].to_numpy()
        ci = values["ci95"].to_numpy()
        ax.plot(x, y, marker=spec["marker"], color=spec["color"], label=spec["title"])
        ax.fill_between(x, y - ci, y + ci, color=spec["color"], alpha=0.15)
    ax.set_title("(b) Historical prototype mismatch")
    ax.set_xlabel("Number of learned tasks")
    ax.set_ylabel("Cosine distance to oracle ↓")
    ax.legend(frameon=False)

    fig.suptitle(
        "ATLAS-v3: representation drift and historical-statistics correction",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "atlas_v3_drift_transport.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "atlas_v3_drift_transport.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--prototype-root", default=str(PROTOTYPE_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    feature = _read_tree(Path(args.feature_root).expanduser().resolve())
    prototype = _read_tree(Path(args.prototype_root).expanduser().resolve())
    for name in ("fold", "after_task", "eval_task"):
        feature[name] = pd.to_numeric(feature[name], errors="raise").astype(int)
        prototype[name] = pd.to_numeric(prototype[name], errors="raise").astype(int)
    folds = sorted(feature["fold"].unique().tolist())
    if args.strict:
        expected = {(fold, task) for fold in range(10) for task in range(1, 10)}
        observed_feature = set(zip(feature["fold"], feature["after_task"]))
        observed_prototype = set(zip(prototype["fold"], prototype["after_task"]))
        if observed_feature != expected or observed_prototype != expected:
            raise RuntimeError(
                "Strict mode requires all 90 fold/checkpoint pairs for both diagnostics"
            )

    feature_curve = _fold_curve(feature, "acl_feature_drift_macro")
    prototype_curve = _prototype_curves(prototype[prototype["fold"].isin(folds)])

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature.to_csv(output / "feature_drift_per_task.csv", index=False)
    feature_curve.to_csv(output / "feature_drift_by_checkpoint.csv", index=False)
    prototype_curve.to_csv(output / "prototype_mismatch_by_checkpoint.csv", index=False)
    _plot(feature_curve, prototype_curve, output)

    manifest = {
        "feature_root": str(Path(args.feature_root).expanduser().resolve()),
        "prototype_root": str(Path(args.prototype_root).expanduser().resolve()),
        "folds": folds,
        "feature_drift": "1 - cosine(frozen_FEATHER(x), ACL_FEATHER_t(x)); macro over classes",
        "prototype_mismatch": "cosine distance between stored and oracle historical prototypes",
        "transport_comparison": {
            "static": "no transport",
            "ungated": "full low-rank transport step (gate=1)",
            "gated": "adaptive class-wise gated low-rank transport step",
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote ATLAS-v3 drift/transport figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
