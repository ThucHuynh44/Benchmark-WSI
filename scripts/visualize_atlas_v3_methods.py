"""Create paired t-SNE and FEATHER Grad-CAM figures for four ATLAS-v3 settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-v3-matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.experiment_loader import config_to_argv, load_experiment_config
from datasets.seq_wsi import TASK_SPECS, resolve_feature_path
from models import get_model
from utils.main import _prepare_fold


METHODS = (
    {
        "id": "atlasv3_frozen_proto",
        "title": "Frozen Proto (NCM)",
        "model": "atlas_v3",
        "root": "ablations/atlas_v3",
        "family": "frozen",
    },
    {
        "id": "atlasv3_frozen_proto_oas_lda",
        "title": "Frozen Proto + OAS-LDA",
        "model": "atlas_v3",
        "root": "ablations/atlas_v3",
        "family": "frozen",
    },
    {
        "id": "atlasv3_acl_normalized_oas_static",
        "title": "ACL + Static Normalized OAS",
        "model": "atlas_v3_acl",
        "root": "ablations/atlas_v3_acl",
        "family": "acl",
    },
    {
        "id": "atlasv3_acl_gated_transport_normalized_oas_no_histneg",
        "title": "ACL + Gated Transport + Normalized OAS",
        "model": "atlas_v3_acl",
        "root": "ablations/atlas_v3_acl",
        "family": "acl",
    },
)
LEGACY_ACL_KEYS = {
    "net.class_task",
    "net.hist_reliability",
    "net.uncertainty_ema",
}


def _parse_tasks(value: str, maximum: int) -> list[int]:
    output = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(part) for part in token.split("-", 1))
            output.extend(range(start, stop + 1))
        else:
            output.append(int(token))
    output = list(dict.fromkeys(output))
    if not output or min(output) < 0 or max(output) > maximum:
        raise ValueError(f"Visualization tasks must be within 0..{maximum}")
    return output


def _checkpoint(spec: Mapping[str, str], fold: int, after_task: int) -> Path:
    description = f"{spec['root']}/{spec['id']}/fold_{fold}"
    return (
        REPO_ROOT
        / "checkpoints"
        / description
        / f"fold_{fold}"
        / f"task{after_task}_checkpoint.pt"
    )


def _payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _args_from_payload(spec, payload, config: str):
    configured = load_experiment_config(
        config, method=spec["model"], backbone="feather"
    )
    module = __import__(f"models.{spec['model']}", fromlist=["get_parser"])
    args = module.get_parser().parse_args(
        config_to_argv(configured)
        + ["--model", spec["model"], "--backbone", "feather"]
    )
    saved = payload.get("method_state", {}).get("config", {})
    if spec["model"] == "atlas_v3":
        args.atlasv3_distribution_mode = saved.get(
            "classifier", args.atlasv3_distribution_mode
        )
    else:
        args.atlasv3_acl_mode = saved.get("mode", args.atlasv3_acl_mode)
        for name, value in saved.get("hyperparameters", {}).items():
            setattr(args, name, value)
    ablation = payload.get("ablation", {})
    args.ablation_id = ablation.get("id")
    args.ablation_group = ablation.get("group")
    args.ablation_config_hash = ablation.get("config_hash")
    return args


def _copy_layout(args, dataset, fold: int) -> None:
    args.fold = int(fold)
    args.n_tasks = int(dataset.N_TASKS)
    args.task_order = list(dataset.task_order)
    args.task_num_classes = list(dataset.task_num_classes)
    args.class_offsets = list(dataset.class_offsets)
    args.num_classes = int(dataset.total_num_classes)
    args.n_classes_per_task = tuple(dataset.task_num_classes)


def _restore(model, payload, model_name: str) -> None:
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed = LEGACY_ACL_KEYS if model_name == "atlas_v3_acl" else set()
    if missing or not unexpected.issubset(allowed):
        raise RuntimeError(
            f"Visualization checkpoint mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    model.load_checkpoint_state(payload["method_state"], strict=False)
    model.eval()


def _load_models(config: str, fold: int, after_task: int):
    payloads = {
        spec["id"]: _payload(_checkpoint(spec, fold, after_task)) for spec in METHODS
    }
    first = METHODS[0]
    first_args = _args_from_payload(first, payloads[first["id"]], config)
    dataset = _prepare_fold(first_args, fold)
    models = {}
    for index, spec in enumerate(METHODS):
        args = first_args if index == 0 else _args_from_payload(
            spec, payloads[spec["id"]], config
        )
        _copy_layout(args, dataset, fold)
        model = get_model(
            args, None, dataset.get_loss(), dataset.get_transform()
        )
        model.to(model.device)
        _restore(model, payloads[spec["id"]], spec["model"])
        models[spec["id"]] = model
    return models, dataset, payloads


def _class_name(dataset, label: int) -> str:
    label = int(label)
    for task, (offset, count) in enumerate(
        zip(dataset.class_offsets, dataset.task_num_classes)
    ):
        if int(offset) <= label < int(offset) + int(count):
            task_name = dataset.task_order[task]
            local = label - int(offset)
            return f"{task_name}:{TASK_SPECS[task_name]['labels'][local]}"
    return str(label)


@torch.no_grad()
def _embedding(model, item) -> torch.Tensor:
    features, coords, patch_size, _ = item
    features, coords, patch_size = model.prepare_inputs(
        features, coords, patch_size, training=False
    )
    return F.normalize(
        model.net.encode(features, coords, patch_size).float(), dim=1, eps=1.0e-8
    ).cpu()


def _sample_test_items(dataset, fold: int, tasks: list[int], maximum: int, seed: int):
    generator = np.random.default_rng(seed)
    output = []
    for task in tasks:
        test_dataset = dataset._datasets_for_task(task, fold)[2]
        labels = test_dataset.slide_data["label"].astype(int).to_numpy()
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label)
            generator.shuffle(indices)
            for index in indices[:maximum]:
                slide_id = str(test_dataset.slide_data.iloc[int(index)].slide_id)
                output.append((task, label, slide_id, test_dataset[int(index)]))
    return output


def _prototype(model, label: int) -> torch.Tensor:
    if hasattr(model.net, "raw_mean") and model.net.raw_mean.numel():
        value = model.net.raw_mean[int(label)]
    elif hasattr(model.net, "lda_means") and model.net.lda_means.numel():
        value = model.net.lda_means[int(label)]
    else:
        value = model.net.prototype_bank[int(label)]
    return F.normalize(value.detach().float(), dim=0, eps=1.0e-8).cpu()


def _create_tsne(
    models,
    dataset,
    fold: int,
    tasks: list[int],
    maximum: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    items = _sample_test_items(dataset, fold, tasks, maximum, seed)
    if len(items) < 4:
        raise RuntimeError("Not enough test WSI for t-SNE")
    family_model = {
        "frozen": models["atlasv3_frozen_proto"],
        "acl": models["atlasv3_acl_normalized_oas_static"],
    }
    family_embeddings = {}
    for family, model in family_model.items():
        family_embeddings[family] = torch.cat(
            [_embedding(model, item[-1]) for item in items]
        )
    labels = [int(item[1]) for item in items]
    class_ids = sorted(set(labels))
    blocks, metadata = [], []
    for family in ("frozen", "acl"):
        blocks.append(family_embeddings[family])
        metadata.extend(
            {
                "kind": "feature",
                "family": family,
                "method": "",
                "class_id": labels[index],
                "slide_id": items[index][2],
            }
            for index in range(len(items))
        )
    for spec in METHODS:
        values = torch.stack(
            [_prototype(models[spec["id"]], label) for label in class_ids]
        )
        blocks.append(values)
        metadata.extend(
            {
                "kind": "prototype",
                "family": spec["family"],
                "method": spec["id"],
                "class_id": label,
                "slide_id": "",
            }
            for label in class_ids
        )
    matrix = torch.cat(blocks).numpy()
    perplexity = min(30.0, max(2.0, (matrix.shape[0] - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        metric="cosine",
        random_state=seed,
    ).fit_transform(matrix)
    for row, point in zip(metadata, coordinates):
        row["x"], row["y"] = float(point[0]), float(point[1])
        row["class_name"] = _class_name(dataset, int(row["class_id"]))
    colors = plt.get_cmap("turbo")(
        np.linspace(0.03, 0.97, max(len(class_ids), 2))
    )
    color = {label: colors[index] for index, label in enumerate(class_ids)}
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    for axis, spec in zip(axes.flat, METHODS):
        family_rows = [
            row for row in metadata
            if row["kind"] == "feature" and row["family"] == spec["family"]
        ]
        for label in class_ids:
            points = np.asarray(
                [[row["x"], row["y"]] for row in family_rows if row["class_id"] == label]
            )
            axis.scatter(
                points[:, 0], points[:, 1], s=17, alpha=0.62,
                color=color[label], linewidths=0,
                label=_class_name(dataset, label),
            )
        prototype_rows = [
            row for row in metadata
            if row["kind"] == "prototype" and row["method"] == spec["id"]
        ]
        for row in prototype_rows:
            axis.scatter(
                row["x"], row["y"], marker="*", s=150,
                color=color[int(row["class_id"])], edgecolor="black", linewidth=0.8,
                zorder=4,
            )
        if spec["id"] == "atlasv3_acl_gated_transport_normalized_oas_no_histneg":
            static = {
                int(row["class_id"]): row for row in metadata
                if row["kind"] == "prototype"
                and row["method"] == "atlasv3_acl_normalized_oas_static"
            }
            for row in prototype_rows:
                label = int(row["class_id"])
                source = static[label]
                axis.annotate(
                    "", xy=(row["x"], row["y"]),
                    xytext=(source["x"], source["y"]),
                    arrowprops={"arrowstyle": "->", "color": color[label], "lw": 1.0, "alpha": 0.75},
                )
        axis.set_title(spec["title"], fontsize=12, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=min(5, len(class_ids)),
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Joint t-SNE of normalized WSI features and class prototypes",
        fontsize=15, fontweight="bold",
    )
    fig.text(
        0.5, 0.045,
        "Stars: class prototypes; arrows: projected static → gated prototype displacement",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "atlas_v3_tsne.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "atlas_v3_tsne.pdf", bbox_inches="tight")
    plt.close(fig)
    with (output / "atlas_v3_tsne_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    return {
        "test_wsis": len(items),
        "class_ids": class_ids,
        "tasks": tasks,
        "perplexity": perplexity,
        "seed": seed,
    }


def _patch_embed_layer(model):
    remote = model.net.backbone.model
    core = getattr(remote, "model", remote)
    layer = getattr(core, "patch_embed", None)
    if layer is None:
        raise AttributeError("Pinned FEATHER does not expose patch_embed for Grad-CAM")
    return layer


def _gradcam(model, item, target: int):
    features, coords, patch_size, _ = item
    features, coords, patch_size = model.prepare_inputs(
        features, coords, patch_size, training=False
    )
    features = features.detach().requires_grad_(True)
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = _patch_embed_layer(model).register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(features, coords, patch_size)[0]
            if not torch.isfinite(logits[0, int(target)]):
                raise RuntimeError(f"Target class {target} is unavailable at this checkpoint")
            prediction = int(logits.argmax(1).item())
            logits[0, int(target)].backward()
        activation = captured["activation"]
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("FEATHER patch embedding did not retain Grad-CAM gradients")
        weights = gradient.float().mean(dim=1, keepdim=True)
        signed = (activation.float() * weights).sum(dim=-1).squeeze(0)
        cam = F.relu(signed)
        fallback = False
        if float(cam.max()) <= torch.finfo(cam.dtype).eps:
            cam = signed.abs()
            fallback = True
        cam = cam.detach().cpu().numpy()
        lower, upper = np.percentile(cam, [5.0, 99.0])
        normalized = np.clip((cam - lower) / max(float(upper - lower), 1.0e-12), 0.0, 1.0)
        return normalized, prediction, fallback
    finally:
        handle.remove()


def _thumbnail_heatmap(
    thumbnail: Image.Image,
    coords: np.ndarray,
    scores: np.ndarray,
    patch_size: int,
    level_width: int,
    level_height: int,
) -> np.ndarray:
    width, height = thumbnail.size
    heat = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    scale_x, scale_y = width / float(level_width), height / float(level_height)
    for (x, y), score in zip(coords, scores):
        x0 = max(0, min(width - 1, int(math.floor(float(x) * scale_x))))
        y0 = max(0, min(height - 1, int(math.floor(float(y) * scale_y))))
        x1 = max(x0 + 1, min(width, int(math.ceil((float(x) + patch_size) * scale_x))))
        y1 = max(y0 + 1, min(height, int(math.ceil((float(y) + patch_size) * scale_y))))
        heat[y0:y1, x0:x1] += float(score)
        count[y0:y1, x0:x1] += 1.0
    heat = heat / np.maximum(count, 1.0)
    heat = gaussian_filter(heat, sigma=max(1.0, min(width, height) / 350.0))
    maximum = float(heat.max())
    return heat / maximum if maximum > 0.0 else heat


def _overlay(thumbnail: Image.Image, heat: np.ndarray) -> np.ndarray:
    base = np.asarray(thumbnail.convert("RGB"), dtype=np.float32) / 255.0
    colored = plt.get_cmap("turbo")(heat)[..., :3]
    alpha = (0.72 * np.power(heat, 0.65))[..., None]
    return np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)


def _thumbnail_from_wsi(wsi_path: Path, max_size: int) -> Image.Image:
    """Read a pyramid level from a TIFF/SVS without decoding the full-resolution level."""
    if not wsi_path.is_file():
        raise FileNotFoundError(wsi_path)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(wsi_path) as slide:
        base_width, base_height = slide.size
        base_ratio = base_width / float(base_height)
        frames = []
        for index in range(getattr(slide, "n_frames", 1)):
            slide.seek(index)
            width, height = slide.size
            ratio_error = abs(width / float(height) - base_ratio) / max(base_ratio, 1.0e-12)
            if ratio_error <= 0.02:
                frames.append((index, width, height))
        if not frames:
            raise RuntimeError(f"No whole-slide pyramid frame found in {wsi_path}")

        non_base = [frame for frame in frames if frame[0] != 0]
        candidates = non_base or frames
        large_enough = [frame for frame in candidates if max(frame[1:]) >= max_size]
        selected = min(large_enough or candidates, key=lambda frame: frame[1] * frame[2])
        slide.seek(selected[0])
        thumbnail = slide.convert("RGB").copy()
    thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return thumbnail


def _load_background(
    thumbnail_path: Path | None,
    wsi_path: Path | None,
    max_size: int,
) -> tuple[Image.Image, str]:
    if wsi_path is not None:
        return _thumbnail_from_wsi(wsi_path, max_size), str(wsi_path)
    if thumbnail_path is None or not thumbnail_path.is_file():
        raise FileNotFoundError(thumbnail_path)
    thumbnail = Image.open(thumbnail_path).convert("RGB")
    thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return thumbnail, str(thumbnail_path)


def _find_slide(dataset, fold: int, task: int, slide_id: str):
    for split_index, split_name in ((0, "train"), (1, "val"), (2, "test")):
        source = dataset._datasets_for_task(task, fold)[split_index]
        ids = source.slide_data["slide_id"].astype(str).tolist()
        if str(slide_id) in ids:
            index = ids.index(str(slide_id))
            return source, index, split_name
    raise ValueError(f"Slide {slide_id!r} is absent from fold={fold}, task={task}")


def _create_gradcam(
    models,
    dataset,
    fold: int,
    task: int,
    slide_id: str,
    thumbnail_path: Path | None,
    wsi_path: Path | None,
    thumbnail_max_size: int,
    output: Path,
) -> dict[str, Any]:
    source, index, split = _find_slide(dataset, fold, task, slide_id)
    item = source[index]
    target = int(item[3])
    feature_path = resolve_feature_path(source.feature_root, slide_id)
    with h5py.File(feature_path, "r") as handle:
        coords = handle["coords"][:]
        attrs = handle["coords"].attrs
        patch_size = int(attrs.get("patch_size_level0", item[2]))
        level_width = int(attrs.get("level0_width", coords[:, 0].max() + patch_size))
        level_height = int(attrs.get("level0_height", coords[:, 1].max() + patch_size))
    thumbnail, background_source = _load_background(
        thumbnail_path, wsi_path, thumbnail_max_size
    )
    outputs = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for axis, spec in zip(axes.flat, METHODS):
        scores, prediction, fallback = _gradcam(
            models[spec["id"]], item, target
        )
        if scores.shape[0] != coords.shape[0]:
            raise RuntimeError("Grad-CAM patch scores do not match stored coordinates")
        heat = _thumbnail_heatmap(
            thumbnail, coords, scores, patch_size, level_width, level_height
        )
        rendered = _overlay(thumbnail, heat)
        axis.imshow(rendered)
        axis.set_title(
            f"{spec['title']}\nPred: {_class_name(dataset, prediction)}",
            fontsize=10, fontweight="bold",
        )
        axis.axis("off")
        individual = output / f"gradcam_{spec['id']}.png"
        Image.fromarray((rendered * 255).astype(np.uint8)).save(individual)
        outputs.append(
            {
                "method": spec["id"],
                "prediction": prediction,
                "prediction_name": _class_name(dataset, prediction),
                "fallback_absolute_cam": fallback,
            }
        )
    fig.suptitle(
        f"FEATHER patch-embedding Grad-CAM | slide {slide_id} | "
        f"true: {_class_name(dataset, target)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "atlas_v3_gradcam.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "atlas_v3_gradcam.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "slide_id": str(slide_id),
        "split": split,
        "true_class": target,
        "true_class_name": _class_name(dataset, target),
        "background": background_source,
        "feature_path": str(feature_path),
        "methods": outputs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/methods.yaml")
    parser.add_argument("--fold", type=int, default=0, choices=range(10))
    parser.add_argument("--after-task", type=int, default=9, choices=range(10))
    parser.add_argument("--tsne-tasks", default="0,1,2,9")
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--heatmap-task", type=int, default=0, choices=range(10))
    parser.add_argument("--slide-id", default="patient_125_node_0")
    parser.add_argument(
        "--thumbnail",
        default=(
            "/datastore/uittogether/LuuTru/Thuchd/Research/dataset/CAMELYON17/"
            "All/thumbnails/patient_125_node_0.jpg"
        ),
    )
    parser.add_argument(
        "--wsi",
        default=None,
        help="Optional original .svs/.tif; takes precedence over --thumbnail.",
    )
    parser.add_argument("--thumbnail-max-size", type=int, default=1600)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.max_per_class <= 0:
        raise ValueError("max-per-class must be positive")
    if args.thumbnail_max_size <= 0:
        raise ValueError("thumbnail-max-size must be positive")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else REPO_ROOT
        / "results/visualizations/atlas_v3"
        / f"fold_{args.fold}_task_{args.after_task}"
    )
    output.mkdir(parents=True, exist_ok=True)
    models, dataset, _ = _load_models(args.config, args.fold, args.after_task)
    tasks = _parse_tasks(args.tsne_tasks, args.after_task)
    tsne = _create_tsne(
        models, dataset, args.fold, tasks, args.max_per_class,
        args.seed, output,
    )
    gradcam = _create_gradcam(
        models, dataset, args.fold, args.heatmap_task,
        str(args.slide_id),
        Path(args.thumbnail).expanduser().resolve() if args.thumbnail else None,
        Path(args.wsi).expanduser().resolve() if args.wsi else None,
        args.thumbnail_max_size,
        output,
    )
    manifest = {
        "audit_only": True,
        "fold": args.fold,
        "after_task": args.after_task,
        "settings": [spec["id"] for spec in METHODS],
        "tsne": tsne,
        "gradcam": gradcam,
        "scientific_note": (
            "Frozen Proto and Frozen OAS-LDA share one feature cloud; ACL Static "
            "and ACL Gated share another. Grad-CAM is computed from the target "
            "class logit at FEATHER patch_embed, not from ATLAS wrapper attention. "
            "t-SNE positions and arrows are qualitative and are not used as distance metrics."
        ),
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote ATLAS-v3 visualizations to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
