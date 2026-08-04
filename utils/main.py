"""Command-line entry point for continual WSI experiments."""

import datetime
import importlib
import os
import socket
import sys
import uuid
from argparse import ArgumentParser

import numpy  # imported before torch for compatibility with the original project
import torch
from tqdm import tqdm

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (main_path, os.path.join(main_path, "datasets"), os.path.join(main_path, "backbone"), os.path.join(main_path, "models")):
    if path not in sys.path:
        sys.path.append(path)

from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from datasets.seq_wsi import parse_folds
from configs.experiment_loader import config_to_argv, load_experiment_config
from models import get_all_models, get_model, validate_model_configuration
from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.training import train

try:
    import setproctitle
except ImportError:
    setproctitle = None


def _resolve_exp_desc(args) -> None:
    """Resolve experiment-name placeholders after all CLI overrides."""
    buffer_size = getattr(args, "buffer_size", None)
    replacements = {
        "method": args.model,
        "backbone": getattr(args, "backbone", "generic_mil"),
        "buffer_tag": (
            f"buffer{buffer_size}" if buffer_size is not None else "nobuffer"
        ),
        "buffer_size": buffer_size if buffer_size is not None else "na",
    }
    try:
        args.exp_desc = str(args.exp_desc).format(**replacements)
    except KeyError as error:
        supported = ", ".join("{" + key + "}" for key in replacements)
        raise ValueError(
            f"Unknown exp_desc placeholder {error.args[0]!r}; supported: {supported}"
        ) from error


def parse_args():
    bootstrap = ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap.add_argument("--model", type=str, default=None)
    bootstrap.add_argument("--backbone", type=str, default=None)
    bootstrap.add_argument("--load_best_args", action="store_true")
    known, _ = bootstrap.parse_known_args()
    configured = load_experiment_config(known.config, method=known.model, backbone=known.backbone)
    model_name = known.model or configured.get("model")
    if not model_name:
        bootstrap.error("--model is required either on the CLI or in --config YAML")
    if model_name not in get_all_models():
        bootstrap.error(f"unknown model {model_name!r}")
    if known.load_best_args and configured:
        bootstrap.error("--config and --load_best_args cannot be combined")

    torch.set_num_threads(4)
    module = importlib.import_module("models." + model_name)

    if known.load_best_args:
        parser = ArgumentParser(description="ConSlide", allow_abbrev=False)
        parser.add_argument("--model", type=str, required=True, choices=get_all_models())
        parser.add_argument("--load_best_args", action="store_true")
        add_management_args(parser)
        parser.add_argument("--dataset", type=str, required=True, choices=DATASET_NAMES)
        if hasattr(module, "Buffer"):
            parser.add_argument("--buffer_size", type=int, required=True)
        initial = parser.parse_args()
        model_key = "sgd" if initial.model == "joint" else initial.model
        selected = best_args[initial.dataset][model_key]
        selected = selected[initial.buffer_size] if hasattr(module, "Buffer") else selected[-1]
        model_parser = getattr(module, "get_parser")()
        command = sys.argv[1:] + [f"--{key}={value}" for key, value in selected.items()]
        command.remove("--load_best_args")
        args = model_parser.parse_args(command)
    else:
        model_parser = getattr(module, "get_parser")()
        yaml_arguments = config_to_argv(configured)
        args = model_parser.parse_args(yaml_arguments + sys.argv[1:])

    _resolve_exp_desc(args)
    if args.seed is not None:
        set_random_seed(args.seed)
    return args


def _prepare_fold(args, fold: int):
    dataset = get_dataset(args)
    if not hasattr(dataset, "preflight"):
        raise ValueError("--folds/preflight integration is currently available for seq-wsi")
    dataset.preflight(fold)
    args.fold = fold
    args.n_tasks = dataset.N_TASKS
    args.task_order = list(dataset.task_order)
    args.task_num_classes = list(dataset.task_num_classes)
    args.class_offsets = list(dataset.class_offsets)
    args.num_classes = dataset.total_num_classes
    # Kept for third-party methods; it is deliberately non-scalar.
    args.n_classes_per_task = tuple(dataset.task_num_classes)
    return dataset


def run_fold(args, fold: int) -> None:
    # Validate method/backbone contracts before resolving any pretrained model.
    # This keeps unsupported combinations from touching the model cache or
    # attempting a download.
    validate_model_configuration(args)
    dataset = _prepare_fold(args, fold)
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    model_module = importlib.import_module("models." + args.model)
    if hasattr(model_module, "Buffer") and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    # Release allocator cache left by the previous fold so each fold's peak
    # reserved-memory measurement starts from a clean baseline.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    backbone = None if hasattr(model_module, "build_model") else dataset.get_backbone()
    model = get_model(args, backbone, dataset.get_loss(), dataset.get_transform())
    if setproctitle is not None:
        setproctitle.setproctitle(f"{args.exp_desc}_fold{fold}")
    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args, fold)
    else:
        ctrain(args)


def main(args=None) -> int:
    args = args or parse_args()
    folds = parse_folds(args.folds)
    args.selected_folds = list(folds)
    failures = []
    if args.preflight_only:
        for fold in folds:
            try:
                dataset = get_dataset(args)
                dataset.preflight(fold)
                print(f"[preflight] fold {fold}: OK")
            except Exception as error:
                failures.append((fold, error))
                print(str(error), file=sys.stderr)
        if failures:
            print(f"[preflight] {len(failures)}/{len(folds)} fold(s) failed", file=sys.stderr)
            return 1
        return 0

    for fold in tqdm(
        folds,
        desc="folds",
        disable=bool(getattr(args, "non_verbose", False)),
    ):
        run_fold(args, fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
