"""Joint-training upper bound for the configured WSI task stream."""

from pathlib import Path

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args
from utils.optim import build_optimizer
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Joint training baseline.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = "joint"
    COMPATIBILITY = ["class-il", "task-il"]

    def end_task(self, dataset, fold):
        if len(dataset.test_loaders) != dataset.N_TASKS:
            raise RuntimeError("Joint training requires all task loaders")
        self.net = dataset.get_backbone().to(self.device)
        self.opt = build_optimizer(self.net.parameters(), self.args)

        # Import lazily to avoid a model-discovery import cycle.
        from utils.training import early_stopping_from_args, evaluate_val, load_checkpoint

        results_dir = Path("checkpoints") / self.args.exp_desc / f"fold_{fold}"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = results_dir / f"task{dataset.N_TASKS - 1}_checkpoint.pt"
        early_stopping = early_stopping_from_args(self.args)

        epoch_bar = tqdm(
            range(self.args.n_epochs),
            desc=f"fold {fold} joint",
            leave=False,
            disable=bool(getattr(self.args, "non_verbose", False)),
        )
        for epoch in epoch_bar:
            self.net.train()
            epoch_loss = 0.0
            epoch_updates = 0
            batch_bar = tqdm(
                dataset.train_loader,
                total=len(dataset.train_loader),
                desc=f"epoch {epoch + 1}/{self.args.n_epochs}",
                leave=False,
                disable=bool(getattr(self.args, "non_verbose", False)),
            )
            for features, coords, patch_size, labels in batch_bar:
                features, coords, patch_size = self.prepare_inputs(
                    features, coords, patch_size, training=True
                )
                labels = labels.to(self.device)
                self.opt.zero_grad()
                logits = self.net([features, coords, patch_size])[0]
                loss = self.loss(logits, labels.long())
                loss.backward()
                self.opt.step()
                epoch_loss += float(loss.item())
                epoch_updates += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
            average_loss = epoch_loss / max(epoch_updates, 1)
            epoch_bar.set_postfix(loss=f"{average_loss:.4f}", refresh=False)
            if not bool(getattr(self.args, "non_verbose", False)):
                tqdm.write(
                    f"[train] fold={fold} joint epoch={epoch + 1}/{self.args.n_epochs} "
                    f"avg_loss={average_loss:.4f} updates={epoch_updates}"
                )
            if evaluate_val(
                self,
                dataset,
                dataset.N_TASKS - 1,
                epoch,
                checkpoint_path,
                fold,
                early_stopping,
            ):
                break
        load_checkpoint(self, checkpoint_path, dataset, fold)

    def observe(self, *args, **kwargs):
        return 0.0
