# ConSlide
[ICCV 2023] ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis.

## Training Data Preparation

We mainly follow the pipeline of [CLAM](https://github.com/mahmoodlab/CLAM). The modified version of the CLAM code for data preparation will be released later.

## Training Example

```
cp configs/datasets.yaml.example configs/datasets.yaml

python utils/main.py --model conslide --dataset seq-wsi \
  --exp_desc conslide --buffer_size 1100 --alpha 0.2 --beta 0.2 \
  --backbone generic_mil --feature_dim 768 --folds all
```

`configs/datasets.yaml` is local and git-ignored. It accepts `{data_root}` in
annotation, feature and split paths. `task_order` is authoritative and may be
reordered or reduced to a subset; use `reverse_task_order: true` in YAML (or
`--reverse_task_order`) for the reverse stream. Do not store Hugging Face
tokens in this file. A custom backbone that needs one should read `HF_TOKEN`
from the environment.

Before a long run, validate one fold or all ten folds without creating a model:

```
python utils/main.py --model sgd --dataset seq-wsi --exp_desc preflight \
  --preflight_only --folds 0

python utils/main.py --model sgd --dataset seq-wsi --exp_desc preflight \
  --preflight_only --folds all
```

Fold syntax also supports lists and inclusive ranges such as `0,2,4-6`.
Training creates a fresh dataset/model for every selected fold and writes
checkpoints below `checkpoints/<exp_desc>/fold_<fold>/`.

## Method YAML configs

Configs for AMIL, A-GEM, DER++, ER-ACE, online EWC, GDumb, Joint, LwF, SGD,
LWSR, MICIL, and QPMIL-VL are combined in `configs/methods.yaml`. Select one with
`--model`:

```
python utils/main.py --config configs/methods.yaml --model agem
python utils/main.py --config configs/methods.yaml --model derpp
python utils/main.py --config configs/methods.yaml --model derpp --backbone titan
python utils/main.py --config configs/methods.yaml --model derpp --backbone feather
```

Explicit command-line values override YAML values, for example:

```
python utils/main.py --config configs/methods.yaml --model ewc_on \
  --folds 0 --n_epochs 3 --exp_desc ewc_debug
```

These files contain baseline starting values rather than tuned best
hyperparameters. Edit `configs/datasets.yaml` for local data paths; keep tokens
out of both dataset and method configs.

### AMIL (PMP + AKD + Logit KD)

AMIL is the benchmark name for a paper reimplementation of Attention Knowledge
Distillation (AKD), logit knowledge distillation, and the Pseudo-Bag Memory
Pool (PMP). The benchmark always enables the complete method, uses only the
MaxMinRand patch selector, and keeps a class-balanced reservoir of 30
pseudo-bags. Run its two supported backbone configurations explicitly:

```
python utils/main.py --config configs/methods.yaml \
  --model amil --backbone generic_mil

python utils/main.py --config configs/methods.yaml \
  --model amil --backbone feather
```

AMIL requires a trainable backbone with genuine patch-level attention and a
full input bag (`backbone_max_patches=0`). TITAN is intentionally unsupported:
its current adapter synthesizes uniform attention when the encoder does not
return patch attention.

After each task's best checkpoint is restored, PMP selects at most 400 patches
per accepted WSI. At `end_task`, the best student re-forwards every retained
pseudo-bag and caches its attention and logits. Those cached outputs implement
the previous-session distillation target for the next task without retaining a
frozen teacher model. This refresh is a Benchmark-WSI adaptation; it is not
claimed as an implementation detail published verbatim by the paper.

The AMIL values `pmp_k=400`, `alpha=1`, `beta=1`, and
`kd_temperature=1` are benchmark-defined defaults because the available paper
and supplementary material do not publish concrete values for the first three
or a separate KD temperature. They must not be described as paper defaults.

### LWSR, MICIL, and QPMIL-VL

The new methods are run through the same ten-task/27-class CLI and evaluator:

```
python utils/main.py --config configs/methods.yaml \
  --model lwsr --backbone titan

python utils/main.py --config configs/methods.yaml \
  --model lwsr --backbone feather

python utils/main.py --config configs/methods.yaml \
  --model micil --backbone titan --no-micil_replay

python utils/main.py --config configs/methods.yaml \
  --model micil --backbone feather --micil_replay

python utils/main.py --config configs/methods.yaml \
  --model qpmil_vl --backbone titan
```

Supported combinations are deliberately narrow:

| Method | TITAN | FEATHER | `generic_mil` | Frozen backbone |
| --- | --- | --- | --- | --- |
| LWSR | yes | yes | no | no |
| MICIL | yes | yes | no | no |
| QPMIL-VL | yes | no | no | TITAN text tower is always frozen |

Unsupported combinations fail before loading a pretrained model. All three
methods require 768-D patch features. LWSR and MICIL fine-tune their slide
backbone, while QPMIL-VL loads only the pinned TITAN text tower and does not use
the TITAN slide aggregator.

The LWSR defaults are `buffer_size=10`, `minibatch_size=4`,
`bags_per_update=4`, `buffer_max_patches=400`, `pair_loss_weight=1.0`,
`ce_loss_weight=1.0`, and `dc_loss_weight=0.01`.

MICIL defaults to the original no-replay mode (`micil_replay=false`) with
classifier weight normalization enabled. Its loss defaults are
`ce_loss_weight=1.0`, `kd_loss_weight=10.0`,
`embedding_loss_weight=1.0`, and `distillation_temperature=2.0`; logical
updates use `bags_per_update=1`. `--micil_replay` enables the explicit replay
extension with `buffer_size=30`, `minibatch_size=4`, and
`buffer_max_patches=400`. Use `--no-micil_replay` to select the original mode
explicitly. Classifier normalization can likewise be selected with
`--micil_weight_norm` or `--no-micil_weight_norm`. For these paired selectors,
a later explicit CLI value wins over its YAML inverse.

QPMIL-VL defaults to `pool_size=20`, `prompt_length=24`, `match_size=5`,
`bags_per_update=16`, `backbone_max_patches=400`, max pooling,
`csm_logit_scale=100`, `classification_logit_scale=1`, `alpha=0.5`, and matching
and class-similarity weights of `0.5` each. It uses AdamW with `lr=1e-5`,
`optim_wd=1e-4`, `adam_eps=1e-8`, and gradient clipping at `1.0`.

Explicit CLI values remain last and therefore override YAML. For example,
`--model lwsr --backbone titan --bags_per_update 2 --buffer_size 20` changes
only those two LWSR values; `--model micil ... --micil_replay` overrides the
YAML no-replay default.

The copied upstream sources are immutable provenance snapshots under
`third_party/upstream/`; active runtime code does not import them. See
`INTEGRATION_CHANGES.md` and each `SOURCE_MANIFEST.json` for source revisions,
checksums, and adaptation details. LWSR retains its MIT license. MICIL and
QPMIL-VL are internal-research-only integrations. In particular, the adapted
QPMIL-VL code must not be pushed to a public repository or distributed without
a separate rights review and any permission required by its CC BY-NC-ND 4.0
license.

The strict preflight checks annotation coverage, disjoint train/val/test IDs,
class coverage, feature presence, and the exact HDF5 shapes. It never filters
slides or guesses labels. With the current local annotation files it therefore
reports blockers for BRCA, RCC and NSCLC; synchronize the annotations/splits
before starting a full run.

## Evaluation metrics and artifacts

Evaluation follows the canonical schema used by the sibling `Benchmark`
repository. After every task, both `class-il-seen` and `task-il` report and
store accuracy, balanced accuracy, macro F1, weighted F1, AUROC, Cohen's kappa,
cross-entropy loss, and sample count. Fold summaries additionally contain
mACC, BWT, forgetting (FGT), training time, total evaluation time, and average
inference time per evaluated task. They also record total/trainable parameter
counts and per-fold peak CUDA allocated/reserved memory (MiB). The same resource
metadata is stored in `run_manifest.json`; CPU runs report CUDA peaks as N/A.

Validation-loss early stopping is configured in `configs/methods.yaml`:

```yaml
early_stopping: true
early_stopping_patience: 3
early_stopping_min_epoch: 1
early_stopping_min_delta: 0.0
early_stopping_verbose: true
```

`early_stopping_min_epoch` counts completed epochs. Setting `early_stopping` to
`false` runs all `n_epochs`, but the best validation checkpoint is still restored
before testing.

By default, `evaluate_fwt: false` skips all test evaluation before learning a
task. The stream therefore runs `train task -> test evaluation -> next task`.
Set `evaluate_fwt: true` only when the initial and future-task evaluations
needed for Forward Transfer are required; otherwise FWT is reported as NaN.

Each experiment writes two directly comparable artifact trees:

```text
results/<exp_desc>/evaluation/class_il/
results/<exp_desc>/evaluation/task_il/
```

Each tree contains:

```text
run_manifest.json
eval_matrix.csv
per_slide_predictions.csv
per_fold_summary.csv
per_task_summary.csv
confusion_matrices/
```

`eval_matrix.csv` has one row per `(fold, after_task, eval_task)`. Per-slide
output includes global/local labels, probabilities and logits for all classes,
patch counts, slide metadata, and correctness. Multiple runs can be aggregated
with:

```bash
python scripts/summarize_results.py \
  --input results \
  --output results/all_methods
```

## Native MIL backbones and single-feature input

The WSI loader uses a backbone-independent HDF5 schema:

- `features`: float tensor with shape `[num_patches, feature_dim]`.
- `coords`: integer tensor with shape `[num_patches, 2]`.
- `coords.attrs["patch_size_level0"]`: optional positive integer metadata.

For the ten-dataset stream, `coords` is required by strict preflight. MIL batch
size is fixed to one because bags have variable patch counts. The default class
counts are `[4,2,3,2,2,2,2,3,2,5]` (27 global outputs), with offsets computed
from the configured task order. Evaluation always reports both Class-IL (only
seen classes) and Task-IL (the current task slice), task ROC-AUC, and global
seen-class macro one-vs-rest ROC-AUC.

`coords` is positional metadata and is no longer treated as a second feature
scale. If its patch-size attr is absent, the loader uses
`--patch_size_level0_fallback` (1024 by default); the HDF5 value always wins.
The built-in `generic_mil` backbone uses gated-attention pooling and accepts any
feature dimension through `--feature_dim`.

The native `titan` and `feather` profiles require 768-D CONCH features. TITAN
uses at most 400 patches per slide by default; training samples randomly and
validation/test use deterministic evenly spaced indices. FEATHER and generic
MIL use the full bag. Both pretrained profiles fine-tune the complete slide
encoder unless `--backbone_freeze` is supplied.

Pinned snapshots are cache-only by default. In the `merge_thuc` environment,
point `HF_HOME` or `--backbone_cache_dir` at the existing Hugging Face cache.
To permit a missing snapshot to be downloaded explicitly, add
`--backbone_allow_download`; authentication is read only from the `HF_TOKEN`
environment variable.

Backbone/buffer-specific experiment names default to
`<method>_<backbone>_<buffer_tag>_10tasks` (for example,
`derpp_titan_buffer30_10tasks` or `ewc_on_titan_nobuffer_10tasks`), preventing
checkpoints from different profiles from overwriting one another. Checkpoints also validate model ID, revision,
freeze state, patch budget, feature dimension and patch-size fallback.

A project-specific backbone can be selected without changing the dataset or
training loop:

```
--backbone my_package.my_backbones:MyMIL \
--backbone_kwargs '{"depth": 2, "num_heads": 4}'
```

The custom class may accept `forward(features)`, `forward(features, coords)`,
or `forward(features, coords, patch_size_level0)` and return a logits tensor, a
dictionary with `logits`, or ConSlide's five-item output tuple.

## Updates / TODOs
Please follow this GitHub for more updates.

- [ ] Refine the code.
- [ ] Provide code for data preparation.
- [ ] Remove dead code.
- [ ] Better documentation on interpretability code example.

## Reference
If you find our work useful in your research please consider citing our [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_ConSlide_Asynchronous_Hierarchical_Interaction_Transformer_with_Breakup-Reorganize_Rehearsal_for_Continual_ICCV_2023_paper.html):

Huang, Y., Zhao, W., Wang, S., Fu, Y., Jiang, Y., & Yu, L. (2023). ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 21349-21360).

```
@inproceedings{huang2023conslide,
  title={ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis},
  author={Huang, Yanyan and Zhao, Weiqin and Wang, Shujun and Fu, Yu and Jiang, Yuming and Yu, Lequan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21349--21360},
  year={2023}
}
```

## Acknowledgements

Framework code for Continual Learning was largely adapted via making modifications to [Mammoth](https://github.com/aimagelab/mammoth)
