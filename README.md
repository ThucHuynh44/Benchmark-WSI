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

Configs for ATLAS-MIL, AMIL, A-GEM, DER++, ER-ACE, online EWC, GDumb, Joint,
LwF, SGD, LWSR, MICIL, OWLoRA, and QPMIL-VL are combined in
`configs/methods.yaml`. Select
one with `--model`:

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

### ATLAS-MIL

ATLAS-MIL combines a prompt-anchored class atlas, multi-positive continual
InfoNCE, prompt-conditioned masked latent reconstruction, MaxMinRand replay,
attention distillation, and fixed-rank semantic soft-orthogonal LoRA merging.
Its slide classifier blends projected TITAN text anchors with empirical FEATHER
centroids and does not require a task ID at inference.

Run the primary configuration explicitly because the common YAML backbone is
TITAN, which does not expose genuine patch attention:

```
python utils/main.py --config configs/methods.yaml \
  --model atlas_mil --backbone feather
```

`generic_mil` is also supported for synthetic tests. Both modes require 768-D
features, a full input bag, and a memory capacity at least as large as the
global class count. The TITAN text model ID/revision is configured separately
through `atlas_text_model_id` and `atlas_text_revision`; it is loaded only to
create fixed class anchors. All ATLAS hyperparameters in the shared YAML are
benchmark-defined starting points rather than published paper defaults.

#### ATLAS-MIL ablations

The declarative ablation matrix is stored in
`configs/atlas_mil_ablations.yaml`. It resolves to 34 unique variants and 340
fold-runs; listing or dry-running the matrix does not start training:

```
python scripts/run_atlas_ablations.py list
python scripts/run_atlas_ablations.py dry-run \
  --variants atlas_ce_noreplay wo_replay full --folds 0
```

Run selected variants in the project's PyTorch environment. Every process owns
one result/checkpoint directory, so completed folds can be resumed safely. A
partial fold is rerun only when explicitly requested:

```
python scripts/run_atlas_ablations.py run \
  --variants full wo_nce wo_reconstruction --folds all --gpus 0,1

python scripts/run_atlas_ablations.py resume \
  --variants full wo_nce wo_reconstruction --folds all --gpus 0,1 \
  --rerun-incomplete
```

Results are isolated below
`results/ablations/atlas_mil/<variant>/fold_<fold>/`. Build the benchmark,
paired-delta, resource, and latent-mechanism tables with:

```
python scripts/summarize_atlas_ablations.py
python scripts/summarize_atlas_ablations.py --strict
```

`sgd_ft` is an external reference, not an additive ATLAS stage. The
`centroid_with_prompt_fallback` variant uses centroids only for finalized
classes and deliberately falls back to their prompt anchors otherwise.

For the architecture-selection stage, run the true no-LoRA control on all ten
folds and the two intermediate Attention-KD weights only on folds 0–2:

```
python scripts/run_atlas_ablations.py run \
  --variants full_no_lora --folds all --gpus 0

python scripts/run_atlas_ablations.py run \
  --variants att_w025 att_w05 --folds 0,1,2 --gpus 0
```

`full_no_lora` keeps replay, losses, prompts, centroids, and the frozen FEATHER
backbone from `full`, but does not attach LoRA modules. This differs from
`atlas_lora_mode=none`, which still trains and merges LoRA but skips the
orthogonal projection. `atlas_ce` and `add_attention` remain the weight-0 and
weight-1 endpoints of the Attention-KD pilot; they do not need to be rerun.

### LWSR, MICIL, OWLoRA, and QPMIL-VL

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
  --model lwsr --backbone gigapath \
  --dataset_config configs/datasets_gigapath.yaml

python utils/main.py --config configs/methods.yaml \
  --model owlora --backbone titan

python utils/main.py --config configs/methods.yaml \
  --model owlora --backbone feather

python utils/main.py --config configs/methods.yaml \
  --model qpmil_vl --backbone titan
```

Supported combinations are deliberately narrow:

| Method | TITAN | FEATHER | GigaPath | `generic_mil` | Frozen backbone |
| --- | --- | --- | --- | --- | --- |
| ATLAS-MIL | no | yes | no | tests | base frozen; LoRA/projector/decoder train |
| LWSR | yes | yes | yes | no | no |
| MICIL | yes | yes | yes | no | no |
| OWLoRA | yes | yes | no | no | no |
| QPMIL-VL | yes | no | no | no | TITAN text tower is always frozen |

Unsupported combinations fail before loading a pretrained model. LWSR and
MICIL use 768-D raw bags with TITAN/FEATHER and 1536-D raw bags with GigaPath;
their slide embedding and classifier input remain 768-D. OWLoRA requires 768-D
features. LWSR, MICIL, and OWLoRA require a trainable slide backbone, while
QPMIL-VL loads only the pinned TITAN text tower and does not use the TITAN slide
aggregator. ATLAS-MIL owns its freezing and adaptation policy internally.

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

OWLoRA adapts eligible encoder `nn.Linear` layers while leaving the global
27-class classifier unwrapped. Task 0 fine-tunes the base model with no LoRA
parameters. After its best checkpoint is restored, OWLoRA truncates each base
weight at 99% singular-value energy, creates the frozen reference, and adds the
first rank-8 task adapter. Later tasks train only the newest adapter and current
classifier rows. Classification uses all seen logits and global labels;
adapters are expanded dynamically and reconstructed from checkpoint tensor
shapes before strict loading. The integration does not include CDATMIL or PPL.

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
checksums, and adaptation details. LWSR retains its MIT license. CoMEL-OWLoRA,
MICIL, and QPMIL-VL are internal-research-only integrations. In particular,
these sources and adaptations must not be redistributed without a separate
rights review and any required permission.

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
- `coords.attrs["patch_size_level0"]`: positive integer metadata (required by
  GigaPath; optional with a configured fallback for older backbones).

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

The native `gigapath` profile consumes the 1536-D tile embeddings produced by
the Prov-GigaPath feature pipeline and emits a 768-D slide embedding. Copy
`configs/datasets_gigapath.yaml.example` to the ignored local
`configs/datasets_gigapath.yaml`, then edit paths if needed. Its preflight
requires the recorded `patch_size_level0`, integer non-negative coordinates,
and positional grids in `[0,999]`; the supplied config also declares
non-overlapping tiling, so duplicate grids are errors. Other configs receive a
collision warning unless they opt into that guarantee. Coordinate columns are
passed through exactly as stored.

Use the dedicated environment without modifying `merge_thuc`:

```bash
conda env create -f environment.gigapath.yaml
conda activate benchmark_gigapath
```

The source package and `slide_encoder.pth` are pinned separately. GigaPath
`--precision auto` resolves to FP16; BF16 is available on supported CUDA GPUs,
and FP32 is rejected because the pinned CUDA FlashAttention path accepts only
FP16/BF16. Other backbones keep the legacy FP32 path under `auto`.

Pinned snapshots/files are cache-only by default. In the relevant environment,
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

## ATLAS-v2 additive ladder

ATLAS-v2 is independent of the historical `atlas_mil` implementation and
registry. Its settings are declared in
`configs/atlas_v2_ablations.yaml`. Replay settings retain at most 30 selected
WSIs total and store every selected WSI's complete pre-extracted feature bag;
there is no patch selection, teacher target, attention KD, reconstruction,
manifold loss, or SOLM projection.

Validate the six-setting, three-fold Phase-1 pilot without launching jobs:

```bash
python scripts/run_atlas_v2_ablations.py dry-run \
  --variants \
    atlasv2_base_frozen \
    atlasv2_base_lora \
    atlasv2_lora_replay \
    atlasv2_lora_replay_proto \
    atlasv2_lora_replay_proto_realign \
    atlasv2_frozen_proto \
  --folds 0,1,2 \
  --gpus 0
```

Replace `dry-run` with `run` to launch those 18 fold-runs. Prompt and NCE are
implemented as Phase-2 settings but are deliberately excluded from this pilot.
Generate the non-ranking summary with:

```bash
python scripts/summarize_atlas_v2_ablations.py
```

The summarizer reads replay accounting from each run manifest and reads every
evaluation matrix only once. Legacy manifests recorded replay accounting before
training; for those runs the first summary uses memory-mapped checkpoint
metadata and writes `replay_memory_cache.json` beside the summary. Later calls
reuse that small cache and never deserialize the full feature-bag storage.

The optional `atlasv2_base_lora_svd_orthogonal` geometry extension otherwise
matches `atlasv2_base_lora`. It uses SVD at task boundaries to track the left
subspace of merged LoRA updates and hard-projects each later LoRA update onto
the orthogonal complement of that historical subspace.

`atlasv2_base_lora_comel_owlora` is a separate LoRA-strategy control adapted
from this repository's CoMEL OWLoRA implementation. At initialization it
SVD-truncates each eligible frozen FEATHER linear weight at 99% energy and
creates a frozen reference adapter. It then learns one weighted low-rank
adapter per task, cumulatively applies all learned task adapters, adds CoMEL's
intra-adapter orthogonality penalty, and projects the current adapter gradients
away from the reference and previous task adapters. As in CoMEL, `qkv` layers
use three times the configured rank. Unlike the original CoMEL trainer, which
full-tunes task 0, this ATLAS-v2 adaptation keeps the pretrained base frozen and
uses an adapter from task 0 so the comparison respects ATLAS-v2's backbone
contract. Run the new setting alone with:

```bash
python scripts/run_atlas_v2_ablations.py dry-run \
  --variants atlasv2_base_lora_comel_owlora \
  --folds 0
```

The strategy-specific controls are `--atlasv2_comel_svd_energy` (default
`0.99`) and `--atlasv2_comel_orthogonal_weight` (default `1.0`). Task adapters
are intentionally not merged, so their parameter memory grows linearly with
the number of tasks.

The prototype-focused LoRA × replay factorial contains all four requested
cells:

| Setting | LoRA | Replay | Prototype | Linear CE |
| --- | --- | --- | --- | --- |
| `atlasv2_frozen_proto` | off | off | on | off |
| `atlasv2_replay_proto` | off | on | on | off |
| `atlasv2_lora_proto` | on | off | on | on |
| `atlasv2_lora_replay_proto` | on | on | on | on |

The two frozen cells disable classifier training because their post-task NCM
inference has no trainable representation. The LoRA cells train the linear CE
head because it supplies the gradient used to learn LoRA. Consequently,
`atlasv2_replay_proto` is an intentional negative control: replay memory is
populated, but a frozen embedding space should make its prototype predictions
match `atlasv2_frozen_proto` up to numerical determinism.

### Train-only frozen prototype extension

The completed results motivate keeping the FEATHER representation frozen:
`atlasv2_frozen_proto` reaches bACC `0.6856` and BWT `-0.0656`, while the
ATLAS-MIL `pruned_neither` control reaches `0.6927` and `-0.0607`; the gap is
small compared with their fold variation. In contrast, ATLAS-v2 LoRA and
prototype realignment increase drift substantially. The controlled extension
`atlasv2_frozen_proto_oas_lda` therefore changes only the prototype metric.

At each task boundary it extracts embeddings from the current **training
split**, accumulates normalized per-class means and pooled within-class scatter,
then fits equal-prior LDA with automatic Oracle Approximating Shrinkage (OAS).
This is a regularized Mahalanobis nearest-class-mean classifier: ordinary cosine
NCM assumes a spherical metric, whereas this setting downweights noisy feature
directions learned from training statistics. Old training data are represented
exactly by sufficient statistics; no WSI replay buffer is needed. Evaluation is
strictly read-only: it never updates means, covariance, shrinkage, or classifier
weights from validation/test inputs.

```bash
python scripts/run_atlas_v2_ablations.py dry-run \
  --variants atlasv2_frozen_proto_oas_lda \
  --folds 0
```

Replace `dry-run` with `run --gpus 0` to train it. This is an experimental
candidate, not a claimed improvement until its fold results are complete.

### ATLAS frozen distribution suite

The distribution suite adds ten frozen-FEATHER settings, from normalized
diagonal and low-rank class distributions through deterministic spherical
multi-prototypes, task-centroid/LME calibration, prototype-only tuning, and an
architecture-faithful RanPAC Phase-2 control. `atlasv2_frozen_atlas_tf` is the
training-free core; `atlasv2_frozen_atlas_pt` is the optional train-only
prototype-offset extension. Neither method adapts on test slides.

All implementation shapes use the runtime FEATHER classifier input dimension;
the pinned FEATHER revision is additionally asserted to produce 512-D slide
embeddings. NCM/ATLAS statistics use normalized embeddings, while RanPAC uses
raw slide embeddings. Checkpoints contain distribution sufficient statistics,
not train/validation slide embeddings.

Model selection is fold-specific and validation-only. It is staged rather than
a cross-stage Cartesian search: covariance, multi-prototype, and task-LME
stages lock their selected predecessors. Calibration manifests are written to
`results/<exp_desc>/evaluation/calibration/fold_<fold>.json` and explicitly
record that no test cache is present.

```bash
python scripts/run_atlas_v2_ablations.py dry-run \
  --variants atlasv2_frozen_proto_diag atlasv2_frozen_atlas_tf \
  atlasv2_frozen_atlas_pt atlasv2_frozen_ranpac \
  --folds 0
```

Experimental `run`/`resume` commands require a clean Git worktree by default.
`--allow-dirty` exists only for development smoke tests; manifests then include
a source-diff hash.

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
