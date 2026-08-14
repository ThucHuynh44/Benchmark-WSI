# ATLAS-MIL, AMIL, LWSR, MICIL, OWLoRA, and QPMIL-VL integration changes

This file records the boundary between the byte-identical upstream snapshots
under `third_party/upstream/` and the active ConSlide implementations under
`models/`. Runtime code must never import from `third_party/`.

## Provenance and source-to-active mapping

| Method | Frozen upstream revision | Core snapshot | Active implementation |
| --- | --- | --- | --- |
| OWLoRA | `9fd667994eb57e3960e36970a9509a8217d84a22` | `third_party/upstream/comel_owlora/continual/main/continual_bag/cdatmil_ppl_owlora_trainer.py` | `models/owlora.py` |
| LWSR | `7620ef944d7dabbb20504744fd244633fc3841d1` | `third_party/upstream/lwsr/models/lwsr.py` | `models/lwsr.py` |
| MICIL | `7c27d197ca522a3cfe3b0629152a07858f707bdf` | `third_party/upstream/micil/code_py/MICIL_train.py` | `models/micil.py` |
| QPMIL-VL | `3a7a7698582dec866d43eb748f8c3599f7be4391` | `third_party/upstream/qpmil_vl/models/model_il.py` | `models/qpmil_vl.py` |

Each snapshot has a `SOURCE_MANIFEST.json` containing its repository URL,
commit, file list, and SHA-256 digests. The provenance test treats those files
as immutable and also checks that active runtime modules do not import them.
The pinned CoMEL and MICIL repositories have no license file. CoMEL's
continual-bag trainer and `MICIL_train.py` are therefore recorded as the core
algorithm sources for their active adapters.

AMIL is deliberately absent from this table and from `third_party/upstream/`.
Its active implementation is a paper reimplementation; no byte-identical
official source snapshot is represented by this repository.

LWSR retains its upstream MIT license. CoMEL-OWLoRA, MICIL, and QPMIL-VL are
marked `INTERNAL_RESEARCH_ONLY`. The adapted implementations must not be pushed
to a public repository or otherwise distributed without a separate rights
review and any required permission.

## Common ConSlide contract

The active methods use ConSlide's variable-length WSI representation rather
than the fixed tensors assumed by the upstream projects:

- one bag contains `features [num_patches, 768]`, `coords [num_patches, 2]`,
  and a scalar `patch_size_level0`;
- TITAN and FEATHER expose `forward_with_embedding(...)`, returning `logits`,
  `embedding`, `attention`, and scalar `auxiliary_loss` in one pass, plus
  `get_classifier()` for methods that operate on the classification head;
- the historical five-item `forward()` result remains unchanged for existing
  ConSlide baselines;
- a logical update can group several variable-length bags while the physical
  `DataLoader` batch size remains one, including the final incomplete group;
- an `observe`/`observe_many` result may be a scalar or a dictionary whose
  required `loss` entry is the scalar used by the trainer; and
- method-owned continual state is serialized through checkpoint hooks after
  the best epoch is restored and the task-ending state update has completed.

Configuration validation rejects unsupported backbone combinations before
pretrained weights are loaded. The four snapshot-backed methods require 768-D
features. LWSR, MICIL, and OWLoRA require a trainable TITAN or FEATHER backbone;
QPMIL-VL uses only its pinned TITAN text tower and rejects FEATHER and generic
MIL.

## ATLAS-MIL active implementation

ATLAS-MIL is a native research implementation rather than a vendored upstream
snapshot. It uses FEATHER (or `generic_mil` for synthetic tests) for genuine
patch attention and loads the separately pinned TITAN text tower only long
enough to produce fixed 27-class semantic anchors. The active model freezes the
base slide aggregator, trains fixed-size active LoRA factors, and compresses
them into fixed-rank merged factors at every task boundary.

Replay stores class-balanced MaxMinRand pseudo-bags with cached attention and
slide embeddings. After best-checkpoint restoration, ATLAS first updates the
reservoir, merges the task adapter, recomputes current-class atlas statistics
with the compressed model, and finally refreshes every replay target. The
hybrid prompt/centroid logits preserve the benchmark's global-classifier output
contract and require no task ID at inference. CICS, CGRL, alternate selectors,
and logit distillation are intentionally outside this implementation.

## OWLoRA active changes

The active adapter retains CoMEL's weighted low-rank residuals, initial SVD
truncation, intra-adapter orthogonality penalty, and projection of each new
adapter's gradients away from the frozen reference and prior adapters. It
deliberately excludes CDATMIL and PPL and instead wraps eligible `nn.Linear`
objects inside the native TITAN or FEATHER slide encoder. The global classifier
is never wrapped.

Task 0 contains no adapter and fine-tunes the base encoder. After the restored
best Task-0 checkpoint, the method creates the frozen reference and first task
adapter. Each subsequent non-final task appends exactly one adapter. Training
uses global labels and cross-entropy over all seen logits, while old and future
classifier rows are restored after AdamW steps so only the current rows change.
Dynamic adapter shapes are inferred from state-dict keys and materialized before
strict checkpoint loading. Projection uses associative products rather than
forming hidden-dimension-square matrices, preserving the source formula with a
smaller memory peak.

## AMIL active implementation

AMIL combines a class-balanced Pseudo-Bag Memory Pool with Attention Knowledge
Distillation and logit knowledge distillation. The benchmark exposes only the
complete method and its MaxMinRand selector. GenericMIL and FEATHER are
supported with trainable, uncapped slide backbones; TITAN is rejected because
its current adapter does not expose genuine patch-level attention.

After the best checkpoint for task `t` is restored, `save_buffer()` uses the
full current-task WSIs to update the class-balanced reservoir. `end_task()`
then re-forwards every retained pseudo-bag with that best model and atomically
refreshes its cached attention, logits, and seen-class boundary. Thus all
replay entries in task `t+1` use outputs from the previous session without
retaining a frozen teacher. An entry's origin task remains unchanged when its
targets are refreshed.

Cached-target refresh is a Benchmark-WSI adaptation used to realize the
previous-session teacher; it is not claimed to be an implementation detail
published verbatim by the paper. Likewise, `pmp_k=400`, `alpha=1`, `beta=1`,
and `kd_temperature=1` are benchmark-defined defaults rather than claimed
paper defaults.

## LWSR active changes

The active adapter preserves the upstream pair, classification, and distance
consistency objectives while making these changes:

- replaces upstream `[batch, 2048, 512]` input and the legacy ViT execution
  path with variable-length 768-D TITAN/FEATHER bags;
- replaces every eight-class assumption with the stream-wide
  `args.num_classes` (27 in the default stream), and constructs one-hot targets
  from integer labels with the correct batch shape;
- always defines pair loss and cross-entropy, including later tasks when the
  replay reservoir is empty;
- stores patch-limited bags on CPU in a seeded reservoir instead of retaining
  GPU tensors;
- concatenates current-group and replay logits/embeddings before computing the
  pair and classification terms;
- computes distance consistency against the rows and columns for the exact
  reservoir indices replayed in that update;
- rebuilds the reference distance matrix only after the task's best model has
  been restored and the reservoir has been updated; and
- checkpoints reservoir entries, number seen, reservoir RNG state, current
  task, and the reference distance matrix.

The upstream retrieval-only mAP, R@3, P@5, cancer grouping, and rank-correlation
evaluation paths are intentionally not active. Evaluation uses ConSlide's
Class-IL and Task-IL protocol with the canonical Benchmark metrics: accuracy,
balanced accuracy, macro/weighted F1, AUROC, Cohen's kappa, loss, BWT, FGT,
timing, per-slide predictions, and confusion matrices.

## MICIL active changes

The active adapter removes upstream CUDA, six-class, iterator-index cache, and
incomplete task-checkpoint assumptions. It uses a single `micil` model with
positive/negative `--micil_replay` and `--no-micil_replay` forms; replay is off
by default. The same form is used for classifier weight normalization. If YAML
and the explicit CLI supply inverse forms, the later CLI occurrence wins.

For both modes, the first task uses class-balanced cross-entropy. On later
tasks, the best student from the preceding task becomes an eval-only frozen
teacher. Distillation is restricted to old-class logits, embedding matching is
student/teacher slide-embedding MSE, cross-entropy sees only the seen-class
slice, and unseen logits are masked from the training objective. Optional
classifier weight normalization operates through `get_classifier()`.
The KD term deliberately retains the upstream
`KL(log_softmax(student/T), softmax(teacher/T))` with `batchmean` reduction and
no additional `T^2` multiplier, so the default `kd_loss_weight=10` preserves
the intended scaling.

Replay mode additionally creates a seeded CPU reservoir of patch-limited WSI
bags. Each update combines current and sampled replay bags; CE, old-class KD,
and embedding matching are evaluated over that combined group. Class weights
at task start include the current split and buffered label distribution. The
reservoir is populated after restoring the best epoch. Its content is present
in checkpoints only when replay is enabled, and restore rejects a replay-mode,
capacity, patch-budget, or feature-dimension mismatch. The no-replay mode does
not create or access a buffer.

## QPMIL-VL active changes

The active adapter keeps the upstream queryable prototype mechanism but moves
it from the 512-D CONCH alignment space to TITAN's 768-D patch/text space:

- the TITAN text tower creates fixed class features and continuous-prompt
  prototypes; the TITAN slide aggregator is not used;
- text-tower parameters remain frozen and in evaluation mode while gradients
  still flow from text features into learned prompt tokens;
- prompt-pool keys and prompts remain trainable across tasks, while only the
  current task's tunable class vectors receive gradients;
- all bags in one logical update vote for a shared majority top-k prototype
  selection, including a final undersized group;
- prior-task key frequencies produce the subsequent task's matching penalty;
- training uses CE plus matching and class-similarity losses over seen classes,
  then pads evaluator logits to the full 27-class width; and
- the active prompt registry covers every class in the configured ten-task
  stream in either task order. The original eight-class JSON remains only in
  the frozen snapshot.

Adaptation checkpoints contain the prompt keys, prompts, tunable vectors,
class features, key-frequency state, task ID, prompt-registry hash, and pinned
TITAN identity. Frozen TITAN weights are reloaded from that exact model
ID/revision instead of being duplicated in the checkpoint; restore rejects a
revision or prompt-hash mismatch.

## Upstream code retained only for comparison

Legacy LWSR ViT/TransMIL helpers and other upstream support files listed by the
manifests remain in the frozen snapshots for provenance. They are not active
ConSlide backbones and are not imported by training or evaluation.
