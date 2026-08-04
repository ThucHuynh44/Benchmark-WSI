# LWSR, MICIL, and QPMIL-VL integration changes

This file records the boundary between the byte-identical upstream snapshots
under `third_party/upstream/` and the active ConSlide implementations under
`models/`. Runtime code must never import from `third_party/`.

## Provenance and source-to-active mapping

| Method | Frozen upstream revision | Core snapshot | Active implementation |
| --- | --- | --- | --- |
| LWSR | `7620ef944d7dabbb20504744fd244633fc3841d1` | `third_party/upstream/lwsr/models/lwsr.py` | `models/lwsr.py` |
| MICIL | `7c27d197ca522a3cfe3b0629152a07858f707bdf` | `third_party/upstream/micil/code_py/MICIL_train.py` | `models/micil.py` |
| QPMIL-VL | `3a7a7698582dec866d43eb748f8c3599f7be4391` | `third_party/upstream/qpmil_vl/models/model_il.py` | `models/qpmil_vl.py` |

Each snapshot has a `SOURCE_MANIFEST.json` containing its repository URL,
commit, file list, and SHA-256 digests. The provenance test treats those files
as immutable and also checks that active runtime modules do not import them.
The pinned MICIL repository has neither a `micil.py` nor a license file;
`MICIL_train.py` is therefore the recorded algorithm source for the active
adapter.

LWSR retains its upstream MIT license. MICIL and QPMIL-VL are marked
`INTERNAL_RESEARCH_ONLY`. QPMIL-VL upstream is CC BY-NC-ND 4.0, so the adapted
implementation must not be pushed to a public repository or otherwise
distributed without a separate rights review and any required permission.

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
pretrained weights are loaded. All three methods require 768-D features. LWSR
and MICIL require a trainable TITAN or FEATHER backbone; QPMIL-VL uses only its
pinned TITAN text tower and rejects FEATHER and generic MIL.

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
