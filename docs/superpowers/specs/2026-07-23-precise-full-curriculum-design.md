# PRECISE-OIA Full-Run Curriculum Design

## Goal

Replace the standalone 3-epoch pilot with one uninterrupted 12-epoch full-data
training run whose first epochs are retained as part of the formal run. Protect
the direct action/reason foundation before progressively activating PRECISE
evidence rereading, semantic exchange, annotation correction, latent messaging,
counterfactual intervention, and train-calib threshold learning.

## Why This Change Is Needed

The current implementation applies reread, exchange, annotation, and latent
reason deltas to final logits from epoch 0. Only auxiliary loss weight has a
short update-level warm-up. This can corrupt immature direct representations and
also consumes each component optimizer's warm-up/cosine schedule before that
component should be trusted.

## Fixed 12-Epoch Curriculum

| Epoch | Foundation | Evidence | Reread | Annotation | Exchange | Reason latent | Intervention | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 1.00 | 1.00 | 0.35 | 0.35 | 0.00 | 0.00 | 0.00 | 0.50 |
| 3 | 1.00 | 1.00 | 0.70 | 0.70 | 0.00 | 0.00 | 0.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.35 | 0.35 | 0.25 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.70 | 0.60 | 1.00 |
| 6-11 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

These values are fixed and reproducible. Test metrics must never control stage
promotion. A non-finite loss is an execution error, not a reason to silently
change the curriculum.

## Runtime Contract

The model receives a typed activation state before every epoch. Activation
scales are applied to the corresponding deltas before final logits are formed,
not only to their losses. The intervention scale multiplies its loss. Threshold
training begins only when its scale is positive.

Inactive owners do not step their optimizer or scheduler. Each owner has its own
local optimizer-step clock, preserved in checkpoints. Resuming at an epoch must
restore both the curriculum state and all owner-local clocks exactly.

The ownership mapping is:

- `action_foundation`, `action_decoder`: always active.
- `evidence_core`: always active.
- `reason_semantic`: direct reason path, always active.
- latent evidence slots and their field parameters belong to `evidence_core` and
  may learn evidence representation from epoch 0; they do not enter final reason
  logits until the separate `reason_latent` message activation begins.
- `reread_adapter`: only rereader parameters; active from epoch 2.
- `exchange_adapter`: only semantic-exchange parameters; active from epoch 4.
- `reason_latent`: only latent query/value/gamma parameters; active from epoch 4.
- `annotation_adapter`: active when annotation scale is positive.
- `threshold_head`: active when threshold scale is positive.

Optimizers and schedulers may be constructed at launch, but an inactive owner
must have empty optimizer state, receive no weight decay, take no optimizer or
scheduler step, and have its gradients explicitly cleared at every accumulation
boundary. Its first active update starts at owner-local step zero.

Each scheduler total is computed from its actual active epochs: 12 epochs for
always-on owners, 10 for reread/annotation/threshold, and 8 for
exchange/reason-latent. Optimizer state and weight decay are not created for an
inactive owner.

The zero-scale anchor is the existing refined-head output over direct category
tokens. This preserves the current PRECISE architecture without silently
replacing it with the first-pass category logits. Artifacts call this branch
`refined_direct_anchor`.

At every boundary the contract is `effective_delta = activation * raw_delta`.
Tests assert this boundary equality rather than assuming the whole nonlinear
model is a linear interpolation. Cached intervention and diagnostic ablation
paths use the same effective deltas exactly once.

Threshold activation applies to deployment itself:

```text
theta_effective = threshold_activation * theta
deploy_logits = raw_logits - theta_effective
```

The threshold teacher remains train-calib-only.

## Artifacts And Monitoring

Every batch record, epoch metric row, checkpoint, and run manifest records:

- curriculum name and version;
- epoch and stage name;
- all eight activation scales;
- each owner active flag and owner-local optimizer step;
- raw delta RMS and effective scaled delta RMS;
- effective delta/direct ratios;
- intervention raw loss and scaled loss.
- curriculum SHA256 and owner-local active-update totals;
- refined-direct anchor logits;
- raw/effective deltas for normal, diagnostic, and cached intervention paths;
- threshold raw/effective theta.

The run remains direct-image, frozen-DINO, no-cache, no-compression, test-only
evaluation, and train-calib-only threshold learning.

## Verification

Tests must prove:

1. Exact scale values for every epoch.
2. Zero-scale deltas produce exact direct/foundation logits.
3. Partial scales alter logits by the requested proportion.
4. Inactive owners do not step optimizer or scheduler.
5. Resume restores owner-local clocks and the same activation state.
6. Prelaunch audit generates `PRECISE_OIA_V1_FULL_CURRICULUM_READY.json`, bound to the
   clean HEAD, source/config/curriculum/skill/DINO/schema hashes, current
   PRE_PILOT_ELIGIBLE checks, runtime profile, and the user's explicit
   replacement of the standalone pilot. The gate replaces only pilot artifacts.
7. Prelaunch audit verifies the static contract and simulated owner-step
   behavior. Runtime assertions verify actual scales, step increments, optimizer
   state, and LR every epoch. At epoch 6, any required owner that has not stepped
   is a protocol error: save diagnostics and abort rather than produce an invalid
   result. The old `FULL_TRAIN_READY` cannot authorize this path.
8. Logs, manifest, epoch artifacts, and checkpoints expose the full curriculum
   state.

The curriculum hash is computed from a canonical JSON serialization of the
normalized schedule, not raw YAML formatting.

The launch remains foreground-only, matching the existing reviewed protocol.
No claim is made that it survives an SSH disconnect.
