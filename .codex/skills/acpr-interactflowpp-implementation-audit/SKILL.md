---
name: acpr-interactflowpp-implementation-audit
description: Blocking adversarial audit for ACPR-InteractFlow++ V1 on the public PSI DAMO-compatible 11,902-sample protocol.
---

# ACPR-InteractFlow++ V1 Implementation Audit Skill

## 1. Authority

This skill is the only formal training authorization gate for ACPR-InteractFlow++ V1.

Formal training is forbidden until this skill writes:

```text
.background_runs/acpr_interactflow_pp_v1_preflight/
REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt
```

The pass is valid only for the exact clean local HEAD equal to `github/acpr_interactflow_pp_v1`.

Any code, config, test, plan, skill, script, or manifest change invalidates it.

---

## 2. Required context

Read before auditing:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/runbooks/ACPR_InteractFlowPP_V1_Implementation_Plan.md
docs/runbooks/ACPR_InteractFlowPP_V1_Implementation_Manifest.json
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml

E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626\PSI_DAMO_COMPATIBLE_DATASET_REPORT.txt
E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626\manifest.json
```

Use an independent Agent B reviewer. Do not trust Agent A's checklist without dynamic evidence.

---

## 3. Git/worktree gate

Expected:

```text
worktree = E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree
branch = acpr_interactflow_pp_v1
remote = github
```

Reject unless:

1. exact worktree and branch;
2. source branch/SHA recorded;
3. `git status --porcelain` empty;
4. `git diff --check` passes;
5. local HEAD equals `git ls-remote github refs/heads/acpr_interactflow_pp_v1`;
6. runtime artifacts, datasets, checkpoints, predictions, caches, and visuals are ignored;
7. plan/config/skill/manifest hashes recorded.

Write `git_provenance.json`.

---

## 4. Formal import graph gate

Formal entrypoints:

```text
fate_oia.engine.train_acpr_interactflow_psi
fate_oia.engine.eval_acpr_interactflow_psi
fate_oia.acpr_interactflow.model.ACPRInteractFlowPPModel
```

Reject reachable formal imports/instantiations of:

```text
ACPROIAModel as main model
train_acpr_oia as trainer
BDDOIAMultiTaskDataset as formal PSI dataset
cached-logit tail adapters
FATE-X / BDD-X trainers
```

Allowed low-level reuse:

```text
ACPRDinoFieldExtractor
ACPRScenePredicateHead
ACPREgoRegionEncoder
AspectRatioLetterboxTransform
ACPR threshold/calibration utilities when train-only
```

Write `formal_import_graph.json`.

---

## 5. Config-binding gate

Every non-comment YAML field must have:

```text
runtime consumer
audit consumer
or explicit documentation-only allowlist
```

Dynamically mutate and prove runtime changes for:

```text
frame count
anchor frames
image resolution
predicate count
lag count
loss weights
best selector
profile candidates
test audit sample count
```

Reject orphan config.

Write `config_binding_report.json`.

---

## 6. Dataset protocol gate

Verify exact PSI package:

```text
train 8873
val 612
test 2417
total 11902
Exp29 dim 29
video leakage 0
action order maintain/reduce/stop
```

Verify sample fields:

```text
input_frames length = 15
target_frame = start + 15
action_soft_target sums to 1
paper_effective_weight present
raw explanation/reasoning text preserved
```

Reject if formal input uses target-frame image.

Write `psi_dataset_contract.json`.

---

## 7. DAMO metric parity gate

Run evaluator on:

1. synthetic exact fixtures;
2. local DAMO reproduction prediction artifacts if available;
3. sanity predictions with known confusion matrix.

Require parity for:

```text
Act_oAcc
Act_mAcc
Maintain/Reduce/Stop recall
Exp_oF1
Exp_mF1
Exp_mAP
joint if local DAMO defines it
```

Reject custom metrics silently replacing DAMO-compatible metrics.

Write `damo_metric_parity.json`.

---

## 8. Direct-image/no-cache gate

Trace file reads in a real smoke.

Require:

```text
frames [B,15,3,H,W]
feature_cache_enabled=false
token_cache_enabled=false
logit_cache_enabled=false
```

Reject:

```text
precomputed visual features
cached DINO tokens
cached logits
target-frame-only formal input
```

Write `direct_image_no_cache_audit.json`.

---

## 9. OIA transfer gate

Require:

- exact 32 OIA predicate names/order;
- source checkpoint path and SHA256;
- query/prototype tensor key;
- true text embedding for predicate names;
- dimension mapper;
- trainable PSI residual;
- gradients to mapper/residual.

Reject:

- anonymous predicates;
- hash-byte pseudo-embeddings;
- unresolved checkpoint;
- mismatch with `configs/acpr_scene_predicates.yaml`.

Write `oia_transfer_audit.json`.

---

## 10. Predicate ontology gate

Require total predicate set:

```text
OIA 32 + PSI-specific predicates
```

Every predicate has:

```text
name
group
region/corridor prior
supervision mode
positive phrases
contradictions or explicit reason why none
structural weak rules where needed
```

Write `predicate_ontology_audit.json`.

---

## 11. Dynamic predicate field gate

On real and synthetic batches verify:

```text
logits/probs [B,15,P]
tokens [B,15,P,D]
anchor evidence maps [B,A,P,H,W]
confidence [B,15,P]
centroids [B,15,P,2]
corridor mass [B,15,P,4]
temporal stats present
```

Require:

- entmax evidence has sparse zeros;
- anchor evidence and interpolated proxy are distinguished;
- temporal order changes predicate states;
- region priors affect expected predicates;
- confidence nonconstant;
- gradients to temporal module and predicate residual.

Write `dynamic_predicate_audit.json`.

---

## 12. nnPU/CalAlign gate

Synthetic text tests must cover:

```text
pedestrian crossing
pedestrian waiting
road/path clear
traffic light red/green
car yielding
occlusion
light traffic exclusion from traffic_light
```

Verify:

- positive labels;
- reliable negatives;
- unlabeled;
- unknown-only rows produce no hard-negative BCE gradient;
- nonnegative PU risk finite;
- class priors train-only;
- thresholds/temperatures train-only;
- test evaluation does not mutate calibration;
- save/resume exact.

Real smoke must show nonzero known positive count for at least some predicates/states.

Write `nnpu_calalign_audit.json`.

---

## 13. Exp29 weak-label gate

Verify:

- Exp29 labels dim 29;
- all-zero rows are masked/unlabeled, not 29 negatives;
- reliability weights loaded/constructed;
- medoid/top phrases attached;
- positive-mask-only metrics computed;
- no all-zero dominance in loss.

Write `exp29_weak_label_audit.json`.

---

## 14. Interaction-flow state gate

Require exact names:

```text
Regime: clear_to_go, caution_required, yielding_required, stop_required
Phase: waiting, approaching, entering, crossing, leaving, uncertain
Source: pedestrian_conflict, crosswalk_context, traffic_signal, front_vehicle_yielding, side_occlusion, intersection_constraint
Corridor: left_sidewalk_zone, center_ego_path, right_sidewalk_zone, crosswalk_zone
```

Verify factor tokens read:

```text
dynamic predicates
temporal stats
corridors
motion tokens
grammar priors
semantic-name embeddings
```

Reject one global vector plus decorative classifiers.

Write `interaction_flow_audit.json`.

---

## 15. Response-lag gate

Require lags `0..4`, causal masks, normalized weights.

Synthetic delayed-event test must recover known lag.

Temporal reverse must change phase/lag behavior.

Disabling lag must change decisions on delayed cases.

Write `response_lag_audit.json`.

---

## 16. Decision ledger gate

Require exact identity:

```text
final_logits = global_logits + sum(gated_state_contributions)
```

Verify:

- action classes maintain/reduce/stop;
- global branch independently supervised;
- state contributions nonzero after mechanism fit;
- benefit gate has detached benefit target;
- non-degradation hinge has correct sign;
- contribution magnitude is not minimized as residual;
- state-off removes/recomputes the exact intended path.

Write `decision_ledger_audit.json`.

---

## 17. Loss gate

Every configured nonzero loss must:

- exist as finite tensor;
- affect total exactly once;
- have raw/weight/weighted logs;
- give gradients to intended modules.

Reject:

```text
hard action CE as sole action loss
all-zero Exp29 negative BCE
contribution magnitude residual loss
state probability mean minimization
pattern logit square-to-zero
test thresholds copied into model
```

Write `loss_audit.json`.

---

## 18. Optimizer/scheduler/precision gate

Verify:

- explicit optimizer groups;
- each trainable parameter exactly once;
- frozen DINO base absent from optimizer unless explicitly adapter/unfreeze mode;
- BF16 autocast active where configured;
- scheduler warmup/cosine saves/resumes;
- gradient clip applied;
- no NaN/Inf.

Write `optimizer_precision_audit.json`.

---

## 19. Real smoke gate

Run a direct-image smoke:

```text
8 train samples
8 test samples
8 optimizer steps
batch 1
full formal losses
test evaluation
checkpoint_latest
```

Require:

- forward/backward;
- finite loss;
- nonzero gradients for formal modules;
- DAMO-compatible metrics JSON;
- no cache;
- no target-frame input.

Write `gate_real_direct_image_smoke.json`.

---

## 20. 128-sample mechanism-fit gate

On deterministic 128 train samples run bounded updates.

Require improvement in:

```text
global action KL
final action KL
ledger residual
predicate known-label nnPU
interaction state weak loss
Exp29 positive-mask loss
contribution alignment
```

Reject collapse:

```text
all maintain predictions
all stop impossible
all predicates constant
all states constant
all benefit gates zero
all contributions zero
Exp29 attention uniform
```

Write `gate_mechanism_fit_128.json`.

---

## 21. Temporal/lag gate

Synthetic and real subset:

```text
reverse/shuffle changes state phase
lag disabled changes delayed-case prediction
last-frame-only differs from full15
prefix 5/10/15 shows expected information growth
```

Write `gate_temporal_lag.json`.

---

## 22. Intervention gate

Supported:

```text
global-only
regime-off
phase-off
source-off
factor-off
predicate-off
evidence-tube-off
equal-mass random
temporal reverse/shuffle
lag disabled
last-frame-only
prefix 5/10/15
```

Every intervention reruns from the earliest affected layer.

Equal-mass random must match evidence mass or patch count.

Require actual action probability deltas and Exp29 deltas.

Write `intervention_audit.json`.

---

## 23. Visualization gate

One real case must produce a complete Dynamic Interaction Decision Ledger PNG/JSON:

```text
frame strip
predicate evidence tubes
interaction-flow ribbons
response-lag panel
exact decision waterfall
Exp29 factor alignment
counterfactual twin
tensor lineage
```

Mini atlas must include grouped cases, failure cases, and intervention distributions.

Reject placeholder HTML/JSON.

Write `visual_artifact_index.json`.

---

## 24. Throughput/memory gate

Run every candidate with:

```text
10 warm-up batches
100 measured full forward/backward batches
real dataloader
all losses
direct images
```

Require:

```text
peak reserved <=44 GiB
preferred 34–42 GiB when throughput-optimal
projected epoch + test eval <=2.5 hours
data_time <=25%
no nonfinite/skip
```

Select highest samples/sec.

No dummy allocation.

Write `throughput_profile.json`.

---

## 25. Test-only protocol and best selector gate

Require:

- full test after every epoch;
- no validation loader in formal training;
- test does not update calibration/thresholds/model/optimizer/scheduler;
- best files separated;
- best selector exactly matches config;
- no metric early stop.

Hash mutable state before/after test.

Write `test_protocol_audit.json`.

---

## 26. Foreground supervisor gate

Static scan forbids:

```text
Start-Process
Start-Job
nohup
shell &
schtasks
hidden window
DETACHED_PROCESS
```

Dynamic smoke requires:

- attached child;
- live stdout/stderr;
- heartbeat;
- review/local/remote SHA check;
- epoch artifact checks;
- OOM fallback;
- latest resume;
- `.tmp` rejection;
- no metric early stop;
- user stop sentinel only if explicitly requested.

Write `foreground_supervisor_audit.json`.

---

## 27. Required report files

Preflight dir must contain:

```text
git_provenance.json
formal_import_graph.json
config_binding_report.json
psi_dataset_contract.json
damo_metric_parity.json
direct_image_no_cache_audit.json
oia_transfer_audit.json
predicate_ontology_audit.json
dynamic_predicate_audit.json
nnpu_calalign_audit.json
exp29_weak_label_audit.json
interaction_flow_audit.json
response_lag_audit.json
decision_ledger_audit.json
loss_audit.json
optimizer_precision_audit.json
gate_real_direct_image_smoke.json
gate_mechanism_fit_128.json
gate_temporal_lag.json
intervention_audit.json
visual_artifact_index.json
throughput_profile.json
test_protocol_audit.json
foreground_supervisor_audit.json
implementation_manifest.json
review_report.json
```

A report without command, exit code, tensor evidence, and artifact paths is insufficient.

---

## 28. Review pass

Only after all gates pass write:

```text
REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt
```

It must contain:

```text
timestamp
reviewer role
source SHA
target worktree/branch/SHA
GitHub SHA
clean status
plan/config/skill/manifest hashes
all commands and exit codes
PSI dataset contract
DAMO metric parity
OIA transfer
dynamic predicate proof
nnPU proof
Exp29 weak-label proof
interaction-flow proof
response-lag proof
decision-ledger exact identity
loss direction proof
gradient proof
intervention proof
visual proof
throughput profile
foreground proof
```

Authorization statement:

```text
ACPR_INTERACTFLOW_PP_V1_IMPLEMENTATION_REVIEW_PASS
The reviewed local HEAD equals the pushed GitHub acpr_interactflow_pp_v1 HEAD.
The worktree is clean.
The PSI public DAMO-compatible data protocol, DAMO metrics, OIA predicate transfer,
dynamic predicates, calibrated nnPU, interaction-flow states, response-lag alignment,
exact decision ledger, weak Exp29 explanation, interventions, visualization,
throughput/memory policy, and foreground supervisor all passed.
Formal execution is authorized for this exact commit only.
```

On any failure:

1. delete stale pass;
2. write blockers;
3. add regression test;
4. fix;
5. commit/push;
6. rerun affected gates and full audit;
7. do not train.
