# ACPR-PMCal-V2 Implementation Audit Skill

Version: 2026-06-28  
Purpose: block incomplete or misleading implementations of ACPR-PMCal-V2 before training.

This audit is intentionally strict. Code that merely trains is insufficient. The implementation must prove that the discussed design has actually been implemented and called in the active training/evaluation path.

---

## 1. Mandatory context

Before any audit, code change, training, evaluation, process management, commit, or push under `E:\sbw\FATE_Drive`, read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Do not create extra training-status Markdown files. Experiment records go only into those three files.

---

## 2. PMCal-V2 method contract

The implementation must be a direct-image model with this active path:

```text
image
  -> frozen DINO
  -> predicate measurement from visual tokens and bounded text prior
  -> train-only noisy predicate observation supervision
  -> PU reason state
  -> formula-guided reason residual
  -> predicate-aware direct action head
  -> PU-CalAlign deploy logits
  -> certified near-boundary HardPair
```

The implementation must be rejected if the active path is any of:

```text
cached-logit adapter
RunC/CalAlign checkpoint distillation
checkpoint soup
PMI/co-occurrence/graph label propagation
expert/MoE/router/selector
generic evidence memory slot bank
action-set marginalization as final action
```

---

## 3. Hard non-negotiables

Reject unless all are true:

1. Direct image training only.
2. No feature cache build.
3. No feature cache read.
4. Token compression is `none`.
5. DINO ViT-S/8 is frozen and no-grad.
6. Test-only evaluation.
7. Best checkpoint selected only from test.
8. No val loader.
9. No `checkpoint_best_val`.
10. No historical best-checkpoint distillation.
11. No RunC / cached-logit / tail adapter.
12. No graph, PMI, co-occurrence, or label graph.
13. No expert/MoE/router/selector architecture.
14. No generic evidence memory.
15. Final action is 4-dimensional sigmoid multi-label logits.
16. Final reason is 21-dimensional sigmoid multi-label logits.
17. Final action does not consume reason logits.
18. Final action does not consume reason labels.
19. Final action does not consume PU state.
20. Final action does not consume reason formula residual.
21. Final action does not consume action-set subset probabilities.
22. BDD100K geometry is train-only supervision, not test-time input.
23. Text prompt prior is bounded and cannot override visual evidence.
24. PU reason loss treats unannotated labels as unknown unless certified negative.
25. HardPair mines only P/N reliable near-boundary pairs.
26. Deploy logits are exactly `base_logits - theta`.
27. Test oracle threshold search is diagnostic only.
28. Every epoch writes required PMCal artifacts.
29. Foreground supervisor streams stdout/stderr and uses no hidden background process.
30. Full training is blocked until this audit writes the review pass for exact HEAD.

---

## 4. Required files

Reject if any are missing:

```text
configs/fate_oia_train_360x640_acpr_pmcal_v2.yaml
configs/acpr_pmcal_v2_text_prompts.yaml

fate_oia/models/acpr_pmcal_v2_model.py
fate_oia/models/acpr_pmcal_label_head.py
fate_oia/models/pmcal_predicate_observation_builder.py
fate_oia/models/pmcal_predicate_measurement.py
fate_oia/models/pmcal_reason_formula_bank.py
fate_oia/models/pmcal_reason_formula_head.py
fate_oia/models/pmcal_pu_reason_state.py
fate_oia/models/pmcal_pu_calalign_head.py
fate_oia/models/pmcal_action_predicate_head.py

fate_oia/losses/pmcal_losses.py
fate_oia/losses/pmcal_certified_pair_loss.py

fate_oia/utils/pmcal_artifacts.py
fate_oia/utils/pmcal_forbidden_scan.py
fate_oia/utils/pmcal_memory_probe.py

fate_oia/optim/pmcal_conflict_aware_optimizer.py

fate_oia/engine/train_pmcal_v2_oia.py
fate_oia/engine/eval_pmcal_v2_oia.py
fate_oia/engine/audit_pmcal_v2_implementation.py
fate_oia/engine/supervise_pmcal_v2_foreground.py

scripts/FATE_OIA_acpr_pmcal_v2_foreground.ps1
```

Reject if tests are missing:

```text
tests/test_pmcal_predicate_observation_builder.py
tests/test_pmcal_predicate_measurement.py
tests/test_pmcal_reason_formula_bank.py
tests/test_pmcal_reason_formula_head.py
tests/test_pmcal_pu_reason_state.py
tests/test_pmcal_action_independence.py
tests/test_pmcal_pu_calalign_head.py
tests/test_pmcal_certified_pair_loss.py
tests/test_pmcal_model_forward.py
tests/test_pmcal_train_protocol.py
tests/test_pmcal_audit.py
tests/test_pmcal_supervisor_foreground.py
tests/test_pmcal_memory_probe.py
tests/test_pmcal_forbidden_patterns.py
```

---

## 5. Forbidden source patterns

Scan only active PMCal files, configs, scripts, and the active supervisor/trainer. Do not fail on legacy ACPR files unless the PMCal trainer imports/calls the forbidden path.

Reject if active files contain or enable:

```text
frozen_run_c
FrozenRunC
run_c_logits
cached_logits
checkpoint_best_test_action_mf1 used as teacher
checkpoint_best_test_deploy_raw used as teacher
distill_from_calalign
distill_from_checkpoint
teacher_checkpoint
tail_residual_adapter
ComplementaryLogitFusionAdapter
PMI
pmi
cooccurrence
co-occurrence
label_graph
graph_delta_to_logits
Graph
expert
Expert
MoE
moe
router
Router
selector
Selector
evidence_memory
generic evidence slot
feature_cache_enabled: true
token_compression: keep_merge
token_compression: true
checkpoint_best_val
best_selection_split: val
eval_splits: val
Start-Process
Start-Job
nohup
hidden
scheduled task
daemon
```

Allowed exceptions:

- Existing legacy files may contain `reason_to_action`; PMCal active files must not call it for final action.
- The audit file may mention forbidden strings as strings to scan for.
- The word `residual` is allowed for neural residuals only.

---

## 6. Static config checks

Load `configs/fate_oia_train_360x640_acpr_pmcal_v2.yaml`.

Reject unless:

```yaml
feature_cache_enabled: false
token_compression: none
eval_splits: [test]
best_selection_split: test
best_selection_metric: deploy_fixed_joint
experiment.direct_image: true
experiment.test_only_evaluation: true
runtime.no_feature_cache: true
runtime.require_no_token_compression: true
model.use_reason_to_action_final: false
model.action_set_affects_final_action: false
model.graph_delta_to_logits: false
```

Check training:

```text
epochs == 18 unless explicitly overridden by user
memory probe ladder exists
target_allocated_gpu_gb_max <= 45.0
absolute_oom_guard_gb <= 46.0
```

Check threshold:

```text
threshold.enabled == true
train_calib_fraction exists
teacher thresholds train_calib only
test oracle diagnostic only
```

---

## 7. Dataset and loader audit

Instantiate train/test datasets with real transforms:

```python
train_ds = BDDOIAMultiTaskDataset(..., split="train", load_image=True)
test_ds = BDDOIAMultiTaskDataset(..., split="test", load_image=True)
```

Verify:

```text
train count > 0
test count > 0
image tensor exists
action shape [4]
reason shape [21]
targets are multi-hot floats
no softmax CE over action
```

Reject if the PMCal trainer builds a val loader or writes val artifacts.

---

## 8. DINO and image path audit

Instantiate `ACPRPMCalV2Model` with real DINO weights and run one real batch.

Verify:

```text
patch_tokens_by_layer shape [B,3,3600,384]
cls_tokens_by_layer shape [B,3,384]
grid_hw == (45,80)
all DINO parameters require_grad=False
DINO forward occurs under no_grad
no cache files are written or read
```

Reject if any cache path is touched.

---

## 9. Predicate measurement audit

### 9.1 Shape and gradient

Run:

```python
out = model(images, split="train", action_labels=..., reason_labels=..., file_names=..., structured_records=...)
```

Verify:

```text
q_pred shape [B,32] or [B,M>=32]
rho_pred shape [B,M]
predicate_tokens shape [B,M,384]
predicate_attention shape [B,M,3600]
predicate logits finite
rho_pred not all 0
rho_pred not all 1
gradients reach predicate queries and visual heads
```

### 9.2 Fair posterior rule

The fair posterior used by final action/reason must be computed from:

```text
visual tokens
bounded text prompt prior
```

It must not depend on:

```text
reason_labels
action_labels
BDD100K structured_records
train/test split label values
```

Dynamic check:

```python
out_a = model(images, split="train", reason_labels=reason_y, structured_records=records)
out_b = model(images, split="train", reason_labels=torch.zeros_like(reason_y), structured_records=different_records)

# q_pred used for final logits must be invariant to train-only observations.
assert max_abs(out_a["q_pred_fair"] - out_b["q_pred_fair"]) < 1e-6
```

The supervised observation targets may differ; final fair posterior may not.

### 9.3 Geometry leakage

Eval/test check:

```python
out_none = model(images, split="test", structured_records=None, reason_labels=None)
out_fake = model(images, split="test", structured_records=fake_records, reason_labels=fake_labels)
assert max_abs(out_none["logits_deploy"] - out_fake["logits_deploy"]) < 1e-6
```

Reject if test forward reads geometry or labels.

---

## 10. Predicate observation builder audit

Run builder with:

```text
real file_names
real reason labels
real BDD100K records
missing records
fake split="test"
```

Verify:

```text
obs_reason_value [B,M]
obs_reason_mask [B,M]
obs_geometry_value [B,M]
obs_geometry_mask [B,M]
obs_geometry_reliability [B,M]
missing records do not crash
unknowns are masked out
proxy reliability <= 0.15
split="test" gives zero train-only masks
```

Geometry parser must support:

```text
box2d
raw poly2d [[x,y,type], ...]
list-of-dict poly2d
lane records
drivable records
```

---

## 11. Reason formula audit

Load formula bank from grammar.

Verify:

```text
exactly 4 actions: forward, stop, left, right
exactly 21 reasons
no placeholder reason names
positive_matrix shape [21,M]
contradiction_matrix shape [21,M]
compatible_action_matrix shape [21,4]
hard_negative_matrix shape [21,21]
tail_indices == [12,9,5,14,6,11,10,13]
```

Run formula head:

```text
reason_formula_logits [B,21]
reason_formula_gate [B,21]
support_score [B,21]
contra_score [B,21]
formula residual cap <= 0.20
formula gate max <= 0.35
```

Reject if graph/PMI/co-occurrence matrices appear.

---

## 12. PU reason state audit

Synthetic test:

```text
case 1: y_r=1 -> P
case 2: y_r=0, no support, high contradiction, high reliability -> N
case 3: y_r=0, low reliability -> U
case 4: y_r=0, support present -> U, not N
```

Verify:

```text
positive_mask + unknown_mask + reliable_negative_mask == 1 per reason
not all y=0 become negatives
negative count is less than all unannotated count
tail labels handled explicitly
```

Loss check:

```text
positive gradients reach logits
reliable negative gradients reach logits
unknown entropy term is finite and bounded
increasing contradiction increases negative penalty only when reliability is high
```

---

## 13. Final action independence audit

This is a blocking check.

### 13.1 No reason logits into final action

Run forward normally and with reason formula branch zeroed:

```python
out_normal = model(images, split="test")
out_no_formula = model(images, split="test", force_zero_reason_formula=True)
assert max_abs(out_normal["action_logits_base"] - out_no_formula["action_logits_base"]) < 1e-6
assert max_abs(out_normal["action_logits_deploy"] - out_no_formula["action_logits_deploy"]) < 1e-6
```

### 13.2 No reason labels into final action during train

Run with different reason labels:

```python
out_a = model(images, split="train", reason_labels=reason_y, action_labels=action_y)
out_b = model(images, split="train", reason_labels=1-reason_y, action_labels=action_y)
assert max_abs(out_a["action_logits_base"] - out_b["action_logits_base"]) < 1e-6
```

Observation losses may differ. Final action logits must not.

### 13.3 No legacy reason-to-action final path

Scan PMCal files. Reject if final action uses:

```text
reason_to_action
action_reason_logits
reason_logits_visual
reason_logits_final
reason_formula_logits
action_set_logits
subset_membership
```

The legacy trunk may still expose these as diagnostics outside PMCal. They must not affect PMCal final action.

---

## 14. PU-CalAlign audit

Run PMCal threshold head with synthetic logits.

Verify:

```text
logits_base = concat(action_logits, reason_logits)
logits_deploy = logits_base - threshold_logit
action_logits_deploy = logits_deploy[:,:4]
reason_logits_deploy = logits_deploy[:,4:]
theta_global exists
theta_instance_delta exists
instance delta is bounded
threshold_prob finite and within configured ranges
```

Verify train/test fairness:

```text
threshold at test depends only on logits/q_pred/rho_pred/cardinality predictions
threshold at test does not depend on labels or structured_records
```

Verify diagnostic separation:

```text
base_fixed metrics saved separately
deploy_fixed primary metrics saved separately
calibrated metrics saved separately
test oracle per-label diagnostic clearly named diagnostic only
test oracle not copied to model parameters
```

---

## 15. Certified HardPair audit

Synthetic miner check:

```text
P sample near boundary, N sample near boundary, high reliability -> pair exists
P sample and U sample -> no pair
P sample and N low reliability -> no pair
P sample and N far from boundary -> no pair
tail reason -> higher max pair budget than common reason
```

Loss check:

```text
z_pos <= z_neg + margin produces positive hinge
z_pos >> z_neg produces zero hinge
loss finite when no pairs
pair cap applies
pair_count_per_reason logged
```

Reject if unknown negatives are mined.

---

## 16. Conflict-aware optimizer audit

Run a toy model with conflicting action/reason gradients.

Verify:

```text
grad cosine logged
projection_applied_count > 0 in conflict case
optimizer updates parameters
no NaN in grads
disabled mode gives ordinary optimizer step
```

This optimizer is not allowed to silently suppress all reason gradients. Log gradient norms for each group.

---

## 17. Training protocol audit

Inspect and smoke `train_pmcal_v2_oia.py`.

Reject unless:

```text
train_loader exists
test_loader exists
val_loader absent
eval_splits == ["test"]
best_selection_split == "test"
checkpoint_best_val absent
no val logits
no feature cache
no token compression
every epoch writes required artifacts
best checkpoint selection uses deploy_fixed_joint
test oracle threshold is diagnostic only
```

Check full smoke command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_pmcal_v2_oia `
  --config configs\fate_oia_train_360x640_acpr_pmcal_v2.yaml `
  --output_dir .background_runs\acpr_pmcal_v2_smoke `
  --epochs 1 `
  --batch_size 1 `
  --gradient_accumulation_steps 2 `
  --max_train_samples 4 `
  --max_test_samples 4 `
  --device cuda `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression `
  --require_review_pass `
  --review_pass_path .background_runs\acpr_pmcal_v2_preflight\REVIEW_PASS_PMCalV2.txt
```

Smoke must emit:

```text
pmcal_train_batch
pmcal_epoch_complete
checkpoint_latest.pth
checkpoint_best_test_deploy_raw.pth
metrics_summary.jsonl
predicate_measurement_stats.jsonl
pu_state_stats.jsonl
threshold_stats.jsonl
action_independence_stats.jsonl
```

---

## 18. Foreground supervisor audit

Inspect:

```text
fate_oia/engine/supervise_pmcal_v2_foreground.py
scripts/FATE_OIA_acpr_pmcal_v2_foreground.ps1
```

Reject if any appear:

```text
Start-Process
Start-Job
nohup
hidden
scheduled task
daemon
metric early stop
```

Accept only if:

```text
foreground child stdout/stderr streamed
review pass required
exact git HEAD checked against audit JSON
memory probe runs before full train
OOM fallback ladder exists
NaN/Inf handling exists
dataloader stall detection exists
no metric early stop
GOAL_COMPLETED written only after configured epochs finish
```

---

## 19. Memory probe audit

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.utils.pmcal_memory_probe `
  --config configs\fate_oia_train_360x640_acpr_pmcal_v2.yaml `
  --device cuda `
  --target_min_gb 40 `
  --target_max_gb 45 `
  --output .background_runs\acpr_pmcal_v2_preflight\memory_probe_result.json
```

Verify:

```text
selected batch/accum in ladder
peak allocated <= 45 GiB
reserved <= 46 GiB
OOM attempts logged
fallback attempts logged
one selected setting writes forward/backward proof
```

If no setting reaches 40GB but training is stable, allow training with a warning. If any setting exceeds 46GB and is selected, reject.

---

## 20. Required preflight commands

Run py_compile:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_pmcal_v2_model.py `
  fate_oia\models\acpr_pmcal_label_head.py `
  fate_oia\models\pmcal_predicate_observation_builder.py `
  fate_oia\models\pmcal_predicate_measurement.py `
  fate_oia\models\pmcal_reason_formula_bank.py `
  fate_oia\models\pmcal_reason_formula_head.py `
  fate_oia\models\pmcal_pu_reason_state.py `
  fate_oia\models\pmcal_pu_calalign_head.py `
  fate_oia\models\pmcal_action_predicate_head.py `
  fate_oia\losses\pmcal_losses.py `
  fate_oia\losses\pmcal_certified_pair_loss.py `
  fate_oia\utils\pmcal_artifacts.py `
  fate_oia\utils\pmcal_forbidden_scan.py `
  fate_oia\utils\pmcal_memory_probe.py `
  fate_oia\optim\pmcal_conflict_aware_optimizer.py `
  fate_oia\engine\train_pmcal_v2_oia.py `
  fate_oia\engine\eval_pmcal_v2_oia.py `
  fate_oia\engine\audit_pmcal_v2_implementation.py `
  fate_oia\engine\supervise_pmcal_v2_foreground.py
```

Run PMCal tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_pmcal_predicate_observation_builder.py `
  tests\test_pmcal_predicate_measurement.py `
  tests\test_pmcal_reason_formula_bank.py `
  tests\test_pmcal_reason_formula_head.py `
  tests\test_pmcal_pu_reason_state.py `
  tests\test_pmcal_action_independence.py `
  tests\test_pmcal_pu_calalign_head.py `
  tests\test_pmcal_certified_pair_loss.py `
  tests\test_pmcal_model_forward.py `
  tests\test_pmcal_train_protocol.py `
  tests\test_pmcal_audit.py `
  tests\test_pmcal_supervisor_foreground.py `
  tests\test_pmcal_memory_probe.py `
  tests\test_pmcal_forbidden_patterns.py `
  -q
```

Run ACPR regression subset:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_calalign_forward.py `
  tests\test_acpr_threshold_head.py `
  tests\test_acpr_threshold_losses.py `
  tests\test_acpr_train_calib_split.py `
  tests\test_acpr_reason_grammar.py `
  tests\test_acpr_predicate_targets.py `
  tests\test_acpr_scene_predicate_head.py `
  tests\test_acpr_train_protocol.py `
  -q
```

Run implementation audit:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_pmcal_v2_implementation `
  --config configs\fate_oia_train_360x640_acpr_pmcal_v2.yaml `
  --output_dir .background_runs\acpr_pmcal_v2_preflight `
  --device cuda `
  --write_review_pass
```

Required pass file:

```text
.background_runs\acpr_pmcal_v2_preflight\REVIEW_PASS_PMCalV2.txt
```

---

## 21. Audit JSON schema

The audit must write:

```text
.background_runs\acpr_pmcal_v2_preflight\implementation_audit_PMCalV2.json
```

Required fields:

```json
{
  "pass": true,
  "git_head": "...",
  "branch": "acpr_pmcal_v2_direct_image",
  "worktree": "E:\\sbw\\FATE_Drive\\fate_oia_acpr_pmcal_v2_worktree",
  "source_branch": "acpr_calalign_v1_2",
  "checked_files": [],
  "forbidden_pattern_results": {},
  "config_checks": {},
  "dataset_checks": {},
  "dino_checks": {},
  "predicate_measurement_checks": {},
  "fair_posterior_checks": {},
  "geometry_leakage_checks": {},
  "reason_formula_checks": {},
  "pu_state_checks": {},
  "action_independence_checks": {},
  "threshold_checks": {},
  "certified_pair_checks": {},
  "conflict_optimizer_checks": {},
  "training_protocol_checks": {},
  "supervisor_checks": {},
  "memory_probe": {},
  "smoke_result": {},
  "review_pass_path": "...",
  "missing_items": [],
  "warnings": [],
  "hard_failures": []
}
```

`REVIEW_PASS_PMCalV2.txt` may be written only when:

```text
pass == true
hard_failures == []
missing_items == []
```

---

## 22. Runtime artifact audit after each epoch

Reject or mark run invalid if any completed epoch lacks:

```text
metrics_summary.jsonl row
metrics_latest.json
epoch_xxx/metrics.json
epoch_xxx/per_label_reason_metrics.json
threshold_stats row
predicate_measurement_stats row
pu_state_stats row
formula_stats row
certified_pair_stats row
grad_conflict_stats row
action_independence_stats row
logits_action_deploy_test.pt
logits_reason_deploy_test.pt
labels_action_test.pt
labels_reason_test.pt
```

Best checkpoint audit:

```text
metrics_best_test.json epoch must match checkpoint_best_test_deploy_raw.pth metadata.
```

---

## 23. Full training launch gate

Full training remains forbidden unless all are true:

```text
canonical md files were read
new worktree created from acpr_calalign_v1_2
source worktree unmodified
py_compile PASS
PMCal tests PASS
ACPR regression tests PASS
implementation audit PASS
REVIEW_PASS_PMCalV2.txt exists
real smoke PASS
code-only commit made
GitHub push succeeded OR push/auth/TLS blocker recorded
memory probe selected a safe batch/accum
```

Full launch command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_pmcal_v2_foreground.ps1 `
  -Epochs 18 `
  -BatchSize 9 `
  -GradAccum 4 `
  -Device cuda `
  -TargetGpuGB 45 `
  -RequireReviewPass
```

---

## 24. Post-run success/failure interpretation

Primary ACPR-CalAlign target to beat:

```text
Act_mF1 >= 0.7200
Exp_mF1 >= 0.4150
deploy_fixed_joint >= 0.5680
```

Also require:

```text
Exp_mAP not materially lower than ACPR-CalAlign reference
tail macro-F1 improves by >= 0.02 for [12,9,5,14,6,11,10,13]
action independence checks pass every epoch
test geometry leakage checks pass
```

If only F1 improves and AP does not, label the gain as calibration/boundary gain, not representation gain.

If action drops while explanation rises, the action-safe design failed.

If explanation rises only under test-oracle threshold, do not claim official improvement.

---

## 25. Final report checklist

Codex final report must include:

```text
worktree path
branch
source branch/head
local HEAD
GitHub HEAD or push blocker
py_compile result
pytest result
audit JSON path
review pass path
smoke path
memory probe result
full run path
selected batch/accum
GPU memory peak
best epoch
best Act_mF1
best Exp_mF1
best joint
best Exp_mAP
tail reason metrics
artifact completeness
whether PMCal-V2 exceeded ACPR-CalAlign
known failures
confirmation .background_runs not committed
confirmation source worktree not modified
```
