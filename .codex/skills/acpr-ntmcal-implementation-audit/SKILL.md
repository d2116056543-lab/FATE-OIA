# ACPR-NTMCal V1 Implementation Audit Skill

## Purpose

Audit that the repository implements **ACPR-NTMCal V1: Native-Text Predicate Measurement Calibration** as a real direct-image BDD-OIA model.

This audit must reject code that only trains but fails to implement the scientific mechanism.

ACPR-NTMCal V1 must treat BDD-OIA native action/reason texts as a structured task-internal predicate measurement language. It must not use external VLMs, text encoders, cached logits, graph/co-occurrence, or checkpoint distillation.

---

## Mandatory context rule

Before any code change, training, evaluation, process management, or GitHub push under `E:\sbw\FATE_Drive`, read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Do not create additional training-status Markdown files. Append remote experiment progress only to those three canonical files.

---

## Hard non-negotiables

Reject unless all are true:

1. Direct image training only.
2. No feature cache build.
3. No feature cache read.
4. No token compression in the primary NTMCal config.
5. DINO ViT-S/8 is frozen and used with no_grad.
6. Input resolution is 360x640, patch_size=8, patch tokens expected [B,3,3600,384].
7. Evaluation split is test only.
8. Best checkpoint is selected only from test deploy-fixed joint.
9. No val loader, no val metrics, no `checkpoint_best_val`.
10. No external VLM/text encoder.
11. No `hashlib` or hash embedding used as text semantics.
12. No graph / PMI / co-occurrence / label-correlation graph.
13. No RunC checkpoint/logit/residual/cache/distillation dependency.
14. No expert/MoE/specialist/router.
15. No action-set marginalization as final action.
16. Final action logits are 4-dimensional multi-label logits.
17. Final reason logits are 21-dimensional multi-label logits.
18. Native predicate posterior is produced from image tokens at test.
19. Reason labels are train-only observation supervision and are never used in test forward.
20. BDD100K geometry, if present, is train-only weak observation/audit and never used in test forward.
21. PU treats reason y=0 as unknown by default, not hard negative.
22. Hard reliable negatives start only after the configured epoch.
23. Pair loss is disabled until the configured pair epoch.
24. NTMCal threshold delta really depends on stopgrad support/contra/rho/margin/cardinality.
25. Deploy logits are exactly `logits - theta`.
26. Test oracle threshold search is diagnostic only and is never assigned to model parameters.
27. Every epoch writes native-text, predicate, PU, threshold, pair, tail, and action-independence artifacts.
28. Full training is forbidden unless `REVIEW_PASS_ACPR_NTMCAL_V1.txt` exists.

---

## Required files

Reject if any are missing:

```text
configs/acpr_ntmcal_native_text_predicates.yaml
configs/acpr_ntmcal_reason_formulas.yaml
configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml

fate_oia/models/acpr_ntmcal_text_atoms.py
fate_oia/models/acpr_ntmcal_predicate_bank.py
fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py
fate_oia/models/acpr_ntmcal_observation_builder.py
fate_oia/models/acpr_ntmcal_pu_state.py
fate_oia/models/acpr_ntmcal_reason_residual.py
fate_oia/models/acpr_ntmcal_action_predicate_head.py
fate_oia/models/acpr_ntmcal_threshold_head.py
fate_oia/models/acpr_ntmcal_pair_memory.py
fate_oia/models/acpr_ntmcal_model.py

fate_oia/losses/acpr_ntmcal_losses.py

fate_oia/engine/train_acpr_ntmcal_oia.py
fate_oia/engine/eval_acpr_ntmcal_oia.py
fate_oia/engine/audit_acpr_ntmcal_implementation.py
fate_oia/engine/supervise_acpr_ntmcal_foreground.py

fate_oia/utils/acpr_ntmcal_artifacts.py
fate_oia/utils/acpr_ntmcal_tensor_asserts.py

tests/test_acpr_ntmcal_text_atoms.py
tests/test_acpr_ntmcal_predicate_bank.py
tests/test_acpr_ntmcal_topk_measurement.py
tests/test_acpr_ntmcal_observation_builder.py
tests/test_acpr_ntmcal_pu_state.py
tests/test_acpr_ntmcal_reason_residual.py
tests/test_acpr_ntmcal_action_predicate.py
tests/test_acpr_ntmcal_threshold_head.py
tests/test_acpr_ntmcal_pair_memory.py
tests/test_acpr_ntmcal_model_forward.py
tests/test_acpr_ntmcal_losses.py
tests/test_acpr_ntmcal_train_protocol.py
tests/test_acpr_ntmcal_audit.py

scripts/FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1
scripts/FATE_OIA_acpr_ntmcal_v1_foreground.ps1
```

---

## Forbidden source patterns

Reject active NTMCal implementation if any NTMCal file contains or enables:

```text
hashlib
_hash_embedding
open_clip
clip.load
CLIPModel
AutoTokenizer
AutoModel
BertModel
SentenceTransformer
sentence_transformers
frozen_run_c
FrozenRunC
run_c_logits
cached_logits
tail_residual_adapter
checkpoint distillation
teacher checkpoint
expert
Expert
moe
MoE
specialist
Specialist
router
Router
graph
pmi
cooccur
co_occurrence
label_correlation
feature_cache_enabled: true
token_compression: keep_merge
token_compression: topk
checkpoint_best_val
best_selection_split: val
eval_splits: val
Start-Job
nohup
scheduled task
daemon
```

The word `residual` is allowed only for the NTMCal bounded reason residual and must not refer to RunC/cached-logit residual adaptation.

The word `graph` is allowed only in comments explaining that graph/co-occurrence is forbidden. It must not be in active model code or config.

---

## Static implementation checks

### 1. Worktree and branch

Verify:

```text
branch == acpr_ntmcal_v1_direct_image
worktree path contains fate_oia_acpr_ntmcal_v1_worktree
base branch is acpr_calalign_v1_2 or local branch derived from it
git status clean after implementation commit
```

### 2. Config

Load `configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml`.

Reject unless:

```yaml
image_height: 360
image_width: 640
patch_size: 8
training.best_selection_split: test
evaluation.splits: [test]
training.no_feature_cache: true
training.token_compression: none
model.dino_no_grad: true
model.predicate_topk <= 96
model.no_dense_predicate_field: true
ntmcal.teacher_source: train_calib
ntmcal.oracle_test_thresholds: diagnostic_only
pu.soft_negative_start_epoch: 3
pu.hard_negative_start_epoch: 7
pair.start_epoch >= 7
```

No val keys may be active.

---

## Functional audit checks

### 1. Dataset and direct-image path

Instantiate `BDDOIAMultiTaskDataset` train/test with `load_image=True`.

Verify:

```text
train count > 0
test count > 0
action shape [4]
reason shape [21]
action/reason are multi-hot floats
image tensor exists
file_name exists
```

Reject if any softmax action CE path is used.

### 2. DINO field

Run `ACPRDinoFieldExtractor` on one real image.

Verify:

```text
patch_tokens_by_layer shape [B,3,3600,384]
cls_tokens_by_layer shape [B,3,384]
grid_hw == (45,80)
all DINO params require_grad == False
no cache file is written
```

### 3. Native text predicate bank

Load `configs/acpr_ntmcal_native_text_predicates.yaml`.

Verify:

```text
predicate_count >= 40
all ids unique
all names unique
all predicates have entity, attribute, spatial, polarity, action_scope, region
all contra_predicates exist in the same bank
left/right mirror pairs complete
required names exist:
  traffic_light_red
  traffic_light_green
  traffic_light_visible
  stop_sign_present
  front_vehicle_close
  pedestrian_front
  cyclist_front
  obstacle_front
  road_clear
  lane_left_available
  lane_right_available
  left_solid_boundary
  right_solid_boundary
  lane_absent_left
  lane_absent_right
  open_left_gap
  open_right_gap
  drivable_center
  drivable_left
  drivable_right
```

Reject if placeholder names appear:

```text
predicate_0
reason_0
unknown
placeholder
tmp
dummy
```

### 4. Native text atom encoder

Instantiate `NativeTextAtomEncoder`.

Verify:

```text
entity/attribute/spatial/polarity/action_scope embeddings exist
predicate embedding shape [P,384]
embeddings are learnable parameters
no hash/text encoder import is used
structural text loss is nonzero on known contradiction/mirror pairs
gradients reach atom embeddings
```

### 5. Reason formulas

Load `configs/acpr_ntmcal_reason_formulas.yaml`.

Verify:

```text
exactly 4 actions: forward, stop, left, right
exactly 21 reasons
reason ids exactly 0..20
each reason has native text string
each reason has support_predicates and contra_predicates fields
all support/contra predicates exist in native predicate bank
compatible_actions subset of {forward, stop, left, right}
tail_reason_indices == [12,9,5,14,6,11,10,13]
```

### 6. Efficient Top-K predicate measurement

Run `NativeTextTopKPredicateMeasurement` on real DINO tokens.

Verify outputs:

```text
q_pred shape [B,P]
rho_pred shape [B,P]
predicate_tokens shape [B,P,384]
predicate_topk_indices shape [B,P,K]
predicate_topk_attention shape [B,P,K]
K <= 96
0 <= q_pred <= 1
0 <= rho_pred <= 1
rho_pred mean in [0.05,0.95]
```

Reject if any intermediate tensor with shape `[B,P,N,D]` or `[B,P,L,N,D]` is created.

Static grep reject:

```text
einsum("ms,bsnd->bmnd"
einsum('ms,bsnd->bmnd'
einsum("ls,bsnd->blnd" in NTMCal predicate measurement
view(b, p, n, d)
reshape(b, p, n, d)
```

`[B,P,L,N]` score tensor is allowed. `[B,P,K,D]` gathered value tensor is allowed.

### 7. Train-only observation builder

Run `NativeTextObservationBuilder` with synthetic reason labels.

Verify:

```text
positive reason creates positive predicate observations
positive reason creates soft-negative observations for contradictory predicates
reason y=0 creates unknown by default
obs_value contains positive and unknown states
soft-negative count can be nonzero when positive reasons have contra predicates
no geometry is required
```

Run with `split="test"` and labels/geometry supplied intentionally.

Verify:

```text
test observation builder ignores labels/geometry
returns no label-derived observations
test logits do not change if fake labels are changed
```

### 8. Predicate measurement loss

Create synthetic observations.

Verify:

```text
positive observations increase q_pred gradient upward
soft-negative observations increase q_pred gradient downward only when epoch >= 3
unknown observations do not become BCE negatives
rho regularizer prevents all-zero/all-one collapse
loss finite when all observations unknown
```

### 9. PU reason state

Run `NativeTextPUReasonState` with synthetic q/rho.

Verify:

```text
epoch 0: hard_negative_count == 0
epoch 0: soft_negative_weight_sum == 0
epoch 3: soft_negative_weight_sum can be > 0
epoch 7: hard negatives can appear if support low, contra high, rho high
reason y=0 is not globally hard negative
support_score shape [B,21]
contra_score shape [B,21]
reason_rho shape [B,21]
```

### 10. PU reason loss

Verify:

```text
positive mask contributes positive ASL
soft negative contributes weighted negative ASL
hard negative contributes negative ASL only when mask true
unknown contributes entropy only, not BCE negative
gradients reach reason_deploy_logits
loss finite if no reliable negatives exist
```

### 11. Bounded reason residual

Run `NativeTextReasonResidual`.

Verify:

```text
reason_delta shape [B,21]
abs(reason_delta).max <= scheduled cap
cap == 0 or near 0 before epoch 3
support/contra/rho inputs are stopgrad
gradients reach residual MLP, not predicate q/rho through threshold inputs
```

Full model independence probe:

```python
out1 = model(images, reason_labels=labels, epoch=8)
out2 = model(images, reason_labels=labels, epoch=8, force_zero_reason_delta=True)
assert max_abs(out1["action_logits_ntmcal"] - out2["action_logits_ntmcal"]) < 1e-6
assert max_abs(out1["reason_logits_ntmcal"] - out2["reason_logits_ntmcal"]) > 0
```

### 12. Action predicate head

Verify:

```text
epoch < 6: action_predicate_delta == 0
epoch >= 6: delta cap <= configured cap
uses q_pred/rho_pred/predicate_tokens
does not use reason logits
does not use reason labels
does not use reason_delta
does not use action combo set output
```

Verify action-predicate loss:

```text
loss references q_pred
loss is margin/compatibility loss
loss is not BCE(action_logits, action_targets)
```

Reject if `action_predicate_consistency_loss` contains only:

```python
binary_cross_entropy_with_logits(action_logits, action_targets)
```

### 13. NTMCal threshold head

Run `NativeTextMetricCalibrator`.

Verify outputs:

```text
theta_action shape [B,4]
theta_reason shape [B,21]
threshold_delta_reason shape [B,21]
threshold_delta_action shape [B,4]
deploy_action == action_logits - theta_action
deploy_reason == reason_logits - theta_reason
```

Verify threshold delta uses:

```text
support_score.detach()
contra_score.detach()
reason_rho.detach()
base_margin.detach()
cardinality feature
text type / label deltas
```

Reject if:

```text
threshold head is an empty subclass of ACPRThresholdHead
threshold_delta_reason is always zero after epoch 3
threshold delta ignores q/rho/support/contra/margin/cardinality
test oracle thresholds are assigned into model parameters
```

Toy optimization:

```text
one threshold parameter or delta MLP parameter must change after a toy loss step
```

### 14. Pair memory

Verify:

```text
epoch < 7: pair loss disabled
epoch >= 7: memory fill can occur
same reason positive vs reliable negative pairs are mined
near-boundary filter is applied
tail labels receive priority
pair loss cap <= 0.05 * main losses
loss finite when no pairs exist
pair_count_by_reason logged
```

### 15. Full model forward

Run `ACPRNTMCalModel` on real images.

Required output keys:

```text
action_logits_base
reason_logits_base
action_logits_ntmcal
reason_logits_ntmcal
theta_action
theta_reason
action_logits_deploy
reason_logits_deploy
logits_deploy
predicate_q
predicate_rho
predicate_tokens
predicate_topk_indices
predicate_topk_attention
support_score
contra_score
reason_reliability
pu_state
reason_delta
action_predicate_delta
threshold_delta_reason
threshold_delta_action
ntmcal_stats
```

Verify:

```text
action logits [B,4]
reason logits [B,21]
final action is not 16-action-set marginalization
final action does not equal action_reason branch unless intentionally by base trunk, but NTMCal must use action_logits_base + action_predicate_delta
reason_delta changes reason only
test forward invariant to fake reason labels
test forward invariant to fake geometry records
```

### 16. Training protocol

Inspect `train_acpr_ntmcal_oia.py`.

Reject unless:

```text
only train/test loaders are built
no val loader
eval_splits == ["test"]
best_selection_split == "test"
best checkpoint name includes test
no checkpoint_best_val
no val logits
no feature cache
no token compression
DINO no_grad
BF16 AMP default enabled on CUDA
test evaluated every epoch
deploy/base/oracle diagnostics separated
```

### 17. Loss schedule

Run one synthetic schedule check.

Verify by epoch:

```text
epoch 0:
  reason_residual_weight == 0
  pair_weight == 0
  action_predicate_weight == 0
  hard_negative_count == 0

epoch 3:
  soft negative enabled
  PMCal threshold delta cap > 0
  reason residual cap > 0
  pair_weight == 0

epoch 7:
  pair memory enabled
  action predicate delta allowed
  hard negative allowed

epoch 13:
  pair weight decayed
  threshold weight remains active
```

Reject static loss schedules that enable all components at epoch 0.

### 18. Artifacts

Run one tiny epoch and verify all exist:

```text
metrics_summary.json
metrics_deploy_fixed.json
metrics_base_fixed.json
metrics_oracle_diagnostic.json
loss_components.jsonl
logits_action_base_test.pt
logits_reason_base_test.pt
logits_action_deploy_test.pt
logits_reason_deploy_test.pt
labels_action_test.pt
labels_reason_test.pt
file_names_test.json
native_text_atom_stats.json
predicate_bank_audit.json
predicate_measurement_stats.jsonl
predicate_topk_stats.jsonl
pu_state_stats.jsonl
reason_delta_stats.jsonl
action_predicate_stats.jsonl
threshold_delta_stats.jsonl
threshold_stats.jsonl
pair_memory_stats.jsonl
tail_reason_metrics.json
grad_conflict_stats.jsonl
action_independence_probe.json
failure_cases.jsonl
run_manifest.json
```

### 19. Memory probe

Run the memory probe script.

Verify:

```text
batch candidates tried in order: 8, 6, 4
probe runs real forward+backward
BF16 active
reserved_memory_gb recorded
step_time_sec recorded
selected candidate is fastest stable, not necessarily largest
selected reserved memory <= 42GB
memory_probe.json written
```

Reject if formal training uses a candidate that was rejected by the memory probe.

### 20. Supervisor

Inspect `supervise_acpr_ntmcal_foreground.py` and PowerShell launcher.

Reject if:

```text
Start-Job
nohup
hidden cmd
daemon
scheduled task
metric early stop before required epochs
training starts without REVIEW_PASS
```

Accept only if:

```text
streams stdout/stderr
runs memory probe
runs implementation audit
runs smoke
checks git branch/status
pushes code branch or records auth failure clearly
uses test-only best
writes supervisor_live_status.json
writes supervisor_decisions.jsonl
handles OOM by fallback batch candidate
handles dataloader stall
```

---

## Required preflight commands

### Py compile

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_ntmcal_text_atoms.py `
  fate_oia\models\acpr_ntmcal_predicate_bank.py `
  fate_oia\models\acpr_ntmcal_topk_predicate_measurement.py `
  fate_oia\models\acpr_ntmcal_observation_builder.py `
  fate_oia\models\acpr_ntmcal_pu_state.py `
  fate_oia\models\acpr_ntmcal_reason_residual.py `
  fate_oia\models\acpr_ntmcal_action_predicate_head.py `
  fate_oia\models\acpr_ntmcal_threshold_head.py `
  fate_oia\models\acpr_ntmcal_pair_memory.py `
  fate_oia\models\acpr_ntmcal_model.py `
  fate_oia\losses\acpr_ntmcal_losses.py `
  fate_oia\engine\train_acpr_ntmcal_oia.py `
  fate_oia\engine\eval_acpr_ntmcal_oia.py `
  fate_oia\engine\audit_acpr_ntmcal_implementation.py `
  fate_oia\engine\supervise_acpr_ntmcal_foreground.py
```

### Tests

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_ntmcal_text_atoms.py `
  tests\test_acpr_ntmcal_predicate_bank.py `
  tests\test_acpr_ntmcal_topk_measurement.py `
  tests\test_acpr_ntmcal_observation_builder.py `
  tests\test_acpr_ntmcal_pu_state.py `
  tests\test_acpr_ntmcal_reason_residual.py `
  tests\test_acpr_ntmcal_action_predicate.py `
  tests\test_acpr_ntmcal_threshold_head.py `
  tests\test_acpr_ntmcal_pair_memory.py `
  tests\test_acpr_ntmcal_model_forward.py `
  tests\test_acpr_ntmcal_losses.py `
  tests\test_acpr_ntmcal_train_protocol.py `
  tests\test_acpr_ntmcal_audit.py -q
```

### Implementation audit

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_ntmcal_implementation `
  --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml `
  --output_dir .background_runs\acpr_ntmcal_v1_preflight `
  --device cuda `
  --write_review_pass
```

Required output:

```text
.background_runs\acpr_ntmcal_v1_preflight\implementation_audit_ACPR_NTMCAL_V1.json
.background_runs\acpr_ntmcal_v1_preflight\REVIEW_PASS_ACPR_NTMCAL_V1.txt
```

### Memory probe

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1
```

Required output:

```text
.background_runs\acpr_ntmcal_v1_memory_probe\memory_probe.json
```

### Smoke

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_ntmcal_oia `
  --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml `
  --output_dir .background_runs\acpr_ntmcal_v1_smoke `
  --epochs 1 `
  --batch_size 2 `
  --gradient_accumulation_steps 2 `
  --max_train_samples 8 `
  --max_test_samples 8 `
  --device cuda `
  --test_only `
  --no_feature_cache `
  --token_compression none
```

---

## Final review JSON schema

Write:

```text
.background_runs\acpr_ntmcal_v1_preflight\implementation_audit_ACPR_NTMCAL_V1.json
```

Required fields:

```json
{
  "pass": true,
  "git_head": "...",
  "branch": "acpr_ntmcal_v1_direct_image",
  "worktree": "...",
  "checked_files": [],
  "forbidden_pattern_results": {},
  "functional_checks": {
    "dataset_direct_image": {},
    "dino_frozen": {},
    "native_text_bank": {},
    "text_atom_encoder": {},
    "reason_formulas": {},
    "topk_measurement": {},
    "observation_builder": {},
    "predicate_loss": {},
    "pu_state": {},
    "reason_residual": {},
    "action_predicate": {},
    "threshold_head": {},
    "pair_memory": {},
    "full_model_forward": {},
    "training_protocol": {},
    "artifact_schema": {}
  },
  "memory_probe_result": {},
  "smoke_result": {},
  "review_pass_path": ".background_runs/acpr_ntmcal_v1_preflight/REVIEW_PASS_ACPR_NTMCAL_V1.txt",
  "missing_items": [],
  "warnings": []
}
```

If any core check fails, `pass` must be false and the pass file must not be written.

---

## Dynamic gate requirements before continuing past specific epochs

During actual full training, the supervisor must check these after each epoch.

### After epoch 2

Stop and inspect code if:

```text
q_pred_mean not in [0.15,0.85]
rho_pred_mean not in [0.10,0.90]
text_obs_positive_count == 0
text_obs_unknown_count == 0
native_text_structure_loss == 0
```

### After epoch 6

Stop and inspect code if:

```text
soft_negative_weight_sum == 0
threshold_delta_reason_abs_mean == 0
reason_delta_abs_mean == 0
reason_pred_positive_rate remains far below train reason positive rate
```

### After epoch 10

Stop and inspect code if:

```text
pair_memory_positive_coverage == 0
pair_memory_negative_coverage == 0
tail_pair_count == 0 for all tail labels
action drops sharply immediately after action predicate delta opens
```

### At final epoch

Report separately:

```text
deploy_fixed formal metrics
base_fixed diagnostic metrics
oracle_threshold diagnostic metrics
Exp_mAP
tail macro F1
tail AP
deploy-oracle gap
```

If Exp_mF1 improves but Exp_mAP is unchanged, label the gain as calibration gain, not representation/ranking gain.

---

## Implementation pass criteria

The implementation is acceptable only when all are true:

```text
py_compile PASS
pytest PASS
audit PASS
REVIEW_PASS file exists
memory probe PASS
tiny smoke PASS
code branch pushed or push failure clearly attributable to auth/TLS and local commit recorded
no forbidden patterns
all core mechanism dynamic checks produce non-empty stats
```

Do not accept a run that merely emits metrics without proving the native-text predicate measurement, PU state, NTMCal threshold, and action-independence invariants.
