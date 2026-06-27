# CALI-Flow++ Current-Branch Audit Skill

Use this skill to review the `acpr_interactflow_pp_v1` branch after implementing CALI-Flow++ directly in the current worktree.

This skill is adversarial. Passing tests by file existence or placeholder code is not sufficient. The reviewer must inspect static code, run dynamic probes, verify artifacts, and bind the review pass to the exact clean GitHub-pushed commit.

---

## 0. Scope

Repository:

```text
E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree
```

Branch:

```text
acpr_interactflow_pp_v1
```

Formal config:

```text
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml
```

Formal train entry:

```text
python -m fate_oia.engine.train_acpr_interactflow_psi
```

Formal eval entry:

```text
python -m fate_oia.engine.eval_acpr_interactflow_psi
```

The branch must implement:

```text
CALI-Flow++:
15-frame clip
→ visual evidence budget
→ dynamic predicate trajectories
→ traffic interaction state grammar
→ per-factor response lag
→ benefit-gated exact decision ledger
→ contribution-grounded Exp29 weak explanation
→ train-only calibrated deploy path
→ intervention-verified model-level counterfactual dependence
```

---

## 1. Hard user constraints

Fail immediately if any is true:

```text
A new worktree was created for this task.
feature_cache_enabled=true
token_cache_enabled=true
logit_cache_enabled=true
Formal training reads val split or selects best on val.
Target frame image is loaded into formal model input.
Training uses cached logits/features/tokens.
`.background_runs`, checkpoint, logits, dataset, or large artifacts are staged for Git.
Worktree is dirty at review-pass generation.
Local HEAD differs from GitHub refs/heads/acpr_interactflow_pp_v1.
```

---

## 2. Pre-audit commands

Run:

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree

git branch --show-current
git status --short
git rev-parse HEAD
git ls-remote github refs/heads/acpr_interactflow_pp_v1
```

Expected:

```text
branch == acpr_interactflow_pp_v1
git status --short empty
local HEAD == GitHub branch HEAD
```

If not true, stop. Do not issue review pass.

---

## 3. Static file presence

Required files:

```text
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml
configs/acpr_interactflow_predicates.yaml
configs/acpr_interactflow_state_grammar.yaml
configs/acpr_interactflow_text_rules.yaml

fate_oia/acpr_interactflow/types.py
fate_oia/acpr_interactflow/model.py
fate_oia/acpr_interactflow/visual_encoder.py
fate_oia/acpr_interactflow/motion_path.py
fate_oia/acpr_interactflow/traffic_event_budget.py
fate_oia/acpr_interactflow/predicate_transfer.py
fate_oia/acpr_interactflow/dynamic_predicate_field.py
fate_oia/acpr_interactflow/interaction_flow.py
fate_oia/acpr_interactflow/response_lag.py
fate_oia/acpr_interactflow/decision_ledger.py
fate_oia/acpr_interactflow/exp29_head.py
fate_oia/acpr_interactflow/calibrated_exp29.py
fate_oia/acpr_interactflow/cluster_semantics.py
fate_oia/acpr_interactflow/reliability.py
fate_oia/acpr_interactflow/interventions.py
fate_oia/acpr_interactflow/timing.py
fate_oia/acpr_interactflow/psi_metrics.py

fate_oia/losses/acpr_interactflow_losses.py

fate_oia/engine/train_acpr_interactflow_psi.py
fate_oia/engine/eval_acpr_interactflow_psi.py
fate_oia/engine/profile_acpr_interactflow.py
fate_oia/engine/run_acpr_interactflow_preflight.py
fate_oia/engine/audit_acpr_interactflow.py
fate_oia/engine/audit_califlowpp_current_branch.py
fate_oia/engine/supervise_acpr_interactflow_foreground.py
fate_oia/engine/export_acpr_interactflow_visuals.py
fate_oia/engine/build_acpr_interactflow_atlas.py

fate_oia/explain/acpr_interactflow_renderer.py
fate_oia/explain/acpr_interactflow_atlas.py
fate_oia/explain/acpr_interactflow_faithfulness.py

scripts/FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1
tests/acpr_interactflow/
```

File presence alone is not enough; continue all checks.

---

## 4. Forbidden static patterns

Search all formal files. Fail if found outside comments explaining prohibition:

```text
ACPROIAModel(
bdd_oia_multitask
train_acpr_oia as formal trainer
feature_cache_enabled: true
token_cache_enabled: true
logit_cache_enabled: true
best_selection_split: val
eval_splits: [val]
checkpoint_best_val as formal checkpoint
Start-Process for formal supervisor
Start-Job
nohup
schtasks
hidden window
use_mock_dino=True in formal train
target_frame_image
formal_input_uses_target_frame: true
all_zero_exp29_negative_bce
unknown_as_negative: true
predicate_nnpu = exp29
F.cross_entropy(final_logits
F.cross_entropy(output.action_logits
num_actions: int = 4 in DecisionLedgerHead
dino input hard-coded 360,640
reshape(...3600...)
grid_hw=(45,80) hard-coded
```

Allowed exceptions:
- Tests may intentionally assert forbidden patterns are absent.
- Audit scripts may contain forbidden strings as string literals to search for them.

---

## 5. Config consumption audit

Read `configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml`.

Fail if any config field is orphaned. In particular verify runtime consumers for:

```text
data.image_height / image_width
data.formal_input_uses_target_frame
data.all_zero_exp29_is_unknown
data.feature_cache_enabled / token_cache_enabled / logit_cache_enabled

model.visual_encoder.anchor_frames
model.visual_encoder.dino_input_height / dino_input_width
model.visual_encoder.dino_chunk_size
model.visual_encoder.event_budget_enabled

model.predicates.require_oia_transfer_source
model.predicates.require_transformer_text
model.nnpu.unknown_as_negative

model.interaction_flow.factor_count
model.interaction_flow.response_lags

model.decision_ledger.exact_additive_decomposition
model.decision_ledger.non_degradation_margin

model.exp29.contribution_alignment
model.exp29.reliability_weighted
model.exp29.calibrated_deploy_path

loss.* all nonzero configured weights
optimization.learning_rates.*
evaluation.eval_splits
evaluation.best_selector
profile.target_peak_reserved_gib
supervisor.require_review_pass
```

The audit report must contain:

```json
{
  "config_runtime_consumption": {
    "field": "consumer_file:function",
    ...
  },
  "orphan_config_fields": []
}
```

---

## 6. Dataset and label checks

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\acpr_interactflow\test_dataset_protocol.py -q
```

Additional dynamic probe required:

```python
ds_train = PSIDAMO11902Dataset(..., split="train")
ds_test = PSIDAMO11902Dataset(..., split="test")
assert len(ds_train) == 8873
assert len(ds_test) == 2417
batch = next(iter(loader))
assert batch.frames.shape[1] == 15
assert batch.action_soft_target.shape[-1] == 3
assert batch.exp29_target.shape[-1] == 29
assert not batch.formal_model_loaded_target_frame_image
```

Exp29 unknown check:

```python
zero_rows = batch.exp29_target.sum(-1) == 0
assert (batch.exp29_mask[zero_rows].sum(-1) == 0).all() or only_reliable_negative_entries_present
```

Fail if all-zero rows become mask-all-one negatives.

---

## 7. Visual encoder audit

Run a forward on real or smoke frames.

Required facts:

```python
visual.stats["formal_target_frame_used"] is False
visual.stats["dino_input_h"] == cfg["model"]["visual_encoder"]["dino_input_height"]
visual.stats["dino_input_w"] == cfg["model"]["visual_encoder"]["dino_input_width"]
visual.stats["grid_h"] == dino_input_h // patch_size
visual.stats["grid_w"] == dino_input_w // patch_size
visual.patch_tokens_by_layer.shape[-2] == grid_h * grid_w
visual.stats["dino_chunk_size"] == selected dino_chunk_size
```

Fail if the code hard-codes 360×640 or 45×80.

If `event_budget_enabled=false`, fixed anchors must be `[0,3,6,9,12,14]`.  
If `event_budget_enabled=true`, selected anchors must be deterministic, observed only, and recorded.

---

## 8. Predicate transfer audit

Inspect `oia_transfer_report.json` and dynamic model report.

Required:

```json
{
  "source_loaded": true,
  "source_tensor_key": "predicate_head.predicate_queries",
  "source_shape": [32,384],
  "mapped_shape": [48,384],
  "oia_name_order_verified": true,
  "loaded_predicate_names": [...32 names...],
  "text_embedding_source": "transformers_frozen",
  "fallback_used": false
}
```

Fail if:
- OIA source checkpoint missing or ambiguous;
- source key is not `predicate_head.predicate_queries`;
- text embedding uses hash/BoW fallback;
- transformer model writes to C盘 default cache;
- transfer residual has zero gradient in gradient-chain check.

---

## 9. Dynamic predicate field audit

Forward output must include:

```python
predicate_logits_trajectory.shape == [B,15,48]
predicate_probs_trajectory.shape == [B,15,48]
predicate_tokens_trajectory.shape == [B,15,48,D]
predicate_evidence_maps.shape[:3] == [B,A,48]
predicate_confidence.shape == [B,15,48]
predicate_centroid.shape == [B,15,48,2]
predicate_relative_motion.shape == [B,14,48,2]
predicate_corridor_mass.shape == [B,15,48,4]
transfer_gate.shape == [48]
```

Fail if:
- only `[B,48]` predicate output exists;
- non-anchor frames are constant copies with no motion update;
- all predicate probabilities are constant or all zero;
- `predicate_positive_rate` is missing;
- `predicate_pu_loss` is absent or zero placeholder.

---

## 10. Interaction-flow / state grammar audit

Required output:

```python
factor_tokens_trajectory.shape == [B,15,F,D]
factor_tokens_lag_aligned.shape == [B,F,D]
factor_to_predicate.shape == [B,15,F,48]
factor_to_corridor.shape == [B,15,F,4]
factor_probs.shape == [B,15,F]
state_group_logits exists
lineage contains top predicates and corridors
```

Fail if:
- state factors are computed directly from action labels only;
- state path bypasses predicates;
- `factor_tokens` are `[B,F,D]` with no 15-frame trajectory before lag;
- weak state loss is zero placeholder.

---

## 11. Response lag audit

Required:

```python
lag_weights.shape == [B,F,5]
lag_weights.sum(-1) ≈ 1
lag_disabled intervention changes downstream action probabilities
temporal_reverse changes phase/lag stats on real smoke subset
synthetic delayed-event test passes
```

Fail if lag is a global `[B,5]` vector shared by all factors, unless explicitly justified and tested as ablation only.

---

## 12. Decision ledger audit

Required:

```python
num_actions == 3
global_logits.shape == [B,3]
raw_state_contributions.shape == [B,F,3]
benefit_gate.shape in {[B,F,1], [B,F,3]}
gated_state_contributions.shape == [B,F,3]
final_logits == global_logits + gated_state_contributions.sum(1) + calibration_delta
identity_error < 1e-6
```

Benefit target required:

```python
benefit_target exists during training
benefit_target detached
benefit_gate_advantage_bce finite
gate_mean not saturated across mechanism fit
```

Fail if `DecisionLedgerHead(num_actions=4)` default remains.

---

## 13. Exp29 ledger-grounding audit

Required `Exp29Output` fields:

```python
logits_raw.shape == [B,29]
logits_calibrated.shape == [B,29]
probs_raw.shape == [B,29]
probs_calibrated.shape == [B,29]
cluster_attention_to_factors.shape == [B,29,F]
cluster_reliability.shape == [29]
cluster_to_state_prior.shape == [29,F]
```

Required code behavior:

```python
Exp29Head.forward(..., gated_state_contributions=ledger.gated_state_contributions, ...)
contrib_mag = normalize(sum(abs(gated_state_contributions), action_dim))
attention score includes contribution term and cluster-to-state prior
```

Fail if Exp29 only attends to `torch.cat([factor_tokens, predicate_tokens])`.

---

## 14. Exp29 calibration audit

Required losses:

```text
exp29_raw_asl
exp29_calibrated_asl
exp29_soft_f1
exp29_positive_rate
exp29_cardinality
exp29_pairwise_rank
exp29_ledger_alignment_js
```

Required eval:

```text
ExpRaw_mF1 / ExpRaw_oF1 / ExpRaw_mAP
ExpCal_mF1 / ExpCal_oF1 / ExpCal_mAP
ExpDiag threshold sweep diagnostic only
primary Exp_mF1 == calibrated fixed-threshold result
```

Fail if:
- ExpCal exists but is not used in primary loss/eval;
- positive-rate calibration uses test statistics;
- `pred_positive_rate@0.5` is missing;
- all predicted positives remain zero in the 128-sample mechanism fit without a failing gate.

---

## 15. Loss audit

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\acpr_interactflow\test_califlowpp_predicate_pu_split.py `
  tests\acpr_interactflow\test_califlowpp_soft_kl_safety.py `
  tests\acpr_interactflow\test_califlowpp_benefit_gate_advantage.py `
  tests\acpr_interactflow\test_califlowpp_exp29_calibration.py `
  tests\acpr_interactflow\test_califlowpp_exp29_ledger_grounding.py -q
```

Required:
- `predicate_pu_loss` uses predicate logits/trajectory, not Exp29 logits.
- `exp29_pu_loss` uses Exp29 logits.
- `non_degradation_soft_kl_hinge` uses `action_soft_target`.
- Every nonzero loss emits raw, weight, weighted, finite, gradient target.
- Total loss finite on real smoke batch.

Fail if any nonzero configured loss has no gradient target.

---

## 16. Gradient chain audit

On a real direct-image smoke batch, after one backward pass:

Required nonzero finite gradients:

```text
visual fast motion / adapter params
predicate transfer residual
dynamic predicate temporal module
motion path
interaction flow factor queries / value/key
response lag module
decision ledger
exp29 head
calibration thresholds/bias
```

Required zero gradients:

```text
frozen DINO backbone, unless adapter mode explicitly unfreezes selected params
frozen BERT/text encoder
OIA source query prior tensor if detached by design
```

Fail if:
- dynamic predicate path has zero grad;
- state path has zero grad;
- Exp29 head has zero grad;
- calibration path has zero grad.

---

## 17. Intervention audit

Run real 16-sample probe:

```python
full = model(frames)
global_only = model(frames, intervention="global_only")
predicate_off = model(frames, intervention="predicate_off")
factor_off = model(frames, intervention="factor_off")
lag_disabled = model(frames, intervention="lag_disabled")
temporal_reverse = model(reverse(frames), intervention="temporal_reverse")
```

Required:

```text
global_only changes final action prob or removes flow_delta
predicate_off changes predicate/flow/ledger/Exp29
factor_off changes ledger/Exp29
lag_disabled changes ledger/Exp29 on at least one delayed sample
temporal_reverse recomputes visual/motion/predicate/flow, not just display
```

Fail if intervention mutates only final logits or JSON artifact.

---

## 18. Timing/profile audit

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.profile_acpr_interactflow `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir .background_runs\cali_flowpp_current_branch_preflight `
  --device cuda `
  --profile_batches 100 `
  --candidate_batch_sizes 8,6,5,4 `
  --candidate_dino_chunk_sizes 8,6,5,4
```

Required in profile JSON:

```text
data_gap_time
h2d_time
visual_dino_time
visual_motion_time
predicate_time
interaction_flow_time
response_lag_time
decision_ledger_time
exp29_time
loss_time
backward_time
optimizer_time
artifact_write_time
samples_per_second
peak_reserved_gib
selected_batch_size
selected_grad_accum
selected_dino_chunk_size
projected_train_epoch_time
projected_test_eval_time
```

Pass condition:

```text
selected peak_reserved_gib <= 46
target preferred 42–45 if feasible
data_time_fraction <= 0.25
no NaN/Inf
no dummy allocation
selected setting has highest stable samples/sec under cap
```

---

## 19. Test-only / best-selection audit

Inspect train/eval/supervisor:

Required:

```text
eval_splits == [test]
no val loader in formal training
checkpoint_best_test.pth exists
checkpoint_best_joint.pth exists
checkpoint_best_action.pth exists
checkpoint_best_exp.pth exists
best selector formula uses test metrics
```

Fail if:
- val is evaluated after every epoch;
- best checkpoint is selected on val;
- test is only diagnostic but not best selector.

---

## 20. Artifact audit

After smoke/preflight, required run-root artifacts:

```text
run_manifest.json
config_resolved.yaml
git_provenance.json
psi_dataset_contract.json
damo_metric_parity.json
oia_transfer_report.json
optimizer_groups.json
throughput_profile.json
timing_summary.json
checkpoint_latest.pth
metrics_summary.jsonl
core_metrics_summary.jsonl
innovation_intermediate_metrics.jsonl
```

Required epoch artifacts:

```text
action_metrics.json
exp29_metrics.json
exp29_raw_metrics.json
exp29_calibrated_metrics.json
exp29_diagnostic_threshold_sweep.json
joint_metrics.json
loss_components.jsonl
timing_epoch.json
gradient_norms.json
predicate_stats.json
cluster_reliability_stats.json
nnpu_calibration.json
interaction_state_stats.json
response_lag_stats.json
decision_ledger_stats.json
exp29_ledger_alignment_stats.json
lightweight_interaction_influence.json
predictions_action.jsonl
predictions_exp29.jsonl
fixed_case_intermediate_outputs.jsonl
```

Fail if missing values are silently zero; missing values require explicit `"available": false, "reason": ...`.

---

## 21. Visualization audit

Run export on smoke/test tensors.

Required per case:

```text
decision_ledger.json
decision_ledger.png
decision_waterfall.png
case_source.json
report.html
```

Required atlas:

```text
atlas.json
atlas.html
```

JSON must include:
- sample ID / video ID / frame indices;
- checkpoint SHA / config SHA;
- exact global logits, gated contributions, calibration delta, final logits;
- Exp29 predicted clusters, medoid text, cluster attention, exact factor contribution;
- intervention full/state-off/evidence-off/random/reverse values.

Fail if:
- renderer fabricates boxes/effects;
- atlas is not tensor-linked;
- placeholder HTML string is used.

---

## 22. Required commands before review pass

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\acpr_interactflow\*.py `
  fate_oia\engine\train_acpr_interactflow_psi.py `
  fate_oia\engine\eval_acpr_interactflow_psi.py `
  fate_oia\engine\audit_acpr_interactflow.py `
  fate_oia\engine\audit_califlowpp_current_branch.py `
  fate_oia\losses\acpr_interactflow_losses.py

E:\Anaconda\envs\sbw39\python.exe -m pytest tests\acpr_interactflow -q

E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.run_acpr_interactflow_preflight `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir .background_runs\cali_flowpp_current_branch_preflight `
  --device cuda `
  --profile_batches 100

E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_califlowpp_current_branch `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir .background_runs\cali_flowpp_current_branch_preflight `
  --device cuda
```

All must pass.

---

## 23. Commit/push audit

After all code/test/config changes:

```powershell
git status --short
git add configs fate_oia tests scripts docs .codex
git commit -m "Implement CALI-Flow++ ledger-grounded PSI interact flow"
git push github acpr_interactflow_pp_v1:acpr_interactflow_pp_v1
git ls-remote github refs/heads/acpr_interactflow_pp_v1
```

Then rerun audit and preflight. Any code/config/script/test change invalidates previous review pass.

---

## 24. Review pass format

Only write review pass if all gates pass after local/GitHub SHA equality.

Path:

```text
.background_runs\cali_flowpp_current_branch_preflight\REVIEW_PASS_CALI_FLOWPP_CURRENT_BRANCH.txt
```

Content:

```json
{
  "pass": true,
  "method": "CALI-Flow++",
  "branch": "acpr_interactflow_pp_v1",
  "git_head": "...",
  "github_remote_head": "...",
  "worktree_clean": true,
  "config_sha256": "...",
  "plan_sha256": "...",
  "skill_sha256": "...",
  "all_tests_passed": true,
  "all_static_checks_passed": true,
  "all_dynamic_checks_passed": true,
  "selected_batch_size": 6,
  "selected_grad_accum": 5,
  "selected_dino_chunk_size": 6,
  "peak_reserved_gib": 42.0,
  "feature_cache_enabled": false,
  "token_cache_enabled": false,
  "logit_cache_enabled": false,
  "eval_splits": ["test"],
  "created_at": "..."
}
```

---

## 25. Formal training authorization

Training can be launched only after review pass exists and supervisor verifies:

```text
review pass git_head == local HEAD
local HEAD == GitHub remote HEAD
worktree clean
cache flags false
test-only eval
profile selected batch exists
```

Recommended launch after pass:

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1 `
  -Config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  -Epochs 30 `
  -BatchSize 6 `
  -GradAccum 5 `
  -DinoChunkSize 6 `
  -Device cuda `
  -RequireReviewPass
```

If profile selected batch=8:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1 `
  -Config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  -Epochs 30 `
  -BatchSize 8 `
  -GradAccum 4 `
  -DinoChunkSize 8 `
  -Device cuda `
  -RequireReviewPass
```

---

## 26. First epoch stop conditions

After epoch0, fail/stop for debugging if any is true:

```text
ExpCal_pred_positive_rate@0.5 == 0
ExpCal_mF1 == 0
identity_error > 1e-6
predicate_positive_rate constant zero
flow_delta_abs_mean == 0
benefit_gate_mean saturated at 0 or 1
lag_disabled does not alter action prob on audit subset
state_off does not alter action prob on audit subset
timing fields missing
NaN/Inf observed
```

Do not blindly run 30 epochs through these failures.

---

## 27. Final experiment complete audit

At the end of 30 epochs, verify:

```text
run_complete.json exists
all 30 epoch dirs exist
full test eval exists for every epoch
checkpoint_best_test.pth exists
best-test full intervention audit exists
visual case exports exist
atlas exists
metrics_summary.jsonl has 30 test rows
core metrics include Act_mAcc, Stop_F1, ExpCal_mF1, ExpRaw_mAP
psi_task_plan.md / psi_findings.md / psi_progress.md updated only for PSI
local/GitHub SHA equality still true
```

Do not claim final paper-level result until this passes.
