# ACPR-TFC V1 Implementation Audit Skill

This skill audits whether the ACPR-TFC implementation actually executes the intended method. Passing py_compile or training a model is not sufficient.

## Method being audited

ACPR-TFC means:

```text
native factor prototypes
+ target-conditioned factor credit
+ same-image deletion contrast
+ PU-safe reason supervision
+ CalAlign deploy boundary
+ action/reason gradient firewall
```

The method is invalid if it degenerates into:

```text
raw predicate probability -> action logit delta
```

or:

```text
reason logits -> final action
```

or:

```text
threshold-only calibration
```

---

## 1. Forbidden patterns

Fail the audit if any of these appear in the TFC final path.

### 1.1 Raw predicate-to-action

Search for and reject direct action construction such as:

```python
action_logits = action_logits + factor_probs
action_logits = action_logits + factor_rho
action_logits = action_logits + q_pred
action_logits = action_logits + action_predicate_delta
```

Allowed only if the delta is explicitly produced by `TFCActionHead` from `credit_action_norm` and deletion-gated target credit.

### 1.2 Reason-to-action final fusion

Reject final action if it consumes:

```python
reason_logits
reason_visual_logits
reason_tfc_delta
reason_to_action(...)
```

Reason-to-action may exist only as an auxiliary diagnostic and must not be `action_logits` used for metrics/checkpoints.

### 1.3 Action-set final

Reject:

```python
action_logits = action_set_logits
action_logits = action_combo_aux(...)
action_logits = action_set_marginalization(...)
```

Action-set or combo priors may be auxiliary diagnostics only.

### 1.4 Graph / PMI / co-occurrence strong bias

Reject files or code names containing:

```text
pmi
cooccurrence
co_occurrence
label_graph
dynamic_graph
graph_attention
correlation_bias
cooc_bias
```

unless the file is an audit note explicitly marking them forbidden. TFC must not use label co-occurrence as causal evidence.

### 1.5 Dense factor-token tensor

Reject materialized tensors with semantic shape:

```text
[B, F, N, D]
[B, F, L*N, D]
[B, P, N, D]
[B, P, L*N, D]
```

Allowed:
```text
scores [B,F,L,N]
topk_values [B,F,K,D]
```

Flag common risky patterns:

```python
einsum("fl,blnd->bfnd")
einsum("fs,bsnd->bfnd")
tokens[:, None].expand(B, F, N, D).clone()
repeat(..., F, ..., D)
```

### 1.6 Test-time leakage

Reject if eval/test forward reads:

```python
reason_labels
action_labels except for metric computation
bdd100k_geometry
grounding_cache
test_per_label_threshold as model threshold
test labels for threshold fitting
```

### 1.7 Feature cache / token compression

Reject:

```text
feature_cache enabled
cached_logits
RunC residual
token_compression true
progressive compression
```

---

## 2. Required files

Audit fails if any are missing:

```text
fate_oia/models/tfc_factor_bank.py
fate_oia/models/tfc_prototype_bank.py
fate_oia/models/tfc_topk_factor_measurement.py
fate_oia/models/tfc_dual_lane_adapter.py
fate_oia/models/tfc_target_credit.py
fate_oia/models/tfc_deletion_contrast.py
fate_oia/models/tfc_action_head.py
fate_oia/models/tfc_reason_head.py
fate_oia/models/tfc_pu_state.py
fate_oia/models/tfc_calalign_head.py
fate_oia/models/acpr_tfc_model.py
fate_oia/losses/tfc_losses.py
fate_oia/optim/tfc_pareto_optimizer.py
fate_oia/engine/audit_tfc_gates.py
fate_oia/engine/train_acpr_tfc_oia.py
fate_oia/engine/eval_tfc_branch_ablation.py
configs/acpr_tfc_factors.yaml
configs/fate_oia_train_360x640_acpr_tfc_v1.yaml
scripts/FATE_OIA_acpr_tfc_v1_foreground.ps1
tests/test_acpr_tfc_factor_bank.py
tests/test_acpr_tfc_model_forward.py
tests/test_acpr_tfc_gates.py
```

---

## 3. Static code audit

Run:

```bash
python -m py_compile $(git ls-files "*.py")
python -m pytest tests/test_acpr_tfc_factor_bank.py tests/test_acpr_tfc_model_forward.py tests/test_acpr_tfc_gates.py -q
```

Then scan source text for forbidden patterns. The audit script should create:

```text
TFC_GATE_A_CODE_AUDIT_PASS.json
```

It must include:

```json
{
  "no_graph_pmi": true,
  "no_action_set_final": true,
  "no_reason_to_final_action": true,
  "no_raw_qrho_to_action_delta": true,
  "no_dense_bpnd": true,
  "no_cache": true,
  "no_token_compression": true
}
```

---

## 4. Factor bank audit

Run:

```bash
python -m fate_oia.engine.audit_tfc_gates --mode factor-bank --config configs/fate_oia_train_360x640_acpr_tfc_v1.yaml
```

Must verify:

```text
all factors have entity/attribute/spatial/polarity/region_prior/target_scope
all action target names resolve to 4 action indices
all reason target names resolve to 21 reason indices
all factor conflicts resolve to existing factors
left/right mirror pairs are complete
traffic light red/green contradiction exists
lane availability / solid boundary conflicts exist
native_similarity matrix is finite and square
```

Fail if any missing reference is silently ignored.

---

## 5. Model forward audit

Run on a synthetic batch and on a small real batch.

Expected output keys:

```text
action_visual_logits
action_tfc_delta
action_logits_base
action_logits_deploy
reason_visual_logits
reason_tfc_delta
reason_logits_base
reason_logits_deploy
factor_probs_action
factor_rho_action
factor_probs_reason
factor_rho_reason
credit_action
credit_reason
credit_confidence_action
credit_confidence_reason
action_theta
reason_theta
theta_delta_action
theta_delta_reason
pu_state
deletion_stats
artifact_stats
```

Fail if any key is missing.

Shape requirements:

```text
action logits: [B,4]
reason logits: [B,21]
factor probabilities: [B,F]
credit_action: [B,F,4]
credit_reason: [B,F,21]
topk_indices: [B,F,K]
```

All outputs must be finite.

---

## 6. Action firewall dynamic probe

This is mandatory.

### 6.1 Reason-zero probe

Forward same image twice:

1. normal forward
2. forward with reason branch disabled / reason delta zeroed / reason labels unavailable

Requirement:

```python
max_abs(action_logits_normal - action_logits_reason_disabled) < 1e-6
```

unless explicitly testing a diagnostic branch that is not used for final action.

### 6.2 Gradient firewall probe

Compute reason loss only:

```python
loss_reason.backward()
```

Requirement:

```text
all action_adapter params have grad None or norm == 0
all action_head final params have grad None or norm == 0
```

Compute action loss only:

```python
loss_action.backward()
```

Requirement:

```text
reason_adapter params have grad None or norm == 0
```

Stop-gradient consistency may update its own tiny projection parameters, but not the wrong lane.

Write:

```text
TFC_GATE_C_ACTION_FIREWALL_PASS.json
```

---

## 7. Target credit audit

Required checks:

```text
credit_action uses factor_probs, factor_rho, learned compatibility, native compatibility, margin gate
credit_reason uses the same target-conditioned logic
credit does not use train label co-occurrence or PMI
credit normalization finite
inhibitory factors can produce negative credit
support factors can produce positive credit
```

Target credit must be target-specific. A factor may support one target and inhibit another.

Fail if credit is only:

```python
factor_probs.mean(...)
```

or a generic pooled predicate feature.

---

## 8. Deletion contrast audit

Run:

```bash
python -m fate_oia.engine.audit_tfc_gates --mode deletion --max_samples 16 --config configs/fate_oia_train_360x640_acpr_tfc_v1.yaml
```

Check:

```text
selected deletion effect tensor exists
random deletion effect tensor exists
selected and random use equal token count
random is same-region, not global arbitrary
DINO is not recomputed
deletion recomputes lightweight heads only
selected_vs_random_gap is finite
selected_gt_random_rate is recorded
```

Pretrain gate only requires functional finite values. After epoch 5, continuation gate requires selected-vs-random gap > 0.

Write:

```text
TFC_GATE_D_FACTOR_GROUNDING_PASS.json
TFC_GATE_E_SELECTED_DELETION_GT_RANDOM_PASS.json
```

---

## 9. PU state audit

Checks:

```text
epoch 0-2: hard_negative_count == 0
epoch 0-2: soft_negative_weight sum == 0
epoch >=3: soft_negative_weight may become nonzero
epoch >=7: hard negatives allowed only if configured gates pass
unknown_mask includes most y=0 reasons early
hard_negative_rate <= max_hard_negative_rate
per-reason counts emitted
```

Fail if reason=0 is treated as hard negative by default.

Write:

```text
TFC_GATE_F_PU_STATE_PASS.json
```

---

## 10. CalAlign / threshold audit

Required:

```text
deploy_logits = base_logits - theta exactly
teacher threshold source is train_calib
test oracle threshold diagnostic only
threshold input credit tensors are detached
action theta does not read raw factor_probs directly
reason theta may read detached support/contra/rho/credit confidence
base/deploy/oracle metrics emitted separately
```

Dynamic check:
- Set `requires_grad=True` on factor_probs.
- Pass `factor_probs.detach()` into threshold.
- Backward threshold loss.
- Assert no gradient flows back into factor_probs through threshold.

Write:

```text
TFC_GATE_G_CALALIGN_PASS.json
```

---

## 11. Memory probe

Run:
```bash
python -m fate_oia.engine.audit_tfc_gates --mode memory --batch_size 4 --grad_accum 8 --config configs/fate_oia_train_360x640_acpr_tfc_v1.yaml
```

Pass if:
```text
forward/backward completes
reserved VRAM < 42GB
step time not pathological
no OOM
no prolonged stall
```

Optional:
```bash
--batch_size 6 --grad_accum 6
```

Use batch 6 only if faster and stable. Do not use batch 9.

Write:
```text
TFC_GATE_H_MEMORY_PROBE_PASS.json
```

---

## 12. Artifact schema audit

A one-epoch smoke must write:

```text
metrics_summary.json
run_manifest.json
loss_components.jsonl
action_branch_metrics.json
factor_measurement_stats.jsonl
target_credit_stats.jsonl
deletion_contrast_stats.jsonl
pu_state_stats.jsonl
threshold_stats.jsonl
pareto_gradient_stats.jsonl
failure_flip_cases.jsonl
```

Fail if any is missing.

`action_branch_metrics.json` must include:
```text
action_visual_only
action_tfc_delta_off
action_tfc_delta_on
action_threshold_delta_off
action_final_deploy
action_oracle
per_action_AP_AUC_F1
FP_to_TP
TP_to_FN
```

`target_credit_stats.jsonl` must include:
```text
target_id
factor_id
credit_mean
deletion_selected
deletion_random
selected_vs_random_gap
positive_credit_sign_acc
inhibitory_credit_sign_acc
```

---

## 13. Training launch audit

Full training command must refuse to start unless:

```text
.review/acpr_tfc_v1_REVIEW_PASS.json exists
TFC_GATE_A_CODE_AUDIT_PASS.json exists
TFC_GATE_B_NO_TEST_LEAKAGE_PASS.json exists
TFC_GATE_C_ACTION_FIREWALL_PASS.json exists
TFC_GATE_F_PU_STATE_PASS.json exists
TFC_GATE_G_CALALIGN_PASS.json exists
TFC_GATE_H_MEMORY_PROBE_PASS.json exists
```

The selected-vs-random deletion gate can be functional before training and strict after epoch 5.

---

## 14. Required REVIEW_PASS JSON

After all strict pretrain gates pass, write:

```text
.review/acpr_tfc_v1_REVIEW_PASS.json
```

Required content:

```json
{
  "review_pass": true,
  "method": "ACPR-TFC-V1",
  "branch": "acpr_tfc_v1_direct_image",
  "base_branch": "acpr_calalign_v1_2",
  "direct_image": true,
  "no_cache": true,
  "no_token_compression": true,
  "test_only_eval": true,
  "best_selection": "test_joint",
  "train_calib_threshold_only": true,
  "test_oracle_diagnostic_only": true,
  "no_graph_pmi": true,
  "no_action_set_final": true,
  "no_reason_to_final_action": true,
  "no_raw_qrho_to_action_delta": true,
  "no_dense_bpnd": true,
  "action_firewall_dynamic_probe": true,
  "target_credit_present": true,
  "deletion_contrast_functional": true,
  "pu_state_schedule_present": true,
  "artifact_schema_complete": true
}
```

---

## 15. Stop conditions during full run

The training script must mark the run as failed or stop if:

```text
NaN or Inf loss.
No log progress for >30 minutes with active process.
q/rho collapse for 2 epochs.
selected-vs-random deletion gap <= 0 after epoch5.
oracle Act_mF1 drops >0.004 for 2 consecutive epochs after action TFC starts.
Act_mAP drops while Exp_mF1 rises only through threshold movement.
TP->FN flips exceed FP->TP flips for any action label after action delta starts.
hard negative rate exceeds configured maximum.
```

The stop reason must be written to:

```text
run_stop_reason.json
```

---

## 16. Final audit judgment

A TFC implementation is **not valid** if:

```text
it trains but target credit tensors are unused;
it trains but action TFC delta does not depend on target credit;
it trains but deletion contrast is always disabled;
it trains but final action still uses reason logits/fused reason-to-action;
it trains but threshold uses test oracle values;
it trains but artifacts do not allow branch-level attribution.
```

A TFC implementation is valid only if the code and dynamic probes prove:

```text
factor measurement -> target-conditioned credit -> deletion-verified residual -> CalAlign deploy
```

is actually executed.
