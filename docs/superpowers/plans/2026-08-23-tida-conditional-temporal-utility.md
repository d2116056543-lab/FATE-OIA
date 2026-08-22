# TIDA Conditional Temporal Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make traffic-flow evidence conditionally stronger for targets that need history while preserving exact image fallback and measurable no-harm behavior.

**Architecture:** Split transition evidence into typed temporal scales, compute per-target utility from image uncertainty, motion salience, persistence, and compatibility, and use that utility to allocate bounded action/reason flow budgets. Train with paired temporal credit plus utility calibration and evaluate with label-independent motion slices and paired contribution metrics.

**Tech Stack:** Python 3.9, PyTorch, pytest, frozen DINO ViT-S/8, YAML configuration.

---

### Task 1: Typed Multi-Scale Transition Bank

**Files:**
- Modify: `fate_oia/models/tida_flow_transition_bank.py`
- Test: `tests/test_tida_multiscale_transition_bank.py`

- [ ] **Step 1: Write failing typed-token tests**

```python
def test_transition_bank_returns_four_typed_scales():
    output = build_bank_output()
    assert output["transition_tokens_by_scale"].shape == (2, 32, 4, 384)
    assert output["transition_scale_names"] == ("velocity", "acceleration", "region_velocity", "persistence")

def test_motion_salience_is_finite_non_saturating_and_zero_without_history():
    real = build_bank_output(valid=True)
    empty = build_bank_output(valid=False)
    assert torch.isfinite(real["motion_salience"]).all()
    assert real["motion_salience"].std() > 0
    assert torch.count_nonzero(empty["motion_salience"]) == 0
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_tida_multiscale_transition_bank.py`
Expected: FAIL because typed tokens and motion salience do not exist.

- [ ] **Step 3: Implement typed projections**

Add four independent projections to `TIDAFlowTransitionBank`, stack their outputs as `[B,P,4,D]`, add a learned scale-type embedding, and return robust log-magnitude salience plus consistency. Preserve the existing `transition_tokens [B,P,D]` output as the mean of typed tokens for checkpoint compatibility.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `pytest -q tests/test_tida_multiscale_transition_bank.py tests/test_tida_flow_transition_bank.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add fate_oia/models/tida_flow_transition_bank.py tests/test_tida_multiscale_transition_bank.py
git commit -m "feat(tida): expose typed multiscale transition tokens"
```

### Task 2: Conditional Temporal Utility Module

**Files:**
- Create: `fate_oia/models/tida_temporal_utility.py`
- Test: `tests/test_tida_temporal_utility.py`

- [ ] **Step 1: Write failing utility tests**

```python
def test_uncertain_image_and_strong_motion_receive_larger_budget():
    module = TIDAConditionalTemporalUtility(max_budget=0.60, min_budget=0.02)
    uncertain = module(torch.zeros(2, 4), motion(1.0), consistency(1.0), compatibility())
    certain = module(torch.full((2, 4), 8.0), motion(1.0), consistency(1.0), compatibility())
    assert torch.all(uncertain["budget"] > certain["budget"])

def test_no_history_has_exact_zero_budget():
    output = module(logits, torch.zeros_like(motion_score), consistency, compatibility)
    assert torch.count_nonzero(output["budget"]) == 0

def test_budget_is_target_specific_and_bounded():
    output = module(logits, motion_score, consistency, compatibility)
    assert output["budget"].shape == logits.shape
    assert output["budget"].min() >= 0
    assert output["budget"].max() <= 0.60
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_tida_temporal_utility.py`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement utility equation**

Implement:

```python
uncertainty = 4 * sigmoid(image_logits) * (1 - sigmoid(image_logits))
need = uncertainty * motion_salience[:, None] * consistency[:, None]
compatibility_weight = 0.5 + 0.5 * torch.sigmoid(compatibility)
budget = history_available[:, None] * max_budget * clamp(need * compatibility_weight, 0, 1)
```

Return `budget`, `uncertainty`, `need`, `compatibility_weight`, and saturation statistics. `motion_salience` and `consistency` are detached measurement inputs; compatibility remains trainable.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_tida_temporal_utility.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add fate_oia/models/tida_temporal_utility.py tests/test_tida_temporal_utility.py
git commit -m "feat(tida): add conditional temporal utility budgets"
```

### Task 3: Integrate Utility Into Action And Reason Readers

**Files:**
- Modify: `fate_oia/models/tida_action_reader.py`
- Modify: `fate_oia/models/tida_reason_reader.py`
- Modify: `fate_oia/models/tida_oia_model.py`
- Test: `tests/test_tida_conditional_readers.py`

- [ ] **Step 1: Write failing reader tests**

Test that utility-disabled logits equal the current reader exactly, history-off produces zero temporal delta, per-target budgets replace the fixed `0.35` mix, typed flow routes sum to their assigned budget, and reason loss has zero gradient to action owners.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_tida_conditional_readers.py`
Expected: FAIL because readers do not accept utility inputs.

- [ ] **Step 3: Integrate utility**

Add reader arguments `image_logits`, `transition_tokens_by_scale`, `motion_salience`, `transition_consistency`, and `history_available`. Flatten typed tokens only at the reader boundary. Compute query compatibility before utility, replace fixed `flow_mix_cap * flow_strength` with returned per-target budget, and expose `action_temporal_budget` and `reason_temporal_budget`. Preserve legacy behavior behind `conditional_utility_enabled=False`.

- [ ] **Step 4: Wire model outputs**

Pass image action/reason logits and transition measurements from `TIDAOIAModel`; export typed tokens, salience, consistency, budgets, and utility diagnostics. Preserve action/reason detach firewalls.

- [ ] **Step 5: Verify GREEN and compatibility**

Run: `pytest -q tests/test_tida_conditional_readers.py tests/test_tida_model_forward.py tests/test_tida_reason_firewall.py tests/test_tida_history_off_mask.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```text
git add fate_oia/models/tida_action_reader.py fate_oia/models/tida_reason_reader.py fate_oia/models/tida_oia_model.py tests/test_tida_conditional_readers.py
git commit -m "feat(tida): condition flow budgets on temporal utility"
```

### Task 4: Utility Credit And No-Harm Losses

**Files:**
- Modify: `fate_oia/losses/tida_flow_credit_losses.py`
- Modify: `fate_oia/losses/tida_losses.py`
- Modify: `fate_oia/losses/tida_loss_registry.py`
- Modify: `configs/fate_oia_train_tida_oia_v1_15f.yaml`
- Test: `tests/test_tida_utility_losses.py`

- [ ] **Step 1: Write failing loss tests**

Test that detached positive counterfactual benefit increases the desired utility target, harmful flow drives utility down, high-need samples receive larger credit weights, low-need samples receive larger no-harm weights, and each loss appears exactly once.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_tida_utility_losses.py tests/test_tida_loss_terms_once.py`
Expected: FAIL because utility calibration terms are absent.

- [ ] **Step 3: Implement losses**

Add `temporal_utility_calibration_loss(budget, real_logits, counterfactual_logits, target)` using detached target-signed benefit as a soft target. Add `conditional_credit_weight` and `conditional_no_harm_weight`. Register action/reason utility calibration at weights `0.05/0.04`; retain existing flow credit and no-harm terms once.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_tida_utility_losses.py tests/test_tida_flow_credit_losses.py tests/test_tida_loss_terms_once.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```text
git add fate_oia/losses/tida_flow_credit_losses.py fate_oia/losses/tida_losses.py fate_oia/losses/tida_loss_registry.py configs/fate_oia_train_tida_oia_v1_15f.yaml tests/test_tida_utility_losses.py
git commit -m "feat(tida): calibrate utility with paired temporal benefit"
```

### Task 5: Temporal Contribution Metrics

**Files:**
- Create: `fate_oia/utils/tida_temporal_metrics.py`
- Modify: `fate_oia/engine/evaluate_tida_oia.py`
- Modify: `fate_oia/utils/tida_artifacts.py`
- Test: `tests/test_tida_temporal_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Test label-independent motion quartiles, benefit/harm coverage, paired-bootstrap intervals with deterministic seed, per-label signed accuracy, and unavailable handling when a slice lacks both classes.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_tida_temporal_metrics.py`
Expected: FAIL with missing module.

- [ ] **Step 3: Implement metrics**

Derive motion score from robust standardized `log1p(velocity)` plus half-weighted `log1p(acceleration)`. Define benefit as target-signed video-minus-image margin greater than `1e-4`, harm below `-1e-4`, and bootstrap the sample-mean signed margin for 2000 deterministic resamples. Export full, motion-quartile, uncertainty-quartile, and per-label records.

- [ ] **Step 4: Wire artifacts**

Write `temporal_contribution_metrics.json` per epoch and append summary fields to `metrics_summary.jsonl`; include budgets and typed-route tensors in epoch artifacts.

- [ ] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_tida_temporal_metrics.py tests/test_tida_artifact_schema.py tests/test_tida_supervisor_protocol.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```text
git add fate_oia/utils/tida_temporal_metrics.py fate_oia/engine/evaluate_tida_oia.py fate_oia/utils/tida_artifacts.py tests/test_tida_temporal_metrics.py
git commit -m "feat(tida): report causal temporal contribution metrics"
```

### Task 6: Audit, Regression, And Checkpoint Pilot

**Files:**
- Modify: `fate_oia/engine/audit_tida_oia_implementation.py`
- Modify: `tests/test_tida_flow_audit.py`
- Create: `scripts/FATE_OIA_tida_ctu_refinement.ps1`

- [ ] **Step 1: Add RED audit checks**

Require typed flow tokens, conditional budgets, exact disabled equivalence, non-saturated budget distribution, metric artifacts, no test parameter fit, action/reason firewalls, and static no-harm.

- [ ] **Step 2: Run RED audit tests**

Run: `pytest -q tests/test_tida_flow_audit.py`
Expected: FAIL until the new checks are implemented.

- [ ] **Step 3: Implement audit and pilot entry point**

The PowerShell entry point loads `checkpoint_best_test_joint.pth`, trains only temporal owners for at most three epochs, uses train-calib/train-audit for action/reason utility strength selection, writes one locked choice, and evaluates test once.

- [ ] **Step 4: Run full verification**

Run: `python -m py_compile fate_oia/models/tida_*.py fate_oia/losses/tida_*.py fate_oia/utils/tida_temporal_metrics.py fate_oia/engine/*.py`

Run: `pytest -q tests/test_tida_*.py`

Expected: all PASS.

- [ ] **Step 5: Run remote pilot and decide**

The pilot must report current-vs-CTU total metrics, full-test flow deltas, dynamic/uncertain slice deltas, bootstrap intervals, benefit/harm coverage, and intervention margins. Start a new full run only if the locked CTU setting improves both total metrics and temporal contribution without a static-slice regression.

- [ ] **Step 6: Commit and push**

```text
git add fate_oia tests configs scripts docs/superpowers
git commit -m "feat(tida): complete conditional temporal utility"
git push github HEAD:tida_oia_conditional_temporal_utility_v2
```
