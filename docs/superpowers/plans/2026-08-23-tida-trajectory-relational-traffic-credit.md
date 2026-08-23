# TIDA Trajectory-Relational Traffic Credit V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace frame-independent patch averaging with cycle-consistent relational trajectory tokens that produce measurable, action-only traffic utility.

**Architecture:** Final-frame action-attentive anchors are propagated through history using bidirectional DINO correspondence. Per-trajectory temporal and inter-trajectory encoders produce a zero-initialized bounded action residual, trained with boundary, ranking, cycle, and deletion objectives while preserving the reason firewall and stable train-calib deployment.

**Tech Stack:** Python 3.9, PyTorch, pytest, frozen DINO ViT-S/8, existing TIDA training/evaluation engine.

---

### Task 1: Cycle-consistent trajectory builder

**Files:**
- Create: `fate_oia/models/tida_traffic_trajectories.py`
- Create: `tests/test_tida_traffic_trajectories.py`

- [ ] Write failing tests for identity-preserving backward tracks, reverse-cycle confidence, invalid-frame masking, finite gradients, and camera-common/exclusive decomposition.
- [ ] Run `pytest tests/test_tida_traffic_trajectories.py -q` and verify failures are due to the missing module.
- [ ] Implement `TIDATrafficTrajectoryBuilder.forward(patch_tokens, patch_xy, patch_weights, frame_valid_mask)` returning trajectory appearance/xy/visibility/cycle confidence/common and exclusive displacement.
- [ ] Re-run the test file and commit.

### Task 2: Relational trajectory action head

**Files:**
- Create: `fate_oia/models/tida_traffic_trajectory_head.py`
- Create: `tests/test_tida_traffic_trajectory_head.py`

- [ ] Write failing tests for eight-bin direction features, trajectory and relation token shapes, strict zero-init equivalence, bounded residual, per-action trust, temporal-order sensitivity, and zero reason gradients.
- [ ] Verify RED, then implement temporal trajectory encoding, inter-trajectory attention, action-query readout, and bounded action-only residual.
- [ ] Verify GREEN and commit.

### Task 3: Integrate into TIDA forward and interventions

**Files:**
- Modify: `fate_oia/models/tida_oia_model.py`
- Modify: `fate_oia/models/tida_context_encoder.py`
- Modify: `fate_oia/engine/evaluate_tida_oia.py`
- Modify: `configs/fate_oia_train_tida_oia_v1_15f.yaml`
- Modify: `tests/test_tida_model_forward.py`
- Modify: `tests/test_tida_temporal_interventions.py`

- [ ] Write failing forward tests for trajectory artifacts, V4 fallback, action-only final integration, and consistent reverse/shuffle/repeated-last transformations.
- [ ] Verify RED, wire V5 behind `traffic_trajectory_enabled`, surface all diagnostics, and preserve V4 compatibility.
- [ ] Verify GREEN and commit.

### Task 4: Utility, boundary, cycle, and deletion losses

**Files:**
- Modify: `fate_oia/losses/tida_losses.py`
- Create: `tests/test_tida_traffic_trajectory_losses.py`

- [ ] Write failing tests proving boundary examples receive higher correction weight, selected deletion exceeds matched control for useful trajectories, cycle errors are penalized, action-specific negative utility cannot open trust, and reason parameters receive no gradients.
- [ ] Verify RED, implement the four losses and add explicit weighted registry entries exactly once.
- [ ] Verify GREEN and commit.

### Task 5: Traffic-specific metrics and visual evidence

**Files:**
- Modify: `fate_oia/engine/evaluate_tida_oia.py`
- Modify: `fate_oia/engine/export_tida_traffic_action_visuals.py`
- Create: `tests/test_tida_traffic_trajectory_metrics.py`

- [ ] Write failing tests for dynamic quantiles, prefix anticipation AUC, earliest-correct gain, corrective/harm ratio, bootstrap deletion gap, cycle coverage, and trajectory visualization schema.
- [ ] Verify RED, implement metrics and exports without using test outputs for parameters.
- [ ] Verify GREEN and commit.

### Task 6: Threshold stability and protocol audit

**Files:**
- Modify: `fate_oia/engine/train_tida_oia.py`
- Modify: `fate_oia/engine/audit_tida_implementation.py`
- Create: `tests/test_tida_traffic_threshold_stability.py`

- [ ] Write failing tests that train-calib chooses thresholds, strong-threshold tie-break prevents unsupported drift, test labels cannot update state, and checkpoints persist trust/threshold history.
- [ ] Verify RED, implement train-only threshold stability and audit artifacts.
- [ ] Verify GREEN and commit.

### Task 7: Verification and focused pilot

**Files:**
- Update: `docs/superpowers/specs/2026-08-23-tida-trajectory-relational-traffic-credit-design.md`

- [ ] Run targeted V5 tests.
- [ ] Run `pytest tests -q -k tida` and require no regression.
- [ ] Run a real-DINO 32-clip smoke and verify finite/nonzero trajectory coverage, cycle confidence, trust gradient, deletion gap, and reason firewall.
- [ ] Run one focused full-data owner-isolated pilot using 2291 train-core, 312 train-calib, and 885 test.
- [ ] Compare absolute online/EMA Act metrics to 0.7838658, plus dynamic, anticipation, deletion, and harm diagnostics. Do not approve formal training on gate results alone.
