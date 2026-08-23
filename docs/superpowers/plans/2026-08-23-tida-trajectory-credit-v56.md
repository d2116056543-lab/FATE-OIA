# TIDA Trajectory Credit V5.6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline with TDD and review each task before the next task.

**Goal:** Learn useful signed action credit from ordered traffic trajectories while using reversed trajectories only as causal controls.

**Architecture:** Shared trajectory encoders produce ordered and reversed contexts. Only their signed/magnitude contrast and learned action identity enter the correction readout; detached base uncertainty only gates its budget. Control corrections are independently decoded with shared weights.

**Tech Stack:** Python 3.9, PyTorch, pytest, frozen DINO video inference.

---

### Task 1: Lock the corrected semantics with RED tests

**Files:**
- Modify: `tests/test_tida_traffic_trajectory_head.py`
- Modify: `tests/test_tida_traffic_trajectory_losses.py`

- [ ] Add a test proving ordered and reversed controls are not forced to exact negatives.
- [ ] Add a test proving base logits condition the correction.
- [ ] Add a test proving zero output still yields exact zero residual.
- [ ] Run the focused tests and verify the new behavior fails before implementation.

### Task 2: Implement non-antisymmetric trajectory credit

**Files:**
- Modify: `fate_oia/models/tida_traffic_trajectory_head.py`
- Modify: `fate_oia/models/tida_oia_model.py`

- [ ] Project detached base logits into the credit feature.
- [ ] Decode ordered and reversed corrections independently with shared weights.
- [ ] Derive a bounded order-discriminability gate from context contrast.
- [ ] Initialize trust at 0.5 and output at zero.
- [ ] Return ordered/control raw deltas and order gate diagnostics.
- [ ] Run focused tests until green.

### Task 3: Train against the explicit reversed control

**Files:**
- Modify: `fate_oia/losses/tida_losses.py`
- Modify: `fate_oia/losses/tida_traffic_trajectory_losses.py`
- Modify: `fate_oia/models/tida_oia_model.py`

- [ ] Surface the internally decoded reversed-control delta.
- [ ] Prefer the internal matched control over recomputing an artificial opposite output.
- [ ] Retain same-clip shuffle/repeat controls as additional hard controls.
- [ ] Verify the reachable margin uses the actual cap, trust, support, and order gate.

### Task 4: Complete diagnostics and verification

**Files:**
- Modify: `fate_oia/engine/evaluate_tida_oia.py`
- Modify: `fate_oia/engine/train_tida_oia.py`
- Modify: `tests/test_tida_trajectory_effectiveness_metrics.py`

- [ ] Persist order gate, ordered/control deltas, and correction-to-control advantage.
- [ ] Report action flip counts, signed GT margin, dynamic quartiles, and causal interventions.
- [ ] Run `py_compile` and all `pytest -k tida` tests.
- [ ] Run one full-data head-only mechanism probe from the same strong checkpoint.
- [ ] Continue to a short multi-epoch probe only if transport becomes positive or improves without baseline damage.
