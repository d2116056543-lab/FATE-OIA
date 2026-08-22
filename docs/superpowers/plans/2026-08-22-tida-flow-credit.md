# TIDA Flow Credit V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ordered traffic flow provide target-specific positive action/reason credit while preserving the exact frozen VETRA image fallback.

**Architecture:** Add a signed transition bank over existing query trajectories, remove the current-frame shortcut from terminal prediction, and supervise action/reason temporal residuals with same-image counterfactual GT-margin losses. Keep all deltas bounded and zero-effect at initialization.

**Tech Stack:** Python 3.9, PyTorch, pytest, frozen DINO/VETRA, BDD-OIA video clips.

---

### Task 1: Shortcut-Free Terminal Predictor

**Files:**
- Modify: `fate_oia/models/tida_terminal_innovation.py`
- Modify: `fate_oia/models/tida_oia_model.py`
- Modify: `fate_oia/losses/tida_losses.py`
- Test: `tests/test_tida_flow_terminal.py`

- [ ] Write RED tests proving current target static context cannot change terminal predictions and no-history reconstruction is diagnostic-only.
- [ ] Run `pytest -q tests/test_tida_flow_terminal.py` and confirm failure from the old static input contract.
- [ ] Change the predictor contract to `(query_identity, history_summary)`, detach terminal targets, and remove `terminal_no_history` from the optimized weighted total while retaining the artifact value.
- [ ] Run the targeted test and existing `tests/test_tida_terminal_innovation.py` to GREEN.
- [ ] Commit `fix(tida): remove target-frame terminal shortcut`.

### Task 2: Signed Flow Transition Bank

**Files:**
- Create: `fate_oia/models/tida_flow_transition_bank.py`
- Test: `tests/test_tida_flow_transition_bank.py`

- [ ] Write RED tests for shapes, finite masked derivatives, reversal sign, repeat collapse, and no dense patch tensor.
- [ ] Run `pytest -q tests/test_tida_flow_transition_bank.py` and confirm import failure.
- [ ] Implement `TIDAFlowTransitionBank.forward(query_trajectory, region_mass, timestamps, valid_mask)` returning `transition_tokens`, `velocity`, `acceleration`, `region_velocity`, `persistence`, and `transition_reliability`.
- [ ] Run the test to GREEN and commit `feat(tida): add signed flow transition bank`.

### Task 3: Target-Conditioned Flow Readers

**Files:**
- Modify: `fate_oia/models/tida_action_reader.py`
- Modify: `fate_oia/models/tida_reason_reader.py`
- Modify: `fate_oia/models/tida_oia_model.py`
- Test: `tests/test_tida_flow_readers.py`

- [ ] Write RED tests proving zero-scale exact fallback, non-zero target-private contributions, bounded deltas, null routing, and reason-to-action gradient firewall.
- [ ] Run the tests and confirm missing transition-factor integration.
- [ ] Append transition tokens as a distinct factor family, expose family route mass, and keep existing bounded trust equations.
- [ ] Surface `action_flow_delta`, `reason_flow_delta`, transition diagnostics, and image/video logits from full forward.
- [ ] Run reader/model regressions to GREEN and commit `feat(tida): route signed flow to action and reason targets`.

### Task 4: Counterfactual GT-Margin Credit

**Files:**
- Create: `fate_oia/losses/tida_flow_credit_losses.py`
- Modify: `fate_oia/losses/tida_losses.py`
- Modify: `fate_oia/engine/train_tida_oia.py`
- Test: `tests/test_tida_flow_credit_losses.py`

- [ ] Write RED tests for action/reason signed margins, satisfied/violated margin behavior, PU reason weighting, and finite gradients.
- [ ] Run the test and confirm import failure.
- [ ] Implement `signed_gt_margin`, `counterfactual_margin_credit_loss`, and `image_fallback_no_harm_loss`.
- [ ] Schedule history-off/repeat every update and shuffle/reverse on alternating updates; feed same-image reruns into the registry without test-derived decisions.
- [ ] Run loss/trainer tests to GREEN and commit `feat(tida): supervise ordered history with target margin credit`.

### Task 5: Phase Schedule And Configuration

**Files:**
- Modify: `configs/fate_oia_train_360x640_tida_oia_v1.yaml`
- Modify: `fate_oia/engine/train_tida_oia.py`
- Test: `tests/test_tida_flow_schedule.py`

- [ ] Write RED tests for `FLOW_FOUNDATION -> FLOW_CREDIT -> SAFE_JOINT`, scale-zero foundation, and test-independent transitions.
- [ ] Add update-based phase fields, counterfactual margins, flow loss weights, transition-bank LR group, and explicit no-cache/no-compression invariants.
- [ ] Run tests to GREEN and commit `feat(tida): add flow credit training schedule`.

### Task 6: Evaluation And Audit Artifacts

**Files:**
- Modify: `fate_oia/engine/eval_tida_oia.py`
- Modify: `fate_oia/engine/audit_tida_oia_implementation.py`
- Modify: `fate_oia/engine/audit_tida_oia_mechanism.py`
- Test: `tests/test_tida_flow_audit.py`

- [ ] Write RED schema tests for per-intervention GT margins, transition sign diagnostics, route-family mass, delta RMS ratio, and gradient isolation.
- [ ] Implement image/real/history-off/repeat/shuffle/reverse metric views and hard forbidden-pattern checks.
- [ ] Require exact fallback and factual artifact availability; keep preferred metric gains as reported evidence rather than arbitrary code-existence gates.
- [ ] Run audit/eval tests to GREEN and commit `feat(tida): audit measurable traffic flow utility`.

### Task 7: Full Verification And Remote Video Pilot

**Files:**
- Modify: `.codex/skills/tida-oia-implementation-audit/SKILL.md`
- Create: `.review/TIDA_FLOW_CREDIT_REVIEW_PASS.json` only after all checks pass.

- [ ] Run `python -m py_compile` over all changed modules.
- [ ] Run all `tests/test_tida_flow_*.py` plus existing TIDA regression tests.
- [ ] Create remote isolated worktree `E:\sbw\FATE_Drive\fate_oia_tida_flow_credit_v1_worktree` from exact HEAD `25184fb6d7205a3289a7725fa4ead1a8ac4659b4`.
- [ ] Run CUDA real-DINO smoke and a bounded real-video pilot using train-core only, with train-audit mechanism reporting.
- [ ] Compare the best pilot to frozen image/history-off and record both absolute task metrics and traffic-flow utility.
- [ ] Start full training only if code correctness passes and traffic flow is stable/non-harmful with positive ordered-history utility.

## Self-Review

- Every design requirement maps to a task.
- Production behavior is preceded by a failing test.
- No optical-flow dependency, stronger backbone, cache, token compression, action-set final path, graph delta, test leakage, or fallback degradation is introduced.
- The plan distinguishes functional correctness from metric preference; a REVIEW_PASS cannot by itself claim traffic-flow effectiveness.

