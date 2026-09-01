# TIDA-LiteFlow V21 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace history DINO with target-conditioned RGB flow and object-track evidence while preserving exact VETRA fallback and improving real action/reason ranking at lower cost.

**Architecture:** Add a lightweight history mode to `TIDAOIAModel`. It reuses terminal DINO query tensors as a schema-only baseline, runs fixed RGB correlation flow and terminal-semantic object intent, and disables every history-DINO-dependent route. Train-calib independently selects bounded action/reason deployment.

**Tech Stack:** Python 3.9, PyTorch, pytest, YAML, PowerShell, BDD-OIA video manifests.

---

### Task 1: Lightweight history contract

**Files:**
- Modify: `fate_oia/models/tida_oia_model.py`
- Test: `tests/test_tida_liteflow_history.py`

- [ ] **Step 1: Write failing tests** proving `history_encoder_mode="terminal_repeat"` does not call `context_encoder`, emits all decoder-compatible history fields, and returns image logits exactly when every temporal scale is zero.
- [ ] **Step 2: Run RED** with `pytest tests/test_tida_liteflow_history.py -q`; expect constructor/field failures.
- [ ] **Step 3: Implement `_terminal_repeat_context`** by expanding detached terminal query/attention/region tensors across history frames and using zero-size or repeated terminal patch fields only where disabled routes require schema values. Reject this mode if any DINO-history-dependent module is enabled.
- [ ] **Step 4: Run GREEN** and the existing TIDA model-forward tests.

### Task 2: Target-conditioned traffic flow

**Files:**
- Modify: `fate_oia/models/tida_geometric_flow.py`
- Modify: `fate_oia/models/tida_oia_model.py`
- Test: `tests/test_tida_liteflow_target_credit.py`

- [ ] **Step 1: Write failing tests** for finite ordered flow features, time-reversal sensitivity, selected-region deletion greater than a deterministic equal-area control, separate action/reason owners, and zero output without history.
- [ ] **Step 2: Run RED** and verify missing target-conditioned outputs cause failure.
- [ ] **Step 3: Extend the fixed flow measurement** with confidence-weighted speed/acceleration and terminal-query regional priors. Implement private bounded action/reason readers and selected/control deletion reruns without dense `[B,L,T,H,W,D]` tensors.
- [ ] **Step 4: Run GREEN** plus geometric-flow regressions.

### Task 3: Object-track terminal-only route

**Files:**
- Modify: `fate_oia/models/tida_oia_model.py`
- Test: `tests/test_tida_liteflow_object_route.py`

- [ ] **Step 1: Write failing tests** proving precomputed tracks call `TIDAObjectIntentTransport` with `temporal_patch_tokens=None`, use target patch semantics, retain selected/control artifacts, and never require history DINO fields.
- [ ] **Step 2: Run RED** against the current always-dense call.
- [ ] **Step 3: Add `object_intent_terminal_semantics_only`** and route only target patches, timestamps, tracks, target nodes, and detached base logits.
- [ ] **Step 4: Run GREEN** and gradient-owner tests.

### Task 4: Losses, telemetry, and config

**Files:**
- Modify: `fate_oia/engine/train_tida_oia.py`
- Modify: `fate_oia/engine/evaluate_tida_oia.py`
- Modify: `fate_oia/losses/tida_losses.py`
- Modify: `fate_oia/losses/tida_loss_registry.py`
- Create: `configs/fate_oia_train_tida_liteflow_v21.yaml`
- Test: `tests/test_tida_liteflow_protocol.py`

- [ ] **Step 1: Write failing tests** for registered flow/object action and PU-reason terms, branch RMS budgets, train-calib-only policy, artifact schema, and the forbidden heavy-module matrix.
- [ ] **Step 2: Run RED** and confirm missing config/protocol keys.
- [ ] **Step 3: Add telemetry and config** with five history frames, `terminal_repeat`, geometric flow and object intent enabled, all history-DINO routes disabled, workers selected from measured throughput, and separate action/reason deployment policy.
- [ ] **Step 4: Run GREEN**, py_compile, and all relevant TIDA tests.

### Task 5: Equal-budget real validation

**Files:**
- Create: `scripts/compare_tida_liteflow_site.py`
- Modify: `scripts/summarize_tida_site_artifacts.py`

- [ ] **Step 1: Run one real batch** and assert finite nonzero flow/object outputs, exact zero-scale fallback, and no history DINO hook calls.
- [ ] **Step 2: Benchmark** the same 256 examples against SITE V20 and report read/target-DINO/history/flow/object/backward seconds and peak memory.
- [ ] **Step 3: Train equal-budget 2,048/512 pilots** from the same frozen checkpoint and seed. Compare image, SITE, flow-only, object-only, and combined candidate/deploy views.
- [ ] **Step 4: Validate acceptance** using real AP/AUC/mF1/oF1, deletion gaps, utility, and no-harm. Do not launch full training if neither task has real ranking gain.

### Task 6: Audit, records, and source control

**Files:**
- Modify: `E:\sbw\FATE_Drive\task_plan.md`
- Modify: `E:\sbw\FATE_Drive\findings.md`
- Modify: `E:\sbw\FATE_Drive\progress.md`

- [ ] **Step 1: Run final py_compile and relevant pytest suite** and preserve exact output.
- [ ] **Step 2: Append research, implementation, speed, pilot metrics, failures, and decision** to the three canonical Markdown files and local Downloads mirrors.
- [ ] **Step 3: Commit code-only changes**; exclude runtime tensors, checkpoints, logs, and copied result JSON.
- [ ] **Step 4: Push and verify** remote branch HEAD with `git ls-remote`.
- [ ] **Step 5: Launch full training only after acceptance**, storing runtime artifacts on G, evaluating the exact official test set, and preserving the target-frame baseline view every epoch.
