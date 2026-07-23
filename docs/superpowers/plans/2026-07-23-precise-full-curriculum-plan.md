# PRECISE-OIA Full-Run Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discarded standalone pilot with a reproducible staged curriculum inside the retained 12-epoch PRECISE full run.

**Architecture:** Add one pure curriculum module that maps epoch to immutable activation scales. The model applies these scales at the actual delta-to-logit boundaries, while the trainer uses the same state for loss scaling, owner-local optimizer clocks, artifacts, checkpoint resume, and supervisor authorization.

**Tech Stack:** Python 3.9, PyTorch, pytest, YAML, PowerShell, frozen DINO ViT-S/8.

---

### Task 1: Curriculum Contract

**Files:**
- Create: `fate_oia/engine/precise_curriculum.py`
- Create: `tests/test_precise_curriculum.py`
- Modify: `configs/fate_oia_train_360x640_precise_oia_v1.yaml`

- [ ] Write tests for exact epoch 0-11 activation values, invalid epoch handling,
  stage names, and owner-active mapping.
- [ ] Run `pytest -q tests/test_precise_curriculum.py` and verify it fails because
  the module and YAML section do not exist.
- [ ] Implement immutable `PRECISECurriculumState` and
  `curriculum_state_for_epoch(config, epoch)`.
- [ ] Add the exact approved table under `curriculum:` in YAML.
- [ ] Re-run the test and require PASS.

### Task 2: Apply Scales At Final-Logit Boundaries

**Files:**
- Modify: `fate_oia/models/precise_oia_model.py`
- Modify: `tests/test_precise_model_forward.py`
- Modify: `tests/test_precise_counterfactual.py`

- [ ] Write tests showing zero reread/exchange/annotation/latent scales reproduce
  the existing refined-head direct anchor exactly.
- [ ] Run the targeted tests and verify RED against the current always-on model.
- [ ] Add `set_curriculum_state()` and apply scales before deltas enter final
  tokens/logits, including cached intervention decode paths and diagnostic
  ablations.
- [ ] Assert each boundary independently:
  `effective_delta == activation * raw_delta`; do not assert global linearity
  across the nonlinear reread/exchange composition.
- [ ] Apply threshold activation as
  `deploy_logits = raw_logits - threshold_activation * theta`.
- [ ] Return refined-direct anchors, raw/effective deltas, raw/effective theta,
  and the active curriculum state.
- [ ] Re-run targeted tests and require PASS.

### Task 3: Loss And Owner-Local Update Clocks

**Files:**
- Modify: `fate_oia/engine/train_precise_oia.py`
- Modify: `tests/test_precise_train_protocol.py`
- Modify: `tests/test_precise_resume_equivalence.py`
- Modify: `tests/test_precise_gradient_ownership.py`

- [ ] Write tests proving intervention loss is multiplied by its activation,
  inactive owners do not step optimizer/scheduler, and resume restores local
  owner clocks.
- [ ] Run tests and verify RED.
- [ ] Set model curriculum state before each epoch.
- [ ] Split ownership into `reason_semantic`, `reread_adapter`,
  `exchange_adapter`, and `reason_latent`; move
  `evidence_fields.latent_parameters()` to always-on `evidence_core`; no
  parameter may occur in two owners.
- [ ] Make `_step_owners` accept owner-active flags and skip optimizer-state
  creation, weight decay, optimizer step, and scheduler step for inactive
  owners.
- [ ] Construct optimizer objects at launch but require inactive
  `optimizer.state` to remain empty; clear inactive-owner gradients at every
  accumulation boundary and assert first activation begins at local step zero.
- [ ] Use exact owner-local scheduler active-epoch totals: always-on=12,
  reread/annotation/threshold=10, exchange/reason-latent=8.
- [ ] Gate threshold calibration by threshold activation.
- [ ] Save/restore activation state, active-update totals, curriculum hash, and
  owner-local update clocks; reject mismatched resume.
- [ ] Add resume-equivalence tests at epoch boundaries 1->2, 3->4, and 5->6.
- [ ] Re-run targeted tests and require PASS.

### Task 4: Artifact And Supervisor Contract

**Files:**
- Modify: `fate_oia/engine/train_precise_oia.py`
- Modify: `fate_oia/engine/supervise_precise_oia_foreground.py`
- Modify: `fate_oia/engine/audit_precise_oia_implementation.py`
- Modify: `tests/test_precise_supervisor.py`
- Modify: `tests/test_precise_audit.py`
- Modify: `tests/test_precise_eval_contract.py`

- [ ] Write tests requiring curriculum identity/scales/owner flags in manifest,
  batch rows, epoch rows, checkpoints, and audit output.
- [ ] Write tests for generation and verification of
  `PRECISE_OIA_V1_FULL_CURRICULUM_READY.json`, bound to HEAD,
  source/config/curriculum/skill/DINO/schema hashes, current preflight gate,
  runtime profile, clean worktree, remote HEAD, and explicit user override.
- [ ] Prove the old `FULL_TRAIN_READY` cannot authorize the curriculum path.
- [ ] Run tests and verify RED.
- [ ] Implement artifact schema and strict override recording.
- [ ] Write exact schema assertions for schedule/hash, all eight scales, every
  owner active/step/LR/total, refined anchor, raw/effective delta RMS and ratios,
  raw/scaled intervention loss, threshold raw/effective theta, and override
  provenance in manifest, batch row, epoch row, checkpoint, and audit JSON.
- [ ] Make audit curriculum-aware: early inactive-owner steps must be zero and
  all owners must be enabled by epoch 6.
- [ ] Separate prelaunch checks from runtime assertions: prelaunch verifies
  static/simulated behavior; each real epoch verifies actual scale, step delta,
  optimizer state, and LR. Abort with a diagnostic artifact if an epoch-6 owner
  has not stepped.
- [ ] Use the fixed flag `--allow_full_with_embedded_curriculum` and manifest
  provenance `override_source=user_approved_2026-07-23`.
- [ ] Hash canonical JSON of the normalized schedule, not YAML source text.
- [ ] Keep no-cache, no-compression, frozen DINO, test-only, and train-calib
  leakage checks unchanged.
- [ ] Re-run targeted tests and require PASS.

### Task 5: Full Verification And Launch

**Files:**
- Modify only if verification identifies a concrete defect.

- [ ] Run `py_compile` on all changed PRECISE files.
- [ ] Run all `tests/test_precise_*.py`; expected result is all PASS.
- [ ] Run implementation audit and verify current git/config/schema hashes.
- [ ] Run same-strength supervisor review of plan coverage and implementation.
- [ ] Commit code/config/tests/docs only and push branch.
- [ ] Re-run clean-HEAD audit and verify remote HEAD equals local HEAD.
- [ ] Launch the 12-epoch full run with the explicit embedded-curriculum override.
- [ ] Launch in foreground, confirm first optimizer steps and curriculum
  artifacts, and verify selected `batch=6, accum=5`. Do not claim SSH-disconnect
  survival.
