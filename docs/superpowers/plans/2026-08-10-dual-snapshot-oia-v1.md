# Dual-Snapshot OIA V1 Implementation Plan

> **For agentic workers:** Execute inline with test-driven development and verify each checkpoint before full training.

**Goal:** Reproduce the proven early/late trajectory complement from a clean official-DINO run and deploy independently calibrated action and reason snapshot ensembles.

**Architecture:** Reuse the audited AIE model unchanged. Add epoch calibration artifacts, a strict dual-task snapshot evaluator, two locked configs, and a restart-safe two-stage supervisor.

**Tech Stack:** Python, PyTorch, pytest, YAML, PowerShell, Git.

---

### Task 1: Calibration artifact contract

**Files:** modify `fate_oia/engine/train_aie_oia.py`; test `tests/test_dual_snapshot_artifacts.py`.

- [ ] Write a failing test requiring aligned train-calib logits, labels, and names.
- [ ] Save those tensors from the already-computed calibration forward without another image/DINO pass.
- [ ] Run the focused test and AIE regression tests.

### Task 2: Dual-task evaluator

**Files:** create `fate_oia/engine/eval_dual_snapshot_oia.py`; test `tests/test_dual_snapshot_oia.py`.

- [ ] Write failing tests for fixed action/reason weights, independent shrinkage, alignment rejection, and metrics.
- [ ] Implement strict artifact loading, blending, train-calib threshold fitting, test evaluation, and bootstrap comparison.
- [ ] Verify tests fail before implementation and pass afterward.

### Task 3: Reproducible two-stage run

**Files:** create two configs and `scripts/FATE_OIA_dual_snapshot_oia_v1_background.ps1`.

- [ ] Copy the audited AIE base protocol and lock 20 base epochs.
- [ ] Lock the three-epoch low-LR consolidation protocol used by the strongest control.
- [ ] Chain base training, consolidation initialization, and final fixed-weight evaluation with restart checks and logs.

### Task 4: Verification and launch

- [ ] Run `py_compile` and focused/regression pytest.
- [ ] Run a four-sample real-DINO smoke and verify finite nonconstant logits and complete artifacts.
- [ ] Commit and push code-only changes.
- [ ] Start the full run detached from SSH and verify the process, GPU utilization, log updates, and first epoch result.
