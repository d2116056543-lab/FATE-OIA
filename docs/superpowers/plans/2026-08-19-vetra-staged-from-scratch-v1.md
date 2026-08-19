# VETRA Staged From-Scratch V1 Implementation Plan

> Execute this plan with test-first changes. The final run may start only from official frozen DINO and random task heads.

**Goal:** Reproduce the historical VETRA range in one auditable staged run: `Act_mF1>=0.731`, `Exp_mF1>=0.405`, and `0.55<=Exp_oF1<=0.57`.

**Architecture:** Stage A trains the verified clean AIE base and selects on `train_audit`. Stage B freezes that exact same-run checkpoint and optionally trains an action-only refiner with an explicit untouched-base fallback. Stage C collects original/flip outputs, fits action TTA/combo by nested train-only OOF, and fits reason thresholds from `train_calib` only.

**Stack:** Python, PyTorch, scikit-learn, YAML, pytest, PowerShell.

## Task 1: Stage Contracts And Lineage

**Files:**
- Create: `fate_oia/utils/vetra_stage_contracts.py`
- Create: `tests/test_vetra_staged_contracts.py`

1. Add failing tests for run IDs, checkpoint hashes, same-root lineage, stage/parent mismatch, historical checkpoint rejection, and atomic completion records.
2. Implement deterministic file/object hashing, run manifest creation, Stage A checkpoint promotion, Stage B lineage validation, and atomic JSON writing.
3. Run `pytest tests/test_vetra_staged_contracts.py -q`.

## Task 2: Independent Deployment Fits

**Files:**
- Modify: `fate_oia/engine/export_vetra_from_scratch_deploy.py`
- Create: `tests/test_vetra_staged_deployment.py`

1. Add failing tests proving action splits and reason splits are independent and both reject `test`.
2. Add `--reason-fit-splits`, defaulting to `--fit-splits` for backward compatibility.
3. Fit reason thresholds only from the declared reason splits and write both policies to the manifest.
4. Run the deployment tests and existing VETRA pipeline tests.

## Task 3: Same-Run Fail-Closed Refiner

**Files:**
- Create: `fate_oia/engine/train_vetra_staged_refine.py`
- Modify: `fate_oia/models/vetra_strong_refiner.py`
- Create: `tests/test_vetra_staged_refine.py`

1. Add failing tests for frozen base parameters, exact reason identity, bounded action delta, zero-effect initialization, same-run source validation, and untouched-base fallback.
2. Wrap the existing action-only refiner with strict Stage A lineage validation.
3. Select between untouched Stage A and refined candidates on train-audit AP/F1 only; record `refiner_selected` and no-regression evidence.
4. Save a full inference checkpoint whose base model remains the same-run Stage A state and whose optional refiner state/gain is explicit.

## Task 4: Refiner-Aware Collection

**Files:**
- Modify: `fate_oia/engine/collect_vetra_tta_outputs.py`
- Create: `tests/test_vetra_staged_collection.py`

1. Add failing tests for original/flip semantic remapping, optional refiner application, zero/no-refiner equivalence, and reason identity.
2. Add optional `--stage-b-checkpoint`; validate its parent hash and apply only when `refiner_selected=true`.
3. Write collection metadata containing source hashes, selected gains, split counts, and reason-identity verification.

## Task 5: Lifecycle Supervisor And Configuration

**Files:**
- Create: `fate_oia/engine/supervise_vetra_staged_from_scratch.py`
- Create: `configs/fate_oia_train_360x640_vetra_staged_from_scratch_v1.yaml`
- Create: `scripts/FATE_OIA_vetra_staged_from_scratch_v1.ps1`
- Create: `tests/test_vetra_staged_supervisor.py`

1. Add failing lifecycle/resume tests.
2. Implement atomic `Stage A -> Stage B -> collect -> Stage C` execution with strict run ID, source tree, split hash, checkpoint hash, and parent hash validation.
3. Stage A must reject task initialization and run 10 epochs; Stage B runs at most three epochs; Stage C uses action `train_calib+train_audit` nested OOF and reason `train_calib` only.
4. Preserve no-cache/no-compression/frozen-DINO and foreground stdout. Record every command and selected checkpoint.

## Task 6: Verification And Full Run

1. Run `py_compile` on all new/modified files.
2. Run all `tests/test_vetra_staged_*.py` plus existing VETRA regression tests.
3. Run a real-DINO tiny smoke through all stages; verify finite artifacts, exact reason identity, no DINO gradients, no external task checkpoint, and no test-fit parameters.
4. Commit and synchronize code to the remote worktree/GitHub branch.
5. Run the full staged experiment on `F:` output storage, supervise through final Stage C, and report both model-internal and train-only deployment metrics.
6. The objective completes only when the final test artifact proves all three metric targets and clean lineage.
