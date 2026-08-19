# VETRA Staged From-Scratch V1 Design

## Objective

Produce one auditable BDD-OIA run that starts from the official frozen DINO ViT-S/8 weights and randomly initialized task parameters, then reaches the historical VETRA strong-refine range without loading any historical task checkpoint. Target test metrics are `Act_mF1=0.731-0.734`, `Exp_mF1=0.405-0.408`, and `Exp_oF1=0.55-0.57`.

## Evidence Behind The Design

The prior clean run already reached `Act_mAP=0.797036` and `Exp_mAP=0.383989`, while the historical strong-refine result had `Act_mAP=0.798167` and `Exp_mAP=0.384217`. The remaining gap is therefore primarily deployment calibration and action refinement, not missing visual ranking capacity.

The historical strong-refine source checkpoint already produced `Act_mF1=0.720359`, `Exp_mF1=0.407559`, and `Exp_oF1=0.574672`. Its refiner changed the final metrics by approximately `+0.012624/+0.000130/+0.000206`, so the valid lesson is to freeze a mature base before action-only refinement. It is not valid to load that source checkpoint in the new run.

The clean deployment experiment also proved that fitting reason thresholds on `train_calib+train_audit` was harmful: it reduced `Exp_mF1/Exp_oF1` from `0.405597/0.554805` to `0.402463/0.536148`. Action and reason deployment fitting must therefore use independent train-only split policies.

## Non-Negotiable Invariants

- The only pretrained state at run start is `ckp/reference/dino_deitsmall8_pretrain.pth`, loaded with `checkpoint_key=teacher` and frozen throughout.
- All action, reason, predicate, evidence, contribution, and refiner parameters start randomly or from exact zero-effect initialization inside this run.
- `external_task_checkpoint` and `historical_source_checkpoint` are null in the run manifest.
- Stage B may load only the Stage A checkpoint whose run ID, source tree, split manifest, and parent run root match the current run.
- Feature cache and token compression remain disabled.
- Test labels never fit thresholds, calibrators, gains, learning rates, schedules, or checkpoint selection.
- Model selection uses `train_audit`; threshold/calibrator fitting uses `train_calib` or nested OOF over explicitly declared train-heldout splits.
- Reason logits are identity-preserved during action refinement.
- A zero-gain action candidate is always available, making the refinement fail closed when train-audit evidence is negative.

## Architecture And Run Lifecycle

### Stage A: Base Representation

Run the existing clean `AIEOIAModel` training path for 10 epochs with the verified clean configuration, `batch_size=6`, `gradient_accumulation_steps=5`, `num_workers=8`, BF16, one DINO call per ordinary batch, EMA decay `0.998`, and the same fixed split seed `20260817`.

The gradient-training split contains 13,450 images. `train_calib` contains 1,608 images and `train_audit` contains 1,024 images; both remain excluded from gradient training. Each epoch may report test diagnostics, but the selected base is the EMA or online state with the best `train_audit deploy_joint`, never the best test epoch.

Stage A writes `checkpoint_stage_a_selected.pth` with `stage=base_selected`, `selected_view`, parent run ID, split hash, source tree hash, model state hash, and the complete selection row. Selection is expected around epochs 6-9, but no epoch is hard-coded.

### Stage B: Same-Run Frozen Action Refinement

Load only `checkpoint_stage_a_selected.pth` from the same run root. Freeze the full AIE base, including DINO, primary action/reason paths, predicate path, reason-private path, and evidence modules. Construct a zero-effect `SelectiveActionPathRefiner` from the selected base action-evidence and action-contribution modules.

Train the refiner for at most three epochs on the same 13,450-image gradient-training split. The objective contains action ASL, pairwise AP, smooth AP, and bounded delta regularization. Reason logits are copied directly from Stage A and receive no gradient. Candidate gains include zero and bounded positive values; per-action gain selection uses `train_calib` thresholds and `train_audit` F1/AP. The selected refiner checkpoint is based only on train-audit action score with an explicit no-regression guard on the frozen reason identity hash.

Stage B writes `checkpoint_stage_b_selected.pth`, `stage_b_gain_audit.json`, gradient ownership evidence, action delta RMS, per-action AP/F1 deltas, and the Stage A parent checkpoint SHA256.

### Stage C: Independent Train-Only Deployment

Collect original and horizontally flipped logits once from the selected Stage B state for `train_calib`, `train_audit`, and test. Fit action TTA/combo hyperparameters by nested OOF over `train_calib+train_audit`; fit stable action thresholds using the declared OOF predictions.

Fit reason thresholds from `train_calib` only. Do not refit reason thresholds on `train_audit`, because the previous experiment demonstrated that this lowers both test `Exp_mF1` and `Exp_oF1`. Stage C constructs one deployment artifact containing the Stage B model hash, action calibrator, action thresholds, and train-calib reason thresholds.

The final deployment equation is evaluated once on test. The manifest separately records raw, original-view, flip-view, calibrated action, and reason metrics. It must state `test_labels_used_for_parameters=false`.

## Supervisor And Recovery

One foreground supervisor owns the lifecycle `Stage A -> Stage B -> Stage C`. Each stage writes an atomic completion JSON before the next stage starts. Resume loads the latest checkpoint only when its run ID, stage, split hash, source tree hash, and parent hash match. An environmental restart cannot substitute a checkpoint from another run.

The supervisor does not stop for weak test metrics. It stops only for NaN/Inf, missing artifacts, lineage mismatch, DINO gradients, reason mutation during Stage B, test leakage, or an unrecoverable runtime error. OOM fallback preserves effective batch size and records the selected batch/accumulation pair.

## Verification

Before full training:

- Unit tests prove same-run lineage acceptance and historical checkpoint rejection.
- Unit tests prove Stage B freezes every base parameter, preserves reason logits exactly, and allows only refiner gradients.
- Unit tests prove zero gain reproduces Stage A action logits.
- Unit tests prove action and reason fit splits are independently configurable and reject `test`.
- Unit tests prove Stage C uses train-calib reason thresholds while action uses nested OOF heldout training data.
- Resume tests prove stage and parent hashes cannot be crossed.
- A real-DINO smoke runs Stage A, Stage B, and Stage C on a tiny real-data subset and emits finite, fully populated artifacts.

Full-run success requires all protocol checks plus final test `Act_mF1>=0.731`, `Exp_mF1>=0.405`, and `0.55<=Exp_oF1<=0.57`. Results outside the target remain valid diagnostics but do not complete the objective.

## Files And Boundaries

- `fate_oia/engine/train_aie_oia.py`: expose Stage A selected-checkpoint metadata without changing the verified base optimization logic.
- `fate_oia/engine/train_vetra_staged_refine.py`: same-run Stage B loader, freeze contract, action-only training, gain audit, and checkpoint output.
- `fate_oia/engine/export_vetra_from_scratch_deploy.py`: independent action and reason fit splits.
- `fate_oia/engine/supervise_vetra_staged_from_scratch.py`: deterministic stage lifecycle and resume validation.
- `fate_oia/utils/vetra_stage_contracts.py`: run identity, lineage, stage transition, and artifact validation.
- `configs/fate_oia_train_360x640_vetra_staged_from_scratch_v1.yaml`: all stage and runtime values.
- `scripts/FATE_OIA_vetra_staged_from_scratch_v1.ps1`: foreground entry point.
- `tests/test_vetra_staged_*.py`: behavior and protocol tests.

No unrelated model module, data loader, metric implementation, or historical branch is changed.
