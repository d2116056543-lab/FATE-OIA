# Dual-Snapshot OIA V1 Design

## Objective

Train from official frozen DINO weights and target `Act_mF1 >= 0.73` and `Exp_mF1 >= 0.40` without introducing a new action/reason coupling path.

## Evidence

- AIE epoch 4 and its low-LR consolidation descendant have complementary action ranking. A locked `0.65 late + 0.35 early` blend reached `Act_mF1=0.730915` with a positive paired-bootstrap interval.
- Their reason logits also have complementary ranking. Train-calib-only five-fold selection chose a late weight near `0.875`; with no cross-label threshold shrinkage, the locked diagnostic reached `Exp_mF1=0.404423` and `Exp_mAP=0.384528`.
- Adding another semantic residual is rejected because prior experiments repeatedly improved explanation loss while damaging action through shared optimization.

## Architecture

The model remains the audited AIE direct-image architecture: one frozen DINO forward, ACPR-CalAlign foundation, action evidence/contribution branch, private reason rereader, and action-to-reason/reason-to-action gradient firewalls. The improvement is a reproducible training and deployment protocol, not a new logit-correction module.

1. Base stage: train AIE from scratch for 20 epochs with its original warmup-cosine schedule.
2. Consolidation stage: initialize from the base deploy-joint checkpoint and train three low-LR epochs with action/reason scales fully enabled.
3. Snapshot deployment: blend base epoch 4 with the best consolidation snapshot using pre-locked task-specific weights: action late weight `0.65`, reason late weight `0.875`.
4. Fit action and reason thresholds independently from train-calib logits. Action retains support shrinkage `50`; reason uses `0`, selected before the new full run by train-calib cross-validation.
5. Test is evaluated only after all weights and thresholds are locked. Individual snapshots, blend deltas, per-label metrics, and bootstrap uncertainty are retained.

## Artifact Contract

Every epoch additionally saves aligned train-calib action/reason logits, labels, and file names. The final evaluator rejects mismatched names or labels and writes weights, thresholds, snapshot hashes, individual metrics, ensemble metrics, and paired-bootstrap results.

## Verification

Tests cover artifact alignment, task-specific fixed blends, independent threshold fitting, mismatch rejection, and preservation of raw ranking metrics. A real four-sample smoke verifies one-DINO forward, calibration artifact creation, and final evaluator execution before full training starts.
