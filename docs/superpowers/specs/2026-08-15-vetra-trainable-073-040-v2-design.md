# VETRA Trainable 0.73/0.40 V2

## Evidence

The clean stage-1 run peaks at deploy Act_mF1 0.72432 (epoch 4) and deploy
Exp_mF1 0.38708 (epoch 9). Test-oracle diagnostics are approximately 0.73175
and 0.39661. The current calibration split is also included in training, so its
per-label F1 of 0.93-0.99 is in-sample and does not transfer. Explanation mAP
peaks near 0.344, so threshold fitting alone cannot reliably reach 0.40.

## Design

1. Build a deterministic, disjoint train-fit/train-calib/train-audit split.
   Calibration remains train-only and test thresholds remain diagnostics.
2. Keep the existing VETRA/AIE image, DINO, action evidence and reason reread
   paths unchanged.
3. Add PU-aware class-wise DR ranking to the detached reason-private branch.
   This improves per-reason ordering without changing action parameters.
4. Maintain an EMA of trainable parameters and evaluate both online and EMA
   models. Selection remains explicit in artifacts.
5. Support a reason-refinement phase that freezes foundation, action evidence,
   action contribution and naming parameters. This phase must preserve action
   logits exactly while optimizing reason ranking.
6. Use a predefined two-stage from-scratch pipeline: leakage-free joint
   training followed by low-LR reason-private refinement from that run's own
   checkpoint. No external task checkpoint or distillation is allowed.

## Acceptance

Unit tests must prove split disjointness, DR gradient direction and PU weighting,
EMA update/state restoration, and action invariance in reason-only refinement.
The implementation is considered promising only if a real pilot improves the
strong checkpoint without test-oracle writeback. Full training metrics, not gate
status alone, decide success.
