# TIDA Logit-Flow V26 Design

## Evidence And Root Cause

The V25c pilot learned sparse, action-specific patch trajectories, but its
trajectory candidate did not generalize from train-calib to test. On the fixed
512-case test subset, the effective residual improved signed target margin on
average while reducing action mAP. The state branch added a class-wide negative
offset, and the utility gate opened more strongly for negatives than positives.
Train-calib OOF selection therefore chose nonzero scales that produced no
FN-to-TP corrections and four TP-to-FN errors on test.

The failure is not caused by another threshold choice. The patch trajectory is
an insufficiently stable carrier of the strong image model's target evidence.

## Architecture

Keep the frozen VETRA image branch as the exact fallback. Reuse each history
frame's already-computed DINO field and run the same frozen image decoder on it;
do not read images twice and do not run DINO twice. This yields per-frame action
and reason logit trajectories in the same representation and calibration space
as the terminal prediction.

An independent per-label Logit-Flow reader consumes robust temporal statistics:
terminal-relative change, recent and global slope, acceleration, robust mean,
variance, monotonicity, terminal uncertainty, and history availability. Action
and reason use separate parameters and separate bounded residuals. Patch-based
trajectory support, cycle confidence, exclusive motion, and interaction risk
may condition utility, but cannot directly alter logits.

The output projections are zero initialized. With the module disabled or at
initialization, action and reason outputs are exactly the frozen image outputs.

## Training And Deployment

Candidate residuals receive label-balanced direction and pair-ranking losses.
Utility learns whether each detached candidate improves the corresponding
terminal target margin. The action loss cannot update reason parameters and the
reason loss cannot update action parameters.

Deployment is fit only on train-calib using grouped out-of-fold folds. Each
label selects a scale and utility cutoff, with zero scale as a mandatory
fallback. A nonzero route must improve out-of-fold F1 and proper scores without
excess fold degradation. Test labels never update the route or thresholds.

## Verification

Tests must prove one DINO pass per frame, frozen per-frame decoding, zero-effect
initialization, time-order sensitivity, action/reason gradient isolation,
bounded residuals, and exact fallback. A CUDA smoke must report candidate and
deployed action/reason metrics, temporal shuffle/repeat ablations, per-label
route selection, proper-score changes, and patch-trajectory support.
