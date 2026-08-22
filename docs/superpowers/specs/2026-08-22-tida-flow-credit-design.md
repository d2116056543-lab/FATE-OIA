# TIDA Flow Credit V1 Design

## Objective

Preserve the frozen VETRA image predictor and make ordered traffic flow provide a measurable, target-specific, positive contribution to both action and explanation. A non-zero temporal delta is not sufficient. Traffic flow is useful only when real ordered history improves ground-truth margins over history-off, repeated-last, shuffled, and reversed histories without degrading the image fallback.

## Evidence And Root Cause

The completed TIDA V1.1 run reached strong final metrics, but its direct temporal utility was small. At the best online epoch, raw action mF1 was unchanged, raw explanation mF1 decreased slightly, and only deploy explanation mF1 improved. Intervention audits detected history presence but barely detected order or direction.

The current terminal predictor receives the current target-frame static context in both its history and no-history branches. The no-history loss therefore rewards a current-frame shortcut. Shuffle/reverse/repeat losses compare terminal reconstruction error only; they do not require ordered history to improve action or reason ground-truth margins. The frozen image anchor, base-protection loss, small bounded delta, and trust cap then make a near-zero action correction the safest optimum.

## Considered Approaches

### A. Explicit Transition Bank And Target Credit (chosen)

Expose signed first- and second-order temporal transitions already available from predicate trajectories, build compact transition tokens, and let action/reason-specific queries read them. Train the resulting bounded residual with same-image counterfactual ground-truth margin credit. This preserves the existing backbone, data protocol, speed, and exact image fallback.

### B. Optical Flow Or Codec Motion

Use optical flow or motion vectors to localize movement. This may improve motion localization, but adds preprocessing/dependencies and does not itself assign motion to BDD-OIA action/reason targets. It is retained only as a future diagnostic, not a formal forward dependency.

### C. Larger Dual Video Encoder

Add a second temporal backbone following large image-to-video transfer architectures. This is more expensive and data-hungry, and risks replacing the verified image predictor rather than safely improving it. It is rejected for V1.

## Architecture

```text
14 ordered context frames + current target frame
  -> frozen DINO/query trajectories
  -> causal temporal encoder
  -> signed transition bank
       velocity / acceleration / region-mass change / persistence
  -> target-conditioned action flow reader
  -> target-conditioned reason flow reader (action-detached)
  -> bounded zero-effect residuals
  -> frozen image logits + temporal residuals
```

### 1. Shortcut-Free Terminal Innovation

The history predictor receives query identity and causal history summary, not current target-frame static features. The terminal target remains detached supervision. The no-history baseline uses learned query priors and zero history, and is not optimized by a standalone target reconstruction term.

For query `q`:

```text
p_hist(q) = Predictor(query_identity(q), history_summary(q))
p_null(q) = Predictor(query_identity(q), 0)
rho(q) = stopgrad(clamp((err_null - err_hist) / (err_null + eps), 0, 1))
xi(q) = rho(q) * LN(p_hist(q) - p_null(q))
```

`terminal_hist`, gain, and counterfactual ordering losses may optimize `p_hist`. `terminal_no_history` is diagnostic-only so it cannot teach a target-frame shortcut.

### 2. Signed Transition Bank

`TIDAFlowTransitionBank` consumes per-frame predicate/query tokens, region mass, timestamps, and valid masks. It returns one token per predicate plus explicit diagnostics:

```text
velocity = weighted slope over valid adjacent transitions
acceleration = weighted slope of velocity transitions
region_velocity = signed target-region mass change
persistence = fraction of valid transitions with consistent velocity sign
transition_token = zero-safe projection([velocity, acceleration, region_velocity, persistence])
```

The implementation must preserve sign. Reversing time must invert velocity-sensitive components; repeating the final frame must collapse velocity/acceleration magnitude. No dense `[B,P,T,N,D]` tensor is allowed.

### 3. Target-Conditioned Flow Credit

Action and reason readers receive transition tokens as a separate factor family. Each target has a private query and a null route. Reliability and route mass bound the correction. Initialization must yield an exact zero residual and exact equality with the frozen image model.

```text
delta_t = scale * trust_t * kappa * tanh(raw_delta_t / kappa)
video_logits_t = image_logits_t + delta_t
```

Reason flow cannot backpropagate into action-only parameters or alter action logits. Action-set marginalization, graph delta, cached logits, and test-derived thresholds remain forbidden.

### 4. Same-Image Counterfactual Credit

For binary target `y` and logit `z`, define signed GT margin `m=(2y-1)z`. For each intervention `c`:

```text
L_credit = mean relu(m_cf - m_real + margin)
L_no_harm = mean relu(m_image - m_real + epsilon)
```

Counterfactuals are `history_off`, `repeated_last`, `time_shuffle`, and `time_reverse`. Credit is computed separately for actions and reasons. Reason negatives use the existing PU weights, while positive labels always retain weight 1. Counterfactual forward passes share the same target image; therefore improvements cannot come from different images or thresholds.

### 5. Training Phases

```text
FLOW_FOUNDATION: terminal/transition/order objectives; output residual scale = 0
FLOW_CREDIT: enable target-specific counterfactual credit and bounded residuals
SAFE_JOINT: retain credit, ranking, PU reason, no-harm, and route diagnostics
```

Phase changes are based on optimizer update count and train-core diagnostics only. Test metrics cannot alter phase, learning rate, gates, thresholds, or loss weights.

## Data Boundary

- 3,115 train clips and 885 test clips across the three audited video batches.
- Each clip contributes 14 context frames plus one current target frame.
- Train-core is the only optimizer source. Train-calib is threshold-only. Train-audit is mechanism reporting only. Test is evaluation only.
- Labels remain four multi-label actions and 21 multi-label reasons.
- No cache, token compression, RunC residual, stronger backbone, validation-best, or label leakage.

## Required Diagnostics

Every evaluation writes image/video/counterfactual metrics and:

- per-target GT-margin real-minus-counterfactual;
- action/reason history utility mF1, oF1, and mAP;
- transition velocity/acceleration RMS and sign-flip checks;
- non-null route mass, trust, delta RMS ratio, and sign agreement with GT margin;
- action/reason gradient isolation;
- exact fallback equality at scale zero.

Traffic flow is considered materially useful only if the ordered-history branch improves both target margins and final metrics. Gate pass alone is not the objective.

## Go/No-Go Criteria

Before full training:

1. Scale-zero logits are bitwise equal to frozen image logits.
2. Time reversal flips signed velocity diagnostics; repeated-last reduces transition magnitude.
3. Action and reason counterfactual-credit losses are finite, non-zero on constructed violations, and zero when real margins satisfy the margin.
4. Reason loss has zero gradient to action-only parameters.
5. On real video pilot data, ordered history produces positive mean GT-margin advantages over history-off/repeat and non-negative advantages over shuffle/reverse.
6. Image fallback metrics remain unchanged. Preferred pilot utility is action mF1 `>= +0.003`, explanation mF1 `>= +0.005`, and non-negative mAP changes. If those numeric preferences are missed but the strongest score remains stable and all history utilities are positive, continue only with explicit evidence, not by relaxing correctness checks.

## Scientific Boundary

The model may be called traffic-flow-aware only when ordered-history interventions reduce target margins and final metrics. If only history-off matters while shuffle/reverse do not, the correct claim is context-aware, not flow-aware.

