# TIDA-LiteFlow V21 Design

## Decision

Replace history-frame DINO inference with a traffic-first temporal path while
keeping the official target-frame VETRA branch bitwise unchanged. The selected
design uses one target-frame DINO pass, five low-resolution RGB frames, fixed
local-correlation flow, and precomputed object tracks. Action and reason use
separate readers and separate train-calib deployment decisions.

## Evidence And Alternatives

TIDA-SITE V20 proves that five history frames are faster than fourteen and that
selected temporal evidence is more causal than random evidence. It does not
improve action or reason ranking: on the 2,048/512 pilot the action mF1 is
unchanged, action mAP changes by only +0.000018, and reason mAP changes by
-0.000025. A five-frame history DINO path still projects to roughly 2.2 hours
per full epoch.

Three alternatives were considered:

1. Keep history DINO and tune the residual. Rejected because the measured
   bottleneck and ranking failure remain.
2. Reduce history to two or three DINO frames. Rejected because it saves only a
   constant factor and weakens acceleration and interaction evidence.
3. Add an external optical-flow or detector network. Rejected because it adds a
   second large visual model and violates the intended efficient direct-video
   comparison.

The chosen route reuses the existing `TIDAGeometricFlowEncoder` and
`TIDAObjectIntentTransport`. This follows the shared pattern in recent video
adaptation and driving-interaction work: preserve the strong image model, use a
small temporal measurement path, retain local order, and expose target-private
interaction evidence rather than a generic clip residual.

## Forward Path

1. Run frozen VETRA/DINO once on the official target frame.
2. Read terminal action, reason, and predicate queries exactly as before.
3. Do not run DINO on history frames. Repeat terminal query tensors only as a
   schema-compatible zero temporal baseline; all heavy semantic, trajectory,
   and relational history-DINO modules are disabled.
4. Measure five-frame RGB motion with fixed multiscale local correlation at
   45x80 resolution. The measurement returns signed flow, confidence, regional
   motion, expansion, rotation, speed, acceleration, and prefix summaries.
5. Read this flow independently for four actions and twenty-one reasons with
   bounded target-conditioned heads.
6. Read precomputed object tracks with `TIDAObjectIntentTransport`, using target
   DINO patches only for terminal object semantics. No history patch tensor is
   supplied. This adds ego-compensated velocity, future approach risk,
   pairwise interaction, role evidence, and selected-vs-control deletion.
7. Add action and reason deltas only through their private owners. The image
   logits remain available as exact fallback.

## Training And Deployment

- Train flow and object readers; keep target VETRA/DINO frozen.
- Use action ASL/ranking and PU reason losses. Reason zero labels remain
  unknown unless contradiction-certified.
- Train selected-vs-control deletion and order credit for each branch.
- Cap each temporal branch relative to its own image-logit RMS and main-task
  loss. Action and reason gradients cannot cross private owners.
- Fit deployment scales only on train-calib. A label uses exact zero temporal
  delta unless it improves its primary ranking/F1 without violating the paired
  no-harm bound. Test never updates policy.

## Required Diagnostics

For action and reason, record image/candidate/deploy AP, AUC, mF1 and oF1;
delta RMS ratio; selected and control deletion effects; order-shuffle drop;
object interaction support; flow confidence; per-label utility; and exact
history-off equality. Record target-DINO, RGB-flow, object-route, decode,
backward, and data-loading time separately.

## Acceptance

The route is eligible for full training only if a real equal-budget pilot shows:

- exact image fallback when temporal deployment is zero;
- no NaN/Inf and strict action/reason gradient isolation;
- materially lower step time than SITE V20;
- selected deletion exceeds control deletion;
- at least one of action mAP/mF1 and one of reason mAP/mF1 improves over the
  same checkpoint image branch, while the other metric stays within 0.001;
- train-calib policy, not test oracle thresholds, selects deployment.

If ranking does not improve, the method is reported as insufficient rather than
starting a costly full run.
