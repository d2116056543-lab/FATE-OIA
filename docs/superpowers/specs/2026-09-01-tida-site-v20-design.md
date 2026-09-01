# TIDA-SITE V20 Design

## Objective

Replace the slow, weakly credited TIDA V19 temporal path with Sparse Interaction
Temporal Evidence (SITE). The official BDD-OIA target frame and 4-action/21-reason
labels remain the only prediction targets. The frozen VETRA image branch remains
an exact fallback.

Success is measured on the same 4,572 official test samples and with the same
Stage-C deployment procedure as the image baseline. Internal trainer metrics are
diagnostic only and must not be compared with the Stage-C result as if they were
the same protocol.

## Evidence From V19

- Fourteen history frames require about three hours per epoch.
- Context DINO plus temporal decoding dominate runtime.
- Best internal video result improved action by about 0.004 but explanation by
  only about 0.0006 relative to the internal image view.
- The relational reason residual is approximately 2e-7 and its deletion gap is
  approximately zero.
- The implemented reason-local query is disabled in the full configuration and
  its training losses have zero weight.

## Research Synthesis

The design combines patterns from AIM, M2-CLIP, TC-CLIP, Tem-Adapter, AdaTAD,
ViT-TAD, Dual-Path Adaptation, Momentor, INT2, JFP, DRAMA, and Drive-WM:

1. Keep the strong image model frozen and train lightweight temporal adapters.
2. Summarize informative temporal evidence instead of retaining every patch from
   every frame.
3. Separate local temporal differences from long-range context.
4. Let target queries read temporal events rather than add a generic clip bias.
5. Model traffic interaction and event order explicitly.
6. Use task-private action and reason paths with exact image fallback.

No architecture or source code is copied verbatim from these works.

## Architecture

### Sparse history

Use five history frames at fixed multi-scale offsets spanning the five-second
clip. This avoids a learned selector that could collapse or leak labels while
reducing history DINO work from fourteen frames to five. The target frame remains
unchanged at 360x640; history remains 192x344.

### Event bank

For each adjacent selected-frame pair, construct target-independent event tokens
from query-token content, signed difference, absolute difference, velocity, and
acceleration. A null event is always present. Event validity follows the frame
mask and timestamps.

### Task-private readers

- Four action queries read the event bank with sparse attention and a bounded
  action residual.
- Twenty-one reason queries read the same measured events through separate
  parameters and label masks. Reason gradients cannot update action-only owners.
- Target-only controls are subtracted so a residual cannot become a static
  per-label bias.

### Training signals

- Main action and PU reason losses remain unchanged.
- Ordered-event versus shuffled-event contrast proves order sensitivity.
- Selected-event versus random-event deletion contrast proves target credit.
- A terminal prediction objective trains history to predict the target query.
- Bounded no-harm losses compare each temporal branch with the frozen image
  branch, but do not suppress ranking improvements by forcing exact imitation.

### Deployment

Action and reason are calibrated independently on train_calib. Each label may
select a bounded temporal scale or exact zero fallback. Test data cannot update
gates, thresholds, learning rates, or checkpoints. Final Stage-C evaluation uses
the existing original-plus-flip deployment evaluator.

## Efficiency Contract

- Five history frames, not fourteen.
- No dense BxTxNxD history artifact is retained after event extraction.
- No second DINO owner, feature cache, token compression, or RunC residual.
- DataLoader keeps workers, prefetch, pinned memory, and persistent workers.
- Runtime telemetry separately records read, target DINO, history DINO, event,
  decode, and backward time.

## Verification

Tests must first fail for sparse-frame shape, event order sensitivity, exact
zero/fallback behavior, action/reason gradient isolation, reason target
specificity, deletion intervention, and no-test-fit deployment. After unit and
integration tests pass, a real-data smoke must show finite nonzero action and
reason event deltas, positive selected-minus-random gaps, exact history-off
fallback, and materially higher throughput than V19 before full training starts.
