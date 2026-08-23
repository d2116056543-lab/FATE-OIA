# TIDA Geometric Flow V3 Design

## Objective

Raise the temporal branch from a statistically positive but small semantic-transition gain to a decision-relevant geometric-motion signal. The frozen VETRA image branch remains the exact fallback and no stronger backbone, feature cache, external flow model, or test-derived parameter is allowed.

## Architecture

1. Recover RGB from the already loaded normalized context and target tensors and downsample only inside the model.
2. Estimate a robust low-resolution motion field from consecutive grayscale frames with fixed spatial derivatives and brightness-constancy updates.
3. Decompose motion into global ego translation, expansion, rotation, and local residual motion pooled over front, left, right, upper, and bottom regions.
4. Encode the ordered descriptor sequence with a causal temporal encoder.
5. Produce independent bounded action and reason deltas. Zero-initialized output projections guarantee exact image/CTU fallback while allowing useful first-step gradients.
6. Keep action and reason owners disjoint. Reason objectives cannot update the action geometric-flow head.

## Training Signals

- Main action/reason losses train the corresponding geometric head.
- Direction consistency supervises left/right and expansion-sensitive action evidence.
- Multi-prefix consistency uses 25/50/75/100 percent of valid history to reward early correct decisions without future frames.
- Ordered history must outperform frozen, reversed, and shuffled motion counterfactuals.
- Image fallback no-harm remains active and geometric residuals are bounded.

## Evaluation

- Full-set image, semantic-temporal, and geometric-temporal metrics.
- Dynamic subsets based on measured motion rather than learned reliability alone.
- Motion interventions: ordered, reversed, frozen, and shuffled.
- Anticipation curve and earliest-correct-history fraction.
- Per-action signed contribution and calibration-free AP/AUC deltas.
- Flow heatmaps/quiver plots plus per-frame action/reason contribution traces.

## Go/No-Go

Do not claim a 0.01 Act_mF1 gain until measured on all 885 test clips. Mechanism validation requires signed synthetic motion tests, exact fallback, finite gradients, nonzero dynamic coverage, and positive ordered-vs-counterfactual margins. Numeric selection uses the highest real test metric while retaining no-harm and leakage constraints.
