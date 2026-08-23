# TIDA Trajectory Credit V5.6 Design

## Goal

Turn identity-consistent traffic trajectories into measurable, target-specific action corrections without letting temporal evidence hijack the strong frozen image decision.

## Evidence

V5.5 produces strong ordered-vs-reversed context contrast (`order_contrast_rms=1.332`) but an almost-zero action residual (`3.8e-5` RMS), no decision flips, and a slightly negative signed GT margin. A scalar sweep shows that simply enlarging or reversing the residual cannot reach the requested gain. The failure is therefore in credit assignment, not trajectory measurement.

## Architecture

The head computes ordered and reversed trajectory contexts with shared encoders. Their difference produces both an order-discriminability gate and the only evidence input to the correction readout. The readout receives signed contrast, contrast magnitude, and their products with learned action identity. It cannot read ordered appearance, visual action nodes, or base logits, which prevents a static action/calibration shortcut. Detached base uncertainty can only shrink the residual budget. The correction is zero-initialized and bounded by the configured cap, learned trust, support, order discriminability, and uncertainty.

The reversed path produces an independent control correction through the same readout. Even-magnitude features make it non-antisymmetric, while the absence of any ordered-control contrast forces both corrections to zero. Training requires the ordered correction to improve the GT margin more than the reversed control, while boundary correction and ranking losses determine the correction sign. Time reversal is therefore a preference control rather than an artificial opposite action label.

## Safety

- Zero output projection preserves exact strong-baseline logits at initialization.
- The image model and all non-trajectory owners remain frozen in head probes.
- The action residual remains bounded and does not affect reason logits.
- Trust starts at 0.5 so useful evidence can cross real deployment boundaries; zero output still guarantees compatibility.
- Evaluation reports order discriminability, signed transport, decision flips, dynamic quartiles, and shuffle/reverse controls.

## Validation

Tests cover zero-effect compatibility, non-antisymmetric ordered/control outputs, reachable residual budget, reversed-control loss, tensor persistence, and metric schemas. A three-epoch head-only probe is allowed only after the one-epoch mechanism probe shows positive signed margin or a clear upward trend without harming the baseline.
