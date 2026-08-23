# TIDA trajectory credit V5.9 design

## Evidence

V5.8 dense reciprocal matching raised mean trajectory support to 0.148 and produced a
positive test signed-margin CI, but the learned residual was negative for every sample
and action.  The ordered and reversed-control readouts shared an output bias, while the
usable signal was multiplied linearly by support.  This turned trajectory credit into a
small instance threshold shift rather than an order-specific traffic contribution.

## Change

Keep the frozen image branch, dense DINO matching, camera-motion compensation, temporal
encoders, and bounded action-only residual unchanged.  Define the credit logit as half
the ordered-minus-reversed readout difference.  Define the control credit as its exact
negative, so any order-independent bias cancels and reversing time cannot introduce a
second learned class prior.

Map reciprocal support `s` to `s / (s + 0.05)` before the residual budget.  This mapping
is exactly zero without matched evidence, remains bounded by one, and does not linearly
suppress moderate support.  Trust, order contrast, frozen-base uncertainty, and the
existing final cap remain active.

## Validation

Tests must prove zero initialization, common-bias cancellation, reversal antisymmetry,
strict zero fallback, saturated support, bounded output, and reason firewall.  A remote
head-only probe must then report full-test action metrics plus signed margin, control
advantage, temporal shuffle/reverse drops, grounding support, and decision flips.
