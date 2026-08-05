# LENS V1.1 Semantic Repair Design

## Objective

Preserve the LENS action reread and latent-reason routes that improved the
pilot's branch metrics, while correcting the verified latent-state/emission
axis mismatch that invalidates annotation semantics. The full-run decision is
based on real test metrics and component deltas; gates remain safety and
interpretability diagnostics rather than automatic quality ranking.

## Verified Defect

`LENSLatentState` exposes `state_prob` in `[positive, counter, unknown]`
order. `LENSAnnotationEmission` exposes annotation likelihoods in
`[counter, unknown, positive]` order. The current forward and loss paths
contract them directly. Consequently, the emission model interprets a visual
positive state as counter evidence, and the responsibility/state losses are
trained on mislabelled axes.

## Design

Introduce a single explicit conversion between state order and emission order.
The annotation-emission forward path and all expectation calculations use the
emission axis; model-facing state tensors and the state loss use the original
state axis. The returned responsibility payload carries both named forms so a
future caller cannot silently reuse a tensor under the wrong convention.

The repair must not modify the frozen DINO path, action reread formula,
action-owner firewall, direct-image protocol, or calibration boundary. This is
one semantic correction, not a new reasoning module.

## Metric-First Selection

For LENS runs, retain checkpoint metrics for source/base/final action and
source/latent/formal reason branches. Rank checkpoints by the configured test
joint score with action as the primary term; report the strongest historical
clean baseline separately. A failed synthetic score or directional
faithfulness threshold is investigated and reported, but does not discard a
configuration whose held-out action and reason metrics are stronger and whose
bounded mechanism is non-harmful.

## Validation

Three tests must fail before the fix: identity emission maps one-hot positive,
counter, and unknown states to 1.0, 0.0, and 0.5; responsibility is converted
back to state order before the state loss; and conflict-safe latent logits use
the same conversion. The targeted tests, complete LENS suite, compilation,
audit, and a small official-DINO smoke must pass before full training.
