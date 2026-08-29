# TIDA Physical Event Conditioning Design

## Goal

Increase the task-specific use of traffic flow without adding an unconstrained
action or reason residual. The existing V10.28/V11 visual, semantic, trajectory,
and relational paths remain the deployment backbone.

## Data Flow

1. Ego-compensated semantic trajectories produce twelve bounded physical events
   for forward, stop, left, and right: approach, crossing, corridor entry and
   occupancy, future collision, slowdown, flow incoherence, visibility, motion,
   distance risk, pair convergence, and temporal asymmetry.
2. Separate zero-output action and reason projections transform those events.
   Action events condition the corresponding action query. A private sparse
   reason router mixes the four action-event rows before conditioning each reason.
3. Event context modifies only target-private relational queries. It cannot add
   standalone logits. Existing support, relevance masks, temporal scale, and
   action/reason caps remain the only route to final predictions.
4. Action and reason event parameters belong to separate optimizer owners and
   preserve the existing gradient firewall.

## Compatibility And Safety

- Final event projection layers are zero initialized, so the initial model is
  exactly equivalent to the prior relational model.
- The projection scale is non-zero, allowing the zero-initialized final layer to
  receive useful gradients once the relational output layer starts learning.
- Event context is bounded component-wise by 0.20 and downstream logits remain
  bounded by the original relational caps.
- Full-test prediction metrics use every official test clip. Repeated mechanism
  interventions use a fixed 32-clip cohort to bound epoch-end cost.

## Evidence And Metrics

- Existing label-effectiveness evidence remains authoritative: proper-score
  information gain, signed margin, and selected-vs-matched-control logit deletion.
- Event necessity is measured independently with batched leave-one-track-out
  attribution. The selected event track is the deletion with maximum target
  context change; its control is visibility-support matched.
- Reports include event occupancy, target routing entropy, event-context RMS,
  event selected-vs-control context gap, risk-stratified utility, temporal-order
  interventions, and per-case trajectory/event visualizations.
- Event necessity alone is not called task utility. A faithful claim requires
  both positive event necessity and positive downstream label-effectiveness.

## Verification

- Unit tests cover zero-effect compatibility, trainability, temporal sensitivity,
  action/reason gradient isolation, leave-one-track-out attribution, metrics, and
  full model owner coverage.
- A real-video smoke must produce finite gradients, non-zero event projections,
  complete tensor artifacts, and renderable case reports before full training.
