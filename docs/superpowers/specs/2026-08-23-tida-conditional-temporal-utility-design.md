# TIDA Conditional Temporal Utility Design

## Goal

Increase the causal contribution of traffic-flow evidence to BDD-OIA action and
reason predictions without weakening the frozen strong image path or selecting
parameters on test labels.

## Diagnosis

The completed TIDA flow-credit run proves that temporal directions are useful
but under-scaled. Action deltas have 72.7% target-signed direction accuracy and
reason deltas have 58.1%. Scaling the existing delta reveals additional headroom,
while the current transition reliability saturates near one and assigns nearly
the same 0.35 flow budget to every sample. The fixed route budget therefore does
not express when temporal evidence is needed, and the downstream trust and kappa
bounds suppress the final effect.

## Architecture

### Multi-Scale Transition Tokens

`TIDAFlowTransitionBank` will preserve separate velocity, acceleration,
region-velocity, and persistence tokens per predicate instead of projecting all
signals into one token. A learned type embedding distinguishes the four temporal
scales. The original combined token remains available as a compatibility view,
but readers consume the typed bank.

### Conditional Temporal Utility

Each action and reason receives a deterministic, differentiable need score from:

- image uncertainty, computed as `4 * sigmoid(z) * (1 - sigmoid(z))`;
- robust motion salience from non-saturating velocity and acceleration scales;
- transition consistency from persistence and valid-frame coverage;
- target-specific compatibility from the existing query-to-transition score.

The need score allocates a bounded per-target flow budget. Action budgets are
bounded by 0.60 and reason budgets by 0.50. Low-need samples retain a small floor
and exact history-off behavior remains the image fallback. No reason feature can
enter the action reader, and no action loss can update reason-only parameters.

### Credit And Safety Objectives

Counterfactual credit remains paired between real history and history-off,
repeated-last, and order interventions. Credit is weighted toward samples with
high independently measured temporal need. Low-motion and high-confidence image
samples receive a stronger no-harm objective. A budget-calibration loss aligns
utility with detached target-signed counterfactual benefit during training only.

## Evaluation

Primary BDD-OIA action/reason metrics remain unchanged. Additional temporal
metrics are required:

- full-test image-to-video mF1, oF1, and mAP deltas;
- paired-bootstrap confidence intervals for signed GT-margin improvement;
- motion-quartile and image-uncertainty-quartile deltas;
- Temporal Benefit Coverage and Temporal Harm Rate;
- history-off, repeated-last, shuffle, and reverse intervention margins;
- per-label delta sign accuracy, budget, route mass, and contribution RMS;
- static-slice no-harm and dynamic-slice gain.

Motion slices must be derived from velocity and acceleration only, never from
labels or the learned utility gate. Test labels may evaluate metrics but may not
select scaling, thresholds, schedules, or checkpoints beyond the existing
test-best protocol explicitly required by the experiment.

## Validation Protocol

1. RED tests prove typed transition tokens, conditional budgets, exact fallback,
   gradient firewalls, and metric schemas are absent before implementation.
2. Targeted unit and regression tests must pass after implementation.
3. A checkpoint compatibility test with utility disabled must reproduce current
   TIDA logits exactly.
4. A short train-calib/train-audit refinement from the epoch-3 checkpoint selects
   independent action and reason utility strengths without test access.
5. A single frozen test evaluation is allowed only after the choice is locked.
6. Full training is allowed only when total metrics are not lower than the current
   best and temporal contribution improves materially.

## Empirical Success Criteria

- Full-test flow gain targets: action mF1 at least +0.002 and reason mF1 at least
  +0.004 over the matched image branch.
- High-motion or high-uncertainty slices should exceed the full-test gain.
- Mean signed margin and bootstrap lower confidence bound must be positive.
- Temporal Harm Rate must not exceed Temporal Benefit Coverage.
- Total deploy metrics must not regress below Act_mF1 0.78349 and Exp_mF1 0.46322.

These are comparison criteria against the measured current run, not arbitrary
requirements for declaring code correctness.

## Research Basis And Boundary

The design adapts cross-snippet temporal propagation, branching temporal
adapters, motion-guided salience, and momentum-aware historical interaction to
the small BDD-OIA clip dataset. It does not copy a full video backbone, introduce
optical-flow dependencies, add an MoE/router, or replace the strong image path.
