# METER-OIA V2 / TESA Implementation and Coverage Plan

## Feature Coverage Matrix

| ID | Required feature or constraint | Implementation target | Verification |
|---|---|---|---|
| F01 | Isolated V2 branch/worktree from clean V1 pilot | `acpr_meter_oia_v2_tesa_direct_image` | HEAD equals `00954c9` before edits; V1 remains dirty but untouched |
| F02 | Preserve frozen DINO 3/7/11, 3600 tokens, 25-query foundation | foundation/model/config | progress-zero and DINO-call tests |
| F03 | Complete 21-factor typed schema | factor and grounding YAML | schema type/cardinality tests |
| F04 | Anchor distribution with explicit null | `meter_signed_factors.py` | normalization, entropy, null noncollapse |
| F05 | Variable-cardinality factor-specific state posterior | typed factor head | state cardinality and unknown tests |
| F06 | Observability and entropy-aware reliability | typed factor head/loss | noncollapse and selective-loss tests |
| F07 | Typed evidence token | typed factor head | output contract and finite-gradient tests |
| F08 | Anchor NLL/Dice and mirror equivariance | grounding losses | targeted unit tests |
| F09 | Conditional state CE, unknown masked | grounding losses/dataset | unknown-not-negative test |
| F10 | Same-type and same-region discrimination | grounding losses | correct-vs-wrong margin tests |
| F11 | Factor-specific rank-16 additive action transport | semantic action module | projection identity and exact additivity tests |
| F12 | Sparse action read with null factor | semantic action module | normalization/sparsity tests |
| F13 | Remove selector/expert mixture/regret | model/loss/train | source scan and dynamic output tests |
| F14 | Global reason primary + bounded correction | reason decoder | global equivalence and correction tests |
| F15 | Groundability mask and latent exact-zero correction | reason decoder/schema | latent-reason test |
| F16 | Reason firewall from action-owned factors | model/train | autograd reachability tests |
| F17 | Remove local full/mix/annotation/action context | reason/model/loss | forbidden-key/source checks |
| F18 | Formal meta disabled, audit-only retained | config/optimizer/audit command | optimizer ownership test |
| F19 | Dense analytic necessity/specificity | counterfactual losses/train | dense coverage and formula tests |
| F20 | Schema-token, state, and cross-sample corruption | eval/loss | three corruption tests |
| F21 | Patch deletion audit-only, all samples, unique IDs | eval/audit | 128 unique-ID tests |
| F22 | PU uses global probability, typed positive state, reliability | PU loss/train | inactive exact-zero gradient and private-only tests |
| F23 | Train-calib-only posthoc calibration | train/eval/artifacts | no-test-update and hash invariance tests |
| F24 | Sequential evaluation and GPU release | evaluator | memory monotonicity test |
| F25 | Complete output/artifact schemas | model/artifact utility | artifact contract test |
| F26 | Continuous 5%/10% ramps only | scheduler/train | boundary-value tests |
| F27 | Resume exactness | checkpoint/train | interrupted-vs-continuous test |
| F28 | Per-epoch action/reason/evidence/intervention/runtime diagnostics | evaluator/artifacts | required-field validation |
| F29 | Real-DINO runtime constraints | profile/smoke | one DINO call, `<45GB`, finite |
| F30 | Pilot gates A-H before full training | supervisor/audit | strict review-pass generation |

## Implementation Sequence

1. Freeze V1 evidence and selectively port only committed pre-eval checkpoint and zero-worker loader safeguards after diff review.
2. Add RED tests for schema, factor head, action formula, reason firewall, identity corruption, dense intervention, unique audit coverage, sequential evaluation, PU, resume, and artifacts.
3. Replace universal signed factors with `TypedEvidenceStateHead`.
4. Replace peer selector with `FactorSpecificActionTransport`.
5. Replace local/mix reason decoder with global primary plus `EvidenceReasonCorrection`.
6. Replace grounding and counterfactual objectives.
7. Remove formal meta and sparse patch-event training.
8. Rewire model, optimizer ownership, warmup, PU, calibration, evaluator, artifacts, and supervisor.
9. Run compile and targeted tests, then the complete regression suite.
10. Run clean-HEAD audit, real-DINO smoke/profile, then the exact four-epoch pilot.
11. Permit full training only if pilot gates A-H all pass.

## Fidelity Rules

- The source plan is authoritative.
- Approximate formulas, placeholder artifacts, aggregate-only factor reports, and source-string-only audits are not acceptable.
- Any technical deviation must be recorded in the supervision log before implementation and must preserve the same scientific contract.
