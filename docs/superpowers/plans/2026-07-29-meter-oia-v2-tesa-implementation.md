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

## Required Test Traceability

| Test | Feature IDs | Hard assertion |
|---|---|---|
| `test_tesa_factor_schema_types.py` | F03/F15 | exact IDs/names/types/states/sources/groundability/owners/observability/mirrors/counter flag |
| `test_tesa_anchor_normalization.py` | F04 | patches+null sum one; finite sparse grad |
| `test_tesa_state_cardinality.py` | F05 | factor-specific masked cardinality |
| `test_tesa_unknown_not_negative.py` | F05/F08/F09 | unknown exact-zero CE gradient; source-weighted anchor NLL/Dice |
| `test_tesa_observability_noncollapse.py` | F06/F08/F10 | exact rho; train-only factor tau; same-region and mirror formulas |
| `test_tesa_factor_specific_projection.py` | F11 | factor-index rank-16 projections |
| `test_tesa_additive_action_identity.py` | F11/F12 | exact bounded additive reconstruction |
| `test_tesa_additive_action_identity.py` | F11/F12 | non-owned weight/contribution/gradient exact zero; no hand-written compatibility mask |
| `test_tesa_no_selector_in_formal_path.py` | F13 | no selector/regret parameters or outputs |
| `test_tesa_reason_global_equivalence.py` | F14 | correction-off equals global |
| `test_tesa_reason_correction_groundability.py` | F14/F15 | rho/mask bounded correction |
| `test_tesa_latent_reason_no_fake_grounding.py` | F15 | factors 14/20 exact-zero correction |
| `test_tesa_reason_firewall.py` | F16 | reason/PU cannot reach factor/action |
| `test_tesa_identity_schema_corruption.py` | F20 | schema-token association corruption |
| `test_tesa_cross_sample_factor_swap.py` | F20 | same-ID swap retains target query |
| `test_tesa_state_corruption.py` | F20 | state corrupts with anchor fixed |
| `test_tesa_dense_intervention_coverage.py` | F19 | three exact objectives and coverage |
| `test_tesa_patch_audit_all_samples.py` | F21 | all eligible targets per sample |
| `test_tesa_patch_audit_unique_ids.py` | F21 | exactly 128 unique IDs |
| `test_tesa_sequential_eval_memory.py` | F24 | fixed order and released tensors |
| `test_tesa_pu_private_only.py` | F22/F16 | admission and exact gradient ownership |
| `test_tesa_progress_zero_equivalence.py` | F02/F26 | three errors below 1e-6 |
| `test_tesa_resume_equivalence.py` | F26/F27 | model/optimizer/ramp equivalence |
| `test_tesa_artifact_contract.py` | F23/F25/F28/F30 | train-calib-only update; threshold/shrinkage/fallback artifacts; complete fields and gates |

## Required Artifact Fields

Per epoch, save action visual/final logits, per-action F1/AP/AUC, correction
RMS, per-factor contribution and sparse weight, identity/factor-off/state-off
deltas; reason global/final logits, per-label F1/AP/AUC, correction RMS,
groundable/latent metrics, PU state, correction-off and identity deltas; typed
anchor entropy/null/observability/state entropy/confusion/source coverage/
wrong-factor margin/mirror; intervention correct/wrong/swap/patch effects,
latest and cumulative unique IDs and action/factor coverage; and data/DINO/
foundation/factor/action/reason/backward/eval-mode timings, memory, and DINO
call count. Aggregate-only factor output is invalid.

Calibration artifacts required by F23/F25/F28 are raw metrics, deploy metrics,
temperature, threshold vector, train-calib raw/deploy joint,
group/per-label-shrinkage state, and fallback reason.

F02 additionally freezes label-specific patch retrieval, 25-token
self-attention, predicate/category interaction, and progress-zero
action/reason-global/label-node equivalence. Merely preserving 25 query
parameters is insufficient.

All exact formula, weight, ownership, ramp, augmentation, pilot, gate, and
full-train values in the frozen design are normative parts of F01-F30.
