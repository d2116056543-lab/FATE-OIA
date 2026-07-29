# METER-OIA V2 / TESA Design

**Status:** user-approved source design, frozen before implementation

**Source plan:** `C:\Users\WLJTXY\Downloads\METER_OIA_V1_Forensic_Audit_and_V2_TESA_Single_Final_ModificationPlan_20260729.md`

**Source plan SHA256:** `1E12A602A0E45623C15AD1B6986C2EA56105FDE48B46DE1E73762D5B6FD354F2`

**V1 base:** `00954c976244e5721ff1d25cae0fae820867f927`

## Design Decision

TESA replaces V1's universal support/counter maps, peer selector, local full reason expert, mix gate, annotation residual, formal meta sharing, and sparse counterfactual training with one typed evidence-state path:

```text
image
  -> frozen DINO layers 3/7/11
  -> CalAlign-compatible 25-query foundation
  -> typed factor {anchor, state, observability, reliability, token}
  -> exact factor-specific additive action transport
  -> global reason + detached groundability-aware evidence correction
  -> train-calib-only deployment calibration
```

The causal and gradient ownership contracts are:

```text
X -> action-owned typed factors -> action
(X, stopgrad(typed factors)) -> benchmark reasons
reason and PU gradients must not update action-owned factors or action parameters
calibration must not update representation parameters
```

## Typed Evidence Contract

Each of the 21 factors has a declared type, variable-cardinality state set, anchor source, state source, groundability, action ownership, observability source, and optional mirror partner.

`TypedEvidenceStateHead` returns:

```text
factor_anchor_map
factor_null_mass
factor_anchor_token
factor_state_logits
factor_state_prob
factor_observability
factor_reliability
factor_typed_token
```

Unknown is never treated as a negative state. Latent relational factors are not forced to have patch grounding.

## Action Contract

The final action is:

```text
action_final = action_visual
             + kappa * tanh(action_factor_contributions.sum(-1) / kappa)
```

Every contribution depends jointly on factor identity and action identity through factor-specific rank-16 projections and action queries. Entmax provides sparse factor weights and includes a null factor. No selector, expert mixture, reason logits, action-set marginal, graph delta, or meta omega may alter final action.

## Reason Contract

The CalAlign-compatible global private reason head remains the primary predictor. A bounded factor-specific correction is permitted only for groundable factors and reads a detached typed token. Latent/unverifiable labels have exactly zero correction. There is no local full classifier, global/local mix, annotation residual, or action decision context.

## Intervention Contract

Formal training uses exact analytic contribution deletion, target specificity, identity corruption, and cross-sample same-factor swap every batch. Patch deletion is audit-only, stratified across all four actions and at least 12 groundable factors, and must audit 128 unique sample IDs rather than one row per batch.

Identity diagnostics must break semantic association:

1. schema-token mismatch;
2. cross-sample same-factor swap;
3. state corruption with the anchor fixed.

Reordering an unordered factor set is not a valid corruption.

## Training Contract

- Direct image, one frozen-DINO call per ordinary batch.
- `360x640`, full 3600 patch tokens, no cache, no compression.
- Batch 6, accumulation 5, bf16, workers 4, prefetch 2.
- Five-percent LR and factor-supervision warmup.
- Ten-percent action/reason correction, softmax-to-entmax, and dense-intervention ramp.
- Label-wise PU may start only after epoch-0 train-audit evidence.
- Test-only epoch evaluation and best selection are retained as an internal protocol and must be labeled `publication_eligible=false`.

## Verification Strategy

Implementation is not accepted by import or forward smoke alone. It must pass:

- all 23 named TESA tests from plan section 23;
- exact progress-zero compatibility;
- dynamic gradient-firewall probes;
- formula reconstruction of action logits;
- factor-specific corruption effects;
- all-sample unique-ID patch audit;
- sequential evaluation memory checks;
- real-DINO smoke and runtime profile;
- four-epoch single-seed pilot gates A-H.

Full training is forbidden until every pilot gate passes.
