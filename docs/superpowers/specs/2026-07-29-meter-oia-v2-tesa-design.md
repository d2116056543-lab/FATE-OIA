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

## Exact Mathematical Contracts

For factor `r`, the formal state is `F_ir=(A_ir,p_state_ir,o_ir,rho_ir,E_ir)`.
The anchor distribution includes a learnable null:

```text
[A_ir1 ... A_irN, A_ir_null]
  = annealed_softmax_to_entmax(
      [q_anchor_ir^T K(F_i)/sqrt(D), b_null_r])
e_anchor_ir = sum_n A_irn V(F_in)
p_state_ir = softmax(W_state_r[H_global_ir;e_anchor_ir;g_i])
o_ir = sigmoid(w_obs_r^T[H_global_ir;e_anchor_ir;A_ir_null;H(p_state_ir)])
rho_ir = o_ir*(1-A_ir_null)*(1-H(p_state_ir)/log(|S_r|))
E_ir = LN(H_global_ir + W_anchor*e_anchor_ir
          + W_state*sum_s p_state_ir(s)e_state_rs)
```

State softmax is factor-specific and masks padding. Unknown is masked, never a
negative. Factors 14/20 are latent and non-action-owned; factor 1 is partially
action-owned as declared by schema.

The 21 entries are fixed:

| id | type | states | anchor | action ownership |
|---:|---|---|---|---|
| 0 | control_attribute | green/red_yellow/unknown | traffic light | full |
| 1 | relational_global | following/not_following/unknown | front traffic | partial |
| 2 | corridor_state | clear/occupied/unknown | center corridor | full |
| 3 | object_presence | present/absent_observable/unknown | traffic light | full |
| 4 | object_presence | present/absent_observable/unknown | sign | full |
| 5 | actor_presence | present/absent_observable/unknown | vehicle | full |
| 6 | actor_presence | present/absent_observable/unknown | pedestrian | full |
| 7 | actor_presence | present/absent_observable/unknown | rider | full |
| 8 | actor_presence | present/absent_observable/unknown | other obstacle | full |
| 9 | corridor_state | unavailable/available/unknown | left corridor | full |
| 10 | corridor_state | occupied/clear/unknown | left corridor | full |
| 11 | boundary_state | solid/non_solid/unknown | left boundary | full |
| 12 | lane_attribute | turn_lane/not_turn_lane/unknown | left lane marking | full |
| 13 | control_attribute | allowed/not_allowed/unknown | left arrow/light | full |
| 14 | latent_relational | left_turn/not_left_turn/unknown | front vehicle cue | none |
| 15 | corridor_state | unavailable/available/unknown | right corridor | full |
| 16 | corridor_state | occupied/clear/unknown | right corridor | full |
| 17 | boundary_state | solid/non_solid/unknown | right boundary | full |
| 18 | lane_attribute | turn_lane/not_turn_lane/unknown | right lane marking | full |
| 19 | control_attribute | allowed/not_allowed/unknown | right arrow/light | full |
| 20 | latent_relational | right_turn/not_right_turn/unknown | front vehicle cue | none |

Every schema row additionally has `counter_localizable=false`. The complete
source/groundability/mirror contract is:

| id | reason name | anchor_source | state_source | groundability | observability_source | mirror |
|---:|---|---|---|---|---|---:|
| 0 | Traffic light is green | traffic-light instance | explicit color attribute | full | light detected + color visible | - |
| 1 | Follow traffic | front traffic context | visual relation cue | partial | front context visible | - |
| 2 | Road is clear | center drivable corridor | object+drivable occupancy | full | corridor source complete | - |
| 3 | Traffic light | traffic-light instance | object presence | full | upper region observable | - |
| 4 | Traffic sign | sign instance | object presence | full | upper region observable | - |
| 5 | Obstacle: car | vehicle instances | object presence/complete absence | full | front corridor complete | - |
| 6 | Obstacle: person | pedestrian instances | object presence/complete absence | full | front corridor complete | - |
| 7 | Obstacle: rider | rider instances | object presence/complete absence | full | front corridor complete | - |
| 8 | Obstacle: others | other obstacle instances | object presence/complete absence | full | front corridor complete | - |
| 9 | No lane on the left | left corridor | lane availability | partial | lane+drivable source | 15 |
| 10 | Obstacles on the left lane | left corridor | object+drivable occupancy | full | left corridor complete | 16 |
| 11 | Solid line on the left | left boundary | explicit lane style | full | boundary style visible | 17 |
| 12 | On the left-turn lane | ego/left lane marking | explicit turn-lane marking | partial | marking visible | 18 |
| 13 | Traffic light allows left | light/arrow instance | explicit directional color | partial | arrow+color visible | 19 |
| 14 | Front car turning left | front vehicle cue | latent single-frame relation | latent | cue explicitly visible only | 20 |
| 15 | No lane on the right | right corridor | lane availability | partial | lane+drivable source | 9 |
| 16 | Obstacles on the right lane | right corridor | object+drivable occupancy | full | right corridor complete | 10 |
| 17 | Solid line on the right | right boundary | explicit lane style | full | boundary style visible | 11 |
| 18 | On the right-turn lane | ego/right lane marking | explicit turn-lane marking | partial | marking visible | 12 |
| 19 | Traffic light allows right | light/arrow instance | explicit directional color | partial | arrow+color visible | 13 |
| 20 | Front car turning right | front vehicle cue | latent single-frame relation | latent | cue explicitly visible only | 14 |

There is no selector or expert mixture. Action transport is exactly:

```text
P_r = B_r A_r, rank=16
pi_iar = masked_entmax_r(q_ia^T K_r(E_ir)+b_type_ar,
                         owner_mask_r,
                         including null factor)
c_iar = owner_mask_r * rho_ir * pi_iar
        * q_ia^T B_r A_r E_ir
null-factor contribution = 0
kappa_a = 0.20*EMA[RMS(z_visual_a)]
delta_ia = kappa_a*tanh(sum_r c_iar/kappa_a)
z_action_final = z_action_visual + ramp_10*delta
```

Action loss weights are fixed to `1.00 final ASL, 0.35 visual ASL, 0.20
correction-mode, 0.05 TwoWay, 0.03 SoftF1, 0.02 cardinality, 0.05 specificity,
0.03 identity`. Correction mode trains delta to repair visual margins, not as
an independent classifier.

Non-action-owned factors must have exact-zero attention weight, contribution,
and gradient. Factor 1 uses a fractional ownership coefficient; 14/20 have zero
ownership for every action. There is no hand-written factor-to-action
compatibility or causal direction mask: compatibility and contribution sign
are learned from `q_a`, `K_r`, `b_type`, and supervised data. The schema
defines measurement and ownership only, never support/veto semantics.

## Exact Grounding Objectives

For a valid normalized anchor target `T`:

```text
L_anchor_nll = -valid_anchor*w_source_ir*sum_n T_n log(A_n)
L_anchor_dice = valid_anchor*w_source_ir*
  [1-(2*sum_n A_n*T_n+eps)/(sum_n A_n+sum_n T_n+eps)]
L_state = valid_state*w_source_ir*CE(state_logits,state_target)
L_obs_bce = valid_obs*w_source_ir*BCEWithLogits(obs_logit,obs_target)
L_obs_coverage = sum_r |E_train[o_r]-tau_r|
L_same_type = valid_pair*w_pair*relu(m+score_same_type_wrong-score_correct)
L_same_region = valid_pair*w_pair*relu(m+score_background-score_correct)
L_mirror = valid_mirror*w_pair*
  [distance(mirror(A_left),A_right)+CE(mirror_state_left,state_right)]
```

Unknown state, invalid anchor, and incomplete absence have zero supervision
weight. Fixed formal weights are `anchor=0.10`, `state=0.10`,
`observability=0.03`, `discrimination=0.05`, and
`dense_intervention=0.05`.

`tau_r` is factor-specific and fixed from train-only source statistics or a
train-only EMA. It never uses test data and never tracks the current batch
target mean. Every weak target carries an explicit source reliability weight;
low-confidence and reliable sources are not treated equally.

Reason correction is exactly:

```text
delta_reason_ir = groundable_r*rho_ir*kappa_r
                  *tanh(w_r^T stopgrad(E_ir)/kappa_r)
z_reason_final = z_reason_global + ramp_10*delta_reason
```

Latent correction is conservatively fixed at zero. Reason loss weights are
`1.00 final weighted-ASL, 0.45 global weighted-ASL, 0.05 rank, 0.05 SoftF1,
0.03 evidence-correction, plus PU-private`.

## Exact Ownership, Intervention, PU, And Calibration

Dynamic autograd probes must establish:

```text
action loss -> foundation/action query/typed factor/action transport
grounding loss -> typed factor only
reason loss -> reason global/reason correction only
PU loss -> reason-private only
threshold fitting -> representation grad exactly zero
```

Reason and PU consume `stopgrad(E_ir)`. Formal meta is absent from forward and
optimizer.

Dense interventions use:

```text
L_necessity = relu(m + loss(z_final,y)-loss(z_final-c_correct,y))
L_specificity = relu(m + |c_wrong|-|c_correct|)
L_identity = relu(m + score_corrupt-score_clean)
```

Identity corruptions are schema-token mismatch, cross-sample same-factor token
swap while retaining the target action query, and state corruption with anchor
fixed.

PU is admitted only when:

```text
u_PU=p_reason_global*p_state_positive*rho
positive_count>=20
hidden-positive AUPRC LCB>baseline AUPRC
0<lambda<=0.15
```

Inactive labels have exact-zero PU loss and gradient. Calibration freezes the
representation and fits temperature, group-shrinkage threshold, then optional
per-label threshold on train-calib only. Test never updates thresholds.

## Exact Data, Ramp, Runtime, And Gate Protocol

Augmentation is light brightness/contrast plus 25% horizontal mirror; strong
hue jitter is forbidden. Mirror synchronously swaps left/right action, reason,
corridor, boundary, anchor, state, and grounding targets.

For total optimizer updates `S`:

```text
r5=clip(step/(0.05*S),0,1)
r10=clip(step/(0.10*S),0,1)
factor supervision=0.25+0.75*r5
action/reason correction=r10
softmax-to-entmax=r10
dense intervention=r10
```

Resume restores exact optimizer-step ramp state.

Fixed config: AdamW, cosine, grad clip 1.0, batch 6, accumulation 5, bf16,
workers 4, prefetch 2, foundation LR `1.5e-4`, typed-factor and action LR
`2e-4`, reason-global and correction LR `2.5e-4`, target reserved 42GB, hard
reserved 45GB.

Pilot is fixed to 4096 train-main, 1024 train-audit, 512 train-calib, 512 test,
4 epochs, seed 20260729. Gate thresholds are:

```text
A: progress-zero action/reason-global/label-node max errors <1e-6
B: per-factor non-max-entropy anchor, null noncollapse, state AP/AUC above
   frequency, correct > same-type wrong, correct mirror
C: final Act_mAP >= visual+0.005; raw Act_mF1 >= visual-0.005; each action
   correction RMS 3%-25% of visual RMS; identity lowers target action AP and
   wrong-target AP drop is significantly smaller
D: final Exp_mAP >= global-0.002; groundable final >= global+0.005; no latent
   fake-grounding degradation; factor identity corruption lowers the
   corresponding reason AP
E: 4/4 actions and >=12 factors; correct effect>wrong; cross-sample swap hurts
F: exactly 128 unique IDs, 4/4 actions, >=12 factors, selected deletion >
   same-region equal-count control, and anchor deletion has the predicted
   target direction
G: inactive PU exact-zero loss/grad; active LCB>baseline; private gradients
H: DINO calls=1; reserved<45GB; sequential eval no persistent growth; finite
```

Sequential evaluation order is main raw/final, visual/factor-off,
reason-global/correction-off, fixed CF subset. Every mode releases tensors,
runs GC, empties CUDA cache, and synchronizes. Ordinary PyTorch warnings are
logged but are not process failures.

Full training is 12 epochs, seed 20260729, test each epoch, primary best deploy
joint and secondary best raw action mAP/mF1, raw explanation mAP, deploy
explanation mF1. Manifest records `internal_test_selected=true` and
`publication_eligible=false`.

Only NaN/Inf, ordinary DINO calls above one, final action mAP below visual by
more than 0.01 for two epochs, null collapse, completely ineffective identity
corruption, or persistent eval-memory growth may stop full training.
