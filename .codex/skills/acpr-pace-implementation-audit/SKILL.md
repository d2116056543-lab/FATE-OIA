# ACPR-PACE V1 Implementation Audit Skill

## Purpose

This skill is the blocking implementation and runtime audit for:

**ACPR-PACE V1 — Predicate-Anchored Coupled Evidence Learning**

The source model is ACPR-CalAlign V1.2.

PACE must solve one precise structural problem:

```text
predicate-conditioned explanation evidence
must be the same evidence used by both
explanation prediction and action reasoning
```

The active chain must be:

```text
DINO patches
  -> scene predicates
  -> predicate-conditioned explanation evidence
  -> explanation prediction
  -> existing reason-to-action mapping
  -> action fusion
  -> CalAlign
```

The audit must reject:
- partial shells,
- placeholder outputs,
- inactive components,
- duplicate objectives,
- test leakage,
- failed-method reintroduction,
- detached/background execution.

An implementation pass is not a research-success claim.


----------------------------------------------------------------------
1. CANONICAL CONTEXT
----------------------------------------------------------------------

Before any remote task under `E:\sbw\FATE_Drive`, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

Do not create additional training-status Markdown files.

Expected source:

```text
github/acpr_calalign_v1_2
```

Expected branch:

```text
acpr_pace_v1
```

Expected worktree:

```text
E:\sbw\FATE_Drive\fate_oia_acpr_pace_v1_worktree
```

Record:

- source SHA
- local HEAD
- remote HEAD
- branch
- worktree
- clean status


----------------------------------------------------------------------
2. HARD PASS CONDITIONS
----------------------------------------------------------------------

Reject unless every item is true.

1. Direct-image training.
2. Frozen no-grad DINO ViT-S/8.
3. 360x640 formal input.
4. No feature cache.
5. No token compression.
6. Final action is four independent sigmoid logits.
7. Final explanation is 21 independent sigmoid logits.
8. Test-only end-of-epoch evaluation.
9. Test deploy-fixed joint selects the primary best checkpoint.
10. Test oracle thresholds are diagnostic only.
11. Scene predicate head remains active.
12. Predicate reasoner remains active.
13. Partial-label PU reason loss remains active.
14. HardPair remains reason-specific and active.
15. HardPair is budgeted.
16. CalAlign remains global per-label.
17. Train-calib teacher uses historical-best lock.
18. The same predicate-conditioned reason evidence drives Exp and action.
19. PACE action coupling uses existing reason_to_action weights.
20. PACE action correction is bounded.
21. PACE adds no large action network.
22. PACE adds no new scalar training loss.
23. Existing soft-F1 and predicate alignment become PU-consistent.
24. Weak predicate targets do not create known contradictory positives.
25. Predicate-core action/Exp gradient coordination is active from epoch 3.
26. Gradient coordination preserves ordinary gradients outside the shared core.
27. Gradient accumulation remains correct.
28. No test metric controls LR, teacher, coupling strength, or cooldown.
29. PACE strength is selected from train_calib signal audit.
30. Signed contribution decomposition is numerically exact.
31. Action->reason->predicate->patch visualization exists.
32. Faithfulness evaluation is evaluation-only.
33. Full training is foreground-only.
34. Weak metrics never early-stop the run.
35. Review pass is tied to exact clean pushed HEAD.


----------------------------------------------------------------------
3. REQUIRED FILES
----------------------------------------------------------------------

### New files

- `configs/fate_oia_train_360x640_acpr_pace_v1.yaml`
- `fate_oia/models/acpr_predicate_action_coupling.py`
- `fate_oia/utils/acpr_pair_budget.py`
- `fate_oia/utils/acpr_pace_gradient_coordinator.py`
- `fate_oia/utils/acpr_pace_training_control.py`
- `fate_oia/utils/acpr_pace_artifacts.py`
- `fate_oia/utils/acpr_teacher_lock.py`
- `fate_oia/engine/audit_acpr_pace_implementation.py`
- `fate_oia/engine/audit_acpr_pace_signal.py`
- `fate_oia/engine/eval_acpr_pace_faithfulness.py`
- `fate_oia/engine/export_acpr_pace_visuals.py`
- `fate_oia/engine/supervise_acpr_pace_foreground.py`
- `scripts/FATE_OIA_acpr_pace_v1_foreground.ps1`
- `.codex/skills/acpr-pace-implementation-audit/SKILL.md`

### Required tests

- `tests/test_acpr_pace_coupling.py`
- `tests/test_acpr_pace_equivalence.py`
- `tests/test_acpr_pace_contributions.py`
- `tests/test_acpr_pace_pu_losses.py`
- `tests/test_acpr_pace_predicate_targets.py`
- `tests/test_acpr_pace_pair_budget.py`
- `tests/test_acpr_pace_gradient_coordinator.py`
- `tests/test_acpr_pace_gradient_accumulation.py`
- `tests/test_acpr_pace_teacher_lock.py`
- `tests/test_acpr_pace_training_control.py`
- `tests/test_acpr_pace_signal_audit.py`
- `tests/test_acpr_pace_visualization.py`
- `tests/test_acpr_pace_faithfulness.py`
- `tests/test_acpr_pace_audit.py`
- `tests/test_acpr_pace_supervisor.py`
- `tests/test_acpr_pace_performance.py`


----------------------------------------------------------------------
4. FORBIDDEN ACTIVE PATTERNS
----------------------------------------------------------------------

Reject active formal implementation containing or enabling:

```text
acpr_semantic_evidence_coattention
acpr_triadic_mediator
predicate_patch_targets
predicate_transport_alignment
predicate_conditioned_threshold
predicate_filtered_hardpair
acpr_action_candidates
acpr_action_utility
acpr_fusionlite
FrozenRunC
frozen_run_c
cached_logits
tail_residual_adapter
expert
Expert
MoE
moe
specialist
Specialist
router
Router
graph_delta_to_logits: true
action_set_affects_final_action: true
feature_cache_enabled: true
token_compression: keep_merge
best_selection_split: val
eval_splits: val
checkpoint_best_val
Start-Process
Start-Job
nohup
daemon
scheduled task
hidden cmd
```

A normal neural residual connection is allowed.
A cached-logit or old-checkpoint residual adapter is not.


----------------------------------------------------------------------
5. STATIC FORWARD-PATH AUDIT
----------------------------------------------------------------------

Inspect `ACPROIAModel.forward`.

Required source order:

```text
field
-> ego
-> predicates
-> trunk
-> predicate_reasoner
```

Required PACE order:

```text
reason_visual =
    trunk.reason_logits_visual

reason_shared =
    reason_visual
    + predicate_reason_delta

action_reason_visual =
    trunk.action_reason_logits_visual

PACE correction =
    current reason_to_action weights
    applied to predicate_reason_delta
    with bounded tanh correction

action_reason_pace =
    action_reason_visual
    + PACE correction

action_base =
    existing fusion gate
    between visual action and action_reason_pace

threshold =
    current CalAlign on action_base/reason_shared
```

Reject if final action is still computed before predicate delta.

Reject if action reads a different explanation tensor than final Exp.

Reject if predicate probability directly replaces action logits.

Reject if a static action/reason grammar is introduced into the coupling.


----------------------------------------------------------------------
6. COUPLING MATH AUDIT
----------------------------------------------------------------------

Synthetic shapes:

```text
action_visual_logits:       [B,4]
action_reason_visual:       [B,4]
fusion_gate:                [B,4]
predicate_reason_delta:     [B,21]
reason_to_action_weight:    [4,21]
```

Require:

```text
raw_contrib:                [B,4,21]
raw_action_delta:           [B,4]
bounded_action_delta:       [B,4]
final_reason_contrib:       [B,4,21]
action_logits_pace:         [B,4]
```

Required formulas:

```python
raw_contrib = W[None, :, :] * delta[:, None, :]
raw_action_delta = raw_contrib.sum(-1)
bounded = max_delta * tanh(kappa * raw_action_delta / max_delta)
action_reason_pace = action_reason_visual + bounded
action_pace = gate * visual + (1-gate) * action_reason_pace
```

Required bounds:

```text
abs(bounded_action_delta) <= max_action_delta + 1e-6
```

Required contribution identities:

```text
bounded_reason_contrib.sum(-1)
    == bounded_action_delta

final_reason_contrib.sum(-1)
    == action_logits_pace - action_logits_legacy
```

Tolerance:
`1e-6`.

Zero-strength test:

```text
kappa=0
action_logits_pace == action_logits_legacy
```

No external fallback or candidate selection may produce this equality.


----------------------------------------------------------------------
7. SHARED EVIDENCE AUDIT
----------------------------------------------------------------------

Require:

```text
reason_logits_base
    == reason_logits_visual
       + predicate_reason_delta
```

The same `predicate_reason_delta` object/value must enter:

- final explanation
- PACE action coupling
- PU diagnostics
- signed contribution artifacts

Reject recomputation with a different detached tensor.

Action loss must have a nonzero gradient to:

- reason_to_action weight
- predicate_reason gate/MLP
- predicate head through predicate probabilities/tokens

when coupling strength is nonzero.

Explanation loss must continue to have normal gradients to the same evidence path.


----------------------------------------------------------------------
8. PREDICATE REASON DECOMPOSITION AUDIT
----------------------------------------------------------------------

Require output shapes:

```text
predicate_reason_contrib_by_predicate: [B,21,P]
predicate_reason_mlp_residual:         [B,21]
predicate_reason_delta:                [B,21]
```

Require:

```text
contrib_by_predicate.sum(-1)
+ mlp_residual
== predicate_reason_delta
```

within `1e-6`.

Reject visualization code that distributes the MLP residual across predicates
without a valid decomposition.

Reports must label the MLP part as an unattributed learned residual.


----------------------------------------------------------------------
9. WEAK PREDICATE TARGET AUDIT
----------------------------------------------------------------------

Use synthetic and real records.

Reject any of the following:

- any traffic light automatically marks green;
- generic traffic sign automatically marks stop sign;
- one front vehicle marks both close and far;
- side vehicle automatically marks parked;
- lane polyline automatically marks turn permission or merging context;
- mere drivable-map availability marks all drivable/open-gap states positive;
- absence of annotated obstacles is treated as certain road clear;
- global scene context is trained as an ordinary binary target without evidence.

Require audit counts:

```text
traffic_light_green_without_color_count == 0
close_far_double_positive_count == 0
parked_without_attribute_count == 0
```

Unknown predicates must have mask=0.


----------------------------------------------------------------------
10. PU-CONSISTENCY AUDIT
----------------------------------------------------------------------

Active PACE losses must use:

- `partial_label_reason_loss`
- `pu_reason_soft_f1_loss`
- `pu_predicate_reason_alignment_loss`

Reject active use of old:

```text
FP = probs * (1-target)
target_sign = target*2-1
```

without contradiction weighting.

Test:

```text
y=0, contradiction=0
```

must receive only `neg_min_weight`.

Test:

```text
y=0, contradiction=1
```

must receive full negative weight.

Positive labels always receive full positive pressure.

Ensure no duplicate second PU loss exists.


----------------------------------------------------------------------
11. HARDPAIR BUDGET AUDIT
----------------------------------------------------------------------

Use:

```text
L_action = 0.10
L_reason = 0.20
pair_raw_weighted = 0.50
ratio = 0.25
```

Expected cap:

```text
0.075
```

Require:

```text
pair_used <= 0.075 + 1e-6
```

Below-cap pair must remain unchanged.

Inspect total loss and reject:
- raw pair plus budgeted pair both added;
- pair cap based on unweighted pair;
- pair cap based on test metrics.

Required logs:

- pair raw
- pair used
- cap
- scale
- active flag
- raw/used ratios


----------------------------------------------------------------------
12. GRADIENT-COORDINATOR AUDIT
----------------------------------------------------------------------

### Parameter groups

Require exact non-overlapping groups:

- label_shared
- predicate_visual
- predicate_reason

Reject duplicate parameter IDs.

Reject inclusion of:
- DINO
- action visual head
- reason_to_action
- fusion gate
- threshold head
- calibration head
- combo head
- pair projections

### Objective groups

Action group must contain action objectives.

Explanation group must contain:
- PU reason,
- PU soft-F1,
- predicate weak,
- PU alignment,
- budgeted pair,
- source predicate compactness.

No test metric may be present.

### Aligned toy case

If dot >= 0:
  output equals action + explanation gradient.

### Conflict toy case

Require:
- finite alpha in [0,1];
- nonnegative dot with both original gradients within tolerance;
- finite norm;
- max scale respected.

### Zero-gradient case

Use nonzero task gradient.

### Accumulation test

Use two microbatches.

Verify:
- previous accumulated gradient is preserved;
- only current shared action+Exp component is replaced;
- non-shared gradients equal ordinary backward;
- other-loss shared gradient is preserved.

### Runtime logs

Require:
- norms,
- dot,
- cosine,
- conflict,
- alpha,
- common norm,
- scale,
- descent checks,
- zero fallback.


----------------------------------------------------------------------
13. CALALIGN TEACHER LOCK AUDIT
----------------------------------------------------------------------

Preserve:

```text
deploy = base - theta
```

No sample-conditioned threshold.

No predicate-conditioned threshold.

No test threshold update.

Feed candidates:

```text
epoch1 joint .50, action .70, exp .40 -> accept
epoch2 joint .49, action .71, exp .39 -> reject
epoch3 joint .51, action .70, exp .41 -> accept
```

Require historical teacher state after each.

Checkpoint must save:

- best teacher theta
- pred rate
- best joint/action/exp
- best epoch

Resume must restore it.


----------------------------------------------------------------------
14. TRAINING-CONTROL AUDIT
----------------------------------------------------------------------

Formal resolved config:

```text
actual epochs = 16
scheduler horizon = 28
warmup = 2
batch = 5
accum = 6
```

Cooldown:

- train-calib joint only;
- min epoch 8;
- patience 2;
- apply once;
- no early stop;
- non-threshold LR x0.20;
- threshold LR x0.50;
- state checkpointed.

Reject any use of test metrics for cooldown.


----------------------------------------------------------------------
15. SIGNAL AUDIT
----------------------------------------------------------------------

Signal audit must load a documented ACPR-CalAlign checkpoint.

Evaluate train_calib at:

```text
kappa = 0, .5, 1, 2
```

No training update.

Select strength from train_calib only.

Pass requires:

- nonzero bounded correction;
- train-calib action not degraded more than .001;
- train-calib joint not degraded more than .001;
- overall action +.001 or one action +.002;
- all-high increase <= .02.

Test may be emitted after selection as diagnostic only.

Formal supervisor must require:

```text
PACE_SIGNAL_PASS.json
pace_selected_strength.json
```


----------------------------------------------------------------------
16. NO-NEW-LOSS AUDIT
----------------------------------------------------------------------

Allowed source objective names remain:

- action_direct
- action_visual_aux
- action_reason_aux
- reason_partial
- reason_soft_f1
- predicate_weak
- predicate_reason_align
- matched_pair_logit
- matched_pair_embed
- action_combo_ce
- action_combo_drop_add
- cardinality
- calibration
- predicate_attention_compactness
- threshold bundle

`reason_soft_f1` and `predicate_reason_align` use PU-consistent formulas.

Reject additional terms functioning as:

- PACE loss
- coupling loss
- attention alignment
- correlation loss
- evidence consistency
- patch alignment
- dynamic threshold regularization
- distillation/teacher loss


----------------------------------------------------------------------
17. VISUALIZATION AUDIT
----------------------------------------------------------------------

Required exact decomposition:

```text
visual action component
+ visual-reason components
+ predicate-reason components
+ action reason bias
== action_logits_base
```

Tolerance:
`1e-5`.

Required chain:

```text
action
-> original BDD-OIA reason
-> predicate
-> DINO patch
```

The report must show:

- signed reason contributions;
- signed predicate contributions;
- support heatmap;
- suppression heatmap;
- MLP residual separately;
- legacy vs PACE action probabilities;
- ground truth separately from model evidence.

No ground-truth reason may construct predicted evidence.


----------------------------------------------------------------------
18. FAITHFULNESS AUDIT
----------------------------------------------------------------------

Faithfulness is eval-only.

Require:

- top reason deletion
- random equal-count reason deletion
- top predicate intervention
- random equal-count predicate intervention
- bounded patch deletion
- random equal-count patch deletion
- sufficiency

No gradients.
No optimizer update.
No teacher/LR feedback.

Required margins:

```text
top_drop - random_drop
```


----------------------------------------------------------------------
19. DATA, MEMORY, AND PERFORMANCE AUDIT
----------------------------------------------------------------------

Require:

```text
feature_cache_enabled=false
token_compression=none
num_workers=4 formal
persistent_workers=true
prefetch_factor=2
```

Primary:

```text
batch 5
accum 6
effective 30
```

Fallback:

```text
[5,6], [4,8], [3,10], [2,15]
```

Performance benchmark uses real DINO and real images.

Pass:

```text
forward overhead <= 5%
active training-step overhead <= 25%
peak allocated increase <= 2.5 GB
```

Do not allocate dummy GPU memory to meet the requested target.


----------------------------------------------------------------------
20. EVALUATION AND CHECKPOINT AUDIT
----------------------------------------------------------------------

Every epoch:
- test only;
- no validation loader;
- no validation metrics.

Primary best:

```text
checkpoint_best_test_deploy_joint.pth
```

Also require:

```text
checkpoint_best_test_action_mf1.pth
checkpoint_best_test_exp_mf1.pth
checkpoint_best_test_exp_map.pth
checkpoint_best_test_base_joint.pth
checkpoint_latest.pth
```

Reject test-oracle threshold checkpoint selection.

Required epoch tensors:

- legacy action base
- PACE action base
- action deploy
- reason visual
- reason shared
- reason deploy
- action/reason labels
- compact signed action/reason contributions


----------------------------------------------------------------------
21. FOREGROUND SUPERVISOR AUDIT
----------------------------------------------------------------------

Reject:

- Start-Process
- Start-Job
- nohup
- daemon
- scheduled task
- detached child
- hidden command window

Require:

1. exact clean pushed HEAD verification;
2. review pass HEAD verification;
3. tiny smoke;
4. signal audit;
5. selected coupling strength;
6. GPU query;
7. line-by-line child output;
8. heartbeat;
9. liveness and stall detection;
10. OOM-only fallback;
11. fresh attempt dirs;
12. failed logs preserved;
13. no metric early stop;
14. technical error surfaced for foreground repair;
15. all 16 epochs verified before completion.


----------------------------------------------------------------------
22. REQUIRED COMPILE COMMAND
----------------------------------------------------------------------

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_predicate_action_coupling.py `
  fate_oia\models\acpr_label_trunk.py `
  fate_oia\models\acpr_oia_model.py `
  fate_oia\models\acpr_predicate_reason.py `
  fate_oia\models\acpr_predicate_targets.py `
  fate_oia\losses\acpr_losses.py `
  fate_oia\utils\acpr_pair_budget.py `
  fate_oia\utils\acpr_pace_gradient_coordinator.py `
  fate_oia\utils\acpr_pace_training_control.py `
  fate_oia\utils\acpr_pace_artifacts.py `
  fate_oia\utils\acpr_teacher_lock.py `
  fate_oia\engine\train_acpr_oia.py `
  fate_oia\engine\eval_acpr_oia.py `
  fate_oia\engine\audit_acpr_pace_implementation.py `
  fate_oia\engine\audit_acpr_pace_signal.py `
  fate_oia\engine\eval_acpr_pace_faithfulness.py `
  fate_oia\engine\export_acpr_pace_visuals.py `
  fate_oia\engine\supervise_acpr_pace_foreground.py
```


----------------------------------------------------------------------
23. REQUIRED TEST COMMAND
----------------------------------------------------------------------

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_pace_coupling.py `
  tests\test_acpr_pace_equivalence.py `
  tests\test_acpr_pace_contributions.py `
  tests\test_acpr_pace_pu_losses.py `
  tests\test_acpr_pace_predicate_targets.py `
  tests\test_acpr_pace_pair_budget.py `
  tests\test_acpr_pace_gradient_coordinator.py `
  tests\test_acpr_pace_gradient_accumulation.py `
  tests\test_acpr_pace_teacher_lock.py `
  tests\test_acpr_pace_training_control.py `
  tests\test_acpr_pace_signal_audit.py `
  tests\test_acpr_pace_visualization.py `
  tests\test_acpr_pace_faithfulness.py `
  tests\test_acpr_pace_audit.py `
  tests\test_acpr_pace_supervisor.py `
  tests\test_acpr_pace_performance.py -q
```

Also run all inherited ACPR-CalAlign regression tests.


----------------------------------------------------------------------
24. REQUIRED AUDIT COMMAND
----------------------------------------------------------------------

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_pace_implementation `
  --config configs\fate_oia_train_360x640_acpr_pace_v1.yaml `
  --output_dir .background_runs\acpr_pace_v1_preflight `
  --device cuda `
  --write_review_pass
```

Required:

```text
implementation_audit_ACPR_PACE_V1.json
performance_audit.json
REVIEW_PASS_ACPR_PACE_V1.txt
```

Review pass contains exact git HEAD.

Any code/config change invalidates it.


----------------------------------------------------------------------
25. AUDIT JSON SCHEMA
----------------------------------------------------------------------

Required top-level fields:

- `pass`
- `git_head`
- `remote_head`
- `branch`
- `worktree`
- `worktree_clean`
- `source_branch`
- `source_sha`
- `checked_files`
- `missing_files`
- `forbidden_patterns`
- `config_checks`
- `forward_path_checks`
- `coupling_math_checks`
- `shared_evidence_checks`
- `predicate_decomposition_checks`
- `predicate_target_checks`
- `pu_checks`
- `pair_budget_checks`
- `gradient_coordinator_checks`
- `gradient_accumulation_checks`
- `teacher_lock_checks`
- `training_control_checks`
- `signal_audit_checks`
- `no_new_loss_checks`
- `visualization_checks`
- `faithfulness_checks`
- `runtime_checks`
- `performance_checks`
- `supervisor_checks`
- `smoke_result`
- `warnings`
- `review_pass_path`

`pass=true` only if all blocking checks pass.


----------------------------------------------------------------------
26. FORMAL TRAINING AUTHORIZATION
----------------------------------------------------------------------

Formal training is authorized only after:

1. code-only commit;
2. branch push;
3. clean worktree;
4. local/remote HEAD match;
5. exact-head audit pass;
6. real tiny smoke pass;
7. performance pass;
8. PACE signal pass;
9. selected nonzero coupling strength;
10. foreground supervisor pass.

Formal command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_pace_v1_foreground.ps1 `
  -Epochs 16 `
  -BatchSize 5 `
  -GradAccum 6 `
  -NumWorkers 4 `
  -Device cuda `
  -ReferenceCheckpoint "<REFERENCE_ACPR_CALALIGN_CHECKPOINT>" `
  -RequireReviewPass
```

Training must remain attached.


----------------------------------------------------------------------
27. VERDICT TERMINOLOGY
----------------------------------------------------------------------

Use three separate verdicts.

### Implementation pass

All code, tests, audits, smoke, signal, exact-HEAD, and performance gates pass.

### Training complete

All 16 epochs, test evaluations, checkpoints, and artifacts exist.

### Research success

Actual metrics and faithfulness exceed or meaningfully improve ACPR-CalAlign.

Never describe implementation pass as research success.
