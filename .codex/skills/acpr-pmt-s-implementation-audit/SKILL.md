# ACPR-PMT-S Implementation Audit Skill

## Purpose

Audit that ACPR-PMT-S V1 is a complete, code-level implementation of:

Predicate-Mediated Triadic Alignment with Stability Scheduling for BDD-OIA.

The audit must reject:
- superficial files with missing semantics
- direct predicate-to-action bypasses
- test-threshold leakage
- feature-cache or token-compression paths
- RunC/cached residual paths
- graph/expert/selector/action-set-final routes
- placeholder artifacts
- missing visualization chains
- Stage/full training without preflight pass

## Required context

Before changing code, running training, evaluating, pushing, or launching a supervisor under E:\sbw\FATE_Drive, read:

- E:\sbw\FATE_Drive\task_plan.md
- E:\sbw\FATE_Drive\findings.md
- E:\sbw\FATE_Drive\progress.md

Only these three Markdown files are durable experiment records. Do not create training/run/audit status Markdown files elsewhere.

## Non-negotiable constraints

Reject unless all are true:

1. Base branch is github/acpr_calalign_v1_2.
2. New worktree/branch is used, not modifying the existing acpr_calalign_v1_2 worktree.
3. Direct image training only.
4. No feature cache generation or read.
5. token_compression is none.
6. Frozen DINO ViT-S/8 dense patch tokens are used.
7. BDD-OIA action is 4-dimensional sigmoid multi-label.
8. BDD-OIA reason is 21-dimensional sigmoid multi-label.
9. Every action/reason final logit is multi-label sigmoid; no softmax four-class action.
10. Action-set auxiliary does not affect final action.
11. No expert/MoE/specialist/router.
12. No selector primary objective.
13. No graph delta to final logits.
14. No RunC checkpoint, cached logits, frozen_run_c_predictor, or residual adapter.
15. Test split is never used to update threshold, gate, teacher, or model parameters.
16. Test per-label threshold remains oracle diagnostic only.
17. Best selection may use test per user protocol, but training decisions/gates/teachers must not use test.
18. Every PMT delta is zero-initialized and bounded.
19. pmt.enabled=false reproduces ACPR-CalAlign behavior.
20. pmt.enabled=true at zero-init reproduces ACPR-CalAlign behavior within numerical tolerance.
21. Predicate cannot bypass reason to directly change action.
22. Visualization exports action -> original BDD-OIA reason -> predicate -> DINO patch chains.

## Required files

Reject if missing:

- configs/fate_oia_train_360x640_acpr_pmt_s_v1.yaml
- configs/acpr_pmt_action_predicate_grammar.yaml
- fate_oia/models/acpr_predicate_patch_targets.py
- fate_oia/models/acpr_predicate_transport_alignment.py
- fate_oia/models/acpr_triadic_mediator.py
- fate_oia/models/acpr_predicate_conditioned_threshold.py
- fate_oia/losses/acpr_pmt_losses.py
- fate_oia/utils/acpr_pmt_artifacts.py
- fate_oia/utils/acpr_pmt_phase_schedule.py
- fate_oia/utils/acpr_pmt_visualization.py
- fate_oia/engine/audit_acpr_pmt_s_implementation.py
- fate_oia/engine/export_acpr_pmt_visuals.py
- fate_oia/engine/supervise_acpr_pmt_s_foreground.py
- scripts/FATE_OIA_acpr_pmt_s_v1_foreground.ps1

Required tests:
- tests/test_acpr_predicate_patch_targets.py
- tests/test_acpr_predicate_transport_alignment.py
- tests/test_acpr_triadic_mediator.py
- tests/test_acpr_predicate_conditioned_threshold.py
- tests/test_acpr_pmt_forward_equivalence.py
- tests/test_acpr_pmt_losses.py
- tests/test_acpr_pmt_pair_filtering.py
- tests/test_acpr_pmt_phase_schedule.py
- tests/test_acpr_pmt_artifacts.py
- tests/test_acpr_pmt_audit.py
- tests/test_acpr_pmt_supervisor.py

## Forbidden active patterns

Reject if active implementation contains:

- frozen_run_c
- FrozenRunC
- run_c_logits
- cached_logits
- tail_residual_adapter
- feature_cache_enabled: true
- token_compression: keep_merge
- Start-Process
- Start-Job
- nohup
- scheduled task
- action_set marginal as final action
- graph_delta_to_logits: true
- selector primary loss
- MoE / expert / specialist routing
- test_threshold_teacher
- update threshold from test
- update gate from test
- test per-label thresholds written into model state

The word "residual" is allowed only for ordinary neural residual blocks, not cached-logit residual adapters.

## Functional audit checks

### 1. ACPR-CalAlign equivalence

Instantiate old ACPR-CalAlign config and new ACPR-PMT-S model with pmt enabled but all PMT deltas zero-init.

Run the same real image batch.

Verify:
- action_logits_base difference < 1e-6
- reason_logits_base difference < 1e-6 except harmless added outputs
- action_logits_deploy difference < 1e-6
- reason_logits_deploy difference < 1e-6
- branch_logits base/deploy/calibrated exist
- pmt.enabled=false path still runs

### 2. Predicate patch target builder

Run synthetic records:
- car bbox in front
- traffic light bbox upper
- lane polyline on left/right
- drivable map available/missing

Verify:
- predicate_patch_targets shape [B,P,3600]
- predicate_patch_mask shape [B,P]
- at least object_box/lane_poly/drivable/weak/missing sources are represented
- missing records do not create high-confidence masks
- mask coordinates correspond to 45x80 grid

### 3. Predicate transport alignment

Run predicate attention and masks.

Verify:
- loss is lower when attention lies inside the target mask
- predicates with predicate_patch_mask=0 are ignored
- entropy regularization finite
- predicate_attention_mass_on_target is logged
- no selected-vs-random deletion objective is used as the primary mechanism

### 4. Triadic mediator

Run ACPRTriadicMediator.

Verify:
- initial triadic_action_delta max abs < 1e-6
- action_reason_logits_triadic equals old action_reason_logits at init
- after nonzero parameter perturbation, output remains bounded by max_action_delta
- output depends on reason logits and predicate evidence
- predicate-only action delta is impossible
- action-reason-predicate chain scores contain real reason ids and predicate ids
- chain export includes original BDD-OIA reason names

### 5. Predicate-mediated reason/action relation

Verify action support goes through:
- reason confidence
- predicate evidence
- grammar compatibility

Reject if action delta is computed only from predicate_probs without reason logits or reason masks.

### 6. Predicate-conditioned PU

Create synthetic reason targets and contradiction scores.

Verify:
- positives weight exactly 1.0
- reason=0 weight is low when contradiction is low
- reason=0 weight is high when predicate contradiction is high
- zeros are not all hard negatives
- tail/common negative weights logged

### 7. Predicate-filtered HardPair

Create synthetic pairs.

Verify:
- pair with weak predicate difference is rejected or downweighted
- pair with strong reason-relevant predicate difference is kept
- pair loss is reason-specific
- pair loss is capped by cap_ratio * main_prediction_loss
- cap activation is logged

### 8. Predicate-conditioned threshold

Run threshold head with predicate_context.

Verify:
- delta_theta zero-init gives exact old deploy logits
- delta_theta bounded by configured max
- disabled mode gives exact old behavior
- test thresholds are not used
- teacher update uses train_calib only
- threshold_delta stats are logged

### 9. Phase schedule

Verify:
- epoch 0-2: triadic delta and predicate-conditioned threshold delta loss disabled or zero-weight
- epoch 3-8: PMT components enabled
- epoch >=9: stable phase lowers pair cap and PMT delta LR/weights
- stable trigger uses train_calib only, not test

### 10. Artifacts

After smoke/training epoch, verify non-placeholder files:
- predicate_patch_alignment.jsonl
- predicate_coverage.jsonl
- predicate_reason_alignment.jsonl
- triadic_mediator_stats.jsonl
- triadic_chain_topk.jsonl
- pmt_hardpair_stats.jsonl
- pmt_pu_stats.jsonl
- threshold_diagnostics.jsonl
- pmt_phase_schedule.jsonl
- loss_group_components.jsonl

Reject if any contain only {"available": true} / {"available": false} placeholders.

### 11. Visualization

Run export_acpr_pmt_visuals on a tiny checkpoint or mock output.

Verify each case contains:
- image file
- predicted action
- predicted reason
- action -> reason -> predicate -> patch chain
- predicate attention heatmap or coordinates
- real BDD-OIA reason names

### 12. Training protocol

Inspect train_acpr_oia.py and supervisor.

Verify:
- train_loader and test_loader exist
- no val-best selection
- eval every epoch only on test
- best selected on test per user protocol
- train_calib split is internal train only
- no cache / no compression
- DINO frozen no_grad
- foreground supervisor
- no metric early stop
- OOM fallback implemented
- GOAL_COMPLETED_ACPR_PMT_S_V1.json written only after full completion

## Required commands

py_compile:
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_predicate_patch_targets.py `
  fate_oia\models\acpr_predicate_transport_alignment.py `
  fate_oia\models\acpr_triadic_mediator.py `
  fate_oia\models\acpr_predicate_conditioned_threshold.py `
  fate_oia\models\acpr_oia_model.py `
  fate_oia\models\acpr_scene_predicate_head.py `
  fate_oia\models\acpr_predicate_targets.py `
  fate_oia\models\acpr_predicate_reason.py `
  fate_oia\models\acpr_label_trunk.py `
  fate_oia\models\acpr_threshold_head.py `
  fate_oia\models\acpr_pair_memory.py `
  fate_oia\losses\acpr_pmt_losses.py `
  fate_oia\engine\train_acpr_oia.py `
  fate_oia\engine\eval_acpr_oia.py `
  fate_oia\engine\audit_acpr_pmt_s_implementation.py `
  fate_oia\engine\export_acpr_pmt_visuals.py `
  fate_oia\engine\supervise_acpr_pmt_s_foreground.py

pytest:
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_predicate_patch_targets.py `
  tests\test_acpr_predicate_transport_alignment.py `
  tests\test_acpr_triadic_mediator.py `
  tests\test_acpr_predicate_conditioned_threshold.py `
  tests\test_acpr_pmt_forward_equivalence.py `
  tests\test_acpr_pmt_losses.py `
  tests\test_acpr_pmt_pair_filtering.py `
  tests\test_acpr_pmt_phase_schedule.py `
  tests\test_acpr_pmt_artifacts.py `
  tests\test_acpr_pmt_audit.py `
  tests\test_acpr_pmt_supervisor.py -q

Audit:
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_pmt_s_implementation `
  --config configs\fate_oia_train_360x640_acpr_pmt_s_v1.yaml `
  --output_dir .background_runs\acpr_pmt_s_v1_preflight `
  --device cuda `
  --write_review_pass

Required pass file:
.background_runs\acpr_pmt_s_v1_preflight\REVIEW_PASS_ACPR_PMT_S_V1.txt

Tiny smoke:
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_oia `
  --config configs\fate_oia_train_360x640_acpr_pmt_s_v1.yaml `
  --output_dir .background_runs\acpr_pmt_s_v1_smoke `
  --epochs 1 `
  --batch_size 1 `
  --gradient_accumulation_steps 2 `
  --max_train_samples 4 `
  --max_test_samples 4 `
  --device cuda `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression

Full train:
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_pmt_s_v1_foreground.ps1 `
  -Epochs 18 `
  -BatchSize 6 `
  -GradAccum 5 `
  -Device cuda `
  -RequireReviewPass

## Pass/fail rubric

PASS only if:
- all code paths implemented
- all tests pass
- audit pass
- tiny smoke pass
- artifacts non-placeholder
- pmt zero-init equivalence verified
- no test leakage
- full training is foreground and complete

FAIL if:
- any PMT component is a placeholder
- predicates bypass original BDD-OIA reasons
- test thresholds influence training
- action-set final returns
- cache/compression returns
- stage schedule missing
- artifacts cannot diagnose predicate/reason/action/threshold chain
