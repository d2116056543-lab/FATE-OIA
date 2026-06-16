
# ACPR-OIA Implementation Audit Skill

## Purpose

Audit that ACPR-OIA V1 is a complete, independent, direct-image implementation of:

Action-Conditioned Predicate Reasoning for Explainable Object-Induced Action Decision.

The implementation must learn finite noisy BDD-OIA reason labels through:
- strong direct multi-label action/reason trunk
- BDD100K-supervised ego-scene predicates
- reason-predicate grammar
- positive-unlabeled partial-label reason learning
- matched counterfactual reason pair mining
- action-composition consistency as auxiliary only
- train-time calibration with real objective
- matched-case visual audit

It must reject selector/expert/router/graph-delta/action-set-final/cached-logit implementations.

## Mandatory context

Before code changes, training, evaluation, pushing, or process management under `E:\sbw\FATE_Drive`, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

Do not create extra training-status Markdown files. Append experiment status only to those three canonical files.

If a "superpowers" planning/review tool is available, use it before coding and before full training. If unavailable, this skill is the blocking implementation review authority.

## Hard non-negotiables

Reject unless all are true:

1. Direct image training only.
2. Frozen DINO ViT-S/8 with no_grad forward.
3. No feature cache build.
4. No feature cache read.
5. token_compression is none in primary config and model path.
6. Test-only evaluation.
7. Best checkpoint selected only from test.
8. No val loader and no val-best artifacts.
9. No RunC checkpoint/logits/residual.
10. No `frozen_run_c`, `cached_logits`, `tail_residual_adapter`, or old calibration-only branch.
11. No expert/MoE/specialist/router architecture.
12. No selector/evidence deletion loss as primary score objective.
13. No graph delta to action/reason logits.
14. No 16-action-set marginalization as final action.
15. Final action is 4-dimensional sigmoid multi-label logits.
16. Final reason is 21-dimensional sigmoid multi-label logits.
17. Scene predicates are predicted from image tokens and weakly supervised by BDD100K structured annotations.
18. Test forward logits are invariant to BDD100K structured records.
19. Reason grammar contains 21 real reason names and no placeholders.
20. Positive-unlabeled reason loss treats reason y=0 as weighted unknown, not hard negative.
21. Matched counterfactual pair mining exists and logs pair counts per reason.
22. Tail reason pair loss multiplier exists for [12,9,5,14,6,11,10,13].
23. Calibration has BCE/NLL/soft-F1 objective, not only identity regularization.
24. Every epoch writes predicate, pair, tail, threshold, calibration, action-combo, and failure-case artifacts.
25. Foreground supervisor streams stdout/stderr and cannot stop on weak metrics.

## Required files

Reject if any missing:

- configs/fate_oia_train_360x640_acpr_oia_v1.yaml
- configs/acpr_reason_predicate_grammar.yaml
- configs/acpr_scene_predicates.yaml
- fate_oia/models/acpr_dino_field.py
- fate_oia/models/acpr_ego_regions.py
- fate_oia/models/acpr_sparse_ops.py
- fate_oia/models/acpr_predicate_targets.py
- fate_oia/models/acpr_scene_predicate_head.py
- fate_oia/models/acpr_label_trunk.py
- fate_oia/models/acpr_reason_grammar.py
- fate_oia/models/acpr_predicate_reason.py
- fate_oia/models/acpr_pair_memory.py
- fate_oia/models/acpr_action_combo_aux.py
- fate_oia/models/acpr_calibration.py
- fate_oia/models/acpr_oia_model.py
- fate_oia/losses/acpr_losses.py
- fate_oia/engine/train_acpr_oia.py
- fate_oia/engine/eval_acpr_oia.py
- fate_oia/engine/audit_acpr_oia_implementation.py
- fate_oia/engine/export_acpr_visuals.py
- fate_oia/engine/supervise_acpr_oia_foreground.py
- fate_oia/utils/acpr_artifacts.py
- fate_oia/utils/acpr_thresholds.py
- fate_oia/utils/acpr_pair_mining.py
- fate_oia/utils/acpr_predicate_audit.py
- fate_oia/utils/acpr_case_export.py
- scripts/FATE_OIA_acpr_oia_v1_foreground.ps1

## Forbidden source patterns

Reject active implementation if any ACPR file contains or enables:

- frozen_run_c
- FrozenRunC
- run_c_logits
- cached_logits
- complementary_logits
- tail_residual_adapter
- expert
- Expert
- moe
- MoE
- specialist
- Specialist
- router
- Router
- graph_delta_to_logits: true
- action_set_probs @ subset_membership used as final action
- feature_cache_enabled: true
- token_compression: keep_merge
- checkpoint_best_val
- best_selection_split: val
- eval_splits: val
- Start-Process
- Start-Job
- nohup
- hidden cmd
- scheduled task
- daemon

The word "residual" is allowed only for internal neural residual connections, not for cached/RunC residual adaptation.

## Functional audit checks

### 1. Dataset and direct image path

Instantiate BDDOIAMultiTaskDataset train/test with load_image=True.

Verify:
- action target shape [4]
- reason target shape [21]
- targets are multi-hot floats
- no softmax CE over actions
- image tensors are produced
- train/test counts nonzero

### 2. DINO field

Instantiate ACPRDinoFieldExtractor and run one real image.

Verify:
- patch_tokens_by_layer shape [B,3,3600,384]
- cls_tokens_by_layer shape [B,3,384]
- grid_hw=(45,80)
- original_tokens=3601
- all DINO params require_grad=False
- no cache file is written

### 3. Ego regions

Run ACPREgoRegionEncoder.

Verify:
- ego_features shape [3600,8]
- region masks exist: front_center, left_corridor, right_corridor, upper_traffic_region, bottom_drivable_region
- patch tokens are modified by ego projection
- region mass statistics are returned

### 4. Scene predicates

Load configs/acpr_scene_predicates.yaml.

Verify:
- at least 32 predicates
- every predicate has id, name, group, region, bdd100k_sources
- required groups exist: object, lane, drivable, composite, traffic, global
- no placeholder predicate names

Run WeakPredicateTargetBuilder on real file_names and structured records.

Verify:
- predicate_targets shape [B,M]
- predicate_mask shape [B,M]
- source counters include object_box, lane_poly, drivable_map, proxy
- proxy targets are never weighted above 0.15
- missing structured records do not crash

### 5. ScenePredicateHead

Run ACPRScenePredicateHead.

Verify:
- predicate_tokens [B,M,D]
- predicate_logits [B,M]
- predicate_probs [B,M]
- predicate_attention [B,M,3600]
- layer weights [M,3]
- gradients reach predicate queries and heads
- attention uses region prior or exposes why not

### 6. Reason grammar

Load configs/acpr_reason_predicate_grammar.yaml.

Verify:
- exactly 4 actions with names forward/stop/left/right
- exactly 21 reason entries
- no names like reason_0, reason_1, unknown_reason
- each reason has positive_predicates, contradictory_predicates, compatible_actions, hard_negative_reasons, spatial_region
- tail_reason_indices equals [12,9,5,14,6,11,10,13]
- matrices have shapes:
  positive_matrix [21,M]
  contradiction_matrix [21,M]
  compatible_action_matrix [21,4]
  hard_negative_matrix [21,21]

### 7. Label trunk

Run ACPRLabelTrunk.

Verify:
- label_tokens [B,25,D]
- action_visual_logits [B,4]
- reason_logits_visual [B,21]
- action_reason_logits [B,4]
- action_logits_direct [B,4]
- action_fusion_gate [B,4]
- action gate clamped within [0.10,0.90]
- no action-set or predicate branch changes action_logits_direct

### 8. Predicate-conditioned reason

Run ACPRPredicateReasoner.

Verify:
- predicate_reason_delta [B,21]
- contradiction_score [B,21]
- required_support_score [B,21]
- max abs delta <= 0.20
- delta affects reason logits only
- action logits unchanged by this module

### 9. Positive-unlabeled reason loss

Create synthetic reason targets and contradiction_score.

Verify:
- positive reason weights exactly 1.0
- negative weights in [0.2,1.0]
- y=0 labels are not all hard negatives
- increasing contradiction_score increases negative penalty
- gradients reach reason logits

### 10. Matched pair mining

Create synthetic memory and batch.

Verify:
- positives with y_r=1 can mine negatives with y_r=0
- pair miner uses action similarity
- pair miner uses visual/global similarity
- pair miner uses predicate similarity
- pair miner uses contradiction threshold
- tail labels receive multiplier
- pair_count_per_reason is logged
- if no pairs exist, loss returns finite zero and logs no_pair_count

### 11. Pair losses

Verify:
- matched_pair_logit_loss penalizes z_r(x+) <= z_r(x-)
- matched_pair_embedding_loss returns finite loss
- pair weights affect loss
- gradients reach reason logits and pair projection

### 12. Action-combo auxiliary

Run ACPRActionComboAux.

Verify:
- all 16 subsets unique
- forward+right maps to subset id 9
- forward+left maps to subset id 5
- stop+right maps to subset id 10
- action_set_logits [B,16]
- cardinality_logits [B,5]
- action_set output is not used as final action

### 13. Calibration

Run calibration head and toy optimization.

Verify:
- calibrated logits separate from raw logits
- temperature in [0.5,3.0]
- bias in [-2,2]
- calibration loss includes BCE/NLL/soft-F1 objective
- toy training changes at least one bias or temperature
- not identity-only regularizer

### 14. Full model forward

Run ACPROIAModel on real images.

Required output keys:
- action_logits_direct
- reason_logits_visual
- action_visual_logits
- action_reason_logits
- action_fusion_gate
- predicate_logits
- predicate_probs
- predicate_attention
- predicate_reason_delta
- contradiction_score
- required_support_score
- action_logits_raw
- reason_logits_raw
- action_logits_calibrated
- reason_logits_calibrated
- action_set_logits
- cardinality_logits
- branch_logits

Verify:
- action logits shape [B,4]
- reason logits shape [B,21]
- final action equals direct action raw source
- final action does not equal action-set marginal
- fair_test_forward with structured_records does not change logits

### 15. Training protocol

Inspect and smoke train_acpr_oia.

Verify:
- train_loader and test_loader only
- no val_loader
- eval_splits only test
- best_selection_split test
- no checkpoint_best_val
- no val logits
- no feature cache
- no token compression
- every epoch writes required artifacts
- raw/global/per-label/calibrated metrics are logged separately

### 16. Foreground supervisor

Inspect supervise_acpr_oia_foreground.py and PowerShell.

Reject if:
- Start-Process
- Start-Job
- nohup
- hidden/detached process
- scheduled task
- metric early stop

Accept only if:
- stdout/stderr stream foreground
- require_review_pass blocks full training
- OOM fallback exists
- NaN/Inf handling exists
- dataloader stall detection exists
- no metric early stop

## Required preflight commands

Run py_compile:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_dino_field.py `
  fate_oia\models\acpr_ego_regions.py `
  fate_oia\models\acpr_sparse_ops.py `
  fate_oia\models\acpr_predicate_targets.py `
  fate_oia\models\acpr_scene_predicate_head.py `
  fate_oia\models\acpr_label_trunk.py `
  fate_oia\models\acpr_reason_grammar.py `
  fate_oia\models\acpr_predicate_reason.py `
  fate_oia\models\acpr_pair_memory.py `
  fate_oia\models\acpr_action_combo_aux.py `
  fate_oia\models\acpr_calibration.py `
  fate_oia\models\acpr_oia_model.py `
  fate_oia\losses\acpr_losses.py `
  fate_oia\engine\train_acpr_oia.py `
  fate_oia\engine\eval_acpr_oia.py `
  fate_oia\engine\audit_acpr_oia_implementation.py `
  fate_oia\engine\export_acpr_visuals.py `
  fate_oia\engine\supervise_acpr_oia_foreground.py
```

Run tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_dino_field.py `
  tests\test_acpr_ego_regions.py `
  tests\test_acpr_predicate_targets.py `
  tests\test_acpr_scene_predicate_head.py `
  tests\test_acpr_reason_grammar.py `
  tests\test_acpr_label_trunk.py `
  tests\test_acpr_predicate_reason.py `
  tests\test_acpr_partial_label_loss.py `
  tests\test_acpr_pair_mining.py `
  tests\test_acpr_action_combo_aux.py `
  tests\test_acpr_calibration.py `
  tests\test_acpr_model_forward.py `
  tests\test_acpr_train_protocol.py `
  tests\test_acpr_audit.py `
  tests\test_acpr_supervisor_foreground.py -q
```

Run implementation audit:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_oia_implementation `
  --config configs\fate_oia_train_360x640_acpr_oia_v1.yaml `
  --output_dir .background_runs\acpr_oia_v1_preflight `
  --device cuda `
  --write_review_pass
```

Required pass file:

```text
.background_runs\acpr_oia_v1_preflight\REVIEW_PASS_ACPR_OIA_V1.txt
```

Run real tiny smoke:

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_oia `
  --config configs\fate_oia_train_360x640_acpr_oia_v1.yaml `
  --output_dir .background_runs\acpr_oia_v1_smoke `
  --epochs 1 `
  --batch_size 1 `
  --gradient_accumulation_steps 2 `
  --max_train_samples 4 `
  --max_test_samples 4 `
  --device cuda `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression
```

Full training remains forbidden until:
- REVIEW_PASS_ACPR_OIA_V1.txt exists
- current git HEAD equals audit JSON git_head
- smoke completed
- code-only commit/push succeeded or push failure is recorded as auth/TLS issue

## Final review JSON

Write:

```text
.background_runs\acpr_oia_v1_preflight\implementation_audit_ACPR_OIA_V1.json
```

Required fields:
- pass
- git_head
- worktree
- checked_files
- forbidden_pattern_results
- functional_checks
- smoke_result
- review_pass_path
- missing_items
- warnings

## ACPR-CalAlign V1.2 addendum

When the active config is `configs/fate_oia_train_360x640_acpr_calalign_v1_2.yaml` or `threshold.enabled=true`, the audit must additionally reject the implementation unless all of these are true:

1. `ACPRThresholdHead` exists and exposes group shrinkage plus label deltas through `theta_group`, `theta_delta`, `label_group_ids`, and `label_delta_scale`.
2. Deploy logits are exactly `deploy_logits = base_logits - theta`; primary raw/fixed metrics use the `deploy_fixed` branch.
3. `base_fixed` metrics and tensors are still saved separately and are not overwritten by deploy metrics.
4. Teacher thresholds are collected only from a deterministic `train_calib` split of the train split.
5. Test per-label threshold search is named oracle/diagnostic only and is never assigned to `threshold_head`, `theta_teacher`, or checkpoint parameters.
6. Threshold losses include soft-F1, predicted positive rate, action cardinality, teacher, and prior terms.
7. Threshold losses use detached base logits by default so they do not damage ACPR ranking unless explicitly configured otherwise.
8. The old ACPR config with `threshold.enabled=false` still forwards and trains without requiring CalAlign.
9. Action-set outputs remain auxiliary only and never become final action logits.
10. No feature cache, token compression, RunC residual, expert, MoE, selector, or graph-delta path is introduced by CalAlign.

Required CalAlign files:

- `fate_oia/models/acpr_threshold_head.py`
- `fate_oia/losses/acpr_threshold_losses.py`
- `fate_oia/utils/acpr_train_calib_split.py`
- `fate_oia/utils/acpr_threshold_search.py`
- `fate_oia/engine/fit_acpr_threshold_head.py`
- `configs/fate_oia_train_360x640_acpr_calalign_v1_2.yaml`
- `scripts/FATE_OIA_acpr_calalign_v1_2_foreground.ps1`

Required CalAlign artifacts per run:

- `threshold_initialization.json`
- `threshold_teacher_epoch_*.json`
- `threshold_stats.jsonl`
- `calibration_diagnostics.jsonl`
- `logits_action_base_test.pt`
- `logits_reason_base_test.pt`
- `logits_action_deploy_test.pt`
- `logits_reason_deploy_test.pt`
- `checkpoint_best_test_deploy_raw.pth`
- `checkpoint_best_test_base_fixed.pth`

## ACPR-ActAlign V1.3.1 Candidate-Probe addendum

When the active config is `configs/fate_oia_train_360x640_acpr_actalign_v1_3_candidate_probe.yaml` or `actalign.stage_mode=candidate_probe`, the audit must additionally reject the implementation unless all of these are true:

1. `ACPRActionCandidates` exists and outputs `fallback`, `visual`, `reason`, `blend`, `predicate`, `blend_predicate`, and `utility_final` action logits.
2. Gate zero is an exact invariant: `action_logits_utility == action_logits_fallback` and final reason logits are unchanged.
3. Stage A trains only candidate heads and predicate micro-delta; trunk, DINO, reason, predicate head, threshold head, and calibration stay frozen.
4. Gate update uses `train_calib` only; test metrics must never update gate, thresholds, or selected candidate state.
5. `candidate_probe.candidate_weight=0.5`, `candidate_probe.nonreg_weight=0.5`, `candidate_probe.gate_ema=0.20`, and `training.lr_action_candidate=0.0005`.
6. `allow_reason_candidate` and `allow_predicate_candidate` are explicit config and gate parameters.
7. Stage A PASS requires all hard gates: nonzero selected gate, per-action delta >= 0.002, gated train_calib Act_mF1 gain >= 0.001, train_calib Exp drop <= 0.005, all_high increase <= 0.02, test Act drop <= 0.005, and test Exp drop <= 0.010.
8. Stage B must be blocked unless a `STAGE_A_CANDIDATE_PROBE_PASS.json` artifact is loaded with `--load_candidate_gate`.
9. Clean full train must remain blocked until Stage A and Stage B both have explicit PASS artifacts.
10. Candidate artifacts must include `action_candidate_train_calib.jsonl`, `action_candidate_metrics.jsonl`, `action_candidate_gate.jsonl`, `candidate_selected_state.json`, and tensor logits for each candidate branch.
11. Evaluation must expose candidate action metrics through `--evaluate_action_candidates`, `--candidate_gate_json`, and `--output_action_candidate_metrics`.
12. Audit must write `implementation_audit_ACPR_ACTALIGN_V1_3_1.json` and `REVIEW_PASS_ACPR_ACTALIGN_V1_3_1.txt` for this config.

Required candidate-probe files:

- `fate_oia/models/acpr_action_candidates.py`
- `fate_oia/losses/acpr_candidate_losses.py`
- `fate_oia/utils/acpr_candidate_gate.py`
- `fate_oia/utils/acpr_candidate_metrics.py`
- `fate_oia/engine/fit_acpr_action_candidates.py`
- `configs/fate_oia_train_360x640_acpr_actalign_v1_3_candidate_probe.yaml`
- `tests/test_acpr_action_candidates.py`
- `tests/test_acpr_candidate_losses.py`
- `tests/test_acpr_candidate_gate.py`
- `tests/test_acpr_actalign_candidate_forward.py`
- `tests/test_acpr_actalign_stageA_protocol.py`
