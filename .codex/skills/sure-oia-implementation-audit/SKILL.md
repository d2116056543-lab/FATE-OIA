# SURE-OIA Implementation Audit Skill

Use this skill for any Codex task implementing SURE-OIA v2.

The purpose is to prevent superficial implementation: code must execute the actual
method, not merely create files or placeholder artifacts.

## Required mindset

This is a GOAL-MODE audit. The task is not done after code writing, smoke, or
preflight. It is done only when the reviewed implementation launches and
foreground-supervises the planned full training run.

## Mandatory checks before training

### 1. Worktree isolation
- Confirm current path is `E:\sbw\FATE_Drive\fate_oia_sure_oia_v2_worktree`.
- Confirm branch is `sure_oia_v2_direct_image`.
- Confirm source main worktree was not modified.
- Confirm `.background_runs`, checkpoints, logits, datasets are not tracked.

### 2. Context compliance
- Confirm `E:\sbw\FATE_Drive\task_plan.md`, `findings.md`, and `progress.md` were read.
- Confirm no new training-status Markdown files were created.
- Confirm durable status will be appended only to those three Markdown files.

### 3. Config invariants
The resolved config must contain:
- `config_version: sure_oia_v2_direct_image`
- `feature_cache.enabled: false`
- `token_compression: none`
- `test_only_evaluation: true`
- `best_selection_split: test`
- `epochs: 24`
- direct image training enabled
- batch fallback plan present
- `bdd100k_root: E:/sbw/BDD100K`

Training cannot start if any invariant is false.

### 4. BDD100K parser
Audit must prove:
- path discovery under `E:\sbw\BDD100K` works.
- label/drivable base-stem match rate is at least 0.90 on sampled train/test.
- parser reads JSON top-level `attributes`, `frames`, `name`.
- parser descends into `frames[0].labels`.
- sampled label count is nonzero.
- object box2d count is nonzero.
- lane/poly2d candidates are reported.
- drivable map count is nonzero.
- semantic segmentation is treated as diagnostic-only.

Reject implementations that only read top-level `labels`.

### 5. Fairness boundary
Primary test path must be image-only.
BDD100K GT structured annotations may be used for:
- training-time relation teacher warmup,
- diagnostic/audit,
- GT-scene upper-bound branch.

BDD100K GT must not be fed into:
- `action_logits`
- `reason_logits`
for the primary image-only result.

Audit must compare forward outputs with and without `structured` in image-only mode;
primary logits must be unchanged.

### 6. Relation proposer
Must implement:
- learnable relation queries
- cross-attention to image/DINO tokens
- source type prediction
- target-category utility prediction
- spatial support or reference support
- GT structured relation token builder for training/audit

Reject if relation proposer is a placeholder returning zeros or random tensors.

### 7. Sparse heterogeneous relation attention
Must implement:
- candidate edge construction
- edge type list
- HGT-style typed Q/K/V or explicit type-specific projections
- Graphormer-style edge/geometry/source/uncertainty bias
- top-k sparse routing
- global and per-category edge budgets
- selected_edges_count < candidate_edges_count
- selected_edges_count <= configured budget

Audit must perturb selected and unselected relation tokens and confirm selected
tokens produce larger average target-logit effect.

### 8. Uncertainty memory gate
Must implement:
- base logit entropy/margin uncertainty
- category uncertainty gate
- relation memory slots by edge type
- gate decreases when base logits are artificially confident
- gate increases when base logits are uncertain

Reject always-on memory or always-zero memory.

### 9. Residual safety
Must implement:
- bounded action residual cap <= 0.06
- bounded reason residual cap <= 0.16, tail <= 0.22
- cap schedule
- dynamic cap by base logit EMA
- action_safe_mode if final action is worse than base action by configured threshold
- audit that residual absolute max does not exceed cap

Reject unconstrained residuals.

### 10. Loss discipline
Primary loss may include only:
- action ASL
- reason ASL
- small scheduled relation teacher warmup loss

Must not include as primary:
- tail rank
- sigmoid F1
- counterfactual logit loss
- edge BCE
- teacher distillation from RunC/TRACE
- PMI or static cooccurrence bias
- action candidate selector loss

GradNorm must update action/reason weights and clamp to [0.7, 1.5].

### 11. Test-only eval
- No validation loader should be active.
- No `checkpoint_best_val` should be written.
- Best checkpoint selected by test metric only.
- Epoch end evaluates test only.

### 12. Foreground supervisor
Reject script if it contains:
- Start-Process
- Start-Job
- Win32_Process
- Invoke-WmiMethod
- nohup
- hidden/detached/background logic

Supervisor must:
- stream stdout/stderr
- handle OOM by batch fallback
- handle dataloader crash by reducing workers
- handle nonfinite loss by restore/reduce LR once
- handle stall by structured restart
- never stop due to low metrics
- run in goal mode until training completes or user stops

### 13. Required artifacts
Tiny smoke and each real epoch must write:
- metrics_summary.json
- metrics_raw_fixed.json
- branch_metrics.json
- relation_stats.json
- gradnorm_stats.json
- bdd100k_structured_stats.json
- action_safe_stats.json
- loss_components.jsonl
- failure_cases.jsonl
- logits/action_final_test.pt
- logits/action_base_test.pt
- logits/reason_final_test.pt
- logits/reason_base_test.pt
- logits/labels_action_test.pt
- logits/labels_reason_test.pt
- logits/file_names_test.json
- sure_visuals_index.jsonl
- selected-vs-random edge deletion diagnostic

Reject placeholder tensors such as all-zero `transport_topk_test.pt` without actual selected edges.

### 14. Review pass
Write:
`.background_runs\sure_oia_v2_preflight\REVIEW_PASS_SURE_OIA_V2.txt`

Only if every check above passes.

Contents must include exactly:
`REVIEW_PASS_SURE_OIA_V2`

No full training may launch without this file.

## Final goal completion

Full task complete only if:
- implementation audit passes,
- foreground full training launched,
- 24 completed epochs or explicit user stop,
- `GOAL_COMPLETED_SURE_OIA_V2.json` is present in output directory,
- canonical three Markdown context files are updated.
