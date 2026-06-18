# ACPR-SECA V1 Implementation Audit Skill

## Purpose

Audit that ACPR-SECA V1 is a complete, efficient, direct-image implementation
of Sparse Evidence Co-Attention on top of the strong ACPR-CalAlign V1.2 model.

The skill is a blocking gate. It must reject incomplete, placeholder,
semantically incorrect, or unnecessarily expanded implementations.

The central semantic chain is:

```text
DINO image patches
  -> predicate vectors
  -> predicate-enriched explanation vectors
  -> sparse action-specific explanation reading
  -> existing action head
  -> existing CalAlign
```

ACPR-SECA must preserve and use:

- predicate-conditioned PU reason learning
- reason-specific HardPair learning
- CalAlign train-calib thresholds
- direct four-label sigmoid action output
- direct 21-label sigmoid explanation output

It must not add a new training loss.

---

## Mandatory remote context

Before code changes, tests, training, evaluation, pushing, or process
management under `E:\sbw\FATE_Drive`, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

Do not create additional experiment-status Markdown files.
Append status only to the three canonical files.

Use Superpowers planning, TDD, review, debugging, and verification skills when
available. This audit remains blocking even if another review passes.

---

## Required branch and worktree

Expected source:

```text
github/acpr_calalign_v1_2
```

Expected new branch:

```text
acpr_seca_v1
```

Expected worktree:

```text
E:\sbw\FATE_Drive\fate_oia_acpr_seca_v1_worktree
```

Reject if formal training runs inside the source ACPR-CalAlign worktree.

Audit records must contain:

- source branch SHA
- current git HEAD
- worktree path
- branch name
- clean/dirty status
- remote branch SHA
- whether local and remote HEAD match

---

## Hard non-negotiables

Reject unless every item is true.

1. Direct image training only.
2. Frozen DINO ViT-S/8 with no-grad backbone forward.
3. No feature cache creation or reading.
4. No token compression.
5. Test-only end-of-epoch evaluation.
6. Best checkpoint selected from test deploy-fixed joint.
7. No validation loader, validation metric, or validation-best checkpoint.
8. Final action is four independent sigmoid logits.
9. Final explanation is 21 independent sigmoid logits.
10. Action-set output remains auxiliary only.
11. ACPR predicate head remains active.
12. Existing predicate-conditioned PU reason objective remains active.
13. Existing reason-specific HardPair mining remains active.
14. HardPair weighted contribution is budgeted relative to action+reason main loss.
15. Existing global per-label CalAlign remains active.
16. Test oracle thresholds are diagnostic only.
17. Train-calib teacher uses best-so-far locking.
18. SECA uses high-dimensional explanation vectors, not reason probabilities only.
19. SECA uses predicate-enriched explanation vectors.
20. Every action independently queries explanations.
21. A null explanation token exists.
22. Action-to-reason attention uses entmax15.
23. SECA has a bounded zero-initialized trainable residual gate.
24. Zero-gate forward is equivalent to ACPR-CalAlign.
25. Zero-gate startup is not gradient-dead.
26. Action-loss gradient into reason/predicate features is scaled, not fully detached.
27. No new SECA/attention/correlation loss exists.
28. No PMT patch supervision exists in the formal trainer.
29. No predicate-conditioned threshold MLP exists.
30. No candidate, FusionLite, graph, expert, MoE, specialist, router, or cached residual path.
31. No test metric updates model parameters, thresholds, teacher, LR control, or cooldown.
32. Full training uses an attached foreground supervisor.
33. Weak metrics never cause early stopping.
34. Review pass is tied to the exact clean pushed HEAD.

---

## Required files

Reject if any are missing.

### New SECA files

- `configs/fate_oia_train_360x640_acpr_seca_v1.yaml`
- `fate_oia/models/acpr_semantic_evidence_coattention.py`
- `fate_oia/utils/acpr_pair_budget.py`
- `fate_oia/utils/acpr_seca_training_control.py`
- `fate_oia/utils/acpr_seca_artifacts.py`
- `fate_oia/engine/audit_acpr_seca_implementation.py`
- `fate_oia/engine/eval_acpr_seca_faithfulness.py`
- `fate_oia/engine/export_acpr_seca_visuals.py`
- `fate_oia/engine/supervise_acpr_seca_foreground.py`
- `scripts/FATE_OIA_acpr_seca_v1_foreground.ps1`

### Required inherited ACPR-CalAlign files

- `fate_oia/models/acpr_dino_field.py`
- `fate_oia/models/acpr_ego_regions.py`
- `fate_oia/models/acpr_sparse_ops.py`
- `fate_oia/models/acpr_scene_predicate_head.py`
- `fate_oia/models/acpr_predicate_targets.py`
- `fate_oia/models/acpr_label_trunk.py`
- `fate_oia/models/acpr_predicate_reason.py`
- `fate_oia/models/acpr_pair_memory.py`
- `fate_oia/models/acpr_action_combo_aux.py`
- `fate_oia/models/acpr_threshold_head.py`
- `fate_oia/models/acpr_oia_model.py`
- `fate_oia/losses/acpr_losses.py`
- `fate_oia/losses/acpr_threshold_losses.py`
- `fate_oia/utils/acpr_train_calib_split.py`
- `fate_oia/utils/acpr_threshold_search.py`
- `fate_oia/engine/train_acpr_oia.py`
- `fate_oia/engine/eval_acpr_oia.py`

### Required tests

- `tests/test_acpr_seca_module.py`
- `tests/test_acpr_seca_equivalence.py`
- `tests/test_acpr_seca_gradient_flow.py`
- `tests/test_acpr_seca_integration.py`
- `tests/test_acpr_seca_pair_budget.py`
- `tests/test_acpr_seca_teacher_lock.py`
- `tests/test_acpr_seca_scheduler.py`
- `tests/test_acpr_seca_artifacts.py`
- `tests/test_acpr_seca_faithfulness.py`
- `tests/test_acpr_seca_audit.py`
- `tests/test_acpr_seca_supervisor.py`
- `tests/test_acpr_seca_performance.py`

---

## Forbidden active source patterns

Reject active formal implementation containing or enabling:

```text
acpr_triadic_mediator
predicate_patch_targets
predicate_transport_alignment
predicate_conditioned_threshold
predicate_filtered_hardpair
acpr_action_candidates
acpr_action_utility
acpr_fusionlite
frozen_run_c
FrozenRunC
cached_logits
run_c_logits
tail_residual_adapter
complementary_logits
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

The string `residual` is allowed for neural residual connections and the SECA
ReZero residual only. It is not allowed for cached-logit or old-checkpoint
adaptation.

---

## Static architecture audit

### 1. SECA constructor

Verify `ACPRSparseEvidenceCoAttention` has:

- dimension 384
- action_dim 4
- reason_dim 21
- four heads
- q/k/v/out projections
- pre-norm action and reason
- learned null reason token
- one trainable gate per action
- max residual scale 0.20
- evidence gradient scale 0.25

Reject if:

- reason logits are the sole key/value evidence
- predicate probabilities directly produce action logits
- ground-truth reasons enter model forward
- a hard action-reason grammar mask is used
- softmax is used instead of entmax15 for action-to-reason attention

### 2. Tensor shapes

Synthetic forward must produce:

```text
action_nodes_seca:                 [B,4,384]
action_reason_attention_heads:    [B,4,4,22]
action_reason_attention:          [B,4,22]
action_reason_attention_no_null:  [B,4,21]
null_attention:                   [B,4]
evidence_context:                 [B,4,384]
residual_scale:                   [4]
```

Attention rows must sum to one within `1e-5`.

At least one synthetic non-degenerate case must contain exact zero attention
entries from entmax.

### 3. Bounded gate

Verify:

```python
residual_scale = max_scale * torch.tanh(residual_gate_raw)
```

or a mathematically equivalent bounded zero-centered parameterization.

Reject if gate initialization changes the source model forward.

### 4. Controlled gradient bridge

Verify forward identity:

```python
reason_for_action == reason_nodes
```

Verify backward scale numerically:

```text
grad(reason_nodes through action path)
approximately equals evidence_grad_scale times unscaled reference gradient
```

Tolerance should account for floating point error.

Reject complete detach and reject unbounded full action gradient if config
requires 0.25.

---

## Dynamic equivalence audit

Build:

- source-style ACPR-CalAlign model with SECA disabled
- ACPR-SECA model with identical inherited weights and zero gate

Use the same deterministic mock-DINO input.

Verify all:

```text
reason_logits_base
action_visual_logits
action_reason_logits
action_logits_base
action_logits_deploy
reason_logits_deploy
threshold_logit
```

Equivalent within:

```text
atol=1e-6
rtol=1e-6
```

Also run a real CUDA image smoke with tolerance `1e-5`.

Reject if equivalence is achieved by skipping module construction or by an
external fallback branch.

---

## Non-dead gradient audit

Use nontrivial action targets.

Step 1:

- zero gradients
- forward with gate zero
- backpropagate action loss

Require:

```text
residual_gate_raw.grad is finite
abs-sum > 0
```

Take one optimizer step.

Require:

```text
at least one residual_gate_raw != 0
```

Step 2:

- forward again
- backpropagate

Require finite nonzero gradients for:

```text
q_proj.weight
k_proj.weight
v_proj.weight
out_proj.weight
```

This check blocks the previous zero-gate deadlock failure.

---

## Integration audit

Inspect `ACPRLabelTrunk`.

Required order:

```text
image label nodes
-> label self-attention
-> reason reads predicate vectors
-> action reads predicate-enriched reason vectors through SECA
-> current action visual head
-> current reason-to-action branch
-> current fusion gate
```

Verify:

- reason logits still use predicate-enriched reason nodes
- SECA updates action vectors, not final logits directly
- existing `reason_to_action` remains
- existing fusion gate remains
- action-set output remains auxiliary
- action logits are not post-hoc candidate selection
- legacy diagnostic outputs do not receive an extra training loss

---

## PU audit

Verify only one PU/partial-label reason objective is active.

Require:

- positive reason weights are full positives
- y=0 reasons are weighted unknown
- contradiction evidence modulates negative weight
- no second `predicate_conditioned_pu` loss
- gradients reach reason logits

---

## HardPair budget audit

Create synthetic:

```text
L_action = 0.10
L_reason = 0.20
weighted pair raw = 0.50
ratio = 0.25
```

Expected cap:

```text
0.25 * (0.10 + 0.20) = 0.075
```

Require:

```text
pair_used <= 0.075 + 1e-6
```

Also test a below-cap case where pair loss is unchanged.

Verify total loss adds only budgeted pair loss.
Reject double addition of raw and budgeted pair terms.

Required runtime logs:

- pair weighted raw
- pair weighted used
- cap
- scale
- raw ratio
- used ratio
- cap active

---

## CalAlign audit

Verify source semantics remain:

```text
deploy = base - theta
```

Require:

- group shrinkage
- label delta
- action/common/tail groups
- threshold ranges
- base detach by default
- train-calib split
- test oracle diagnostic only
- action set does not determine final action

### Teacher best lock

Feed candidate teacher scores:

```text
epoch 1 joint 0.50 -> accepted
epoch 2 joint 0.49 -> rejected
epoch 3 joint 0.51 -> accepted
```

Require:

- teacher after epoch 2 equals epoch 1 teacher
- teacher after epoch 3 equals epoch 3 teacher
- accepted/rejected states are logged
- no test metric is read by update function

---

## Scheduler and cooldown audit

Configuration must resolve to:

```text
epochs=14
warmup_epochs=2
scheduler=warmup_cosine_by_update
min_lr_ratio=0.05
```

Verify per-update multiplier:

- increases during warmup
- decreases monotonically after warmup
- ends near 0.05
- uses optimizer updates, not raw batches

Cooldown test:

- monitor train-calib joint only
- minimum epoch 6
- patience 2
- one-time trigger
- non-threshold LR multiplier 0.20
- threshold LR multiplier 0.50
- does not stop training
- persists across checkpoint resume

Reject use of test metrics for cooldown.

---

## No-new-loss audit

Inspect total objective.

Allowed source objectives:

- action direct
- action visual auxiliary
- action reason auxiliary
- PU reason partial
- reason soft-F1
- weak predicate
- predicate-reason alignment
- budgeted HardPair logit/embed
- action-combo auxiliary
- action-count auxiliary
- source calibration
- CalAlign threshold bundle
- predicate attention compactness if already in source config

Reject new terms named or functioning as:

- seca loss
- attention alignment
- attention entropy
- correlation loss
- triadic consistency
- patch alignment
- predicate-conditioned PU duplicate
- dynamic threshold delta regularization

SECA must learn from existing action/reason objectives.

---

## Data and runtime audit

Verify:

- images are loaded directly
- no feature cache
- no token compression
- formal num_workers=4
- persistent workers active when workers > 0
- prefetch factor 2
- tiny smoke uses workers 0

Formal primary batch:

```text
batch 5
grad accumulation 6
effective batch 30
```

Fallback ladder:

```text
[5,6], [4,8], [3,10], [2,15]
```

---

## Performance audit

Run a real 16-sample benchmark for source-style ACPR-CalAlign and ACPR-SECA
under the same batch, device, and number of warmup iterations.

Use medians after warmup.

Require:

```text
SECA median step-time overhead <= 10%
SECA peak allocated increase <= 2.0 GB
```

Write:

```text
.background_runs/acpr_seca_v1_preflight/performance_audit.json
```

Reject formal training if the limit is exceeded.

Do not pass using mock DINO only.

---

## Evaluation artifact audit

Every epoch must contain:

- `metrics_summary.json`
- base/deploy/calibrated branch metrics
- action and reason labels
- legacy base action logits
- SECA base action logits
- deploy action/reason logits
- compact action-to-reason attention
- compact reason-to-predicate attention
- SECA metrics JSON
- threshold stats
- pair-budget stats
- training-control state
- evidence chain JSONL
- lightweight faithfulness JSON

Required checkpoint files:

- `checkpoint_best_test_deploy_joint.pth`
- `checkpoint_best_test_action_mf1.pth`
- `checkpoint_best_test_exp_mf1.pth`
- `checkpoint_best_test_exp_map.pth`
- `checkpoint_best_test_base_joint.pth`
- `checkpoint_latest.pth`

Best selection must not use test oracle per-label threshold metrics.

---

## Visualization audit

Required evidence chain:

```text
action -> original BDD-OIA reason -> predicate -> DINO patch
```

Verify:

- action-to-reason weights come from SECA
- reason-to-predicate weights come from existing predicate cross-attention
- predicate-to-patch weights come from predicate head attention
- matrices are multiplied only for visualization/diagnostics
- no label truth is used to construct predicted evidence
- report shows ground truth separately from predicted evidence
- top reasons and null mass are visible

Do not claim attention alone is causal explanation.

---

## Faithfulness audit

Faithfulness is evaluation-only.

Required:

- top-reason deletion
- random same-count reason deletion
- top-reason sufficiency
- deterministic case subset
- no gradient
- no training update
- no metric feedback into training

Require:

```text
deletion_margin =
top_reason_deletion_drop - random_reason_deletion_drop
```

Write per-case and aggregate results.

---

## Foreground supervisor audit

Reject if source contains:

- `Start-Process`
- `Start-Job`
- `nohup`
- hidden window
- daemon
- scheduled task
- detached process

Require:

- line-by-line stdout/stderr streaming
- exact-head review-pass check
- real tiny smoke before formal run
- GPU memory query
- OOM-only fallback
- fresh attempt directory per fallback
- heartbeat
- stall/liveness checks
- no metric early stop
- full 14-epoch completion check
- truthful completion JSON
- failed attempts preserved
- non-OOM failure surfaced for Codex diagnosis

The supervisor may not stop because the model is below the historical best.

---

## Required compile command

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_semantic_evidence_coattention.py `
  fate_oia\models\acpr_label_trunk.py `
  fate_oia\models\acpr_oia_model.py `
  fate_oia\losses\acpr_losses.py `
  fate_oia\utils\acpr_pair_budget.py `
  fate_oia\utils\acpr_seca_training_control.py `
  fate_oia\utils\acpr_seca_artifacts.py `
  fate_oia\engine\train_acpr_oia.py `
  fate_oia\engine\eval_acpr_oia.py `
  fate_oia\engine\eval_acpr_seca_faithfulness.py `
  fate_oia\engine\export_acpr_seca_visuals.py `
  fate_oia\engine\audit_acpr_seca_implementation.py `
  fate_oia\engine\supervise_acpr_seca_foreground.py
```

---

## Required test command

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_seca_module.py `
  tests\test_acpr_seca_equivalence.py `
  tests\test_acpr_seca_gradient_flow.py `
  tests\test_acpr_seca_integration.py `
  tests\test_acpr_seca_pair_budget.py `
  tests\test_acpr_seca_teacher_lock.py `
  tests\test_acpr_seca_scheduler.py `
  tests\test_acpr_seca_artifacts.py `
  tests\test_acpr_seca_faithfulness.py `
  tests\test_acpr_seca_audit.py `
  tests\test_acpr_seca_supervisor.py `
  tests\test_acpr_seca_performance.py -q
```

Also run all inherited ACPR-CalAlign tests.

---

## Required audit command

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_seca_implementation `
  --config configs\fate_oia_train_360x640_acpr_seca_v1.yaml `
  --output_dir .background_runs\acpr_seca_v1_preflight `
  --device cuda `
  --write_review_pass
```

Required files:

```text
.background_runs\acpr_seca_v1_preflight\implementation_audit_ACPR_SECA_V1.json
.background_runs\acpr_seca_v1_preflight\performance_audit.json
.background_runs\acpr_seca_v1_preflight\REVIEW_PASS_ACPR_SECA_V1.txt
```

The pass file must contain the exact git HEAD.

Any code or formal config change invalidates the pass.

---

## Required real tiny smoke

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_oia `
  --config configs\fate_oia_train_360x640_acpr_seca_v1.yaml `
  --output_dir .background_runs\acpr_seca_v1_smoke `
  --epochs 1 `
  --batch_size 1 `
  --gradient_accumulation_steps 2 `
  --max_train_samples 8 `
  --max_test_samples 8 `
  --num_workers 0 `
  --device cuda `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression
```

The smoke is a structural test, not a performance result.

---

## Final review JSON schema

Write:

```text
.background_runs\acpr_seca_v1_preflight\implementation_audit_ACPR_SECA_V1.json
```

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
- `static_architecture_checks`
- `equivalence_checks`
- `gradient_checks`
- `pu_checks`
- `hardpair_budget_checks`
- `calalign_checks`
- `scheduler_checks`
- `artifact_checks`
- `visualization_checks`
- `faithfulness_checks`
- `supervisor_checks`
- `performance_checks`
- `smoke_result`
- `warnings`
- `review_pass_path`

`pass=true` only if every blocking check passes.

---

## Formal training authorization

Formal training is authorized only after:

1. code-only commit exists
2. branch is pushed
3. local and remote HEAD match
4. worktree is clean
5. exact-head audit passes
6. performance gate passes
7. real tiny smoke passes
8. `REVIEW_PASS_ACPR_SECA_V1.txt` exists

Formal command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_seca_v1_foreground.ps1 `
  -Epochs 14 `
  -BatchSize 5 `
  -GradAccum 6 `
  -NumWorkers 4 `
  -Device cuda `
  -RequireReviewPass
```

Training must remain attached in the foreground.

---

## Completion verdict

This skill distinguishes:

### Implementation pass

All planned semantics, tests, audit, smoke, performance, and exact-head checks pass.

### Training complete

All 14 epochs complete with required test evaluation, checkpoints, and artifacts.

### Research success

Only after results exceed or meaningfully improve the strong ACPR-CalAlign
baseline in action, explanation, joint, stability, or faithfulness.

Never equate implementation pass with research success.
