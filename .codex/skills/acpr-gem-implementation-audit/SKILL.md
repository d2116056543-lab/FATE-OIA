# ACPR-GEM V1 Implementation Audit Skill

## Purpose

Audit the implementation of:

**ACPR-GEM V1 — Predicate-Consistent Grounded Evidence Memory**

The method must create a shared, grounded, intervenable evidence memory for
BDD-OIA action and explanation decisions. It must preserve ACPR-CalAlign's
strong path and add evidence tokens as a controlled supplemental memory.

GEM is not:
- a logit residual,
- a visual token adapter,
- a candidate branch,
- a graph delta,
- an expert router,
- an action-set final classifier,
- a patch-mask-heavy PMT branch.

This audit blocks formal training until all code, gate, and runtime requirements
are satisfied.

---

## 1. Required context

Before running remote tasks under `E:\sbw\FATE_Drive`, Codex must read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

No repo-local training status markdown files may be created.

Expected source branch:

```text
github/acpr_calalign_v1_2
```

Expected new branch:

```text
acpr_gem_v1
```

Expected worktree:

```text
E:\sbw\FATE_Drive\fate_oia_acpr_gem_v1_worktree
```

---

## 2. Blocking pass conditions

Reject unless all are true.

1. Direct-image training only.
2. Frozen no-grad DINO ViT-S/8.
3. No feature cache.
4. No token compression.
5. Final action is four independent sigmoid logits.
6. Final explanation is 21 independent sigmoid logits.
7. Test-only epoch evaluation.
8. Best primary checkpoint uses test deploy-fixed joint.
9. Action-set head remains auxiliary only.
10. ACPR-CalAlign original patch path remains active.
11. GEM evidence tokens are supplemental, not replacements for raw DINO tokens.
12. Evidence tokens are consumed by both action and reason label decoding.
13. Evidence tokens are consumed by predicate decoding.
14. Evidence memory uses fixed named evidence slots.
15. Evidence attention uses entmax15 or top-k sparse attention.
16. Initial GEM output is exactly equivalent to ACPR-CalAlign.
17. Equivalence is achieved through zero-output projections, not bypassing modules.
18. First backward gives nonzero evidence output projection gradients.
19. Learned evidence attention is grounded to object/lane/drivable when available.
20. Oracle evidence mode exists for gates only.
21. Oracle evidence mode is forbidden in formal training/test inference.
22. BDD100K masks are not stored as persistent feature caches.
23. PU-consistent explanation losses are active.
24. Weak predicate target cleanup is active.
25. HardPair is reason-specific and budgeted.
26. Predicate-filtered HardPair is not reintroduced.
27. CalAlign remains global per-label deploy=base-theta.
28. CalAlign teacher best-lock is true and blocks rejected candidates before update.
29. No test metric updates training state.
30. Faithfulness/deletion is evaluation-only.
31. Full train is foreground-attached.
32. Weak metrics never stop training.
33. Exact clean pushed HEAD audit passes.

---

## 3. Required files

### New files

- `configs/acpr_gem_evidence_slots.yaml`
- `configs/fate_oia_train_360x640_acpr_gem_v1.yaml`
- `fate_oia/models/acpr_grounded_evidence_memory.py`
- `fate_oia/grounding/acpr_gem_grounding.py`
- `fate_oia/utils/acpr_gem_artifacts.py`
- `fate_oia/utils/acpr_gem_teacher_lock.py`
- `fate_oia/utils/acpr_pair_budget.py`
- `fate_oia/utils/acpr_gem_training_control.py`
- `fate_oia/engine/audit_acpr_gem_implementation.py`
- `fate_oia/engine/audit_acpr_gem_gates.py`
- `fate_oia/engine/probe_acpr_gem_memory.py`
- `fate_oia/engine/eval_acpr_gem_faithfulness.py`
- `fate_oia/engine/export_acpr_gem_visuals.py`
- `fate_oia/engine/supervise_acpr_gem_foreground.py`
- `scripts/FATE_OIA_acpr_gem_v1_foreground.ps1`
- `.codex/skills/acpr-gem-implementation-audit/SKILL.md`

### Tests

- `tests/test_acpr_gem_evidence_memory.py`
- `tests/test_acpr_gem_oracle_pooler.py`
- `tests/test_acpr_gem_grounding.py`
- `tests/test_acpr_gem_trunk_integration.py`
- `tests/test_acpr_gem_predicate_integration.py`
- `tests/test_acpr_gem_equivalence.py`
- `tests/test_acpr_gem_gradient_flow.py`
- `tests/test_acpr_gem_pu_losses.py`
- `tests/test_acpr_gem_predicate_targets.py`
- `tests/test_acpr_gem_pair_budget.py`
- `tests/test_acpr_gem_teacher_lock.py`
- `tests/test_acpr_gem_gates.py`
- `tests/test_acpr_gem_faithfulness.py`
- `tests/test_acpr_gem_visualization.py`
- `tests/test_acpr_gem_audit.py`
- `tests/test_acpr_gem_supervisor.py`
- `tests/test_acpr_gem_memory_probe.py`

---

## 4. Forbidden active patterns

Reject if formal branch contains or enables:

```text
acpr_visual_token_adapter
acpr_predicate_action_coupling
acpr_semantic_evidence_coattention
acpr_triadic_mediator
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

A normal residual inside a neural layer is allowed. A cached-logit residual
adapter is not.

---

## 5. Evidence slot audit

Verify `configs/acpr_gem_evidence_slots.yaml` contains named slots.

Required slot families:

- front/left/right object
- pedestrian or rider
- traffic control
- front obstacle
- crosswalk or intersection
- front/left/right lane boundary
- center/left/right drivable
- context slots

Each slot must specify:
- group
- grounding sources
- region prior
- reliability
- whether oracle pooling is allowed

Reject anonymous-only slots.

Default slot count must be 16-24, with default 20.

---

## 6. Evidence memory architecture audit

Inspect `ACPRGroundedEvidencePooler`.

Required:

- learned evidence queries [M,384]
- query/key/value projections
- patch memory from DINO selected-layer mean
- spatial region prior
- entmax15 or top-k entmax attention
- evidence_tokens [B,M,384]
- evidence_attention [B,M,N]
- per-slot names/groups in output
- grounding target/mask in output when available

Reject:
- average pooling all patches as every evidence token;
- softmax-only full dense attention without sparsity option;
- using ground-truth masks at test inference;
- replacing DINO tokens with evidence tokens.

---

## 7. Oracle evidence audit

Oracle pooler must:

- accept [B,M,N] masks;
- pool DINO tokens by normalized masks;
- return oracle_available flags;
- fallback to learned queries only when mask unavailable;
- be disabled in formal training/test inference.

Gate B must use oracle mode.
Formal config must set oracle_mode=false.

---

## 8. Label trunk integration audit

Inspect `ACPRLabelTrunk`.

Required source path remains:

```text
label queries -> patch attention -> label self-attention -> predicate conditioning
```

Then evidence path:

```text
label nodes attend to evidence tokens
zero-init evidence output projection
bounded evidence delta
label_nodes = label_nodes_patch + evidence_delta
```

Both action and reason labels must consume evidence.

Required outputs:

- label_nodes_patch
- label_nodes_evidence_context
- label_nodes_evidence_delta
- label_evidence_attention [B,25,M]
- action_evidence_attention [B,4,M]
- reason_evidence_attention [B,21,M]
- label_evidence_delta_norm

Zero-init equivalence must pass.

Reject if evidence is used only for explanations or only for actions.

---

## 9. Predicate head integration audit

Inspect `ACPRScenePredicateHead`.

Predicate queries must read evidence tokens through a zero-init residual.

Required outputs:

- predicate_tokens_patch
- predicate_tokens_evidence_context
- predicate_evidence_attention [B,P,M]
- predicate_evidence_delta_norm

Reject if predicate head ignores evidence tokens.
Reject if predicate scalar probabilities directly change action logits.

---

## 10. Full forward-path audit

Inspect `ACPROIAModel.forward`.

Required order:

```text
DINO once
ego region from raw DINO layer 0
evidence memory from raw patch tokens
predicate head with evidence tokens
label trunk with evidence tokens and predicate tokens
predicate reasoner
CalAlign
```

DINO forward count must be one.

Final prediction uses:
- raw patch path plus evidence memory inside trunk/predicate head;
- no oracle masks;
- no branch voting.

---

## 11. Equivalence audit

With evidence output projections zero:

Compare GEM and ACPR-CalAlign.

Required equal:

- action logits base
- reason logits base
- deploy logits
- calibrated logits
- predicate probabilities
- action-combo logits

Tolerance:

- mock CPU: 1e-6
- real CUDA: 1e-5

The evidence modules must execute. Do not bypass them.

---

## 12. Gradient audit

First backward:
- evidence output projection gradient nonzero.

After one optimizer step:
- output projection leaves zero.

Second backward:
- evidence queries gradient nonzero;
- evidence q/k/v projection gradient nonzero;
- label evidence cross-attention gradients nonzero;
- predicate evidence cross-attention gradients nonzero.

Action loss, reason loss, and predicate loss must each reach evidence memory.

---

## 13. Grounding audit

Grounding masks:

- object boxes
- lane polylines
- drivable maps

No semantic segmentation dependency.

No persistent disk cache.

In-memory LRU allowed but must be disabled by `persistent_mask_cache=false`.

Grounding loss:
- attention mass objective
- low entropy penalty
- weight default 0.05

Reject:
- high-weight patch BCE over all patches;
- semantic segmentation as required source;
- unsupported high-level predicate masks.

---

## 14. PU and predicate-target audit

PU losses:
- partial-label PU reason loss
- PU soft-F1
- PU predicate/reason alignment

Reject active:
```text
FP = probs * (1-target)
target_sign = target*2-1
```
without contradiction weighting.

Predicate targets:
- traffic_light_green without color count = 0
- close/far double-positive count = 0
- parked without parked attribute count = 0

Unknown states use mask=0.

---

## 15. HardPair budget audit

Pair miner remains source reason-specific miner.

Budget:

```text
L_main = weighted action_direct + weighted reason_partial
cap = ratio(epoch) * stopgrad(L_main)
L_pair_used <= cap
```

Schedule:
- epoch 3-7 ratio 0.20
- epoch 8+ ratio 0.10

Reject raw+budgeted double addition.

---

## 16. CalAlign teacher lock audit

Preserve:

```text
deploy = base - theta
```

Reject:
- sample-conditioned threshold;
- predicate-conditioned threshold;
- test threshold update.

Candidate must be evaluated before update.

Rejected candidate must not change threshold_head teacher state.

State must be checkpointed and restored.

---

## 17. Gate audit

Formal training requires all:

- Gate A zero evidence equivalence
- Gate B oracle evidence upper bound
- Gate C learned evidence grounding
- Gate D 128-sample mechanism overfit
- Gate E strong checkpoint train-calib sanity
- Gate F faithfulness sanity
- Gate G performance/memory

Gate output files:

- `GEM_GATE_A_EQUIVALENCE.json`
- `GEM_GATE_B_ORACLE_UPPER_BOUND.json`
- `GEM_GATE_C_LEARNED_GROUNDING.json`
- `GEM_GATE_D_MECHANISM_OVERFIT.json`
- `GEM_GATE_E_TRAIN_CALIB_SANITY.json`
- `GEM_GATE_F_FAITHFULNESS.json`
- `GEM_MEMORY_PASS.json`
- `GEM_GATES_PASS.json`

If any gate fails, full train is forbidden.

---

## 18. Runtime and performance audit

Formal config:

- no feature cache
- no token compression
- num_workers=6
- persistent_workers=true
- prefetch_factor=4
- pin_memory=true
- TF32=true

Memory candidates:
- [6,5]
- [5,6]
- [4,8]
- [3,10]
- [2,15]

Select fastest stable candidate:
- own peak allocated <= 30 GiB
- no CUDA OOM

Performance requirements:
- DINO forward count remains 1
- forward overhead <= 15%
- active train-step overhead <= 25%
- peak allocated increase <= 3 GiB

---

## 19. Evaluation/artifact audit

Every epoch must save:

- metrics summary
- branch metrics
- evidence metrics
- compact evidence attention subset
- action/reason/predicate logits
- action/reason labels
- evidence chains JSONL
- evidence report HTML
- light faithfulness JSON
- pair budget summary
- teacher state

Best checkpoints:
- best test deploy joint
- best test action mF1
- best test Exp mF1
- best test Exp mAP
- best test base joint
- latest

No test oracle threshold for checkpoint selection.

---

## 20. Visualization and faithfulness audit

Evidence chain must include:

```text
action -> evidence token -> predicate -> reason -> patch/box/lane/drivable region
```

Reports must show:
- predicted action
- predicted explanation
- ground truth separately
- evidence token name/group
- evidence attention map
- predicate links
- reason links
- top deletion impact
- random deletion baseline

Faithfulness is eval-only:
- no gradients
- no optimizer step
- no teacher/LR feedback

---

## 21. Foreground supervisor audit

Reject:
- Start-Process
- Start-Job
- nohup
- daemon
- scheduled task
- hidden cmd
- detached process

Supervisor must:
- verify exact clean pushed HEAD
- verify review pass HEAD
- run smoke
- run gates
- run memory probe
- choose batch/accum
- stream stdout/stderr
- heartbeat every <=300s
- OOM-only fallback
- fresh attempt dirs
- preserve failed logs
- no metric early stop
- completion only after all epochs and artifacts

---

## 22. Commands

Compile:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\acpr_grounded_evidence_memory.py `
  fate_oia\grounding\acpr_gem_grounding.py `
  fate_oia\models\acpr_oia_model.py `
  fate_oia\models\acpr_label_trunk.py `
  fate_oia\models\acpr_scene_predicate_head.py `
  fate_oia\models\acpr_predicate_targets.py `
  fate_oia\losses\acpr_losses.py `
  fate_oia\utils\acpr_gem_artifacts.py `
  fate_oia\utils\acpr_gem_teacher_lock.py `
  fate_oia\utils\acpr_pair_budget.py `
  fate_oia\utils\acpr_gem_training_control.py `
  fate_oia\engine\train_acpr_oia.py `
  fate_oia\engine\eval_acpr_oia.py `
  fate_oia\engine\audit_acpr_gem_implementation.py `
  fate_oia\engine\audit_acpr_gem_gates.py `
  fate_oia\engine\probe_acpr_gem_memory.py `
  fate_oia\engine\eval_acpr_gem_faithfulness.py `
  fate_oia\engine\export_acpr_gem_visuals.py `
  fate_oia\engine\supervise_acpr_gem_foreground.py
```

Tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_gem_evidence_memory.py `
  tests\test_acpr_gem_oracle_pooler.py `
  tests\test_acpr_gem_grounding.py `
  tests\test_acpr_gem_trunk_integration.py `
  tests\test_acpr_gem_predicate_integration.py `
  tests\test_acpr_gem_equivalence.py `
  tests\test_acpr_gem_gradient_flow.py `
  tests\test_acpr_gem_pu_losses.py `
  tests\test_acpr_gem_predicate_targets.py `
  tests\test_acpr_gem_pair_budget.py `
  tests\test_acpr_gem_teacher_lock.py `
  tests\test_acpr_gem_gates.py `
  tests\test_acpr_gem_faithfulness.py `
  tests\test_acpr_gem_visualization.py `
  tests\test_acpr_gem_audit.py `
  tests\test_acpr_gem_supervisor.py `
  tests\test_acpr_gem_memory_probe.py -q
```

Audit:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_gem_implementation `
  --config configs\fate_oia_train_360x640_acpr_gem_v1.yaml `
  --output_dir .background_runs\acpr_gem_v1_preflight `
  --device cuda `
  --write_review_pass
```

Formal command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_gem_v1_foreground.ps1 `
  -Epochs 16 `
  -NumWorkers 6 `
  -Device cuda `
  -ReferenceCheckpoint "<REFERENCE_ACPR_CALALIGN_CHECKPOINT>" `
  -RequireReviewPass
```

---

## 23. Audit JSON schema

Required top-level keys:

- `pass`
- `git_head`
- `remote_head`
- `branch`
- `worktree`
- `source_branch`
- `source_sha`
- `config_checks`
- `forbidden_patterns`
- `evidence_slot_checks`
- `evidence_memory_checks`
- `oracle_checks`
- `trunk_integration_checks`
- `predicate_integration_checks`
- `forward_path_checks`
- `equivalence_checks`
- `gradient_checks`
- `grounding_checks`
- `pu_checks`
- `predicate_target_checks`
- `pair_budget_checks`
- `teacher_lock_checks`
- `gate_results`
- `memory_results`
- `runtime_checks`
- `artifact_checks`
- `visualization_checks`
- `faithfulness_checks`
- `supervisor_checks`
- `warnings`
- `review_pass_path`

`pass=true` only if every blocking check passes.

---

## 24. Verdict terminology

Implementation pass:
  code and audits match the GEM plan.

Gate pass:
  oracle/learned evidence and action-safety have been proven before full train.

Training complete:
  all 16 epochs, checkpoints, and artifacts complete.

Research success:
  metrics and evidence diagnostics beat ACPR-CalAlign.

Never equate implementation pass with research success.
