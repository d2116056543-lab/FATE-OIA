# PRECISE-OIA V1 Implementation Audit Skill

## 1. Purpose

This skill is the blocking implementation authority for:

> PRECISE-OIA V1  
> Predicate-Regulated Evidence-Certified Inter-task Semantic Exchange

It audits that the code is not merely trainable, but actually implements:

- frozen direct-image DINO field;
- action/reason category-specific representations;
- structured multi-part evidence fields;
- predicate-guided second-pass visual rereading;
- explicit-evidence-certified action–reason semantic exchange;
- semantic/annotation reason bifurcation;
- same-image target-specific interventions;
- PCVL conditional-value diagnostics;
- test-only internal evaluation and test-selected checkpoints;
- no feature cache and no token compression.

A passing result means the planned mechanisms are present, called, receive the intended gradients, obey information firewalls, emit required diagnostics, and can enter the short pilot. It does not guarantee performance.

---

# 2. Mandatory context

Before any code modification, test, training, evaluation, process management, commit, or push under:

```text
E:\sbw\FATE_Drive
```

read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Do not create new repository-local training status Markdown files.

Append status only to those three canonical records.

---

# 3. Immutable repository contract

Expected:

```text
repository: d2116056543-lab/FATE-OIA
base branch: acpr_calalign_v1_2
base commit: 373aa49feac17372574fd7fb056c1d79c7c848fe

implementation branch: acpr_precise_oia_v1_direct_image
worktree: E:\sbw\FATE_Drive\fate_oia_precise_oia_v1_worktree
```

Reject if:

- current branch is not `acpr_precise_oia_v1_direct_image`;
- implementation worktree equals the existing baseline worktree;
- base commit is not an ancestor;
- original `acpr_calalign_v1_2` was modified;
- source changes are uncommitted before pilot;
- remote branch does not point to local HEAD before full train.

---

# 4. Hard non-negotiables

Reject unless every item is true.

1. Single-image input only.
2. Official DINO ViT-S/8.
3. Input 360×640, patch=8.
4. Exactly 3600 full patch tokens retained.
5. DINO parameters frozen.
6. DINO forward under no_grad.
7. One DINO call per train batch, including mirror-paired batch.
8. No feature cache build.
9. No feature cache read.
10. No token compression/drop/merge in primary path.
11. No Run C/CalAlign/MOSAIC checkpoint initialization.
12. No cached logits or historical residual.
13. Final action is 4 independent sigmoid logits.
14. Final reason is 21 independent sigmoid logits.
15. Test evaluator only reads images and BDD-OIA action/reason labels.
16. Test model forward has no structured-record argument.
17. BDD100K is train-only weak supervision and pilot oracle evidence only.
18. No scene token is presented as object/lane/drivable evidence.
19. No fixed PMI/co-occurrence graph is added to logits.
20. No 16-action-set marginalization as final action.
21. No expert/MoE/router selector architecture.
22. No cross-image HardPair as primary mechanism.
23. PU is disabled in the first formal configuration.
24. Action never reads reason GT.
25. Action never reads observed reason logits.
26. Action never reads annotation adapter outputs.
27. Action receives reason semantic tokens only through stop-gradient.
28. Certified exchange uses explicit evidence only.
29. Latent evidence never receives a human-readable predicate name.
30. Observed reason loss updates annotation adapter only.
31. All core components are active from the first optimizer step through continuous warm-up.
32. Full training is blocked without current-hash `FULL_TRAIN_READY`.

---

# 5. Required files

Reject if any are missing.

## Config

```text
configs/fate_oia_train_360x640_precise_oia_v1.yaml
configs/precise_evidence_fields.yaml
configs/precise_reason_semantics.yaml
configs/precise_action_semantics.yaml
```

## Data

```text
fate_oia/datasets/bdd100k_task_aware_index.py
fate_oia/datasets/precise_grounding_adapter.py
fate_oia/transforms_precise.py
```

## Models

```text
fate_oia/models/precise_dino_field.py
fate_oia/models/precise_visual_field.py
fate_oia/models/precise_category_decoder.py
fate_oia/models/precise_evidence_fields.py
fate_oia/models/precise_visual_rereader.py
fate_oia/models/precise_semantic_exchange.py
fate_oia/models/precise_annotation_head.py
fate_oia/models/precise_pcvl_probes.py
fate_oia/models/precise_oia_model.py
```

## Losses and utils

```text
fate_oia/losses/precise_losses.py
fate_oia/losses/precise_intervention_losses.py
fate_oia/utils/precise_schema.py
fate_oia/utils/precise_artifacts.py
fate_oia/utils/precise_gradient_ownership.py
fate_oia/utils/precise_runtime.py
```

## Engines and scripts

```text
fate_oia/engine/train_precise_oia.py
fate_oia/engine/eval_precise_oia.py
fate_oia/engine/run_precise_pcvl.py
fate_oia/engine/profile_precise_oia.py
fate_oia/engine/audit_precise_oia_implementation.py
fate_oia/engine/export_precise_cases.py
fate_oia/engine/supervise_precise_oia_foreground.py
scripts/FATE_OIA_precise_oia_v1_foreground.ps1
```

## Skill

```text
.codex/skills/precise-oia-implementation-audit/SKILL.md
```

---

# 6. Allowed reuse from ACPR-CalAlign

The new primary path may import only these prior components:

```text
fate_oia.datasets.bdd_oia_multitask
fate_oia.metrics
fate_oia.threshold_tuning
fate_oia.models.acpr_sparse_ops.entmax15_bisect
fate_oia.models.acpr_threshold_head.ACPRThresholdHead
fate_oia.losses.acpr_threshold_losses
fate_oia.utils.acpr_artifacts
fate_oia.utils.acpr_threshold_search
fate_oia.utils.acpr_thresholds
fate_oia.utils.acpr_train_calib_split
vision_transformer
utils
```

Reject if the PRECISE model, trainer, or evaluator imports:

```text
ACPROIAModel
ACPRLabelTrunk
ACPRScenePredicateHead
WeakPredicateTargetBuilder
ACPRPredicateReasoner
ACPRPairMemory
ACPRActionComboAux
ACPRCalibrationHead
```

---

# 7. Forbidden source patterns

Reject active PRECISE source/config if any of the following are enabled:

```text
frozen_run_c
FrozenRunC
run_c_logits
cached_logits
cached_features
feature_cache
tail_residual_adapter
complementary_logits
action_set_probs @ subset_membership as final action
graph_delta_to_logits
label_pmi
cooccurrence_bias
reason_gt_to_action
reason_logits_observed_to_action
structured_records in model.forward
token_compression: keep_merge
token_compression: topk
checkpoint_best_val
val_loader
best_selection_split: val
eval_splits: val
Start-Process
Start-Job
Register-ScheduledTask
nohup
daemon
hidden window
```

Also reject any PRECISE source containing unresolved:

```text
TODO
FIXME
pass
NotImplementedError
return torch.zeros(...) as a placeholder component output
```

A graph-connected zero for a genuinely empty masked loss is allowed only when accompanied by `valid_count=0`.

---

# 8. Schema audit

## 8.1 Reason schema

Verify:

```text
exactly 21 reason rows
names match configs/bdd_oia_reason_names_external.yaml
each row has entity/state/sector/decision_role
allowed_evidence_families exists
explicit_certifiable exists
mirror_partner exists
```

Mirror pairs must be:

```text
9<->15
10<->16
11<->17
12<->18
13<->19
14<->20
```

Rows 12,13,14,18,19,20 must not be marked fully explicit-certifiable.

## 8.2 Action schema

Verify exactly:

```text
0 forward
1 stop
2 left
3 right
```

Left/right share a side base query plus side embeddings.

## 8.3 Evidence schema

Verify initial fields:

```text
traffic_light
traffic_sign
actor_left
actor_center
actor_right
drivable_left
drivable_center
drivable_right
boundary_left
boundary_right
```

Each field must define:

```text
family
sector
part_type
num_parts
supervision_sources
state/type schema
geometry requirement
minimum coverage
```

Latent slots must have `name: null` or equivalent non-human identifier and `certifiable: false`.

---

# 9. Task-aware BDD100K audit

Instantiate `BDD100KTaskAwareIndex` on the real root.

Verify:

1. detection, lane, drivable sources are stored separately;
2. same stem can contain all three;
3. no last-write-wins path overwrite;
4. JSON is parsed once at startup, not in every batch;
5. metadata manifest is emitted;
6. no visual features are serialized.

Synthetic tests:

- traffic-light object without color:
  - presence valid;
  - color state mask=0;
  - green target must not be 1.
- source complete and no center actor:
  - actor_center observability=1;
  - occupancy=0;
  - valid negative=1.
- missing source:
  - valid mask=0;
  - not converted to negative.
- lane style absent:
  - boundary presence may be valid;
  - style target unknown.
- detection and lane JSON for same stem:
  - both are present after merge.

Preflight reject if any enabled field has:

```text
positive <64
reliable_negative <128
geometry_valid <64 when geometry_required
```

Unsupported fields must be pruned before model construction.

---

# 10. Frozen DINO audit

Instantiate `PRECISEDinoFieldExtractor` with official checkpoint.

Verify:

```text
patch_tokens_by_layer [B,3,3600,384]
cls_tokens_by_layer [B,3,384]
grid_hw == (45,80)
original_tokens == 3601
```

Verify all DINO parameters:

```python
assert all(not p.requires_grad for p in dino.parameters())
```

Run backward from a downstream adapter loss and verify:

```text
DINO grad count = 0
adapter grad count > 0
```

After forward, for every ViT block:

```text
block.attn.attention_map is None
block.attn.attn_gradients is None
block.attn.input is None
block.attn.v is None
block.attn.vproj is None
```

Wrap/monkeypatch DINO forward to count calls. A train batch with canonical and selected mirror pairs must produce exactly one call.

Reject if any `.pt/.pth/.npy/.npz` feature file appears outside checkpoint/output artifacts.

---

# 11. Visual field audit

Verify:

```text
full 3600 tokens remain available at all selected layers
no token index pruning
no merge map
no keep mask
```

Perspective encoding checks:

- left coordinate changes sign after mirror;
- perspective distance remains unchanged after horizontal mirror;
- layer embeddings differ;
- all outputs finite.

Gradient ownership:

- action loss updates action foundation adapter;
- semantic reason loss does not update action foundation adapter;
- evidence loss does not update action foundation adapter;
- observed reason loss updates no visual field parameter.

---

# 12. Category decoder audit

First pass required outputs:

```text
action_tokens_direct [B,4,D]
reason_tokens_direct [B,21,D]
action_logits_direct [B,4]
reason_logits_direct [B,21]
action_entropy [B,4]
reason_entropy [B,21]
```

Verify compositional reason query equals:

```text
entity + state + sector + role + label residual
```

and not a single independent 21×D parameter alone.

Verify action left/right:

```text
Q_left - E_left == Q_right - E_right == Q_side
```

within numerical tolerance.

First pass must not use cross-task exchange.

---

# 13. Evidence-field audit

Required outputs:

```text
explicit_tokens
latent_tokens
presence_logits
observability_logits
state/type logits
part_coordinates
part_scales
soft_masks
derived_atom_probs
reliability
```

Verify:

- traffic/actor fields have 4 parts;
- drivable fields have 8 region parts;
- boundary fields have 8 ordered curve parts;
- latent slots have 4 parts;
- curve part order is preserved;
- no field is reduced to one centroid before rereading;
- latent slots cannot enter explicit certificate.

Reliability monotonicity:

- higher observability with same inputs cannot lower reliability;
- higher positive-vs-negative prototype margin cannot lower reliability;
- lower view consistency cannot raise reliability.

Reject if:

```text
all reliability <1e-4
all reliability >0.999
one field occupies >80% attention for all samples
all masks are zero
all masks are identical
```

---

# 14. Second-pass rereader audit

Required:

```text
reference_points [B,25,3,4,2] or schema-equivalent
reread_delta action/reason
sampling_weights
reference entropy
center-collapse statistic
```

Verify:

1. reference points depend on evidence coordinates;
2. changing evidence coordinates changes sampled visual features;
3. no additional DINO call;
4. hard-example refinement loss rewards lower second-pass loss;
5. easy-example non-regression penalizes degradation;
6. mirror transforms reference x coordinate correctly.

Real-smoke health:

```text
reference x/y variance > 0
center-collapse rate <0.70
out-of-bounds rate <0.05
reread/direct RMS ratio between 0.02 and 0.30
```

---

# 15. Certified semantic exchange audit

Required tensors:

```text
action_evidence_attention [B,4,E]
reason_evidence_attention [B,21,E]
exchange_overlap [B,4,21]
exchange_gate [B,4,21]
action_exchange_delta
reason_exchange_delta
```

Structural invariants:

```python
explicit_reliability.zero_()
assert max_abs(action_exchange_delta) < 1e-7
assert max_abs(reason_exchange_delta) < 1e-7
```

Action firewall:

```text
action receives stopgrad(reason semantic tokens)
action does not read reason semantic logits
action does not read reason observed logits
action does not read annotation delta
action loss produces zero grad on reason owner
```

No fixed correlation:

- no static 4×21 PMI matrix added to logits;
- family mask may only be 0/-inf;
- overlap must be sample-specific.

Ablation:

```text
certified
ungated
off
evidence shuffled
reason tokens shuffled
```

must be computable from one encoded visual field.

Health:

```text
action exchange/direct RMS ratio 0.01–0.25
reason exchange/direct RMS ratio 0.01–0.30
gate not all zero/not all one
wrong-target message ratio < correct-target ratio
```

---

# 16. Semantic–annotation bifurcation audit

Required logits:

```text
reason_logits_direct
reason_logits_semantic
reason_logits_observed
```

Implementation invariant:

```python
reason_observed = reason_semantic.detach() + annotation_delta
annotation_delta = annotation_head(
    reason_semantic_token.detach(),
    context.detach()
)
```

Observed loss gradient test:

```text
annotation adapter grad >0
action grad =0
action foundation grad =0
reason semantic owner grad =0
evidence core grad =0
```

Verify annotation head is sample-dependent:

- same label bias alone is insufficient;
- two distinct detached contexts can produce different deltas;
- delta is capped.

Official test Exp branch must be observed; semantic branch must still be saved and evaluated separately.

---

# 17. Counterfactual audit

Construct synthetic selected/control/wrong effects.

Verify loss directions:

```text
selected effect > control effect lowers loss
wrong target effect > correct effect raises loss
```

Matched control must satisfy:

```text
same family
same sector
same number of parts
mask mass relative error <=5%
selected/control overlap ==0
```

Runtime invariant:

```text
no backbone/DINO rerun
no visual-field rerun
only evidence/exchange/head rerun
```

At least one real smoke batch must have:

```text
valid_counterfactual_count >0
selected_effect != control_effect
```

---

# 18. PCVL audit

PCVL probes:

```text
U0 category base
U1 oracle evidence
U2 learned evidence + oracle compatibility
U3 learned full exchange
```

Verify:

- same probe architecture/capacity;
- base category input detached;
- probes do not update primary model;
- train_main and train_audit disjoint;
- oracle evidence only from train records;
- PCVL code not imported by official test evaluator;
- no PCVL output becomes final action.

Required artifacts:

```text
pcvl_metrics.json
pcvl_per_action.json
pcvl_bootstrap.json
pcvl_value_decomposition.json
```

If U1 does not exceed U0, output:

```text
predicate_action_value_supported: false
```

Do not fail implementation completeness solely because scientific value is negative, but do not write `FULL_TRAIN_READY` for a dual-task predicate claim.

---

# 19. Loss and owner audit

Build an explicit parameter ownership table at runtime.

Expected owners:

```text
action_foundation
action_decoder
reason_semantic
evidence_core
exchange_reread
annotation_adapter
threshold_head
```

For each loss, compute `torch.autograd.grad(..., allow_unused=True)` and emit norm by owner.

Required matrix:

| loss | action foundation | action decoder | reason semantic | evidence core | exchange/reread | annotation | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| action | >0 | >0 | 0 | bounded/declared | >0 | 0 | 0 |
| reason semantic | 0 | 0 | >0 | bounded/declared | >0 | 0 | 0 |
| reason observed | 0 | 0 | 0 | 0 | 0 | >0 | 0 |
| evidence | 0 | 0 | 0 | >0 | 0 | 0 | 0 |
| intervention | 0 | target-specific | 0 | projected/bounded | >0 | 0 | 0 |
| threshold | 0 | 0 | 0 | 0 | 0 | 0 | >0 |

Evidence target-credit projection must satisfy:

```text
projected credit dot grounding grad >= -1e-8
projected credit norm <=0.20 * grounding grad norm
```

---

# 20. Full model forward contract

Run real and mock forward.

Required keys:

```text
action_logits_direct
action_logits_reread
action_logits_exchange_ungated
action_logits_exchange_certified
action_logits_final_raw
reason_logits_direct
reason_logits_semantic
reason_logits_observed
reason_logits_final_raw
action_logits_deploy
reason_logits_deploy
action_logits_calibrated
reason_logits_calibrated
explicit_evidence_tokens
latent_evidence_tokens
derived_atom_probs
evidence_reliability
evidence_part_coordinates
evidence_masks
action_evidence_attention
reason_evidence_attention
exchange_overlap
exchange_gate
action_exchange_delta
reason_exchange_delta
action_reread_delta
reason_reread_delta
reference_points
annotation_delta
branch_logits
diagnostics
```

Shapes:

```text
action [B,4]
reason [B,21]
exchange [B,4,21]
```

Final source:

```text
action_logits_final_raw == certified refined action head
reason_logits_final_raw == reason_logits_observed
```

Reject if final action equals:

```text
reason_to_action(reason probabilities)
action-set marginal
oracle branch
PCVL probe
```

---

# 21. Trainer protocol audit

Config must be:

```text
eval_splits: test
best_selection_split: test
token_compression: none
feature_cache_enabled: false
epochs: 12
precision: bf16
```

Trainer must build:

```text
train loader
train_calib loader
test loader
```

No val loader.

Every epoch:

1. train representation;
2. update train-calib threshold;
3. evaluate test once;
4. write all branch metrics;
5. save latest and best test checkpoints.

Primary checkpoint:

```text
checkpoint_best_test_deploy_joint.pth
```

Manifest must say:

```text
internal_test_selected: true
publication_eligible_selection: false
```

Gradient accumulation logging must use optimizer-step count, not only micro-step count.

Checkpoint/resume must save and restore:

```text
model
all optimizers/groups
scheduler
epoch/global micro step/optimizer step
RNG states
threshold teacher
evidence prototype EMA
view-consistency EMA
active field schema hash
config hash
source hash
```

Resume-equivalence test must match the next optimizer update within tolerance.

---

# 22. Runtime audit

Actual path profiler must include:

```text
full DINO
visual field
evidence extraction
two-pass reread
certified exchange
annotation head
all losses
backward
optimizer step
25% mirror pairing
```

Compare profiles:

```text
10/3
8/4
6/5
```

Reject a profile if:

```text
peak reserved >46.5GB
OOM
NaN/Inf
any core mechanism disabled
DINO call count !=1
```

Select highest samples/sec under45GB; if difference<3%, choose lower-memory profile.

Required runtime artifacts:

```text
runtime_profile.json
runtime_steps.jsonl
selected_runtime_profile.json
```

---

# 23. Diagnostics/artifact audit

Per run required:

```text
run_manifest.json
config_resolved.yaml
implementation_fingerprint.json
loss_components.jsonl
gradient_ownership.jsonl
mechanism_batch_stats.jsonl
metrics_summary.jsonl
```

Per epoch required:

```text
metrics_summary.json
branch_metrics.json
per_action_metrics.json
per_reason_metrics.json
evidence_family_stats.json
evidence_reliability.json
exchange_stats.json
reread_stats.json
annotation_gap.json
counterfactual_stats.json
gradient_firewall.json
failure_cases.jsonl
evidence_cases.jsonl
logits/labels tensors
```

Audit must verify values are computed from actual tensors, not placeholder constants.

Required diagnostic branches:

```text
action direct
action reread-no-exchange
action ungated
action certified
action final/deploy

reason direct
reason semantic
reason observed/deploy

exchange off
evidence shuffled
reason-token shuffled
explicit only
latent only
annotation off
```

---

# 24. Required tests

Run at minimum:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_precise_bdd100k_task_aware_index.py `
  tests\test_precise_evidence_targets.py `
  tests\test_precise_reason_schema.py `
  tests\test_precise_dino_field.py `
  tests\test_precise_visual_field.py `
  tests\test_precise_category_decoder.py `
  tests\test_precise_evidence_fields.py `
  tests\test_precise_visual_rereader.py `
  tests\test_precise_semantic_exchange.py `
  tests\test_precise_annotation_firewall.py `
  tests\test_precise_counterfactual.py `
  tests\test_precise_pcvl.py `
  tests\test_precise_model_forward.py `
  tests\test_precise_gradient_ownership.py `
  tests\test_precise_eval_contract.py `
  tests\test_precise_train_protocol.py `
  tests\test_precise_runtime_contract.py `
  tests\test_precise_supervisor.py `
  tests\test_precise_audit.py -q
```

Also run the existing ACPR-CalAlign tests to prove no regression:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_calalign_forward.py `
  tests\test_acpr_calalign_train_protocol.py `
  tests\test_acpr_threshold_head.py `
  tests\test_acpr_threshold_losses.py `
  tests\test_acpr_threshold_search.py -q
```

---

# 25. Required audit commands

## Source/static audit

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_precise_oia_implementation `
  --config configs\fate_oia_train_360x640_precise_oia_v1.yaml `
  --output_dir .review\precise_oia_v1 `
  --device cuda `
  --mode preflight `
  --write_pre_pilot_eligible
```

## Runtime profile

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.profile_precise_oia `
  --config configs\fate_oia_train_360x640_precise_oia_v1.yaml `
  --output_dir .review\precise_oia_v1\runtime `
  --device cuda
```

## Pilot

The historical pilot path below remains available for ordinary PRECISE runs.
It is not the authorization path for the user-approved embedded-curriculum
full run dated 2026-07-23.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\FATE_OIA_precise_oia_v1_foreground.ps1 `
  -Mode pilot `
  -Epochs 3 `
  -BatchSize 8 `
  -GradAccum 4 `
  -NumWorkers 8
```

## Post-pilot audit

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_precise_oia_implementation `
  --config configs\fate_oia_train_360x640_precise_oia_v1.yaml `
  --output_dir .review\precise_oia_v1 `
  --pilot_dir .background_runs\precise_oia_v1_360x640_testprimary\pilot `
  --device cuda `
  --mode post_pilot `
  --write_full_train_ready
```

---

# 26. Review artifacts

## PRE_PILOT_ELIGIBLE

`.review/PRECISE_OIA_V1_PRE_PILOT_ELIGIBLE.json`

Required fields:

```json
{
  "status": "PASS",
  "git_head": "...",
  "branch": "acpr_precise_oia_v1_direct_image",
  "base_commit": "373aa49feac17372574fd7fb056c1d79c7c848fe",
  "config_sha256": "...",
  "skill_sha256": "...",
  "source_tree_sha256": "...",
  "training_source_sha256": "...",
  "tests_passed": true,
  "real_forward_passed": true,
  "gradient_firewall_passed": true,
  "dino_call_contract_passed": true,
  "runtime_profile_passed": true,
  "unresolved": []
}
```

## FULL_TRAIN_READY

`.review/PRECISE_OIA_V1_FULL_TRAIN_READY.json`

Required fields:

```json
{
  "status": "PASS",
  "git_head": "...",
  "config_sha256": "...",
  "skill_sha256": "...",
  "pilot_complete": true,
  "mechanisms_active": true,
  "pcvl": {
    "u0": {},
    "u1": {},
    "u2": {},
    "u3": {},
    "predicate_action_value_supported": true
  },
  "runtime_selected": {},
  "unresolved": []
}
```

If predicate action value is negative:

```text
status must not silently remain PASS for the dual-task predicate claim
```

Use:

```json
{
  "status": "SCIENTIFIC_GATE_FAILED",
  "predicate_action_value_supported": false
}
```

## FULL_CURRICULUM_READY

`.review/PRECISE_OIA_V1_FULL_CURRICULUM_READY.json`

This is the only artifact allowed to authorize the embedded-curriculum run.
`FULL_TRAIN_READY cannot authorize the embedded-curriculum run`.

Required fields:

```json
{
  "status": "FULL_CURRICULUM_READY",
  "git_head": "...",
  "config_sha256": "...",
  "skill_sha256": "...",
  "source_tree_sha256": "...",
  "curriculum_sha256": "...",
  "curriculum_epochs": 12,
  "provenance": "user_approved_2026-07-23",
  "prelaunch_assertions_passed": true,
  "runtime_assertions_required": true,
  "runtime_selected": {},
  "unresolved": []
}
```

The gate must bind the canonical curriculum JSON, per-owner active epoch
counts, clean git HEAD, exact skill/config hashes, real-DINO forward,
gradient-firewall checks, and the selected runtime profile. Source, config,
skill, or curriculum changes invalidate it. Runtime assertions must then
verify epoch-boundary scales, inactive-owner zero updates and empty optimizer
state, local scheduler clocks, and all mechanisms active at epoch 6.

The full command must include:

```text
--allow_full_with_embedded_curriculum
```

---

# 27. Review states

Only these states are allowed:

```text
CHANGES_REQUIRED
PRE_PILOT_ELIGIBLE
SCIENTIFIC_GATE_FAILED
FULL_TRAIN_READY
FULL_CURRICULUM_READY
FULL_RUN_COMPLETE
```

`REVIEW_PASS` text files without hashes are forbidden.

Any source/config/skill change invalidates prior gate artifacts.

---

# 28. Completion report required from Codex

Codex must return:

```text
branch
worktree
local HEAD
remote HEAD
base ancestor
changed/new files
test command and counts
py_compile result
real forward shapes
gradient ownership matrix
DINO call count
GPU profile
pilot metrics U0–U3
mechanism ranges
review artifact paths
full train command
unresolved issues
```

It must not say “implemented” if a required module is:

- only instantiated;
- not called by formal forward;
- called only in a smoke helper;
- not included in optimizer;
- returning placeholder values;
- excluded from the actual training command.
