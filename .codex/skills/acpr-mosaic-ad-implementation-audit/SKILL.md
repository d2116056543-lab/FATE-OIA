---
name: acpr-mosaic-ad-implementation-audit
description: Fail-closed implementation audit for ACPR-MOSAIC-AD V1, including mechanism, leakage, runtime, pilot, and artifact gates.
---
# ACPR-MOSAIC-AD V1 Implementation Audit Skill

**Install target:** `.codex/skills/acpr-mosaic-ad-implementation-audit/SKILL.md`  
**Applies to branch:** `acpr_mosaic_ad_v1_direct_image`  
**Source branch:** `acpr_calalign_v1_2`  
**Audit policy:** fail closed; full training is forbidden until all static, dynamic, numerical, leakage, runtime, and artifact gates pass.

---

## 1. Skill objective

This skill audits whether Codex implemented the complete ACPR-MOSAIC-AD method rather than a runnable approximation.

A passing implementation must prove all of the following:

```text
observable predicate measurement is visual and geometry-typed
visibility and presence are not conflated
multiple prototypes are actually used
support and veto evidence remain separate
action is independent of reason supervision
latent reasons are separated from recorded reasons
the selective observation posterior is mathematically correct
posterior-weighted ranking replaces hard negative pairing
the action-anchored update is active
calibration is train-calib-only
test forward contains no labels or geometry metadata
formal training uses direct images, no cache, and no token compression
```

The audit must not infer correctness from file names, class names, config flags, or emitted log strings. It requires dynamic evidence.

---

## 2. Required start procedure

Before inspecting or modifying the repository:

```powershell
Get-Content E:\sbw\FATE_Drive\task_plan.md
Get-Content E:\sbw\FATE_Drive\findings.md
Get-Content E:\sbw\FATE_Drive\progress.md

cd E:\sbw\FATE_Drive\fate_oia_acpr_mosaic_ad_v1_worktree
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Audit must fail if:

```text
current branch != acpr_mosaic_ad_v1_direct_image
worktree path is the original acpr_calalign_v1_2 worktree
uncommitted unrelated changes are present
source/new branch manifests are missing
```

Write:

```text
.review/audit_context.json
```

---

## 3. Expected files

Fail if any required file is absent.

```text
configs/mosaic_label_schema.yaml
configs/mosaic_observable_factors.yaml
configs/mosaic_decision_states.yaml
configs/mosaic_reason_observation.yaml
configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml

fate_oia/models/mosaic_native_semantics.py
fate_oia/models/mosaic_visual_pyramid.py
fate_oia/models/mosaic_geometry_typed_attention.py
fate_oia/models/mosaic_observable_predicates.py
fate_oia/models/mosaic_state_composer.py
fate_oia/models/mosaic_sparse_label_decoder.py
fate_oia/models/mosaic_action_decoder.py
fate_oia/models/mosaic_reason_decoder.py
fate_oia/models/mosaic_selective_observation.py
fate_oia/models/mosaic_group_threshold.py
fate_oia/models/acpr_mosaic_ad_model.py

fate_oia/losses/mosaic_action_losses.py
fate_oia/losses/mosaic_factor_losses.py
fate_oia/losses/mosaic_reason_observation_losses.py
fate_oia/losses/mosaic_posterior_ranking.py
fate_oia/losses/mosaic_state_losses.py

fate_oia/optim/mosaic_action_anchor.py
fate_oia/optim/mosaic_soft_rank_queue.py

fate_oia/datasets/mosaic_grounding_observations.py
fate_oia/datasets/mosaic_multiview.py
fate_oia/datasets/mosaic_train_calib_split.py

fate_oia/engine/train_acpr_mosaic_ad.py
fate_oia/engine/eval_acpr_mosaic_ad.py
fate_oia/engine/profile_acpr_mosaic_ad.py
fate_oia/engine/audit_acpr_mosaic_ad.py
fate_oia/engine/export_mosaic_visual_audit.py
fate_oia/engine/build_mosaic_ablation_table.py

scripts/FATE_OIA_acpr_mosaic_ad_v1_foreground.ps1
tests/test_mosaic_*.py
```

---

## 4. Forbidden-pattern scan

Scan all formal model/training/config files.

Fail on active formal-path matches for:

```text
RunC
cached_logits
feature_cache_enabled: true
token_compression: anything other than none
ACPROIAModel(
ACPRLabelTrunk(
ReasonToAction
reason_to_action
action_set_affects_final_action: true
action-set marginalization as final
PMI
cooccurrence graph
label graph
test threshold copied to model
test oracle copied to theta
geometry passed to model.forward
reason labels passed to model.forward
action labels passed to model.forward
raw q/rho added directly to action logits
```

Allowed only in:
- explicit negative tests;
- comments documenting a prohibition;
- ablation scripts clearly excluded from formal training.

Write:

```text
.review/forbidden_pattern_scan.json
```

with path, line, match, disposition.

---

## 5. Compilation and test gate

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m compileall fate_oia
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_mosaic_*.py -q
```

Also run relevant legacy regression tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_bdd_oia_dataset.py `
  tests\test_fate_oia_metrics.py `
  tests\test_token_provenance.py `
  -q
```

Fail on skipped core tests, xfail core tests, import warnings caused by missing modules, or any test whose only assertion is “no exception”.

Write:

```text
.review/compile_test_gate.json
```

---

## 6. Label-schema and ontology gate

Load all four MOSAIC YAML files.

Required assertions:

```python
action indices == [0,1,2,3]
reason indices == [0,...,20]
factor names unique
state names unique
all factor references resolve
all state references resolve
state dependency graph is acyclic
all 21 reasons mapped
every reason has visibility factors
every factor has exactly one geometry type
mirror links are symmetric
contradiction links are valid
```

Audit must print the official action and reason names and their indices.

Fail if Codex silently ignores missing factors or states.

Write:

```text
.review/schema_ontology_gate.json
```

---

## 7. Formal forward-signature gate

Inspect and dynamically call:

```python
MOSAICADModel.forward(images, prior_mode="full", return_masks=True)
```

Fail if the formal forward accepts or requires:

```text
action labels
reason labels
geometry records
BDD100K masks
threshold labels
test metadata
```

Required output keys:

```text
factor_presence_logits
factor_visibility_logits
factor_positive_evidence
factor_negative_evidence
factor_uncertainty
factor_soft_masks
decision_state_prob
decision_state_support
decision_state_veto
action_logits_visual
action_logits_state
action_state_gate
action_logits_raw
reason_logits_latent
```

Write:

```text
.review/forward_contract_gate.json
```

---

## 8. No-cache/no-compression/direct-image gate

Resolve the formal config and launcher.

Required:

```yaml
image_height: 360
image_width: 640
patch_size: 8
feature_cache_enabled: false
token_compression: none
direct_image: true
backbone.freeze_backbone: true
backbone.selected_layers: [3,7,11]
```

Dynamic test:

1. instrument DINO forward call count;
2. feed a real image batch;
3. prove DINO executes on the current batch;
4. prove no cached feature file is read;
5. prove the patch count remains 3600 per selected layer.

Write:

```text
.review/direct_image_gate.json
```

---

## 9. Geometry-typed attention gate

### 9.1 Shape and vectorization

Use a synthetic batch and assert:

```text
point factors use point sampler
curve factors use curve sampler
region factors use region sampler
```

Capture tensor shapes.

Fail if any materialized tensor matches:

```text
[B,F,N,D]
[B,F,L*N,D]
```

or if the formal path loops in Python over all factors and calls `grid_sample` once per factor.

### 9.2 Gradient reachability

Backpropagate a factor loss and assert nonzero finite gradients for:

```text
prototype tensors
context router
anchor scorer
point offsets
curve tangent/normal parameters
region width/height or grid parameters
presence head
visibility head
```

### 9.3 Prior isolation

Run the same images with:

```text
full
content_only
prior_only
```

Verify outputs differ and prior scale is bounded.

Write:

```text
.review/typed_attention_gate.json
```

---

## 10. True multi-prototype gate

For every factor with `num_prototypes > 1`:

1. verify all prototypes independently participate in score computation;
2. verify the factor query is not produced by a simple prototype mean before matching;
3. verify prototype weights sum to one;
4. verify at least two prototypes receive nonzero probability on a mixed synthetic batch;
5. verify each prototype can receive gradient;
6. report pairwise cosine and dead prototypes.

Fail if:

```text
prototype mean is the only query
dominant prototype rate >0.95 in smoke
all prototype gradients are identical
unused prototypes have zero gradients throughout
```

Write:

```text
.review/prototype_gate.json
```

---

## 11. Visibility/presence semantics gate

Construct controlled tensors and verify:

```text
visibility=0, presence=1 -> positive evidence near 0
visibility=0, presence=0 -> negative evidence near 0
visibility=1, presence=1 -> positive evidence high
visibility=1, presence=0 -> negative evidence high
```

Exact contract:

```python
e_pos == visibility_prob * presence_prob
e_neg == visibility_prob * (1 - presence_prob)
```

Fail if absence is represented as `1-presence` without visibility gating.

Write:

```text
.review/visibility_presence_gate.json
```

---

## 12. Grounding-observation and leakage gate

### 12.1 Train-only builder

Check that BDD100K geometry is used only by:

```text
MOSAICGroundingObservationBuilder
factor grounding losses
visual audit
```

It must not be stored in model state or passed to formal eval forward.

### 12.2 Unknown handling

Test:

```text
missing annotation -> mask=0 / source=unknown
reason zero -> no negative factor target
positive reason -> permitted weak-positive factors only
```

### 12.3 Eval invariance

Run test forward twice:

```text
with geometry metadata present in batch dictionary
with geometry metadata removed
```

Action and reason raw logits must be bitwise equal within deterministic tolerance.

Write:

```text
.review/grounding_no_leakage_gate.json
```

---

## 13. Support–veto state gate

Run analytical interventions.

Required monotonic checks:

```text
left_solid_boundary ↑ -> left_veto nondecreasing
left_obstacle ↑ -> left_veto nondecreasing
left_drivable ↑ -> left_affordance nondecreasing
front_risk ↑ -> stop_obligation nondecreasing
center_occupied ↑ -> forward_feasible nonincreasing
```

Verify all learned support/veto weights are nonnegative after transform.

Verify the bounded residual cannot overwhelm the full support–veto range.

Write:

```text
.review/state_monotonicity_gate.json
```

---

## 14. Action firewall gate

This gate is mandatory and dynamic.

For the same image batch:

1. run formal action logits;
2. randomize reason labels in the external loss batch;
3. randomize reason posterior;
4. randomize propensity parameters;
5. zero reason decoder parameters;
6. remove reason observation mapping;
7. remove geometry metadata.

The model forward action logits must remain unchanged for changes 2–7 unless the changed object is an explicitly shared A/B visual parameter modified by an optimizer step. Merely changing training labels without an update must never alter forward action logits.

Use gradient probes:

```text
reason-only loss gradient on action decoder parameters == 0
propensity loss gradient on action decoder parameters == 0
reason posterior loss gradient on action decoder parameters == 0
action loss gradient on reason decoder parameters == 0
```

Write:

```text
.review/action_firewall_gate.json
```

---

## 15. Sparse label decoder gate

Verify action and reason decoders have:

```text
label queries
context cross-attention
high/mid-resolution sparse retrieval
entmax or equivalent sparse attention
label self-attention
separate action/reason parameters
```

Fail if the decoder degenerates to global mean pooling or CLS-only classification.

Verify:
- action visual branch is nonconstant;
- reason visual branch is nonconstant;
- all action/reason queries receive gradients.

Write:

```text
.review/label_decoder_gate.json
```

---

## 16. Selective observation mathematical gate

Use random, edge, and hand-computed values.

Verify:

\[
P(Y=1)=\pi p+\epsilon(1-p).
\]

For observed zero:

\[
q=
\frac{p(1-\pi)}
{p(1-\pi)+(1-p)(1-\epsilon)}.
\]

For observed one:

```text
q == 1
```

Requirements:

```text
log-space stable computation
finite gradients
pi in [pi_min, pi_max]
epsilon in [0, false_positive_max]
q detached before posterior-target BCE/ranking
```

Fail if the propensity model reads raw image tokens, action logits, or undetached reason logits.

Write:

```text
.review/selective_observation_math_gate.json
```

---

## 17. Synthetic missing-positive gate

Create a batch with known positive reasons.

Hide a known fraction as observed zero.

Verify:

```text
hidden labels are not supplied to the observation model
hidden-positive mask is retained only for recovery loss/audit
posterior q for hidden positives receives recovery supervision
unmasked real zeros do not become hard negatives
```

Run a 512-sample recovery smoke and compare against zero-as-negative.

Pass criterion:

```text
posterior recovery AUPRC > zero-as-negative AUPRC
```

Write:

```text
.review/synthetic_missing_recovery_gate.json
```

---

## 18. Posterior-ranking gate

Verify reason ranking:

```text
uses q*(1-q) soft weights
does not threshold q
operates across samples for the same reason
uses detached queue entries
excludes same sample id
normalizes by total valid weight
```

Verify action ranking:

```text
uses true action labels
operates across samples for the same action
does not compare four actions only within one image
```

Test queue reset, wraparound, device movement, and deterministic state serialization.

Write:

```text
.review/posterior_ranking_gate.json
```

---

## 19. Action-anchored trust-update gate

Construct two synthetic losses with:

```text
positive gradient cosine
negative gradient cosine
zero action gradient
unused parameters
```

Verify the implementation computes:

```text
gA
gE
dot
lambda_star
combined shared gradient
```

For conflicting gradients, verify:

\[
g_A^\top(g_A+\lambda^\star g_E)
\ge
\kappa\|g_A\|^2-\text{tolerance}.
\]

Verify:
- task-specific gradients are not projected unnecessarily;
- shared gradients are not accumulated twice;
- BF16 model still computes dot/norm in FP32;
- gradient accumulation works;
- logged `projection_count` is nonzero in a forced-conflict test.

Write:

```text
.review/action_anchor_gate.json
```

---

## 20. Calibration gate

Requirements:

```text
representation model frozen
threshold head is low capacity
theta uses group shrinkage and bounded label deltas
deploy logits equal raw logits - theta exactly
threshold source is train_calib
test oracle remains diagnostic only
```

Dynamic leakage probe:

1. fit calibration on train-calib;
2. evaluate test;
3. perturb test labels;
4. refit nothing;
5. verify learned thresholds are unchanged.

Fail if threshold head reads sample logits as neural features or uses reason posterior/propensity as input.

Write:

```text
.review/calibration_gate.json
```

---

## 21. Schedule gate

Parse the resolved config and simulate epochs 0–14.

Assert exact phase switches:

```text
0–2: state gate 0, propensity fixed, posterior rank off
3–5: state ramp on, action anchor on, propensity fixed
6–8: learned propensity, synthetic missingness, posterior warmup
9–11: full posterior rank and action anchor
12: prototypes/propensity frozen, LR reduced
13–14: representation frozen, threshold-only
```

The training engine must query one canonical schedule function. Duplicated epoch conditions in multiple files are forbidden.

Write:

```text
.review/schedule_gate.json
```

---

## 22. Artifact-schema gate

Run a two-epoch smoke and verify all required root and epoch artifacts exist, are valid JSON/JSONL/PT, and contain non-placeholder values.

Fail if:
- required arrays are empty without an explicit valid-count reason;
- files contain only zeros from hardcoded placeholders;
- branch names are ambiguous;
- `action_logits_raw` already contains threshold subtraction;
- threshold source is missing;
- sample IDs are absent.

Write:

```text
.review/artifact_schema_gate.json
```

---

## 23. Runtime and memory gate

Run `profile_acpr_mosaic_ad`.

The profile must exercise the Phase D path, not a reduced smoke path.

Required metrics per candidate:

```text
batch_size
grad_accum
num_workers
median_step_ms
p95_step_ms
samples_per_sec
max_allocated_gb
max_reserved_gb
cuda_retries
dataloader_stalls
nan_count
```

Pass only if the chosen configuration:
- has the highest stable samples/sec among tested candidates;
- uses <=43 GB reserved VRAM;
- completes the 15-minute stability probe;
- writes `.review/mosaic_runtime_selection.json`.

Write:

```text
.review/runtime_memory_gate.json
```

---

## 24. Pilot gate

Run three seeds on the fixed pilot subsets.

Required comparisons:

```text
MOSAIC base visual branch
MOSAIC combined action
zero-as-negative reason baseline
selective-observation reason
content-only factors
prior-only factors
full factors
```

Required pass criteria:

```text
no NaN/Inf
no action collapse
posterior recovery beats zero-as-negative
content-only factor signal is nontrivial
prior-only does not explain nearly all full factor performance
combined action is noninferior to action visual branch
raw Exp_mAP does not collapse
action-anchor pass rate >=95%
```

Write:

```text
.review/pilot_gate.json
```

No full-run pass may be issued before this file reports PASS for all three seeds.

---

## 25. Review pass generation

`audit_acpr_mosaic_ad.py` may write:

```text
.review/acpr_mosaic_ad_v1_REVIEW_PASS.json
```

only when every gate is PASS.

Required contents:

```json
{
  "status": "PASS",
  "git_head": "...",
  "branch": "acpr_mosaic_ad_v1_direct_image",
  "source_branch": "acpr_calalign_v1_2",
  "config_hash": "...",
  "schema_hash": "...",
  "runtime_selection_hash": "...",
  "tests_passed": 0,
  "gates": {
    "code": true,
    "schema": true,
    "direct_image": true,
    "typed_attention": true,
    "prototype": true,
    "visibility_presence": true,
    "grounding_no_leakage": true,
    "state_monotonicity": true,
    "action_firewall": true,
    "label_decoder": true,
    "selective_observation": true,
    "synthetic_missing": true,
    "posterior_ranking": true,
    "action_anchor": true,
    "calibration": true,
    "schedule": true,
    "artifacts": true,
    "runtime": true,
    "pilot": true
  },
  "timestamp_utc": "..."
}
```

Fail closed:
- do not write PASS with warnings on core gates;
- do not create PASS before pilot;
- do not allow a `--force` flag;
- do not let the launcher bypass the pass file.

---

## 26. Foreground launcher enforcement

`FATE_OIA_acpr_mosaic_ad_v1_foreground.ps1` must:

1. verify branch and worktree;
2. read `REVIEW_PASS`;
3. compare its git/config/runtime hashes to the current state;
4. refuse launch if stale;
5. read selected batch/worker settings;
6. run in the foreground;
7. stream stdout/stderr;
8. write process and command manifests;
9. evaluate test at every epoch;
10. save best test joint/action/reason checkpoints.

The launcher must not:
- detach with an unreliable nested PowerShell process;
- silently reduce the model;
- silently disable failed components;
- change batch without updating the runtime manifest;
- continue after a hard scientific gate failure.

---

## 27. GitHub synchronization gate

After implementation:

```powershell
git status --short
git rev-parse HEAD
git push github acpr_mosaic_ad_v1_direct_image
git ls-remote github refs/heads/acpr_mosaic_ad_v1_direct_image
```

Fresh-clone compile and tests are mandatory.

Write:

```text
.review/github_sync_pass.json
```

The local and remote commit hashes must match.

---

## 28. Audit verdict categories

Use only these verdicts:

```text
PASS
FAIL_CODE
FAIL_SCHEMA
FAIL_LEAKAGE
FAIL_MECHANISM
FAIL_NUMERICAL
FAIL_RUNTIME
FAIL_ARTIFACT
FAIL_PILOT
STALE_REVIEW
```

Do not use “mostly pass”, “pass with minor issues”, or “training can start while fixing later”.

The full experiment may start only with `PASS`.
