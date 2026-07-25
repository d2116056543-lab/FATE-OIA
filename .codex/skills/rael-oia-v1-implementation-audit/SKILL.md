# RAEL-OIA V1 Implementation Audit Skill

## 1. Role

This skill is the fail-closed implementation authority for:

> RAEL-OIA V1  
> Reliability-Admitted, Relation-Aware Evidence Ledger

It must determine whether the implementation:

- exists in the intended isolated worktree;
- starts from the exact ACPR-CalAlign commit;
- implements every planned scientific path;
- calls those paths in the formal model and trainer;
- assigns correct gradients and optimizer ownership;
- emits non-placeholder diagnostics;
- obeys image-only test-time firewalls;
- is safe to run in the short pilot and full train.

Passing this skill does not guarantee performance. It guarantees implementation completeness and scientific-contract fidelity.

---

# 2. Canonical project records

Before any code, test, run, commit or push under:

```text
E:\sbw\FATE_Drive
```

read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Do not create experiment-status Markdown mirrors in the repository. Append experiment state only to those three files.

---

# 3. Repository gate

Expected:

```text
repository: d2116056543-lab/FATE-OIA
base branch: acpr_calalign_v1_2
base commit: 373aa49feac17372574fd7fb056c1d79c7c848fe
implementation branch: acpr_rael_oia_v1_direct_image
worktree: E:\sbw\FATE_Drive\fate_oia_acpr_rael_oia_v1_worktree
push remote: github
```

Reject if:

- current branch differs;
- base commit is not an ancestor;
- original worktree has RAEL changes;
- worktree path is reused from another experiment;
- `github` remote does not point to the intended repository;
- local and remote HEAD differ before full train;
- worktree is dirty before gate artifact creation.

---

# 4. Non-negotiables

Reject unless all are true.

```text
single-image input
360×640
4 actions
21 reasons
official frozen DINO ViT-S/8
selected layers 3/6/9/12
3600 full patch tokens retained
no visual feature cache
no token compression
no video
no distillation
no VLM/MLLM
no per-image caption
no graph/PMI/static co-occurrence bias
no action-set marginalization as final action
BDD100K train-only
test forward RGB-only
```

Action firewall:

```text
no reason GT to action
no reason logits/probabilities to action
no reason-private adapter to action
no Pu targets to action
no BDD100K geometry to action at test
no text encoder to action
```

---

# 5. Forbidden imports in formal RAEL path

Reject if formal RAEL model/trainer imports or instantiates:

```text
ACPROIAModel
ACPRLabelTrunk
ACPRScenePredicateHead
ACPRPredicateReasoner
ACPRPairMemory
ACPRActionComboAux
ACPRCalibrationHead
ACPRThresholdHead as a representation-trained module
```

Allowed utility reuse must be explicit and limited to dataset, metrics, DINO loading, entmax and post-hoc threshold search.

---

# 6. Required files

Verify all files listed in the implementation plan exist, compile and are imported by the formal RAEL model/trainer.

Reject unresolved:

```text
TODO
FIXME
pass
NotImplementedError
placeholder tensors
hard-coded zero diagnostics
mock-only implementation
```

A masked loss may return a graph-connected zero only with an explicit `valid_count=0`.

---

# 7. DINO gate

Instantiate real DINO.

Required shapes:

```text
patch_tokens_by_layer [B,4,3600,384]
cls_tokens_by_layer   [B,4,384]
grid_hw               (45,80)
```

Assertions:

```python
assert all(not p.requires_grad for p in dino.parameters())
assert dino.training is False
assert dino_call_count == 1
```

Run backward from RAEL downstream loss:

```text
DINO grads = 0
RAEL field grads >0
```

After forward, third-party stored attention/input/value tensors must be cleared.

Search runtime/output for feature-cache files. Reject if model features are serialized for reuse.

---

# 8. Multilayer field gate

Verify:

- 4 layer projections exist;
- local depthwise adapters exist and receive gradients;
- query-conditioned layer weights are sample/query dependent;
- no direct layer mean;
- no 14400-token concatenation attention;
- all 3600 tokens remain accessible.

Perturb one selected layer while holding others fixed. At least some action, reason and slot queries must change their layer weights and outputs.

Reject:

```text
all layer weights identical
one layer weight >0.99 for all queries/samples
local adapter gradients permanently zero after 2 updates
```

---

# 9. Slot ledger gate

Required internal tensors:

```text
12 entity/control slots
5 road slots
3 latent slots
1 background sink
20 explainable slots total
2 slot iterations
```

Verify patch normalization is over slots, not over patches independently for every slot.

For every patch:

```text
sum over all slots including background ≈1
```

Verify iteration-2 logits include iteration-1 mask bias.

Health:

```text
active entity slots >=6 on representative batch
background mass <0.80
slot masks not all zero
slot masks not all identical
pairwise mask IoU not all >0.90
latent slots have no human name
background cannot enter contributions
```

---

# 10. Attribute and absence gate

Entity type must be conditional softmax, not independent sigmoids.

Traffic-control state must be conditional softmax.

Road style must be conditional softmax.

Verify reliable absence:

```text
visible sector + no entity occupancy -> clear evidence high
presence low alone does not force evidence reliability low
```

Test left/right mirror transforms all slot sectors and road identities correctly.

---

# 11. Grounding gate

Real task-aware index must preserve detection/lane/drivable for the same stem.

No last-write-wins overwrite.

JSON metadata must be preloaded once, not parsed in every batch.

Hungarian entity matching must be active on real BDD100K samples.

Required stats:

```text
matched entity count
unmatched positive count
reliable negative count
unknown count
traffic state valid count
drivable valid count
boundary valid count
```

Reject an active boundary loss if `valid_count=0`.

No semantic-seg-only dependency is allowed.

---

# 12. Reason schema gate

Exactly 21 rows.

Each row has:

```text
entity
state
sector
role
mirror_partner
explicit_evidence_families
pu_eligible
```

Role must distinguish support and veto.

No hard action compatibility mask may be derived from role/sector.

Verify compositional query construction and bounded residual norm.

---

# 13. Action–semantic reason bridge gate

Required:

```text
4 action visual tokens
21 semantic reason tokens
action cross-attention to semantic reason tokens
```

Action may not read reason logits or reason-private outputs.

Gradient tests:

```text
action loss -> semantic reason token grad >0 after ReZero bootstrap
action loss -> reason-private grad =0
reason-private loss -> action params grad =0
```

ReZero bootstrap:

```text
update0: bridge output scalar/projection grad >0
update1: bridge internal attention grad >0
```

Action-semantic delta/global ratio during pilot must be finite and nonzero.

---

# 14. Unary gate

General alpha-entmax must be implemented and differentiable.

Verify:

```text
alpha in [1.05,1.50]
initial alpha≈1.10
null slot exists
background slot excluded
unary contributions enter final logits
```

At init, final-global difference must be below configured function-preserving tolerance.

After 2 updates, unary contribution gradients and values must be nonzero.

---

# 15. Pairwise relation gate

Required:

```text
190 pairs from 20 slots
relative geometry features
relation hidden dim 64
vectorized implementation
no fixed adjacency
no PMI
```

Pairwise contributions must enter final action and reason logits.

Changing relative slot geometry while fixing slot features must change pairwise contribution.

Turning pairwise off must change only the pairwise branch, not unary/global values.

After 3 updates, pairwise internal gradients must be nonzero.

---

# 16. Additive explanation gate

For every target:

```text
final_logit ≈ global_logit + sum(unary) + sum(pairwise)
```

Check numerical reconstruction tolerance `<1e-6` in fp32 diagnostics.

Required contribution partitions:

```text
named
latent
global
positive
negative
```

Latent slots must not be reported with human predicate names.

---

# 17. Reason-private gate

Required:

```text
R = S + stopgrad(action_context) + private_adapter
```

Private rank=64 and norm cap active.

Action must be invariant to:

- shuffling private reason deltas;
- zeroing private adapter;
- changing Pu targets.

Semantic reason shuffle should change action if bridge is active; private reason shuffle must not.

---

# 18. Pu gate

Initial formal config:

```text
all Pu lambdas =0
```

Epoch0 audit may activate labels only if:

```text
positive count >=20
LCB95 hidden-positive AUPRC gain >0
```

Pu gradients:

```text
reason-private >0
semantic reason =0
ledger =0
action =0
```

Reliability/soft targets must be detached.

Reject if Pu activation is global/all-label by default.

---

# 19. Gradient admission gate

The implementation must compute action, reason, grounding and counterfactual gradients at:

```text
evidence slots
semantic reason tokens
```

EMA beta=0.95.

For each slot/token, verify:

```text
negative action-parallel component removed
aligned component retained
orthogonal component retained
reason norm <=0.25 action reference
ground norm <=0.15
cf norm <=0.05
```

Owner tests:

```text
reason-private receives full reason grad
grounding heads receive full ground grad
action params receive zero reason-private grad
shared boundaries receive replaced admitted grad
```

Run synthetic vectors with known angles and verify exact projection direction.

Reject if this is only logged but not used to replace backward gradients.

---

# 20. Counterfactual gate

Analytical deletion must equal additive contribution decomposition.

Feature intervention must:

```text
not rerun DINO
use background-mean replacement
use equal-mass same-sector control
have overlap <5%
```

Pilot coverage:

```text
4/4 action targets
>=11/21 reason targets
```

Loss direction:

```text
selected > control lowers loss
wrong-target > correct raises loss
```

---

# 21. Calibration gate

There must be no trainable threshold/calibration parameter in the representation optimizer.

Each epoch:

```text
freeze model
fit on train-calib
evaluate test
```

Verify mAP unchanged by thresholding.

Reject calibration if:

```text
threshold RMS >=0.35 raw-logit RMS
deploy mF1 drops >0.005 without fallback
```

---

# 22. Full forward contract

Required keys are those listed in the implementation plan.

Test shapes and finiteness.

Formal forward signature must not accept:

```text
labels
BDD100K records
text
oracle evidence
epoch
cache paths
```

Test-time code must not instantiate the BDD100K grounding index.

---

# 23. Training protocol gate

Internal protocol:

```text
test only each epoch
best selected on test deploy joint
14 epochs
one seed
bf16
no metric early stop
```

Must write:

```text
internal_test_selected=true
publication_eligible=false
```

Representation training uses fixed train-audit and train-calib protocol.

Checkpoint/resume must restore:

```text
model
optimizer
scheduler
epoch/micro/optimizer step
RNG states
slot/action gradient EMA
Pu lambdas
view consistency EMA
posthoc thresholds
active schema hash
source/config hash
```

Resume-equivalence test must match next update within tolerance.

---

# 24. Diagnostics gate

Verify every required batch and epoch diagnostic exists and is computed from tensors.

Reject:

```text
constant zero values
missing owner stats
metrics emitted only by smoke helper
branch not evaluated in formal test loop
```

Required branch evaluation includes global, unary, pairwise, full, semantic-shuffle, private-shuffle, named-only, latent-only, evidence-shuffle and Pu-off.

---

# 25. Runtime gate

Actual-path profiles:

```text
8/4
6/5
4/8
```

All mechanisms must be enabled.

Reject if:

```text
reserved>45GB
core component disabled
DINO calls>1
NaN/Inf
```

Select fastest profile under42GB target.

Required artifacts:

```text
runtime_profile.json
runtime_steps.jsonl
selected_runtime_profile.json
```

---

# 26. Required tests

Run all RAEL tests plus ACPR-CalAlign regression tests.

Minimum:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_rael*.py -q
```

Then run source baseline tests to verify no regression in old path.

---

# 27. Gate artifacts

## PRE_PILOT

```text
.review/RAEL_OIA_V1_PRE_PILOT_READY.json
```

Must bind:

```text
git HEAD
branch
base commit
source tree hash
config hash
skill hash
test result
real DINO result
gradient result
runtime result
```

## FULL_TRAIN_READY

```text
.review/RAEL_OIA_V1_FULL_TRAIN_READY.json
```

Must additionally bind:

```text
pilot artifacts hash
mechanism ranges
counterfactual coverage
selected runtime profile
Pu audit
unresolved=[]
```

Any source/config/skill change invalidates prior gate artifacts.

---

# 28. Completion report

Codex must return:

```text
worktree
branch
local HEAD
remote HEAD
base ancestor
new/changed files
test counts
DINO shapes/call count
model output shapes
owner matrix
gradient admission proof
runtime profile
pilot metrics
mechanism diagnostics
Pu labels
review artifact paths
full command
unresolved issues
```

It must not say “implemented” if a component:

- exists but is not called;
- is called only in a smoke helper;
- is not in the optimizer;
- has zero parameter delta beyond bootstrap;
- returns placeholders;
- is disabled by the formal launch command.
