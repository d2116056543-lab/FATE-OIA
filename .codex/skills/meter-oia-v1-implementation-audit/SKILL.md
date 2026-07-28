# METER-OIA V1 Implementation Audit Skill

## 1. Authority

This is the fail-closed audit contract for:

> METER-OIA V1  
> Meta-Validated Evidence Transport and Explanation Rectification

It verifies implementation completeness, invocation, gradients, optimizer ownership, diagnostics, runtime and pilot readiness.

Passing does not guarantee final performance. Passing means the planned method—not a weakened surrogate—has been implemented and exercised.

---

# 2. Mandatory project records

Before any repository action, read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Experiment status must be appended only to those files.

Reject creation of duplicate implementation/training/audit status Markdown files inside the repository.

---

# 3. Repository identity

Expected:

```text
repo: d2116056543-lab/FATE-OIA
base branch: acpr_calalign_v1_2
base SHA: 373aa49feac17372574fd7fb056c1d79c7c848fe
branch: acpr_meter_oia_v1_direct_image
worktree: E:\sbw\FATE_Drive\fate_oia_acpr_meter_oia_v1_worktree
remote: github
```

Reject if:

- wrong branch;
- base SHA not ancestor;
- source worktree modified;
- local/remote HEAD mismatch before pilot/full;
- worktree dirty when a readiness artifact is generated.

---

# 4. Non-negotiable task constraints

```text
single RGB frame
360×640
4 actions + 21 reasons
frozen official DINO ViT-S/8
layers 3/7/11
full 3600 patch tokens
no feature cache
no token compression
no video
no distillation
no VLM/MLLM
no per-image caption
no graph/PMI/static adjacency
BDD100K train-only
test-time RGB only
test-every-epoch internal protocol
test-selected best marked non-publication-eligible
```

---

# 5. Required files and call graph

Verify all files listed in the implementation plan exist, compile and are imported by the formal METER entrypoint.

Reject unresolved:

```text
TODO
FIXME
pass
NotImplementedError
constant zero diagnostics
placeholder tensor
mock-only branch
branch called only by tests
```

Every formal component must be reachable from:

```text
train_acpr_meter_oia.py
→ METEROIAModel.forward/decode_from_field
→ loss
→ optimizer
```

---

# 6. Forbidden formal-path components

Reject if METER final prediction depends on:

```text
ACPRPairMemory
matched-pair mining
action-set marginalization
trainable threshold head
trainable calibration head
token compressor
cached logits/features
history checkpoint teacher
reason label/logit fed into action
BDD100K geometry fed into test forward
```

---

# 7. DINO contract

Instantiate real DINO.

Expected:

```text
patch_tokens_by_layer [B,3,3600,384]
cls_tokens_by_layer   [B,3,384]
grid=(45,80)
```

Assertions:

```python
all(not p.requires_grad for p in dino.parameters())
dino.training is False
dino_call_count == 1  # ordinary batch
```

Backward from METER losses:

```text
DINO grads =0
downstream grads >0
```

No visual features may be written to disk.

Counterfactual must reuse in-memory field.

Meta event may run one additional audit-batch DINO call, which must be separately logged.

---

# 8. Foundation equivalence gate

Load source ACPROIAModel with threshold disabled and METER foundation with mapped identical state.

On the same real images:

```text
action max abs error <1e-6
reason max abs error <1e-6
label node error <1e-6
label attention error <1e-6
```

The comparison must include:

```text
ego-region update
scene predicate head
predicate cross-attention
predicate reason delta
reason-to-action
fusion gate
```

Reject “same architecture in principle” without numerical equality.

At `progress=0`, METER final action/reason must equal foundation outputs.

---

# 9. Signed grounding gate

The signed builder must emit support, counter and unknown separately.

Run synthetic and real-record tests.

Reject if:

```text
any traffic light -> green
any traffic sign -> stop sign
any lane -> solid boundary
missing object -> reliable absence
missing annotation -> negative
front-car turning inferred from static box alone
```

For each factor report:

```text
support count
counter count
unknown count
source coverage
groundability
```

Any named grounded factor without enough real support/counter examples must be downgraded.

---

# 10. Factor map gate

For each factor, support/counter distributions must normalize over:

```text
3600 patches + null
```

They must not normalize over factors.

Tests:

```text
sum(patches + null)≈1
multiple factors may attend same patch
all-null input can select null
maps not forced to area 1/21
```

Reject:

```text
map entropy always ≈ln(3600)
all factor maps identical
null always 0 or 1
support/counter identical
```

Only residual gamma may start at zero. Q/K/V and evidence projections must have normal initialization.

After two optimizer updates:

```text
factor internal grads >0
support/counter maps change
factor output parameter delta >0
```

---

# 11. Layer reading gate

Foundation must use the source-compatible layer mean.

Signed detail layer weights must remain bounded near uniform at initialization.

Reject query layer routing if one layer exceeds0.85 for almost all factors/samples during pilot without explicit justification.

Log per-factor layer weights and entropy.

---

# 12. Reliability gate

Verify:

```text
null high -> reliability lower
support≈counter -> reliability lower
counter evidence is not represented as 1-reliability
```

Early reliability floor must be implemented and scheduled.

The semantic expert must still receive gradients when learned reliability is initially low.

---

# 13. Semantic action expert gate

Required:

```text
21 factors + null
SECA-style sparse weights
factor value per action
exact signed contribution
semantic logits directly supervised
```

Numerical identity:

```text
semantic_logit == bias + contribution.sum(reason)
error <1e-6 fp32
```

Reject if semantic expert is only:

```text
bounded residual on visual logits
diagnostic head
auxiliary loss without final path
```

After warm-up, semantic contribution RMS/visual RMS must be measurable and nonzero.

---

# 14. Peer selector gate

Required branches:

```text
visual
semantic
peer candidate
final
```

`progress=0` final=visual exactly.

`progress=1` final=peer candidate exactly.

Selector must be sample/action dependent.

Reject:

```text
all lambdas constant
all lambdas≈1 forever
all lambdas≈0 forever
semantic branch stronger but final systematically weaker without selector-regret activation
```

Selector-regret must be in formal loss and have nonzero gradient.

---

# 15. Private reason decoder gate

Required:

```text
global private expert
grounded local private expert
decision context
factor context
mixture gate
annotation residual
final reason
```

Global and local experts each receive direct loss from step0.

Action must be invariant to:

```text
shuffle private reason tokens
zero annotation residual
change PU target
change private thresholds
```

Reason may read detached action/factor context.

---

# 16. Reason firewall gate

Dynamic autograd tests:

```text
d(z_action)/d(z_reason_private)=0
d(z_action)/d(annotation_delta)=0
d(z_action)/d(PU_target)=0
d(L_reason_private)/d(action_head)=0
d(L_reason_private)/d(foundation_core)=0
```

Only meta adapter may receive controlled reason gradient.

---

# 17. Meta adapter gate

Every factor has a low-rank adapter.

Action path uses its output and action loss updates it.

Reason input uses:

```text
detach(action factor)
+ omega * gradient bridge through meta adapter
```

Tests:

```text
omega=0 -> reason grad to adapter=0
omega=1 -> reason grad to adapter>0
action grad to adapter>0 regardless of omega
reason grad to core factor=0
```

---

# 18. Meta utility gate

Meta utility must:

1. select configured factors;
2. compute action and reason gradients on current train batch;
3. form action-only and action+reason virtual adapter states;
4. encode one train-audit batch once;
5. decode both candidates from the same field;
6. compute held-out action losses;
7. update utility EMA and omega;
8. leave real parameters unchanged until normal optimizer step.

Reject if utility uses:

```text
test
official validation
cached visual features
current-batch action loss only
gradient cosine in place of audit utility
```

Synthetic test:

- construct a reason gradient that improves audit action;
- omega must increase;
- construct a harmful reason gradient;
- omega must decrease/remain zero.

Log wall time and DINO calls.

---

# 19. PU gate

Initial:

```text
all PU lambdas=0
```

Activation requires hidden-positive audit and per-label evidence.

PU gradients allowed:

```text
private reason decoder
annotation residual
```

PU gradients forbidden:

```text
foundation
factor maps
semantic action
selector
meta adapter
```

Reliability and PU targets must be detached.

---

# 20. Counterfactual gate

Counterfactual must use factor patch maps, not slots.

Verify:

```text
selected top-mass patch set
equal-count control
same region
low overlap
feature-norm match
neighbor-mean replacement
no DINO rerun
```

Required directions:

```text
support deletion lowers factor support
counter deletion raises factor posterior / weakens veto
selected effect > control
target effect > wrong-target effect
```

Pilot coverage:

```text
4/4 actions
>=12/21 factors
```

A zero valid count is allowed only with explicit reason and cannot pass pilot if persistent.

---

# 21. Loss gate

Verify formal loss contains and updates:

```text
action final
action visual
action semantic
selector regret
reason final
reason global
reason local
ranking
SoftF1
grounding
counterfactual
```

No branch may depend on its final gate before receiving any direct gradient.

Check actual configured weights, not only default constants.

---

# 22. Warm-up gate

All components must exist from step0.

Only continuous ramps are allowed.

At progress0:

```text
final action=CalAlign action
final reason=CalAlign reason
semantic/global/local branches still compute and receive direct losses
```

At progress1:

```text
full METER paths active
```

Reject epoch-based hard activation of factor, selector, reason local or meta modules.

PU data-driven activation is allowed.

---

# 23. Calibration gate

No trainable threshold parameter in representation optimizer.

Each epoch:

```text
freeze model
fit train-calib
evaluate test
```

Verify:

```text
mAP unchanged
threshold RMS bounded
fallback when deploy mF1 degrades
```

Manifest must mark results as internal test-selected.

---

# 24. Formal forward gate

Default test call:

```python
out = model(images)
```

No labels/geometry/text/epoch/cache arguments.

Required shapes and finite values for all outputs listed in the plan.

`decode_from_field` must support branch diagnostics without another DINO call.

---

# 25. Diagnostic branch gate

Each epoch formal evaluator must calculate:

Action:

```text
visual
semantic
peer
final
factor_off
factor_shuffle
support_only
counter_only
meta_off
```

Reason:

```text
CalAlign reason
global
local
mix
final
annotation off
factor context off
map shuffle
decision context off
meta off
```

Reject branch metrics generated by a separate smoke-only helper.

---

# 26. Optimizer ownership gate

Required owner groups and nonzero parameter deltas:

```text
foundation
factor evidence
semantic action
selector
reason global
reason local
annotation
meta adapter
PU private
```

No parameter may appear in two optimizer groups.

DINO appears in none.

Record step count, grad norm and parameter delta.

---

# 27. Runtime gate

Profile:

```text
16/2
12/3
8/4
6/5
workers 4/8
```

All formal core mechanisms enabled.

Separate amortized timing for:

```text
counterfactual
meta utility
calibration
```

Reject profile if:

```text
reserved>=45GB
NaN/Inf
DINO ordinary call>1
core branch disabled
```

Select fastest under target constraints.

---

# 28. Training protocol gate

Internal protocol:

```text
12 epochs
one seed
test each epoch
test deploy joint selects primary best
train-audit only for meta/PU
train-calib only for calibration
```

Artifacts must say:

```text
internal_test_selected=true
publication_eligible=false
```

---

# 29. Resume gate

Checkpoint must restore all states listed in the plan.

Run resume-equivalence test for one update.

Reject partial model-only resume as formal full-train resume.

---

# 30. Required tests

Run:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_meter*.py -q
```

Then source CalAlign regression tests.

Also run:

```text
real DINO forward
3 real optimizer updates
branch evaluation
counterfactual event
meta event
```

---

# 31. Readiness artifacts

## PRE_PILOT

```text
.review/METER_OIA_V1_PRE_PILOT_READY.json
```

Bind:

```text
branch
HEAD
base SHA
source tree hash
config hash
schema hash
Skill hash
test results
real DINO result
foundation equivalence
runtime selection
```

## FULL_TRAIN_READY

```text
.review/METER_OIA_V1_FULL_TRAIN_READY.json
```

Also bind:

```text
pilot artifact hashes
action branch ranges
reason branch ranges
meta omega distribution
counterfactual coverage
PU state
unresolved=[]
```

Any source/config/schema/Skill change invalidates prior readiness artifacts.

---

# 32. Completion report

Codex must return:

```text
worktree
branch
local SHA
remote SHA
base ancestor
changed files
test counts
foundation equivalence error
DINO shapes/call count
model output shapes
optimizer owners
branch parameter deltas
runtime profile
pilot branch metrics
factor map statistics
meta utility statistics
PU active labels
counterfactual coverage
readiness artifacts
full command
unresolved issues
```

It must not report “complete” if a component merely exists but is not in the formal path.
