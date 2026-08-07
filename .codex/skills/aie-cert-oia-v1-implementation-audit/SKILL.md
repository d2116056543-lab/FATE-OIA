# AIE-CERT-OIA V1 Implementation Audit Skill

## Purpose

本 Skill 对以下实现执行 fail-closed 审查：

```text
repository: d2116056543-lab/FATE-OIA
source:     acpr_aie_oia_v1_direct_image@8a324b94b1cd6b4a4377655a1bd426f7d854fec0
target:     acpr_aie_cert_oia_v1_direct_image
method:     AIE-CERT-OIA V1
```

审查目标不是确认“代码能跑”，而是确认：

> **AIE-CERT 的 clean predicate、evidence atom conservation、bias-free contribution、multi-control certificate、primal-dual budget、signed Reason、ECPO、read-only naming、continuous schedule 和完整诊断均真实进入 formal forward/loss/backward/evaluator/checkpoint。**

`REVIEW_PASS` 只证明实现完整和机制真实，不证明训练指标必然超过基线。

---

# 1. 强制上下文

任何 Git、代码、测试、训练、评估、进程管理或 push 前读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

不得创建额外训练状态 Markdown。

再读取目标 worktree：

```text
docs/superpowers/plans/2026-08-07-aie-cert-oia-v1-implementation.md
configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml
configs/aie_cert_reason_counter_evidence.yaml
```

若计划、Skill、代码或 config 冲突：

```text
REVIEW_FAIL
记录冲突
不得静默选择容易实现的一方
```

---

# 2. 审查状态

## 2.1 REVIEW_PASS

证明：

```text
required files存在
formal import graph正确
所有核心公式/shape/owner正确
真实DINO动态审查通过
无placeholder/旁路/错误gradient
artifact与resume合同完整
```

输出：

```text
.review/aie_cert_oia_v1/REVIEW_PASS_AIE_CERT_OIA_V1.json
.review/aie_cert_oia_v1/AIE_CERT_REQUIREMENT_MATRIX.json
```

## 2.2 PILOT_PASS

证明唯一 3-epoch pilot：

```text
所有核心路径激活
数值有限
证书/ECPO/dual/预算健康
final未明显破坏primary
诊断artifact完整
```

## 2.3 FULL_TRAIN_READY

必须：

```text
REVIEW_PASS
RUNTIME_PROFILE_PASS
PILOT_PASS
HEAD/config/source tree hash一致
worktree clean
local HEAD == GitHub HEAD
```

---

# 3. Git/worktree hard checks

执行：

```powershell
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
git worktree list --porcelain
git rev-parse github/acpr_aie_oia_v1_direct_image
git ls-remote github refs/heads/acpr_aie_oia_v1_direct_image
git ls-remote github refs/heads/acpr_aie_cert_oia_v1_direct_image
```

要求：

```text
branch = acpr_aie_cert_oia_v1_direct_image
source remote HEAD = 8a324b94b1cd6b4a4377655a1bd426f7d854fec0
target worktree path =
  E:\sbw\FATE_Drive\fate_oia_acpr_aie_cert_oia_v1_worktree
target != original AIE worktree
original AIE worktree unchanged
target clean when REVIEW_PASS/full starts
local target HEAD == GitHub target HEAD
```

失败代码：

```text
WRONG_TARGET_BRANCH
SOURCE_HEAD_MISMATCH
TARGET_IS_SOURCE_WORKTREE
SOURCE_WORKTREE_MUTATED
DIRTY_TARGET_WORKTREE
REMOTE_HEAD_MISMATCH
```

---

# 4. Required files

根目录 `.gitignore` 必须包含精确规则 `.review/`，且不得以更宽泛规则误忽略 AIE-CERT 源码、测试、计划或 Skill。

必须存在实施计划中列出的所有新 config/model/dataset/loss/utils/engine/script/test/skill 文件。审查器必须硬编码 required list，不能只检查目录。

关键 required files：

```text
configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml
configs/aie_cert_reason_counter_evidence.yaml

fate_oia/models/aie_cert_calalign_foundation.py
fate_oia/models/aie_cert_sparse.py
fate_oia/models/aie_cert_predicate_bank.py
fate_oia/models/aie_cert_deformable_reread.py
fate_oia/models/aie_cert_atom_transport.py
fate_oia/models/aie_cert_evidence_interface.py
fate_oia/models/aie_cert_contribution_head.py
fate_oia/models/aie_cert_reason_rereader.py
fate_oia/models/aie_cert_naming.py
fate_oia/models/aie_cert_oia_model.py

fate_oia/datasets/aie_cert_structured_evidence.py

fate_oia/losses/aie_cert_losses.py
fate_oia/losses/aie_cert_constraints.py
fate_oia/losses/aie_cert_loss_registry.py

fate_oia/utils/aie_cert_counterfactual.py
fate_oia/utils/aie_cert_preference_queue.py
fate_oia/utils/aie_cert_calibration.py
fate_oia/utils/aie_cert_metrics.py
fate_oia/utils/aie_cert_artifacts.py

fate_oia/engine/train_aie_cert_oia.py
fate_oia/engine/eval_aie_cert_oia.py
fate_oia/engine/profile_aie_cert_oia.py
fate_oia/engine/audit_aie_cert_oia_implementation.py
fate_oia/engine/evaluate_aie_cert_oia_pilot.py
fate_oia/engine/supervise_aie_cert_oia_foreground.py

scripts/FATE_OIA_aie_cert_oia_v1_preflight.ps1
scripts/FATE_OIA_aie_cert_oia_v1_pilot.ps1
scripts/FATE_OIA_aie_cert_oia_v1_foreground.ps1

.codex/skills/aie-cert-oia-v1-implementation-audit/SKILL.md
docs/superpowers/plans/2026-08-07-aie-cert-oia-v1-implementation.md
```

失败：

```text
REQUIRED_FILE_MISSING
REVIEW_ARTIFACT_NOT_IGNORED
SOURCE_FILE_ACCIDENTALLY_IGNORED
```

---

# 5. Formal import graph

从：

```text
train_aie_cert_oia
eval_aie_cert_oia
profile_aie_cert_oia
supervise_aie_cert_oia_foreground
```

构建 AST/import graph。

允许：

```text
AIECertOIAModel
AIECert* modules
ACPR DINO/ego/trunk/predicate/reason grammar utilities
AIE generic artifact/hash/calibration utilities
```

禁止 formal graph 到达：

```text
AIEOIAModel
AIEEvidenceInterface
AIEContributionHead
AIEReasonRereader
AIEPredicateNaming
AIECounterfactualEngine

pair memory
hard pair
action-set final/marginalization
graph
PMI/co-occurrence delta
latent state / unknown state
cached logits / FrozenRunC
distillation
VLM/MLLM
feature cache
token compression
video input
second backbone
trainable threshold head
```

旧文件可以存在，但不得被 formal path import。

失败：

```text
OLD_AIE_FORMAL_PATH
FORBIDDEN_PAIR_PATH
FORBIDDEN_ACTION_SET_PATH
GRAPH_OR_PMI_FOUND
CACHED_OR_DISTILLED_PATH
CACHE_OR_COMPRESSION_FOUND
VLM_OR_VIDEO_PATH
TRAINABLE_THRESHOLD_FOUND
```

---

# 6. Config contract

检查：

```text
360x640
patch8
layers 3/7/11
action4 reason21 predicate32
direct_image=true
feature_cache_enabled=false
token_compression=none
eval_splits=[test]
best_selection_split=test
best_selection_metric=deploy_joint
internal_test_selected=true
publication_eligible=false
epochs=16
bf16
max_reserved_memory_gb=44.5
```

Counterfactual：

```text
interval=4
2 matched controls
min valid controls=3
wrong-probe
wrong-action
rerun_dino=false
```

ECPO：

```text
capacity=512
max_age=64
age_tau=32
verified threshold=0.50
```

失败：

```text
CONFIG_CONTRACT_MISMATCH
COUNTERFACTUAL_PROTOCOL_MISMATCH
ECPO_PROTOCOL_MISMATCH
PUBLICATION_FLAG_MISSING
```

---

# 7. Source forward equivalence

实例化：

```text
source AIECalAlignFoundation
AIECertCalAlignFoundation
```

加载相同 state，真实图像比较：

```text
action_logits_primary
reason_logits_primary
label_nodes
label_attention
predicate_logits
predicate_probs
predicate_attention
action_visual_logits_primary
action_reason_logits_primary
```

要求：

```text
fp32 max abs <1e-6
bf16 max abs <5e-4
```

detach 只改变梯度，不得改变 forward 数值。

失败：

```text
SOURCE_FORWARD_MISMATCH
```

---

# 8. DINO contract

真实图像：

```text
input [B,3,360,640]
patch [B,3,3600,384]
cls [B,3,384]
```

硬要求：

```text
all DINO params requires_grad=false
eval
no_grad
ordinary batch DINO calls=1
full test batch DINO calls=1
same-field branch audit extra calls=0
CF extra calls=0
```

失败：

```text
DINO_TRAINABLE
WRONG_DINO_SHAPE
DINO_CALL_DUPLICATION
COUNTERFACTUAL_REENCODES_DINO
BRANCH_AUDIT_REENCODES_DINO
```

---

# 9. Owner exact cover and gradient firewalls

owner：

```text
primary_core
predicate_visual
action_evidence
action_contribution
reason_private
naming_readout
```

全部 trainable parameters 恰好一个 owner。

构造独立 loss probes：

| loss | 允许 owner | 必须为0 |
|---|---|---|
| primary Action/Reason | primary_core | predicate_visual, all CERT owners |
| predicate structured | predicate_visual | primary_core, all CERT owners |
| final Action | action_evidence, action_contribution | primary_core, predicate_visual, reason_private, naming |
| final Reason | reason_private | all others |
| ECPO | reason_private | all others |
| naming | naming_readout | all others |
| CF necessity | action_evidence | primary/predicate/reason/naming |
| CF effect/budget | action_evidence and/or contribution per registry | primary/predicate/reason/naming |

额外硬检查：

```text
Reason targets不能对predicate_visual/action owners产生grad
naming不能对shared_predicate_keys产生grad
DINO grad=0
```

失败：

```text
OWNER_NOT_EXACT
PRIMARY_CONTAMINATED
REASON_TO_PREDICATE_LEAK
REASON_TO_ACTION_LEAK
NAMING_TO_EVIDENCE_LEAK
DINO_GRADIENT_FOUND
OWNER_NO_GRAD
```

---

# 10. Clean structured evidence

动态 synthetic + real-record 测试：

```text
predicate_source_complete逐predicate
object complete不自动令lane/drivable complete
verified counter需要显式contradictory predicate
Reason=0不能单独创建counter
model outputs不参与counter builder
```

输出 shape/coverage完整。

失败：

```text
SOURCE_COMPLETENESS_BROADCAST
UNVERIFIED_REASON_NEGATIVE
MODEL_SELF_COUNTER_FOUND
STRUCTURED_SHAPE_MISMATCH
```

---

# 11. Entmax and predicate mixture

## 11.1 entmax

测试：

```text
nonnegative
sum1
exact zeros exist
uniform for equal logits
finite grad
bf16 stable
```

## 11.2 shared identity

对象身份检查：

```python
model.evidence.predicate_bank is model.naming.predicate_bank
```

或 naming 显式接收同一 bank；禁止第二份 parameter。

state_dict 中 predicate key parameter 只能一份。

## 11.3 arithmetic mixture

AST 和 numerical test 必须证明：

```text
mix = sum(pi * predicate_map)
bias = log(mix)
```

构造两张互斥 map，arithmetic mixture应保留两处质量；geometric mixture会近零。动态结果必须符合 arithmetic 预期。

禁止源码出现等价几何公式：

```text
einsum(pi, log(map))
sum(pi * log(map))
prod(map ** pi)
```

## 11.4 fallback

predicate presence低于0.30：

```text
mixture zero
prior strength zero
visual map仍正常
```

失败：

```text
ENTMAX_INVALID
PREDICATE_KEY_DUPLICATED
GEOMETRIC_PREDICATE_PRIOR
VISUAL_FALLBACK_MISSING
PREDICATE_PRIOR_OUT_OF_BOUND
```

---

# 12. Evidence-conditioned local reread

通过 forward hook 捕获 offset head输入。

分别只改变：

```text
global token
map summary
probe
```

offset/input/local token必须变化。

要求：

```text
q_local包含三项
abs(offset)<=0.25
weights sum1
3 layers x 8 points
grid_sample或等效
```

失败：

```text
LOCAL_QUERY_NOT_EVIDENCE_CONDITIONED
LOCAL_REREAD_PLACEHOLDER
OFFSET_OUT_OF_RANGE
SAMPLING_WEIGHT_INVALID
LOCAL_REREAD_NO_EFFECT
```

---

# 13. Map-token co-transport

动态检查：

```text
transport matrix [B,4,4,4]
跨Action矩阵不存在
token_post变化
map_post变化
二者使用同一matrix和gamma
map sum1
probe permutation equivariant
```

设置 gamma=0：

```text
token_post==token_pre
map_post==map_pre
```

设置非零且修改一个 probe：

```text
同Action其他probe按A变化
其他Action不变化
```

正式下游 hook 必须确认 contribution/CF/Reason/naming读取 post-transport tensor。

失败：

```text
TOKEN_ONLY_TRANSPORT
MAP_ONLY_TRANSPORT
TRANSPORT_MATRIX_MISMATCH
CROSS_ACTION_TRANSPORT
PRE_TRANSPORT_BYPASS
TRANSPORT_NOT_EQUIVARIANT
```

---

# 14. Overlap ceiling

构造 overlap：

```text
0.40 -> zero loss
0.65 -> approximately zero
0.90 -> positive
inactive contribution -> no pair penalty
```

禁止旧 raw mean cosine objective进入 formal loss。

失败：

```text
RAW_ORTHOGONALIZATION_FOUND
OVERLAP_CEILING_INVALID
```

---

# 15. Background centering and contribution

检查：

```text
atom own region
selected topk excluded
fallback on insufficient background
centered token changes with background
```

contribution head：

```text
no bias attribute
state_dict无bias
zero centered -> zero raw contribution
field shuffle changes contribution
```

精确重建：

```text
fp32 <1e-6
bf16 <5e-4
```

失败：

```text
BACKGROUND_CENTER_MISSING
WRONG_REGION_BACKGROUND
CONTRIBUTION_BIAS_FOUND
IMAGE_INDEPENDENT_CONTRIBUTION
CONTRIBUTION_RECONSTRUCTION_FAIL
```

---

# 16. Multi-control counterfactual

必须生成四 control slots：

```text
matched1
matched2
wrong_probe
wrong_action
```

动态检查：

```text
matched seeds不同
mass match
overlap<=0.20
wrong probe own region
wrong action own region
>=3 valid才event valid
invalid reason真实记录
DINO calls=0
```

故意使一个 control invalid，应使用其余 valid controls；少于3时 entire event invalid。

失败：

```text
SINGLE_CONTROL_ONLY
CONTROL_SEED_DUPLICATED
CONTROL_MASS_MISMATCH
CONTROL_OVERLAP_EXCEEDED
WRONG_CONTROL_REGION
INVALID_EVENT_ACCEPTED
COUNTERFACTUAL_REENCODES_DINO
```

---

# 17. Robust certificate

numerical unit test：

```text
selected=0.6
controls=[0.1,0.2,0.0,0.1]
cert = selected - (mean + std)
```

实现结果必须匹配。

检查：

```text
control_std_multiplier=1
reliability=exp(-std/tau)
all finite
grad selected mask/evidence存在
control target按合同detach
```

禁止使用单 control gap作为 formal certificate。

失败：

```text
CERTIFICATE_FORMULA_MISMATCH
CERTIFICATE_SINGLE_CONTROL
CERTIFICATE_RELIABILITY_INVALID
CERTIFICATE_NO_GRAD
```

---

# 18. Primal-dual constraints

检查四个 lambda/EMA buffers：

```text
effect
necessity
action_budget
reason_budget
```

构造 violated/satisfied synthetic constraints：

```text
violation -> lambda上升
satisfied -> lambda不继续无界上升
clamp [0,10]
eval no update
```

checkpoint round trip：

```text
lambda
EMA
update count
```

resume后下一步与 uninterrupted一致。

禁止 lambda 是 AdamW parameter。

失败：

```text
DUAL_STATE_MISSING
DUAL_WRONGLY_OPTIMIZED
DUAL_UPDATE_DIRECTION
DUAL_EVAL_MUTATION
DUAL_RESUME_MISMATCH
```

---

# 19. Signed Reason priors

源码和动态检查禁止：

```python
abs(contribution)
predicate_signed.clamp_min(0)
```

构造同 map、相反 contribution：

```text
support prior与inhibit prior交换
```

构造 positive/contradictory predicate：

```text
support/counter prior均非零且不同
```

关闭 support/inhibit branch应分别改变 Reason logits。

失败：

```text
CONTRIBUTION_SIGN_LOST
PREDICATE_COUNTER_DROPPED
SIGNED_PRIOR_NO_EFFECT
SUPPORT_INHIBIT_IDENTICAL
```

---

# 20. Dynamic Reason budget

检查：

```text
uncertainty in [0,1]
agreement in [0,1]
budget in [0.10,current_max]
```

synthetic：

```text
low uncertainty -> near min
high uncertainty + high agreement -> high
high uncertainty + disagreement -> near min
```

Reason delta物理受 budget*kappa限制。

失败：

```text
REASON_BUDGET_RANGE
REASON_BUDGET_NOT_SAMPLE_LABEL_SPECIFIC
REASON_DELTA_EXCEEDS_BUDGET
REASON_AGREEMENT_PLACEHOLDER
```

---

# 21. ECPO and queue

## 21.1 pair validity

只允许：

```text
positive Reason target
vs external verified counter negative
```

禁止：

```text
model contradiction
attention max
unverified zero label
```

## 21.2 primary reference

numerical test必须匹配：

```text
-log sigmoid(beta*((final_i-final_j)-(primary_i-primary_j)))
```

primary logits detach。

## 21.3 queue

测试：

```text
capacity<=512
age<=64
age decay exp(-age/32)
per-label balance
duplicate sample pair禁止
checkpoint roundtrip
resume deterministic
```

无 pair：

```text
loss differentiable zero
inactivity reason logged
```

失败：

```text
ECPO_UNVERIFIED_PAIR
ECPO_PRIMARY_REFERENCE_MISSING
ECPO_PRIMARY_GRAD
QUEUE_OVER_CAPACITY
STALE_QUEUE_PAIR
QUEUE_UNBALANCED
QUEUE_RESUME_MISMATCH
```

---

# 22. Read-only naming

对象检查：

```text
no local predicate_keys parameter
shared keys passed detached
atom token/map detached
predicate outputs detached
certificate detached
```

gradient probe：

```text
naming_readout >0
all other owners =0
```

coverage 0 可通过实现审查，但 artifact必须包含 raw quality/confidence/margin/IoU；不得伪造 name。

失败：

```text
NAMING_KEY_DUPLICATION
NAMING_NOT_READONLY
NAMING_FAKE_COVERAGE
NAMING_ARTIFACT_MISSING
```

---

# 23. Schedule

检查 `schedule_values`：

```text
progress 0
0.05
0.08
0.10
0.15
0.20
0.50
1.00
```

必须连续、单调、有界。

resume同 update完全相同。

禁止：

```text
epoch==5 enable
epoch>=7 branch switch
replace model/optimizer stage
```

失败：

```text
SCHEDULE_DISCONTINUITY
SCHEDULE_OUT_OF_RANGE
RESUME_SCHEDULE_MISMATCH
HARD_STAGE_SWITCH_FOUND
```

---

# 24. Formal loss registry

每个 configured loss每 forward恰好一次。

inactive：

```text
differentiable zero
inactivity reason
active count
owner
```

禁止：

```text
placeholder constant detached zero
duplicate weight
loss computed but omitted total
loss in total但owner无grad
```

失败：

```text
LOSS_MISSING
LOSS_DUPLICATED
LOSS_DOUBLE_WEIGHTED
LOSS_PLACEHOLDER
LOSS_OWNER_NO_GRAD
```

---

# 25. Evaluation and calibration

每 epoch：

```text
only test evaluated
train-calib fits threshold
test never fits threshold
single DINO call per batch
full test primary/final from same field
```

calibration guard：

```text
EMA
max step
joint guard
Action guard
reject keeps previous accepted threshold
state checkpointed
```

fixed 128 audit：

```text
same sample IDs
same field
all branch variants real
not copied metrics
no extra DINO
not used for best
```

失败：

```text
VAL_EVALUATED
TEST_THRESHOLD_LEAK
TEST_ORACLE_WRITEBACK
CALIBRATION_GUARD_MISSING
BRANCH_METRICS_COPIED
BRANCH_REENCODES_DINO
BRANCH_USED_FOR_BEST
```

---

# 26. Required online diagnostics

审查 training source 与 3-update smoke，确保计划第28节所有关键 keys真实产生。

最低硬字段：

```text
schedule values
all raw/weighted losses
owner gradients
Action delta/contribution/reconstruction
predicate mixture/fallback
pre/post map entropy/overlap
transport matrix/gamma/token/map delta
local offsets/local-global ratio
background/centered norms
all 4 control drops
certificate/reliability
all constraints/lambdas
signed Reason priors
Reason uncertainty/agreement/budget/delta
ECPO pairs/queue age
naming raw fields
structured coverage
DINO calls/grad
memory/timing
```

字段存在但恒为 placeholder 0/None 失败，除非真实 inactivity reason允许且测试另有 active case。

失败：

```text
DIAGNOSTIC_FIELD_MISSING
DIAGNOSTIC_PLACEHOLDER
MECHANISM_NOT_OBSERVABLE
```

---

# 27. Artifact schema

检查计划第29节全部根目录和 per-epoch artifacts。

必须验证：

```text
JSON可解析
tensor可加载
file_names/targets/logits行数一致
threshold维度25
fixed audit IDs跨epoch一致
checkpoint pre-eval/latest/best存在
manifest记录完整命令/HEAD/config/source
```

失败：

```text
ARTIFACT_MISSING
ARTIFACT_SCHEMA_INVALID
FIXED_AUDIT_IDS_CHANGED
CHECKPOINT_MISSING
MANIFEST_INCOMPLETE
```

---

# 28. Checkpoint/resume exactness

保存：

```text
model
optimizer
scheduler/update
AMP scaler if used
dual state
ECPO queue
calibration EMA/accepted thresholds
RNG states
split/audit IDs
best metrics
global micro/update
```

2-update uninterrupted vs 1+resume+1：

```text
trainable parameters max abs <1e-7 fp32 test
dual exact
queue exact
schedule exact
calibration exact
```

失败：

```text
RESUME_MODEL_MISMATCH
RESUME_OPTIMIZER_MISMATCH
RESUME_DUAL_MISMATCH
RESUME_QUEUE_MISMATCH
RESUME_CALIBRATION_MISMATCH
RESUME_RNG_MISMATCH
```

---

# 29. Static forbidden patterns

AST/semantic scan，不可仅字符串误报。硬检查：

```text
AIECertContributionHead has no bias
Reason formal path no abs(contribution)
Reason counter path not clamp-to-zero
predicate mixture arithmetic
one shared predicate key parameter
naming detach
CF controls count>=4
queue age filter called
dual update called after optimizer update
schedule called every update
```

失败代码对应具体合同，不允许只写 generic `STATIC_FAIL`。

---

# 30. Targeted test suite

preflight 必须运行：

```powershell
python -m pytest `
  tests/test_aie_cert_source_regression.py `
  tests/test_aie_cert_sparse_predicate.py `
  tests/test_aie_cert_atom_transport.py `
  tests/test_aie_cert_local_reread.py `
  tests/test_aie_cert_contribution.py `
  tests/test_aie_cert_counterfactual.py `
  tests/test_aie_cert_constraints.py `
  tests/test_aie_cert_reason_signed.py `
  tests/test_aie_cert_ecpo_queue.py `
  tests/test_aie_cert_naming.py `
  tests/test_aie_cert_owner_firewalls.py `
  tests/test_aie_cert_schedule.py `
  tests/test_aie_cert_eval_artifacts.py `
  tests/test_aie_cert_runtime_contract.py `
  tests/test_aie_cert_static_contracts.py -q
```

同时运行相关旧 AIE/CalAlign regression tests，确保 dataset/DINO/foundation通用路径未破坏。

测试通过数量必须写入 REVIEW artifact，不能硬编码预期数量冒充执行。

---

# 31. Real-DINO dynamic audit

使用真实 train 图像，不使用随机 tensor：

```text
official checkpoint loaded
one 360x640 image
formal model
formal config
```

执行：

```text
forward
all shapes
all owner gradient probes
one valid multi-control CF event
one ECPO synthetic/real pair event
naming gradient probe
three optimizer updates
checkpoint/resume probe
```

要求 all finite，显存低于合同。

mock DINO 只能用于快速 unit tests，不足以生成 REVIEW_PASS。

---

# 32. Runtime profile gate

候选：

```text
8/4/16
7/4/16
6/5/16
5/6/16
6/5/8
```

每个包含真实 CF。选择 event-adjusted throughput最快且：

```text
reserved <44.5GB
growth <=0.25GB
```

profile artifact绑定：

```text
git HEAD
config hash
source tree hash
GPU name
torch/cuda
candidate table
selected candidate
```

失败：

```text
NO_RUNTIME_CANDIDATE
PROFILE_WITHOUT_CF
PROFILE_SHARED_GPU
PROFILE_BINDING_MISMATCH
```

---

# 33. Requirement Matrix

审查器必须生成实施计划定义的 C01-C32。

每项：

```json
{
  "id": "C12",
  "name": "map_token_cotransport",
  "implementation_symbols": [],
  "static_tests": [],
  "dynamic_checks": [],
  "runtime_artifact_keys": [],
  "status": "PASS|FAIL",
  "evidence": {}
}
```

全部 PASS 才能 REVIEW_PASS。

禁止：

```text
status默认PASS
仅文件存在即PASS
dynamic check缺失仍PASS
```

---

# 34. REVIEW_PASS 写入规则

只有以下全部成立才写：

```text
missing=[]
forbidden=[]
compile pass
targeted tests pass
old regressions pass
source equivalence pass
real-DINO pass
owner/firewalls pass
functional checks pass
resume pass
diagnostic keys pass
artifact schema smoke pass
runtime profile pass
C01-C32 all pass
worktree clean except ignored .review
local HEAD == GitHub HEAD
```

`REVIEW_PASS_AIE_CERT_OIA_V1.json` 必须包含：

```text
git_head
source_head
branch
worktree
remote_head
config_hash
source_tree_hash
skill_hash
plan_hash
required_files
test command/results
functional checks
gradient owners
runtime profile
requirement matrix hash
warnings
```

生成 REVIEW_PASS 后若代码/config/plan/skill改变，旧 PASS 自动失效。

---

# 35. Pilot gate

pilot配置：

```text
4096 train
512 calib
512 audit
512 test
3 epochs
single seed
```

严格读取真实 artifacts，不根据 stdout猜测。

硬 gates 使用实施计划第26节。尤其：

```text
CF events>=64
certificate positive rate>=0.40
correlation>0
ECPO pairs>=100
>=8 labels covered
reason delta RMS<1.5
final Action/Reason mAP each not >0.02 below primary
```

Naming coverage可0，但必须 raw quality非零且 no fake claim。

输出：

```text
AIE_CERT_PILOT_GATE.json
AIE_CERT_FULL_TRAIN_READY.json
```

若 fail，后者不得存在。

---

# 36. Full supervisor gate

full start前复核：

```text
REVIEW PASS binding
runtime binding
pilot binding
config binding
source tree binding
clean worktree
remote sync
GPU free
no stale duplicate trainer
```

foreground supervisor必须 stream child output，保存 command 和 exit code。不得用无日志的隐形替代进程。

---

# 37. Full-run completion audit

16 epochs全部完成后验证：

```text
16 test rows
16 pre-eval checkpoints
16 epoch directories
latest
all five best checkpoints
no val metrics
no test threshold fitting
no nonfinite
DINO frozen
component diagnostics present
```

输出：

```text
GOAL_COMPLETED_AIE_CERT_OIA_V1.json
```

内容必须同时报告同一 best-joint checkpoint：

```text
Act mF1/oF1/mAP
Exp mF1/oF1/mAP
joint
primary/final delta
threshold source
late drift
CF certificate
ECPO/Reason budget
component diagnosis
naming boundary
```

若未达到 `0.73/0.38`，仍可写 run completed，但：

```text
metric_goal_pass=false
```

不得把完成训练写成达到数值目标。

---

# 38. 审查者最终决策模板

```text
REVIEW_PASS / REVIEW_FAIL
PILOT_PASS / PILOT_FAIL
FULL_TRAIN_READY / BLOCKED
```

并按严重度输出：

```text
Hard blockers
Functional defects
Scientific-claim blockers
Runtime risks
Non-blocking warnings
```

每个 defect 必须包含：

```text
requirement ID
file:function
observed evidence
expected contract
required fix
```

不得只写泛化建议。
