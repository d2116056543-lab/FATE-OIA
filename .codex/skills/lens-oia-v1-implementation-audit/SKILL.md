# LENS-OIA V1 Implementation Audit Skill

## Purpose

本 Skill 用于 fail-closed 审查：

```text
branch: acpr_lens_oia_v1_direct_image
method: LENS-OIA V1
source: acpr_calalign_v1_2@373aa49feac17372574fd7fb056c1d79c7c848fe
```

审查目标不是“能运行”，而是确认上一轮 LENS-OIA 的核心概率模型、信息流、梯度所有权、运行协议和诊断输出都被完整实现并进入正式 forward/backward。

---

# 1. 强制前置读取

执行任何远程操作前读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

训练/实验状态只能追加到这三份 Markdown。

再读取：

```text
docs/superpowers/specs/2026-08-05-lens-oia-v1-design.md
docs/superpowers/plans/2026-08-05-lens-oia-v1-implementation.md
configs/fate_oia_train_360x640_lens_oia_v1.yaml
configs/lens_reason_state_schema.yaml
```

---

# 2. 审查状态

## REVIEW_PASS

证明：

```text
代码完整
formal路径真实调用
公式、shape、grad owner正确
real-DINO smoke正确
```

不证明指标有效。

## PILOT_PASS

证明唯一4轮pilot通过所有机制与非坍缩gate。

## FULL_TRAIN_READY

必须同时满足：

```text
REVIEW_PASS
PILOT_PASS
当前HEAD与pilot HEAD一致
config/source/schema/split hash一致
worktree clean
local HEAD==remote HEAD
```

---

# 3. Git/worktree审查

执行：

```powershell
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
git ls-remote origin refs/heads/acpr_lens_oia_v1_direct_image
git worktree list --porcelain
git rev-parse origin/acpr_calalign_v1_2
```

要求：

```text
target branch正确
source HEAD=373aa49...
target不是source worktree
source worktree未修改
target clean
local==remote
```

失败代码：

```text
SOURCE_HEAD_MISMATCH
WRONG_BRANCH
TARGET_IS_SOURCE_WORKTREE
SOURCE_WORKTREE_MUTATED
DIRTY_WORKTREE
REMOTE_HEAD_MISMATCH
```

---

# 4. Required files

必须存在：

```text
configs/fate_oia_train_360x640_lens_oia_v1.yaml
configs/lens_reason_state_schema.yaml
configs/lens_observability_groups.yaml

fate_oia/models/lens_calalign_foundation.py
fate_oia/models/lens_adaptive_evidence.py
fate_oia/models/lens_latent_state.py
fate_oia/models/lens_annotation_emission.py
fate_oia/models/lens_action_reread.py
fate_oia/models/lens_oia_model.py

fate_oia/datasets/lens_structured_evidence.py
fate_oia/datasets/lens_splits.py
fate_oia/datasets/lens_mirror.py

fate_oia/losses/lens_action_losses.py
fate_oia/losses/lens_reason_losses.py
fate_oia/losses/lens_latent_losses.py
fate_oia/losses/lens_grounding_losses.py
fate_oia/losses/lens_loss_registry.py

fate_oia/engine/train_lens_oia.py
fate_oia/engine/eval_lens_oia.py
fate_oia/engine/profile_lens_oia.py
fate_oia/engine/audit_lens_oia_implementation.py
fate_oia/engine/evaluate_lens_oia_pilot.py
fate_oia/engine/supervise_lens_oia_foreground.py

fate_oia/utils/lens_artifacts.py
fate_oia/utils/lens_calibration.py
fate_oia/utils/lens_metrics.py
fate_oia/utils/lens_contracts.py
fate_oia/utils/lens_hashes.py

scripts/FATE_OIA_lens_oia_v1_pilot.ps1
scripts/FATE_OIA_lens_oia_v1_foreground.ps1
```

---

# 5. Forbidden formal paths

静态 import graph、model constructor、trainer和config不得启用：

```text
ACPRPairMemory
matched_pair
pair_memory
ACPRActionComboAux
action_set_logits as final
ACPRThresholdHead inside formal model
trainable threshold optimizer
SAVEUtilityBridge
utility predictor
clean/private reason mixture
PCGrad
graph
PMI
cached_logits
FrozenRunC
tail_residual_adapter
VLM
MLLM
token compression
feature cache
```

旧文件可以保留，但 LENS formal import graph必须不可达。

失败：

```text
FORBIDDEN_PAIR_PATH
FORBIDDEN_ACTION_SET_PATH
TRAINABLE_THRESHOLD_FOUND
UTILITY_GATE_FOUND
PRIVATE_REASON_ROUTER_FOUND
CACHE_OR_COMPRESSION_FOUND
```

---

# 6. Source foundation审查

## 6.1 模块范围

`LENSCalAlignFoundation`只能包含：

```text
DINO
ego
scene predicate context
label trunk
source predicate reason
```

## 6.2 Real-image equivalence

相同随机种子、相同模块state、相同真实图像，对比：

```text
ACPROIAModel(threshold_enabled=False)
LENSCalAlignFoundation
```

检查：

```text
action source
reason source
label nodes
label attention
visual action
reason visual
reason-to-action
fusion gate
```

误差：

```text
fp32 <1e-6
bf16 <5e-4
```

失败：

```text
FOUNDATION_MODULE_MISSING
SOURCE_ACTION_MISMATCH
SOURCE_REASON_MISMATCH
SOURCE_TOKEN_MISMATCH
```

---

# 7. DINO审查

必须：

```text
input=360×640
layers=[3,7,11]
patch=[B,3,3600,384]
cls=[B,3,384]
requires_grad=False
eval mode
no_grad
普通batch calls=1
无cache read/write
```

失败：

```text
DINO_TRAINABLE
DINO_CALL_DUPLICATION
WRONG_PATCH_SHAPE
FEATURE_CACHE_FOUND
```

---

# 8. Adaptive evidence审查

输入：

```text
reason nodes [B,21,384]
patches [B,3,3600,384]
```

输出：

```text
map [B,21,3600]
null [B,21]
token [B,21,384]
temperature [B,21]
SNR [B,21]
entropy [B,21]
```

检查：

```text
sum(map)+null=1
tau∈[0.35,2.0]
region prior是有限soft bias
无hard top-k/ROI
无[B,21,3600,384]物化
evidence parameter grad>0
```

功能干预：

```text
shuffle reason query改变对应map
shuffle patch位置改变map
null高时map mass低
```

失败：

```text
MAP_NOT_NORMALIZED
NULL_MISSING
TEMPERATURE_CONSTANT_PLACEHOLDER
HARD_ROI_FOUND
DENSE_FACTOR_TOKEN_MATERIALIZED
EVIDENCE_NO_GRAD
```

---

# 9. 三状态posterior审查

必须使用可识别分解：

```text
v+ = (1-u)*sigmoid(z)
v- = (1-u)*(1-sigmoid(z))
v? = u
```

检查：

```text
shape=[B,21,3]
sum=1
state顺序固定
progress0 unknown=0
progress0 support logit=source reason visual
不能mean(dim=-1)替代state
```

progress0：

```text
clean log odds == source reason visual
```

失败：

```text
FREE_STATE_PERMUTATION
STATE_NOT_NORMALIZED
UNKNOWN_ACTIVE_AT_ZERO
STATE_MEAN_COLLAPSE
CLEAN_LOGODDS_MISMATCH
```

---

# 10. Ordered emission审查

对每标签：

```text
T+ > T? > T-
```

检查：

```text
有序参数由softplus增量构造
不是自由3参数排序后处理
group shrinkage存在
label delta bounded
identity ramp存在
train-main频率初始化
```

progress0：

```text
formal reason == source reason
```

测试 full progress：

```text
formal reason == logit(sum(state_prob*T))
```

失败：

```text
EMISSION_NOT_ORDERED
FREE_TRANSITION_MATRIX
IDENTITY_RAMP_MISSING
FORMAL_REASON_NOT_MARGINAL
```

---

# 11. Responsibility与冲突审查

检查实现：

```text
mV
mR
mA
JS conflict
annotation mass→unknown discount
gamma normalize
gamma.detach
```

数值：

```text
gamma sum=1
conflict∈[0,1]
相同mV/mR conflict≈0
对立mV/mR conflict高
conflict提高时unknown责任提高
```

Gradient attenuation：

```text
positive floor=0.25
zero floor=0.05
share weight不为0
高conflict共享梯度小于低conflict
```

失败：

```text
GAMMA_NOT_DETACHED
CONFLICT_NO_EFFECT
HARD_ZERO_GATE
ANNOTATION_DIRECT_TO_ACTION
```

---

# 12. Action base审查

检查：

```text
visual action来自source trunk
reason_to_action使用clean latent log odds
fusion gate使用source gate
raw benchmark marginal不进入action
emission参数不进入action graph
```

progress0：

```text
LENS action base == source action
```

失败：

```text
WEAK_NEW_ACTION_BASE
BENCHMARK_REASON_TO_ACTION
EMISSION_TO_ACTION_LEAK
ACTION_BASE_MISMATCH
```

---

# 13. Factor-local reread审查

必须：

```text
entmax over21+null
Action patch score读取完整3600
factor map只作为soft log bias
支持factor chunk
```

输出：

```text
local evidence
state-specific contribution
expected contribution
unnamed contribution
state substitution logits
```

检查：

```text
unknown named contribution=0
unnamed不恒0
factor-off可测
4个action factor分布不同
```

失败：

```text
NO_NULL_FACTOR
NOT_FULL_FIELD_REREAD
UNKNOWN_NAMED_CONTRIBUTION
FACTOR_ROUTE_NO_GRAD
```

---

# 14. Contribution守恒审查

检查：

\[
final-base
=
\alpha
\left(
sum_r bounded\_contribution_r
+
bounded\_unnamed
\right).
\]

误差：

```text
fp32 <1e-6
bf16 <5e-4
```

State substitution必须只替换目标factor contribution并重用同一cap。

失败：

```text
CONTRIBUTION_NOT_ADDITIVE
STATE_SUBSTITUTION_REENCODES
WRONG_CAP_IN_VARIANT
```

---

# 15. Conservative grounding审查

对 `lens_structured_evidence.py` 做静态与真实记录测试。

禁止：

```text
light presence→green
sign presence→stop
car→close and far
lane poly→solid
lane poly→turn lane
missing object→negative
```

检查：

```text
support/counter/unknown
source_complete
reliability
map target
source inventory
```

模型test forward不得接收structured data。

失败：

```text
OVERCLAIMED_GROUNDING
MISSING_AS_NEGATIVE
TEST_STRUCTURED_INPUT
```

---

# 16. Mirror审查

Action permutation：

```text
0→0,1→1,2↔3
```

Reason/factor：

```text
9↔15
10↔16
11↔17
12↔18
13↔19
14↔20
```

检查：

```text
labels交换
state交换
map channel交换+水平翻转
original/paired一次DINO
```

失败：

```text
LEFT_RIGHT_LABEL_NOT_SWAPPED
MAP_NOT_MIRRORED
PAIRED_VIEW_DUPLICATES_DINO
```

---

# 17. Loss registry审查

每个配置loss：

```text
调用一次
加权一次
owner唯一
raw value非placeholder
```

不得出现：

```text
配置有权重但raw loss永远0
模型输出key与loss读取key不一致
嵌套loss未被registry读取
```

失败：

```text
LOSS_MISSING
LOSS_DUPLICATED
LOSS_DOUBLE_WEIGHTED
LOSS_KEY_MISMATCH
PLACEHOLDER_ZERO_LOSS
```

---

# 18. Autograd owner审查

分别运行：

```text
action-only backward
reason-annotation-only backward
latent-state-only backward
emission-only backward
grounding-only backward
```

硬要求：

```text
reason annotation → action head=0
reason annotation → action reread=0
action → emission=0
grounding → DINO=0
action → evidence/state/reread>0
emission loss → emission>0
state loss → state/evidence>0
```

失败：

```text
REASON_TO_ACTION_GRAD_LEAK
ACTION_TO_EMISSION_GRAD_LEAK
DINO_GRAD_LEAK
OWNER_NO_GRAD
```

---

# 19. Optimizer owner审查

Trainable参数必须恰好属于一个group：

```text
foundation
adaptive_evidence
latent_state
action_reread
annotation_emission
```

运行两个optimizer updates后记录：

```text
grad norm
parameter delta
lr
clip前后norm
```

失败：

```text
UNOWNED_PARAMETER
DUPLICATE_OWNER
OWNER_NO_UPDATE
```

---

# 20. Calibration/test协议审查

必须：

```text
train-main/audit/calib互斥
train-calib拟合threshold/temperature
calibration前后model hash相同
每epochtest一次
best=test deploy_joint
manifest标internal_test_selected
```

禁止：

```text
val loader
checkpoint_best_val
test threshold写回
test oracle选best
```

失败：

```text
SPLIT_OVERLAP
CALIBRATION_MUTATES_MODEL
TEST_THRESHOLD_LEAK
VAL_PROTOCOL_FOUND
MANIFEST_MISSING_INTERNAL_FLAG
```

---

# 21. Runtime审查

Real-DINO profile覆盖：

```text
batch6/chunk21/workers4
batch6/chunk21/workers8
batch6/chunk7/workers4
batch5/chunk21/workers8
```

硬要求：

```text
reserved<45GB
无OOM
普通batch DINO=1
branch test不重编码
无持续显存增长
```

失败：

```text
MEMORY_OVER_LIMIT
PROFILE_OMITS_CORE_PATH
TEST_BRANCH_REENCODES
MEMORY_LEAK
```

---

# 22. Artifact/resume审查

Checkpoint保存：

```text
model
optimizer
scheduler/update count
RNG
split manifest
emission initialization
running statistics
calibration
config/source/schema hashes
```

Resume下一step应在容差内复现不中断路径。

每epoch artifacts必须包含：

```text
raw/deploy metrics
source/base/final branches
per-label AP/F1/AUC
state/emission/conflict summaries
factor contributions
gradient owner
runtime
fixed audit subset
synthetic flip audit
```

失败：

```text
RESUME_INCOMPLETE
ARTIFACT_HASH_MISSING
BRANCH_ARTIFACT_MISSING
```

---

# 23. Required commands

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\lens_*.py `
  fate_oia\losses\lens_*.py `
  fate_oia\datasets\lens_*.py `
  fate_oia\engine\*lens_oia*.py `
  fate_oia\utils\lens_*.py
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_lens_*.py -q
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_dino_field.py `
  tests\test_acpr_label_trunk.py `
  tests\test_acpr_model_forward.py `
  tests\test_bdd_oia_dataset.py -q
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_lens_oia_implementation `
  --config configs\fate_oia_train_360x640_lens_oia_v1.yaml `
  --output-dir .background_runs\lens_oia_v1_preflight `
  --device cuda
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.profile_lens_oia `
  --config configs\fate_oia_train_360x640_lens_oia_v1.yaml `
  --device cuda
```

---

# 24. REVIEW输出

生成：

```text
LENS_IMPLEMENTATION_REVIEW.json
LENS_RUNTIME_PROFILE.json
```

字段：

```text
status
git_head
source_head
config_hash
schema_hash
split_seed
checked_files
forbidden_paths
equivalence
functional_checks
gradient_ownership
optimizer_ownership
runtime
failures
warnings
```

只有所有硬检查通过：

```text
status=REVIEW_PASS
```

---

# 25. 唯一Pilot审查

Pilot：

```text
4096 main
1024 audit
512 calib
512 test
4 epochs
1 seed
```

审查器必须从原始：

```text
logits
labels
file order
state posterior
emission
conflict
contributions
deletion variants
gradients
runtime
```

独立重算gate，不能信任trainer自报pass。

## Gate A

```text
progress0 exact
one DINO
no cache/compression
runtime healthy
```

## Gate B

```text
ordered emission
state not all unknown/present
anchor direction correct
```

## Gate C

```text
flip AUROC>=0.70
flip unknown>clean
gradient robustness ratio<=0.70
```

## Gate D

```text
Action no collapse
至少一轮mAP base+0.001
factor route active
contribution exact
```

## Gate E

```text
Exp mAP source-0.005以内
Exp mF1 source-0.010以内
latent branch active
```

## Gate F

```text
selected>control
target>wrong factor
state swap target-specific
```

## Gate G

```text
owner firewalls
all owners update
logs complete
```

---

# 26. Pilot pass输出

生成：

```text
LENS_PILOT_RAW_EVIDENCE.json
LENS_PILOT_GATES.json
LENS_PILOT_PASS.json
LENS_FULL_TRAIN_READY.json
```

绑定：

```text
git_head
config_hash
source_hash
schema_hash
split_hash
checkpoint_hash
logits_hash
labels_hash
file_order_hash
```

任何核心代码/config/schema变化：

```text
invalidate PILOT_PASS
invalidate FULL_TRAIN_READY
重新pilot
```

---

# 27. Full train启动审查

启动前重新核对：

```text
branch正确
worktree clean
local==remote
HEAD匹配review/pilot
14 epochs
test every epoch
best test deploy_joint
bf16
no cache/compression
```

不得因低metric自动停止；只允许结构性故障停止。

---

# 28. 审查报告必须回答

```text
1. source CalAlign是否真正恢复？
2. 21个evidence map是否真实读取3600 patches？
3. unknown是否是可学习状态而非常数？
4. emission是否严格有序？
5. formal Exp是否就是state marginal？
6. raw reason annotation是否被conflict折扣？
7. reason annotation是否完全无法更新Action参数？
8. Action是否使用latent clean state？
9. factor-local reread是否进入formal final？
10. named+unnamed是否严格重构Action residual？
11. grounding是否保守且train-only？
12. mirror是否正确交换左右语义？
13. 每个loss是否真正进入total/backward？
14. 每个owner是否有非零更新？
15. 日志是否足以无需额外重跑判断组件？
16. pilot是否满足full train条件？
```

任一问题不能由代码和artifact回答：

```text
REVIEW_FAIL
```

---

# 29. 审查边界

`REVIEW_PASS`只能证明：

```text
代码完整
机制真实调用
梯度/公式/协议正确
```

不能证明：

```text
Act_mF1必然达到0.73
Exp_mF1必然达到0.38
```

数值结论必须来自真实pilot/full run。
