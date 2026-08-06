# AIE-OIA V1 Implementation Audit Skill

## Purpose

本 Skill 对以下实现执行 fail-closed 审查：

```text
repository: d2116056543-lab/FATE-OIA
source:     acpr_calalign_v1_2@373aa49feac17372574fd7fb056c1d79c7c848fe
target:     acpr_aie_oia_v1_direct_image
method:     AIE-OIA V1
```

审查目标不是确认“模型能跑”，而是确认：

> Action-Induced Evidence Interface 的所有核心机制均已在正式 forward、正式 loss、正式 backward、正式 evaluator 和正式 artifact 中实现，没有空占位、错误调用、被旁路、错误梯度所有权或更强但未被 formal 选择的隐藏分支。

`REVIEW_PASS` 只能证明代码和机制完整；不能证明 Action/Explanation 指标一定超过强基线。

---

# 1. 强制上下文

执行任何 Git、代码、测试、训练、评估、进程管理或 push 前读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

不得创建额外训练状态 Markdown。

再读取目标 worktree 中：

```text
docs/superpowers/specs/2026-08-06-aie-oia-v1-design.md
docs/superpowers/plans/2026-08-06-aie-oia-v1-implementation.md
configs/fate_oia_train_360x640_aie_oia_v1.yaml
configs/aie_scene_predicates.yaml
configs/aie_reason_counter_evidence.yaml
```

如果实现计划与代码冲突，以实施计划和本 Skill 的硬合同为准；必须记录冲突，不能静默选择较容易实现的一方。

---

# 2. 审查状态定义

## 2.1 `REVIEW_PASS`

证明：

```text
required files存在
formal路径真实调用
形状/公式/梯度owner正确
真实DINO smoke通过
无禁用路径
runtime与artifact合同存在
```

不证明性能。

## 2.2 `PILOT_PASS`

证明：

```text
唯一4轮pilot完成
primary未被破坏
Action/Reason新增路径真实激活
counterfactual方向正确
probe/predicate/naming不坍缩
owner firewall通过
```

## 2.3 `FULL_TRAIN_READY`

必须同时满足：

```text
REVIEW_PASS
PILOT_PASS
当前HEAD等于review/pilot绑定HEAD
config/schema/source/split hash一致
worktree clean
local HEAD == remote HEAD
```

---

# 3. Git/worktree 审查

执行：

```powershell
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
git ls-remote origin refs/heads/acpr_aie_oia_v1_direct_image
git rev-parse origin/acpr_calalign_v1_2
git worktree list --porcelain
```

要求：

```text
branch = acpr_aie_oia_v1_direct_image
source HEAD = 373aa49feac17372574fd7fb056c1d79c7c848fe
target worktree不是source worktree
source worktree状态未改变
target worktree clean
local target HEAD == GitHub target HEAD
```

失败代码：

```text
SOURCE_HEAD_MISMATCH
WRONG_TARGET_BRANCH
TARGET_IS_SOURCE_WORKTREE
SOURCE_WORKTREE_MUTATED
DIRTY_TARGET_WORKTREE
REMOTE_HEAD_MISMATCH
```

---

# 4. Required files

必须存在：

```text
configs/fate_oia_train_360x640_aie_oia_v1.yaml
configs/aie_scene_predicates.yaml
configs/aie_reason_counter_evidence.yaml

fate_oia/models/aie_calalign_foundation.py
fate_oia/models/aie_evidence_interface.py
fate_oia/models/aie_deformable_reread.py
fate_oia/models/aie_contribution_head.py
fate_oia/models/aie_predicate_naming.py
fate_oia/models/aie_reason_rereader.py
fate_oia/models/aie_oia_model.py

fate_oia/datasets/aie_structured_evidence.py
fate_oia/datasets/aie_splits.py

fate_oia/losses/aie_losses.py
fate_oia/losses/aie_loss_registry.py

fate_oia/utils/aie_counterfactual.py
fate_oia/utils/aie_calibration.py
fate_oia/utils/aie_metrics.py
fate_oia/utils/aie_artifacts.py
fate_oia/utils/aie_contracts.py
fate_oia/utils/aie_hashes.py

fate_oia/engine/train_aie_oia.py
fate_oia/engine/eval_aie_oia.py
fate_oia/engine/profile_aie_oia.py
fate_oia/engine/audit_aie_oia_implementation.py
fate_oia/engine/evaluate_aie_oia_pilot.py
fate_oia/engine/supervise_aie_oia_foreground.py

scripts/FATE_OIA_aie_oia_v1_pilot.ps1
scripts/FATE_OIA_aie_oia_v1_foreground.ps1

.codex/skills/aie-oia-v1-implementation-audit/SKILL.md
docs/superpowers/specs/2026-08-06-aie-oia-v1-design.md
docs/superpowers/plans/2026-08-06-aie-oia-v1-implementation.md
```

失败：

```text
REQUIRED_FILE_MISSING
```

---

# 5. Forbidden formal paths

静态 import graph、model constructor、trainer、config 和正式 evaluator 不得启用：

```text
ACPRPairMemory
matched pair mining
hard pair
pair memory enqueue
ACPRActionComboAux
16-action-set as final
action_set marginalization
ACPRCalibrationHead as representation
trainable ACPRThresholdHead in model optimizer
SAVEUtilityBridge
LENS latent state
annotation emission
unknown absorbing state
clean/private route selector
graph
PMI
static co-occurrence logit delta
FrozenRunC
RunC checkpoint
cached_logits
tail residual adapter
distillation
VLM
MLLM
feature cache
token compression
second visual backbone
video input
```

旧源文件可以继续存在，但 AIE formal import graph不得到达上述实现。

允许：

```text
source ACPR utility functions
source DINO
source ego encoder
source scene predicate head
source label trunk
source predicate reasoner
source metric/threshold search utility
```

失败代码：

```text
FORBIDDEN_PAIR_PATH
FORBIDDEN_ACTION_SET_PATH
TRAINABLE_THRESHOLD_FOUND
LATENT_EMISSION_FOUND
GRAPH_OR_PMI_FOUND
CACHED_OR_DISTILLED_PATH
VLM_OR_VIDEO_PATH
CACHE_OR_COMPRESSION_FOUND
```

---

# 6. Config 审查

检查：

```text
image = 360×640
patch_size = 8
DINO layers = [3,7,11]
action_dim = 4
reason_dim = 21
direct_image = true
feature_cache_enabled = false
token_compression = none
best_selection_split = test
best_selection_metric = deploy_joint
internal_test_selected = true
publication_eligible = false
epochs = 20
precision = bf16
```

Counterfactual：

```text
interval = 4 optimizer updates
batch fraction <=0.50
rerun_dino = false
```

Runtime：

```text
max_reserved_memory_gb = 45
test_every_epoch = true
```

失败：

```text
CONFIG_CONTRACT_MISMATCH
PUBLICATION_FLAG_MISSING
COUNTERFACTUAL_PROTOCOL_MISMATCH
```

---

# 7. Source foundation 审查

## 7.1 模块范围

`AIECalAlignFoundation` 只能包含：

```text
ACPRDinoFieldExtractor
ACPREgoRegionEncoder
ACPRScenePredicateHead
ACPRLabelTrunk
ACPRPredicateReasoner
```

不得包含 pair、combo、calibration、threshold。

## 7.2 API

必须实现：

```python
encode_images(images)
decode_field(field)
forward(images)
load_from_acpr_state_dict(state_dict)
```

## 7.3 Real-image numerical equivalence

实例化：

```text
source ACPROIAModel(threshold_enabled=False)
AIECalAlignFoundation
```

将 source state映射到 foundation。

相同真实图像比较：

```text
action_logits_base ↔ action_logits_primary
reason_logits_base ↔ reason_logits_primary
label_nodes
label_attention
predicate_logits
predicate_probs
predicate_attention
action_visual_logits
action_reason_logits
action_fusion_gate
```

要求：

```text
fp32 max abs <1e-6
bf16 max abs <5e-4
```

失败：

```text
FOUNDATION_MODULE_MISSING
SOURCE_ACTION_MISMATCH
SOURCE_REASON_MISMATCH
SOURCE_TOKEN_MISMATCH
SOURCE_PREDICATE_MISMATCH
```

---

# 8. Primary direct supervision 审查

检查 loss registry中从第一个 update 到最后一个 update存在：

```text
primary_action
primary_action_visual
primary_action_reason
primary_reason_partial
primary_reason_soft_f1
predicate_cls
predicate_map
predicate_reason_align
```

任何 primary loss不得被 progress/ramp关闭。

检查：

```text
primary owner grad >0
primary parameter delta >0
primary logits每轮保存
```

失败：

```text
PRIMARY_LOSS_MISSING
PRIMARY_LOSS_SCHEDULED_OFF
PRIMARY_OWNER_NO_GRAD
PRIMARY_BRANCH_LOG_ONLY
```

---

# 9. Primary trajectory isolation

这是 AIE 最重要的保护检查。

## 9.1 构造两个完全相同的模型

```text
A：只计算primary losses
B：计算primary + final Action + final Reason + predicate/naming/CF
```

同：

```text
initial state
batch order
random seed
optimizer
LR
weight decay
accumulation
```

连续2个 optimizer updates。

## 9.2 比较 primary 参数

参数范围：

```text
foundation.ego
foundation.predicate_head
foundation.trunk
foundation.predicate_reason
```

要求：

```text
max abs parameter difference <1e-7
```

如果 CF event涉及随机性，固定 event off或固定相同seed；该测试只验证 owner isolation。

失败：

```text
PRIMARY_TRAJECTORY_CONTAMINATED
```

---

# 10. DINO 审查

真实图像 forward：

```text
input [B,3,360,640]
patch_tokens_by_layer [B,3,3600,384]
cls_tokens_by_layer [B,3,384]
grid_hw=(45,80)
```

硬要求：

```text
all params requires_grad=False
backbone eval
forward no_grad
ordinary batch calls=1
CF event extra DINO calls=0
branch eval extra DINO calls=0
```

失败：

```text
DINO_TRAINABLE
WRONG_DINO_SHAPE
DINO_CALL_DUPLICATION
COUNTERFACTUAL_REENCODES_DINO
TEST_BRANCH_REENCODES_DINO
```

---

# 11. Multi-layer conditioner 审查

输入：

```text
[B,3,3600,384]
```

检查：

```text
每层独立Linear
每层独立RMSNorm
2D/ego位置编码存在
source primary patch未被修改
K/V每层只投影一次
```

禁止物化：

```text
[B,16,3,3600,384]
[B,21,3,3600,384]
[B,Q,N,D]
```

使用 profiler/memory hook捕获最大中间 tensor shape。

失败：

```text
MULTILAYER_CONDITIONER_MISSING
PRIMARY_PATCH_MUTATED
DENSE_QND_MATERIALIZED
```

---

# 12. Action evidence probes 审查

## 12.1 Shapes

必须：

```text
probe_queries              [B,4,4,384]
global_attention           [B,4,4,3600]
global_token               [B,4,4,384]
layer_mixture              [B,4,4,3]
evidence_map               [B,4,4,3600]
evidence_token             [B,4,4,384]
reference_point            [B,4,4,2]
sampling_offsets           [B,4,4,3,8,2]
sampling_weights           [B,4,4,3,8]
```

## 12.2 Action-conditioned initialization

验证：

```text
probe = stopgrad(primary Action node)+role embedding
```

Final Action loss对 primary Action node梯度必须为0。

## 12.3 Global inquiry

干预：

```text
shuffle patch tokens → global attention/token变化
shuffle Action node → 对应Action probes变化
```

## 12.4 Groupwise specialization

实现必须 reshape：

```text
[B,4,4,D] → [B*4,4,D]
```

只在同一 Action 内 self-attention。

构造测试：仅改变 Action 0 probes，Action 1/2/3 group attention结果不应直接变化。

失败：

```text
PROBE_SHAPE_MISMATCH
PROBE_NOT_ACTION_CONDITIONED
GLOBAL_INQUIRY_PLACEHOLDER
CROSS_ACTION_PROBE_COMPETITION
PROBE_NO_GRAD
```

---

# 13. Predicate low-bandwidth interface 审查

允许进入 Action evidence的 predicate信息仅为：

```text
predicate_attention.detach()
predicate_probs.detach()
learnable class-level key table [32,64]
scalar compatibility
```

检查 autograd graph，不得包含：

```text
source predicate_token [B,32,384] as value
reason logit/token
BDD GT
```

Predicate bias强度：

```text
0 <= lambda <=0.25
```

Ablation：

```text
predicate_bias_off只移除空间bias
global visual inquiry仍工作
```

失败：

```text
PREDICATE_HIGH_BANDWIDTH_LEAK
REASON_TO_ACTION_EVIDENCE_LEAK
PREDICATE_BIAS_OUT_OF_BOUND
BDD_GT_TO_ACTION_FORWARD
```

---

# 14. Local deformable reread 审查

必须使用：

```text
grid_sample或等效可微采样
3 layers
8 points/layer
reference point来自evidence map
offset经过tanh限制
```

检查：

```text
abs(offset)<=0.25
sampling weights sum=1
local token随offset/patch变化
local_reread_off时输出不同
```

禁止：

```text
global token + MLP冒充local reread
top-k直接平均
再次DINO
```

失败：

```text
LOCAL_REREAD_PLACEHOLDER
OFFSET_OUT_OF_RANGE
SAMPLING_WEIGHT_INVALID
LOCAL_REREAD_NO_EFFECT
```

---

# 15. Evidence map 审查

要求：

```text
shape [B,4,4,3600]
finite
nonnegative
sum over patch =1
entropy非固定
map不全部均匀
map不全部单patch
```

干预：

```text
predicate bias off应改变部分map但不能使全部map失效
local reread与map使用同一combined score语义
```

失败：

```text
EVIDENCE_MAP_NOT_NORMALIZED
EVIDENCE_MAP_CONSTANT
EVIDENCE_MAP_ALL_UNIFORM
EVIDENCE_MAP_ALL_ONE_HOT
```

---

# 16. Contribution head 审查

## 16.1 Early activation

最终Linear不得全零初始化。

在第一个真实 batch：

```text
raw contribution std >0
action_evidence grad >0
action_contribution grad >0
```

## 16.2 Final Action

必须：

```text
action_final = action_primary + bounded delta
```

训练 loss 使用：

```text
stopgrad(action_primary)+bounded delta
```

推理数值使用非detach primary。

## 16.3 Exact additivity

要求：

\[
action\_final-action\_primary
=
\sum_k bounded\_contribution_k.
\]

误差：

```text
fp32 <1e-6
bf16 <5e-4
```

## 16.4 Cap

检查：

```text
kappa=3
direction-preserving L2 logit cap
无逐标签hard clamp
```

失败：

```text
CONTRIBUTION_ZERO_INIT_STARVATION
FINAL_ACTION_FORMULA_MISMATCH
CONTRIBUTION_NOT_ADDITIVE
ACTION_CAP_CHANGES_DIRECTION
```

---

# 17. Counterfactual engine 审查

## 17.1 Trigger

检查 optimizer-update计数，不得按 micro-step错误触发。

```text
interval=4 optimizer updates
batch fraction<=0.5
max actions/sample=2
```

## 17.2 Mask

```text
top-64 hard forward
soft straight-through backward
```

检查 mask：

```text
selected support count=64或受有效region限制
gradient可回到evidence map
```

## 17.3 Same-region substitution

验证：

```text
背景来自同图同层同ego region
不是全零
不是全图均值
```

## 17.4 Matched control

必须满足：

```text
same region
same support count
same value distribution
overlap<=0.20
deterministic seed
```

无法匹配时 fail-closed并记录，不得退化为任意 random。

## 17.5 No DINO rerun

CF只重跑：

```text
evidence interface
contribution head
```

## 17.6 Target-signed margin

单元测试覆盖：

```text
y=1: 降低正logit应降低margin
y=0: 提高logit应降低margin
```

## 17.7 Loss direction

构造 synthetic：

```text
selected effect > control → necessity loss较小
selected effect < control → necessity loss较大
```

Contribution-effect target必须 detach。

失败：

```text
CF_TRIGGER_WRONG_UNIT
CF_MASK_NO_GRAD
CF_BACKGROUND_OOD
CF_CONTROL_NOT_MATCHED
CF_REENCODES_DINO
CF_MARGIN_SIGN_WRONG
CF_LOSS_DIRECTION_WRONG
```

---

# 18. Probe collapse 审查

每轮必须计算：

```text
contribution std
dominant probe share
effective probe count
map entropy
pairwise overlap
```

硬失败：

```text
全部contribution=0
全部map相同
单probe在>80%样本承担>90%总贡献
所有map均匀
所有map单patch
```

不要求平均使用四个 probes。

失败：

```text
PROBE_ALL_ZERO
PROBE_SINGLE_DOMINANCE
PROBE_MAP_DUPLICATION
```

---

# 19. Conservative predicate target 审查

静态扫描和真实记录测试必须证明：

```text
traffic light presence不会自动产生green/red
traffic sign presence不会自动产生stop sign
front vehicle不会同时close/far
generic lane不会自动solid/turn
missing object不会自动negative
```

Map rasterization：

```text
box到45×80
polyline到45×80
drivable到45×80
```

输出形状：

```text
targets [B,32]
target_mask [B,32]
counter_target [B,32]
counter_mask [B,32]
map_target [B,32,3600]
map_mask [B,32]
```

Test model forward只接收RGB。

失败：

```text
OVERCLAIMED_PREDICATE_TARGET
MISSING_AS_NEGATIVE
PREDICATE_MAP_SHAPE_WRONG
TEST_STRUCTURED_INPUT
```

---

# 20. Predicate prediction 审查

检查：

```text
predicate logits/probs/maps形状正确
map loss仅在map_mask=1计算
classification loss仅在target_mask=1计算
reliability被使用
compactness权重极小
```

Predicate loss不得更新 Action evidence。

失败：

```text
PREDICATE_LOSS_MASK_IGNORED
PREDICATE_MAP_LOSS_MISSING
PREDICATE_TO_ACTION_GRAD_LEAK
```

---

# 21. Predicate naming 审查

## 21.1 无 null competition

代码中不得对：

```text
32 predicates + null
```

做统一 softmax/entmax。

## 21.2 Naming quality

必须计算：

```text
space SoftIoU
low-dimensional compatibility
predicate presence
optional CF effect
```

## 21.3 Abstention

未达到：

```text
confidence>=0.45
margin>=0.08
presence>=0.30
```

必须：

```text
name_id=-1
```

## 21.4 Training gate

Naming loss只在：

```text
valid CF
supportive contribution
positive selected-control effect
reliable grounded predicate
```

同时成立时计算。

失败：

```text
NULL_COMPETITION_FOUND
NAMING_FORCED_ON_ALL_ATOMS
NAMING_IGNORES_EFFECT
NAMING_USES_UNRELIABLE_GT
```

---

# 22. Reason rereader 审查

## 22.1 Inputs detached

检查：

```text
reason primary node detached
Action evidence token/map/contribution detached
predicate map/prob detached
```

## 22.2 Full-field reread

Reason query必须重新读取：

```text
3×3600 DINO field
```

不能仅：

```text
MLP(action evidence)
MLP(predicate token)
```

## 22.3 Private Reason interaction

一层21-token private self-attention存在。

## 22.4 Formal output

必须：

```text
reason_final = reason_primary + bounded reason delta
```

训练 final reason使用 detach primary。

Evaluator formal Exp只能使用 `reason_logits_final`。

失败：

```text
REASON_INPUT_NOT_DETACHED
REASON_REREAD_PLACEHOLDER
REASON_PRIVATE_INTERACTION_MISSING
FORMAL_REASON_NOT_FINAL
HIDDEN_STRONG_REASON_BRANCH
```

---

# 23. Evidence-censored negative loss 审查

检查：

```text
positive weight=1
zero weight=0.25+0.75*counter_conf
counter_conf detached
zero weight∈[0.25,1]
```

不得：

```text
zero全部hard negative
zero伪标positive
online pseudo-label
trainable reliability自我控制
```

Synthetic：

```text
counter_conf=0 → weight=0.25
counter_conf=1 → weight=1
positive target → weight=1
```

失败：

```text
REASON_ZERO_HARD_NEGATIVE
REASON_PSEUDO_LABEL_FOUND
COUNTER_WEIGHT_FORMULA_WRONG
```

---

# 24. Loss registry 审查

每个配置项：

```text
存在raw loss
进入total一次
权重一次
owner明确
```

检查 key 一致性：

```text
model output key
loss function input key
registry key
artifact key
```

不得：

```text
嵌套loss未读取
配置非零但raw loss恒0
同一loss双重加权
placeholder zero
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

# 25. Autograd owner 审查

分别运行：

```text
primary-only backward
final-Action-only backward
final-Reason-only backward
predicate-only backward
CF-only backward
naming-only backward
```

硬要求：

```text
primary-only:
  primary >0
  AIE Action =0
  Reason private=0

final-Action-only:
  Action evidence >0
  contribution >0
  primary=0
  Reason private=0

final-Reason-only:
  Reason private>0
  Action evidence=0
  contribution=0
  primary=0
  predicate=0

predicate-only:
  predicate>0
  Action evidence=0

CF-only:
  Action evidence/contribution>0
  primary=0
  Reason private=0

naming-only:
  Action evidence/compatibility>0
  predicate=0
  primary=0

DINO=0 in all cases
```

失败：

```text
PRIMARY_TO_AIE_GRAD_LEAK
FINAL_ACTION_TO_PRIMARY_LEAK
FINAL_REASON_TO_ACTION_LEAK
FINAL_REASON_TO_PREDICATE_LEAK
PREDICATE_TO_ACTION_LEAK
DINO_GRAD_LEAK
OWNER_NO_GRAD
```

---

# 26. Optimizer exact-cover 审查

Trainable参数必须恰好属于一个 owner：

```text
primary
action_evidence
action_contribution
reason_private
```

检查：

```text
unowned set = empty
duplicate-owned set = empty
DINO absent
```

连续2 optimizer updates后记录：

```text
grad norm
parameter delta
LR
weight decay
```

失败：

```text
UNOWNED_PARAMETER
DUPLICATE_OWNER
DINO_IN_OPTIMIZER
OWNER_NO_UPDATE
```

---

# 27. Accumulation 审查

构造：

```text
7 micro-batches
accum=5
```

要求：

```text
optimizer steps=2
最后2个micro-batches被flush
尾窗口按实际2步缩放
```

失败：

```text
ACCUMULATION_TAIL_DROPPED
ACCUMULATION_TAIL_WRONG_SCALE
```

---

# 28. BF16 审查

检查：

```text
autocast dtype=bfloat16
无FP16 GradScaler
metric/CF margin/reconstruction用FP32
```

失败：

```text
WRONG_MIXED_PRECISION
FP16_SCALER_WITH_BF16
LOW_PRECISION_METRICS
```

---

# 29. Calibration/test 协议审查

必须：

```text
train-calib拟合threshold
test只应用threshold
calibration前后model state hash相同
每epoch只执行一次test image encoding
best=test deploy_joint
```

Manifest：

```text
internal_test_selected=true
publication_eligible=false
```

禁止：

```text
test阈值搜索写回
test oracle选best
val best
模型参数被calibration修改
```

失败：

```text
TEST_THRESHOLD_LEAK
CALIBRATION_MUTATES_MODEL
VAL_PROTOCOL_FOUND
MANIFEST_FLAG_MISSING
```

---

# 30. Runtime profiler 审查

Profiler必须覆盖：

```text
official DINO
primary foundation
32 predicate maps
16 global probes
16 local rereads
21 Reason rereads
CF event摊销
BF16
```

比较配置：

```text
bs6/acc5/chunk16
bs6/acc5/chunk8
bs5/acc6/chunk16
bs4/acc8/chunk16
```

硬要求：

```text
reserved<45GB
无OOM
普通batch DINO=1
CF extra DINO=0
branch extra DINO=0
无持续memory growth
```

失败：

```text
PROFILE_OMITS_CORE_PATH
MEMORY_OVER_LIMIT
MEMORY_LEAK
```

---

# 31. Artifact 审查

## 31.1 Run-level

必须：

```text
config_resolved.yaml
run_manifest.json
source_contract.json
owner_map.json
split_manifest.json
train_calib_ids.json
train_audit_ids.json
AIE_IMPLEMENTATION_REVIEW.json
AIE_RUNTIME_PROFILE.json
```

## 31.2 Step-level

必须：

```text
loss_components.jsonl
owner_gradients.jsonl
runtime_components.jsonl
evidence_components.jsonl
```

## 31.3 Epoch-level

必须：

```text
metrics_summary.json
branch_metrics.json
per_label_action_metrics.json
per_label_reason_metrics.json
calibration.json
predicate_metrics.json
naming_metrics.json
probe_metrics.json
counterfactual_metrics.json
owner_metrics.json
runtime_metrics.json
```

固定 audit-128：

```text
evidence maps
contributions
names
selected/control masks
CF effects
reason evidence attention
ablation logits
```

失败：

```text
ARTIFACT_MISSING
BRANCH_METRICS_MISSING
COUNTERFACTUAL_ARTIFACT_MISSING
OWNER_ARTIFACT_MISSING
```

---

# 32. Resume 审查

Checkpoint必须包含：

```text
model
optimizer
scheduler/update count
epoch
micro-step/global update
RNG states
split manifests/hash
calibration
audit IDs
config/source/schema hashes
best states
```

中断前后下一 optimizer update应在容差内复现。

失败：

```text
RESUME_INCOMPLETE
RESUME_NONDETERMINISTIC
BEST_STATE_LOST
```

---

# 33. Required commands

```powershell
E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\models\aie_*.py `
  fate_oia\losses\aie_*.py `
  fate_oia\datasets\aie_*.py `
  fate_oia\engine\*aie_oia*.py `
  fate_oia\utils\aie_*.py
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_aie_*.py -q
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_acpr_dino_field.py `
  tests\test_acpr_label_trunk.py `
  tests\test_acpr_model_forward.py `
  tests\test_bdd_oia_dataset.py -q
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_aie_oia_implementation `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --output-dir .background_runs\aie_oia_v1_preflight `
  --device cuda
```

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.profile_aie_oia `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --device cuda
```

---

# 34. REVIEW 输出

生成：

```text
AIE_IMPLEMENTATION_REVIEW.json
AIE_RUNTIME_PROFILE.json
```

字段：

```text
status
git_head
source_head
config_hash
predicate_schema_hash
counter_evidence_schema_hash
checked_files
forbidden_paths
source_equivalence
primary_trajectory_isolation
functional_checks
gradient_ownership
optimizer_ownership
runtime
failures
warnings
```

只有全部硬检查通过：

```text
status=REVIEW_PASS
```

---

# 35. 唯一 Pilot 审查

Pilot：

```text
4096 train
1024 audit
512 calib
512 test
4 epochs
seed 20260806
```

审查器必须从原始：

```text
logits
labels
file order
contributions
evidence maps
CF variants
predicate maps
naming
owner gradients
runtime
```

独立重算 gate，不得只信任 trainer 自报。

## Gate A：foundation与primary

```text
equivalence通过
primary trajectory isolation通过
primary owner active
DINO frozen
```

## Gate B：evidence activation

前50 updates：

```text
evidence grad>0
contribution grad>0
contribution std>1e-3
local reread非占位
```

## Gate C：Action

最后两轮至少一轮：

```text
final Act_mAP >= primary Act_mAP +0.003
final Act_mF1 >= primary Act_mF1 -0.002
```

## Gate D：Reason

最后两轮至少一轮：

```text
final Exp_mAP >= primary Exp_mAP +0.003
final Exp_mF1 >= primary Exp_mF1 -0.003
```

## Gate E：CF

```text
valid events>0
selected-control macro mean>0
3/4 actions positive
contribution-effect Spearman>0.30
control overlap<=0.20
```

## Gate F：probe health

```text
not all zero
not single-probe collapse
not map duplication
```

## Gate G：naming

```text
quality>random
5%<named coverage<90%
unnamed>0
```

## Gate H：firewalls

所有严格0梯度合同通过。

## Gate I：runtime/artifacts

```text
reserved<45GB
one DINO
no cache/compression
all required artifacts
```

---

# 36. Pilot 输出

生成：

```text
AIE_PILOT_RAW_EVIDENCE.json
AIE_PILOT_GATES.json
AIE_PILOT_PASS.json
AIE_FULL_TRAIN_READY.json
```

绑定：

```text
git_head
source_head
config_hash
schema_hashes
split_hash
checkpoint_hash
logits_hash
labels_hash
file_order_hash
```

核心代码/config/schema修改后：

```text
invalidate PILOT_PASS
invalidate FULL_TRAIN_READY
重新pilot
```

---

# 37. Full train 启动审查

启动前重新验证：

```text
target branch正确
worktree clean
local==remote
HEAD匹配review/pilot
20 epochs
BF16
profiled batch/accum
test every epoch
best test deploy_joint
no cache/compression
```

Supervisor必须：

```text
前台stream stdout/stderr
先audit
再pilot pass check
再full train
结构性错误才停止
```

不得：

```text
隐藏Start-Process后不留日志
弱指标自动停止
修改门槛后继续
```

---

# 38. 审查报告必须回答

```text
1. CalAlign raw primary是否数值恢复？
2. Primary是否从头到尾直接训练？
3. AIE final loss是否严格不能改变primary参数？
4. 16 probes是否真实读取三层3600 patch？
5. Local reread是否真实可微采样？
6. Predicate是否只通过低带宽空间接口进入Action？
7. Reason或BDD GT是否泄漏到Action evidence？
8. 每个evidence atom是否进入formal final Action？
9. contribution是否精确重构final residual？
10. selected substitution是否优于matched control？
11. contribution是否与CF effect相关？
12. probes是否坍缩？
13. Predicate naming是否质量准入并允许unnamed？
14. Formal Reason是否为高容量direct refined Reason？
15. Reason loss是否完全无法更新Action evidence？
16. 每个loss是否真实进入total/backward？
17. 每个owner是否有非零更新且exact-cover？
18. 每个epoch日志是否足以判断组件有用/无用？
19. Test是否只运行一次图像编码？
20. Pilot是否满足full train条件？
```

任一问题无法由代码与 artifact 回答：

```text
REVIEW_FAIL
```

---

# 39. 审查边界

`REVIEW_PASS` 只证明：

```text
代码完整
机制真实进入正式路径
公式、shape、梯度、协议正确
```

`PILOT_PASS` 只证明：

```text
短训练中机制激活且方向合理
```

都不能证明：

```text
Act_mF1必然达到0.73
Exp_mF1必然超过0.38
```

性能结论只能来自真实 full train。
