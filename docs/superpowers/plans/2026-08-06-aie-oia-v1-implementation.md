# AIE-OIA V1：Codex 完整代码级实施计划

## Action-Induced Evidence Interface for Explainable Object-Induced Action Decision

**日期：** 2026-08-06  
**仓库：** `d2116056543-lab/FATE-OIA`  
**唯一源分支：** `acpr_calalign_v1_2`  
**已核对源 HEAD：** `373aa49feac17372574fd7fb056c1d79c7c848fe`  
**目标分支：** `acpr_aie_oia_v1_direct_image`  
**目标 worktree：** `E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree`  
**任务：** BDD-OIA 单帧 RGB，4 维 Action + 21 维 Reason 多标签联合预测  
**硬件：** NVIDIA RTX 5880，48 GB  
**训练协议：** direct image、Frozen DINO、无 feature cache、无 token compression  
**用户内部评估协议：** 每个 epoch 只评估 test，以 test deploy-joint 选 best  
**论文边界：** manifest 必须写入 `internal_test_selected=true`、`publication_eligible=false`

---

# 0. 文件地位与唯一方案

本文件是 Codex 实现 AIE-OIA V1 的唯一代码合同。不得把它解释成多个备选方案，不得在实施过程中自行改回 LENS、SAVE、METER、PROBE、DEFT、BASIS、HardPair、graph/PMI、VLM 或 cached-logit 路线。

AIE-OIA 的中心科学命题只有一个：

> **先由 Action 监督学习真正改变决策的视觉证据，再让 Predicate 只命名其中可验证的部分；Reason 只读取 stop-gradient 后的 Action evidence，而 noisy Reason label 永远不能定义或改写 Action evidence。**

统一数据流：

```text
RGB
 ↓
Frozen DINO ViT-S/8, layers 3/7/11, 3600 patches
 ↓
CalAlign-compatible 25-query primary route
 ├── 4 primary Action queries/logits
 └── 21 primary Reason queries/logits
 ↓
16 Action-Induced Evidence Probes（4 actions × 4 probes）
 ├── global multi-layer inquiry
 ├── bounded predicate-spatial prior
 ├── local deformable reread
 ├── exact signed contribution
 └── matched counterfactual effect
 ↓
final Action
 ↓ stop-gradient evidence interface
21 Evidence-Conditioned Reason queries
 ↓
final benchmark Reason
```

AIE 的解释对象是一个 evidence atom：

\[
\mathcal E_{iak}
=
\left(
M_{iak},
e_{iak},
c_{iak},
n_{iak}
\right),
\]

其中：

```text
M_iak  spatial evidence map
e_iak  evidence representation
c_iak  signed contribution to final Action logit
n_iak  predicate name or explicit unnamed/abstain
```

每个 evidence atom 必须同时满足：

1. 进入正式 final Action；
2. 有非零、可审计的梯度；
3. 对 final Action residual 有精确加性贡献；
4. selected deletion effect 高于 matched control；
5. contribution 与干预 effect 方向一致；
6. 可被可靠 predicate 命名，或诚实输出 unnamed；
7. 不接受 Reason loss 的梯度。

---

# 1. 开始远程任务前的强制规则

在任何 Git、文件读取、代码修改、测试、训练、评估、进程管理或 push 前，Codex 必须先读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

这三份文件是唯一持久训练/实验状态 Markdown：

```text
task_plan.md  计划、边界、约束
findings.md   发现、根因、审查结论
progress.md   时间顺序实施、测试、训练进展
```

不得新建：

```text
implementation_status.md
audit_status.md
pilot_status.md
run_status.md
training_status.md
```

本实施计划与配套 Skill 属于方法规范，可以放入：

```text
docs/superpowers/specs/
docs/superpowers/plans/
.codex/skills/
```

但不能替代三份 canonical 状态文件。

每次开始远程会话都必须在 `progress.md` 追加：

```text
AIE task start
当前时间
当前branch
当前HEAD
已读取三份canonical文件
```

---

# 2. 源分支与 worktree 合同

## 2.1 核对远程源 HEAD

在现有 FATE-OIA 管理 worktree 中执行：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_worktree
git fetch origin --prune
git rev-parse origin/acpr_calalign_v1_2
```

必须输出：

```text
373aa49feac17372574fd7fb056c1d79c7c848fe
```

若不同：

```text
STOP
将实际SHA与差异写入findings.md
不得静默基于新HEAD实施
```

## 2.2 记录源 worktree 状态

```powershell
git -C E:\sbw\FATE_Drive\fate_oia_worktree status --porcelain --untracked-files=all
git -C E:\sbw\FATE_Drive\fate_oia_worktree rev-parse HEAD
git worktree list --porcelain
```

任务结束时重复执行，源 worktree 状态必须完全不变。

## 2.3 检查目标 branch/worktree 不存在

```powershell
git show-ref --verify --quiet refs/heads/acpr_aie_oia_v1_direct_image
Test-Path E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree
```

若任一已存在：

```text
STOP
不得覆盖、删除、reset或复用未知目录
在findings.md记录
```

## 2.4 新建 worktree

```powershell
git worktree add `
  -b acpr_aie_oia_v1_direct_image `
  E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree `
  origin/acpr_calalign_v1_2
```

验证：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
```

必须满足：

```text
branch = acpr_aie_oia_v1_direct_image
HEAD   = 373aa49feac17372574fd7fb056c1d79c7c848fe
status = clean
```

## 2.5 立即创建远程 branch

```powershell
git push -u origin acpr_aie_oia_v1_direct_image
git ls-remote origin refs/heads/acpr_aie_oia_v1_direct_image
```

以后每一实施阶段：

```text
commit
push
验证 local HEAD == remote HEAD
```

---

# 3. 当前 `acpr_calalign_v1_2` 的代码事实

Codex 必须先通读源代码，不得按文件名猜测。

## 3.1 DINO visual field

源文件：

```text
fate_oia/models/acpr_dino_field.py
```

当前行为：

```text
固定输入 360×640
DINO ViT-S/8
selected layers = 3, 7, 11
patch_tokens_by_layer = [B,3,3600,384]
cls_tokens_by_layer   = [B,3,384]
backbone eval
backbone requires_grad=False
forward no_grad
```

AIE 必须直接复用，不能增加第二个视觉 backbone，不能重新运行 DINO 计算 counterfactual。

## 3.2 25-query primary trunk

源文件：

```text
fate_oia/models/acpr_label_trunk.py
```

当前包含：

```text
25 learnable label queries
label-specific full-patch retrieval
entmax1.5
25-token self-attention
4 Action nodes
21 Reason nodes
visual Action head
reason_to_action
sample-dependent fusion gate
predicate-conditioned Reason cross-attention
```

当前 raw Action：

\[
z_a^{primary}
=
g_a z_a^{visual}
+
(1-g_a)W_{R\rightarrow A}z_R^{visual}.
\]

AIE 不能重建弱 primary predictor，也不能让 final branch 替代 primary 直接监督。

## 3.3 Scene predicates

源文件：

```text
fate_oia/models/acpr_scene_predicate_head.py
configs/acpr_scene_predicates.yaml
```

当前已有：

```text
32 predicates
per-predicate layer weights
full-field sparse attention
predicate token
predicate probability
predicate attention map [B,32,3600]
ego-region log prior
```

AIE 复用这套 image-predicted predicate map，但：

```text
Action只能读取detach后的空间map和标量compatibility
不得读取完整高维predicate token作为Action语义捷径
不得使用BDD100K GT进入test forward
```

## 3.4 当前 `ACPROIAModel` 中的历史模块

源模型同时实例化：

```text
ACPRPairMemory
reason_pair_proj
ACPRActionComboAux
ACPRCalibrationHead
ACPRThresholdHead
```

这些模块不决定 raw `action_logits_base/reason_logits_base`，但属于历史训练辅助和部署路径。

AIE 正式模型不得实例化或调用：

```text
pair memory
matched pair mining
16 action-set final/marginalization
legacy calibration as representation
trainable threshold inside representation optimizer
```

**分支特定处理：**

- AIE 新建 `AIECalAlignFoundation`，只复用构成 raw CalAlign logits 的模块；
- 不直接继承整个 `ACPROIAModel`；
- numerical equivalence 只针对 source raw Action/Reason logits、label nodes、attention 和 predicate outputs；
- threshold 改为模型外 train-calib post-hoc calibration；
- pair/action-combo 不进入 AIE 正式方法。

## 3.5 当前 weak predicate target 的过强推断

源文件：

```text
fate_oia/models/acpr_predicate_targets.py
```

当前存在以下不安全推断：

```text
traffic light存在 → traffic_light_green
traffic sign存在  → stop_sign_present
front car存在     → close、far、road_crowded同时正
lane poly存在     → solid/turn/merge同时正
没有前方对象      → road_clear
```

AIE 不得复用 `WeakPredicateTargetBuilder`。必须新建 fail-closed 的 `AIEPredicateTargetBuilder`。

---

# 4. 分支特定设计决策

## 4.1 Primary route 必须持续直接训练

LENS 的主要错误之一是：

```text
progress=0等价
训练后primary/source只剩日志
```

AIE 强制：

```text
primary Action loss从第1步到最后1步存在
primary Reason loss从第1步到最后1步存在
primary predicate loss从第1步到最后1步存在
```

## 4.2 Final branch 不能反向破坏 primary

训练时：

\[
z_A^{final,train}
=
sg(z_A^{primary})
+
\Delta z_A^{AIE},
\]

\[
z_R^{final,train}
=
sg(z_R^{primary})
+
\Delta z_R^{AIE}.
\]

推理时数值仍为：

\[
z_A^{final}
=
z_A^{primary}
+
\Delta z_A^{AIE},
\]

\[
z_R^{final}
=
z_R^{primary}
+
\Delta z_R^{AIE}.
\]

因此 final loss 对 primary 参数的梯度必须严格为 0。

## 4.3 Predicate 只作低带宽空间接口

Action path允许：

```text
predicate map [B,32,3600]，detach
predicate probability [B,32]，detach
probe-predicate scalar compatibility
```

Action path禁止：

```text
predicate token直接拼接到Action evidence
reason logits进入Action evidence
BDD100K GT map进入Action forward
```

## 4.4 Named/unnamed 不竞争

AIE 不设置：

```text
named factors + null softmax
route entropy
non-null coverage
```

Evidence atom先由Action训练。命名是后验质量判定：

```text
可靠匹配 → predicate name
无可靠匹配 → unnamed
```

Unnamed 不会取消该 atom 的 contribution。

---

# 5. 目标目录结构

## 5.1 新增配置

```text
configs/fate_oia_train_360x640_aie_oia_v1.yaml
configs/aie_scene_predicates.yaml
configs/aie_reason_counter_evidence.yaml
```

## 5.2 新增模型

```text
fate_oia/models/aie_calalign_foundation.py
fate_oia/models/aie_evidence_interface.py
fate_oia/models/aie_deformable_reread.py
fate_oia/models/aie_contribution_head.py
fate_oia/models/aie_predicate_naming.py
fate_oia/models/aie_reason_rereader.py
fate_oia/models/aie_oia_model.py
```

## 5.3 新增数据/grounding

```text
fate_oia/datasets/aie_structured_evidence.py
fate_oia/datasets/aie_splits.py
```

## 5.4 新增损失

```text
fate_oia/losses/aie_losses.py
fate_oia/losses/aie_loss_registry.py
```

## 5.5 新增工具

```text
fate_oia/utils/aie_counterfactual.py
fate_oia/utils/aie_calibration.py
fate_oia/utils/aie_metrics.py
fate_oia/utils/aie_artifacts.py
fate_oia/utils/aie_contracts.py
fate_oia/utils/aie_hashes.py
```

## 5.6 新增 engine

```text
fate_oia/engine/train_aie_oia.py
fate_oia/engine/eval_aie_oia.py
fate_oia/engine/profile_aie_oia.py
fate_oia/engine/audit_aie_oia_implementation.py
fate_oia/engine/evaluate_aie_oia_pilot.py
fate_oia/engine/supervise_aie_oia_foreground.py
```

## 5.7 新增脚本、Skill 与设计文档

```text
scripts/FATE_OIA_aie_oia_v1_pilot.ps1
scripts/FATE_OIA_aie_oia_v1_foreground.ps1

.codex/skills/aie-oia-v1-implementation-audit/SKILL.md
docs/superpowers/specs/2026-08-06-aie-oia-v1-design.md
docs/superpowers/plans/2026-08-06-aie-oia-v1-implementation.md
```

---

# 6. `AIECalAlignFoundation`

实现：

```text
fate_oia/models/aie_calalign_foundation.py
```

## 6.1 模块构成

只实例化：

```python
self.dino = ACPRDinoFieldExtractor(...)
self.ego = ACPREgoRegionEncoder(...)
self.predicate_head = ACPRScenePredicateHead(...)
self.trunk = ACPRLabelTrunk(...)
self.predicate_reason = ACPRPredicateReasoner(...)
```

不得实例化：

```text
ACPRPairMemory
ACPRActionComboAux
ACPRCalibrationHead
ACPRThresholdHead
```

## 6.2 API

```python
class AIECalAlignFoundation(nn.Module):
    def encode_images(self, images: Tensor) -> dict[str, Any]:
        ...

    def decode_field(
        self,
        field: dict[str, Any],
    ) -> dict[str, Tensor | dict]:
        ...

    def forward(self, images: Tensor) -> dict[str, Any]:
        field = self.encode_images(images)
        return self.decode_field(field)
```

## 6.3 `encode_images`

只能执行：

```python
field = self.dino(images)
```

每个普通 batch只能调用一次 DINO。

## 6.4 `decode_field`

必须逐行复现 source `ACPROIAModel.forward` 的 raw foundation部分：

1. 取 `patch_tokens_by_layer`；
2. 只对 layer 0 调用 ego encoder；
3. clone patch field并写回 layer 0；
4. scene predicate head；
5. label trunk；
6. predicate reason delta；
7. 构造 raw primary Action/Reason。

定义：

```python
action_logits_primary = trunk["action_logits_direct"]
reason_logits_primary = (
    trunk["reason_logits_visual"]
    + predicate_reason["predicate_reason_delta"]
)
```

## 6.5 必须输出

```text
patch_tokens_by_layer_raw          [B,3,3600,384]
patch_tokens_by_layer_ego          [B,3,3600,384]
cls_tokens_by_layer                [B,3,384]
grid_hw                            (45,80)
ego_features                       [3600,8]
ego_region_masks                   dict

predicate_tokens                   [B,32,384]
predicate_logits                   [B,32]
predicate_probs                    [B,32]
predicate_attention                [B,32,3600]
predicate_layer_weights            [32,3]

label_nodes                        [B,25,384]
label_attention                    [B,25,3600]
action_nodes_primary               [B,4,384]
reason_nodes_primary               [B,21,384]

action_visual_logits_primary       [B,4]
action_reason_logits_primary       [B,4]
action_fusion_gate_primary         [B,4]
action_logits_primary              [B,4]

reason_logits_visual_primary       [B,21]
predicate_reason_delta_primary     [B,21]
reason_logits_primary              [B,21]
```

## 6.6 Source state-dict 映射

实现：

```python
def load_from_acpr_state_dict(
    self,
    source_state_dict: Mapping[str, Tensor],
    strict: bool = True,
) -> dict[str, list[str]]:
    ...
```

映射：

```text
dino.*             → dino.*
ego.*              → ego.*
predicate_head.*   → predicate_head.*
trunk.*            → trunk.*
predicate_reason.* → predicate_reason.*
```

## 6.7 数值等价

在同一真实图像、同一权重下：

```text
action_logits_primary
reason_logits_primary
label_nodes
label_attention
predicate_logits
predicate_attention
```

与 source `ACPROIAModel(threshold_enabled=False)` 对应输出：

```text
fp32 max abs error < 1e-6
bf16 max abs error < 5e-4
```

---

# 7. Multi-layer conditioning

Source primary trunk保持原样，不改其 `mean(1)` 行为，保证数值等价。

AIE 新增 evidence/reason reread使用独立的多层规范化视图：

\[
\widehat F^l
=
RMSNorm(W_lF^l),
\qquad l\in\{3,7,11\}.
\]

实现约束：

```text
每层一个Linear(384,384)
每层一个RMSNorm
不生成第二套backbone
不把三层直接concat为1152维
不物化[B,Q,3,3600,384]
K/V每层只投影一次并复用
```

额外加入固定 2D/ego位置编码：

```text
normalized x
normalized y
distance to ego bottom-center
front-center score
left-corridor score
right-corridor score
upper-traffic score
bottom-drivable score
```

使用小型 `Linear(8,384)` 投影并加到 AIE K/V，不修改 primary patch tensor。

---

# 8. `AIEEvidenceInterface`

实现：

```text
fate_oia/models/aie_evidence_interface.py
fate_oia/models/aie_deformable_reread.py
```

## 8.1 Probe 数量与初始化

固定：

```text
action_dim = 4
probes_per_action = 4
total_probes = 16
```

四个 role embeddings：

```text
role_0: dominant/local hazard
role_1: traffic-control/context
role_2: lane/drivable
role_3: residual/global
```

role 名称只用于初始化/日志，不是固定语义标签。

Probe query：

\[
q_{iak}^{0}
=
sg(h_{ia}^{primary})
+
r_{ak}.
\]

Action node必须 stop-gradient，防止 AIE final loss回写 primary。

## 8.2 Global inquiry

输入：

```text
probe_queries                 [B,4,4,384]
conditioned_patch_layers      [B,3,3600,384]
```

每层共享 K/V 投影：

```text
K_l [B,3600,D]
V_l [B,3600,D]
```

Probe-layer score：

\[
s_{iakln}^{global}
=
\frac{
Q(q_{iak})^\top K_l(F_{iln})
}{
\sqrt d
}.
\]

每 probe 的 layer mixture：

\[
\pi_{iak}^{layer}
=
softmax(MLP_{layer}(q_{iak})).
\]

Global attention：

\[
A_{iakn}^{global}
=
\sum_l
\pi_{iakl}
softmax_n(s_{iakln}^{global}).
\]

Global token：

\[
g_{iak}
=
\sum_{l,n}
\pi_{iakl}
A_{iakln}^{l}
V_l(F_{iln}).
\]

必须输出：

```text
global_attention             [B,4,4,3600]
global_token                 [B,4,4,384]
layer_mixture                [B,4,4,3]
```

## 8.3 Predicate-bounded spatial prior

输入只能使用：

```text
predicate_attention.detach() [B,32,3600]
predicate_probs.detach()     [B,32]
```

兼容度由 probe token与一个**单独的低维 predicate key table**计算：

```python
predicate_key = nn.Parameter([32, 64])
probe_key = Linear(384,64)
compat = sigmoid(einsum(...))
```

禁止把 source `predicate_tokens [B,32,384]` 输入 Action evidence value path。

兼容度归一化：

\[
\bar\omega_{iakp}
=
\frac{
\omega_{iakp}p_{ip}
}{
\sum_p\omega_{iakp}p_{ip}+\epsilon
}.
\]

每 probe 的 predicate bias强度：

\[
\lambda_{iak}
=
0.25\sigma(MLP_\lambda(g_{iak})).
\]

约束：

```text
0 <= lambda <= 0.25
sum_p omega <= 1
predicate bias可关闭
```

Predicate prior：

\[
B_{iakn}^{P}
=
\sum_p
\bar\omega_{iakp}
\log(P_{ipn}+\epsilon).
\]

Combined score：

\[
s_{iakn}^{combined}
=
s_{iakn}^{visual}
+
\lambda_{iak}B_{iakn}^{P}.
\]

Evidence map：

\[
M_{iak}
=
softmax_n(s_{iakn}^{combined}).
\]

## 8.4 Local deformable reread

由 `M_iak` 计算 reference center：

\[
\mu_{iak}
=
\sum_nM_{iakn}(x_n,y_n).
\]

每 probe预测：

```text
3 layers × 8 points × 2 offsets
3 layers × 8 sampling weights
```

Offset限制：

```text
x/y normalized offset经过tanh
max absolute normalized offset <= 0.25
```

使用 `torch.nn.functional.grid_sample` 从 `[B,D,45,80]` 采样。

Local token：

\[
l_{iak}
=
\sum_{l,m}
a_{iaklm}
F_i^l(\mu_{iak}+\Delta_{iaklm}).
\]

最终 evidence token：

\[
e_{iak}^{pre}
=
LN(g_{iak}+l_{iak}+FFN(g_{iak}+l_{iak})).
\]

## 8.5 同 Action 内 group self-attention

reshape：

```text
[B,4,4,D] → [B*4,4,D]
```

一层 MHA：

\[
E_{ia}
=
GroupSelfAttn(
[e_{ia1}^{pre},...,e_{ia4}^{pre}]
).
\]

禁止跨 Action probes 做同层 self-attention。

输出：

```text
evidence_token               [B,4,4,384]
evidence_map                 [B,4,4,3600]
reference_point              [B,4,4,2]
sampling_offsets             [B,4,4,3,8,2]
sampling_weights             [B,4,4,3,8]
predicate_compatibility      [B,4,4,32]
predicate_bias_strength      [B,4,4]
```

## 8.6 不允许的退化实现

不得将 local reread实现为：

```text
对global token再过一个MLP
从top-k patch直接平均
只读取source action attention
只读取predicate token
再次运行DINO
```

---

# 9. `AIEContributionHead`

实现：

```text
fate_oia/models/aie_contribution_head.py
```

## 9.1 Raw contribution

每个 Action 有独立权重：

\[
c_{iak}^{raw}
=
w_a^\top
LN(e_{iak})
+
b_{ak}.
\]

输出：

```text
raw_contribution [B,4,4]
```

最终线性层使用小尺度初始化：

```text
weight std = 1e-3
bias = 0
```

不能全零初始化，否则第一步 evidence probes梯度为0。

## 9.2 Action residual

\[
S_{ia}
=
\sum_k c_{iak}^{raw}.
\]

\[
\Delta z_{ia}^{bounded}
=
\kappa_A
\tanh(S_{ia}/\kappa_A).
\]

默认：

```text
kappa_action = 3.0
```

训练 scale：

```text
alpha_action starts at 0.10
alpha_action reaches 1.0 by 10% updates
```

推理：

\[
z_{ia}^{final}
=
z_{ia}^{primary}
+
\alpha_A
\Delta z_{ia}^{bounded}.
\]

训练：

\[
z_{ia}^{final,train}
=
sg(z_{ia}^{primary})
+
\alpha_A
\Delta z_{ia}^{bounded}.
\]

## 9.3 Exact bounded contribution

\[
r_{ia}
=
\frac{
\alpha_A\kappa_A\tanh(S_{ia}/\kappa_A)
}{
S_{ia}+\epsilon
}.
\]

\[
\widetilde c_{iak}
=
r_{ia}c_{iak}^{raw}.
\]

严格满足：

\[
z_{ia}^{final}-z_{ia}^{primary}
=
\sum_k\widetilde c_{iak}.
\]

输出：

```text
action_logits_primary
action_logits_final
action_logits_final_train
raw_contribution
bounded_contribution
action_delta
contribution_reconstruction_error
```

误差要求：

```text
fp32 < 1e-6
bf16 < 5e-4
```

## 9.4 Direction-preserving logit cap

保留方向的 L2 cap：

```text
action_logit_norm_cap = 20.0
```

不能逐标签 clamp 改变相对排序。

---

# 10. Counterfactual Engine

实现：

```text
fate_oia/utils/aie_counterfactual.py
```

## 10.1 触发频率

训练时：

```text
每4个optimizer updates触发一次
最多使用当前micro-batch的50%
每样本最多2个Action
每Action只取target-signed contribution最大的1个probe
max_cf_atoms_per_event = 8
```

Action选择采用 round-robin 与当前损失结合，保证4个Action长期均有覆盖。

## 10.2 Target-signed margin

\[
t_{ia}=2y_{ia}^{A}-1.
\]

\[
m_{ia}(F)
=
t_{ia}z_{ia}^{final}(F).
\]

该定义同时处理：

```text
GT=1需要提高logit
GT=0需要降低logit
```

## 10.3 Straight-through evidence mask

从 `M_iak` 生成固定 support 数的 hard mask：

```text
topk_patches = 64
```

forward：

```text
hard top-64 binary mask
```

backward：

```text
soft M gradient
```

实现：

```python
mask_st = hard.detach() - soft.detach() + soft
```

对 mask 做 stop-gradient 的 ablation必须保留用于诊断。

## 10.4 Same-region background

根据 evidence center选择一个 ego region：

```text
front_center
left_corridor
right_corridor
upper_traffic_region
bottom_drivable_region
```

每层背景：

\[
\bar F_{il}^{region}
=
\frac{
\sum_nR_{in}F_{iln}
}{
\sum_nR_{in}+\epsilon
}.
\]

## 10.5 Selected substitution

\[
F_{sel}^{-}
=
F-
D_{iak}\odot
(F-\bar F^{region}).
\]

不置零，不使用全图均值。

## 10.6 Matched control

Control 必须保持：

```text
同一ego region
相同patch数量
相同mask质量
相同mask值分布
低selected overlap
```

实现为区域内 deterministic derangement/permutation：

```text
由 file_name + action_id + probe_id + global_update 生成seed
在同区域索引内最多尝试4个roll/permutation
选择与selected overlap最小者
```

若无法达到：

```text
overlap <= 0.20
```

该 event fail-closed，不计入 loss，记录原因。

## 10.7 Counterfactual decode

不得重跑：

```text
DINO
primary 25-query trunk
predicate head
```

重用：

```text
primary action nodes.detach()
predicate maps.detach()
modified patch field
```

只重跑：

```text
AIEEvidenceInterface
AIEContributionHead
```

Primary Action logit作为固定 base。

## 10.8 Necessity

\[
d_{iak}^{sel}
=
m_{ia}(F)-m_{ia}(F_{sel}^{-}),
\]

\[
d_{iak}^{ctl}
=
m_{ia}(F)-m_{ia}(F_{ctl}^{-}).
\]

\[
L_{nec}
=
softplus(
m_{nec}
-
d^{sel}
+
d^{ctl}
).
\]

默认：

```text
necessity_margin = 0.05
```

## 10.9 Contribution–effect consistency

Target-signed contribution：

\[
c_{iak}^{support}
=
t_{ia}\widetilde c_{iak}.
\]

Counterfactual effect：

\[
e_{iak}^{cf}
=
d_{iak}^{sel}-d_{iak}^{ctl}.
\]

\[
L_{effect}
=
Huber(
c_{iak}^{support},
sg(e_{iak}^{cf})
).
\]

## 10.10 Set sufficiency

对某 Action 所有 target-supportive probes取 union mask：

\[
D_{ia}^{union}
=
1-\prod_k(1-D_{iak}).
\]

构造 only-selected field：

\[
F_{keep}
=
\bar F^{region}
+
D_{ia}^{union}
\odot
(F-\bar F^{region}).
\]

要求：

\[
m_{ia}(F_{keep})
\ge
\rho_{suf}
m_{ia}(F)
-
m_{suf}.
\]

默认：

```text
rho_sufficiency = 0.50
sufficiency_margin = 0.05
```

## 10.11 Counterfactual 输出

```text
cf_valid_count
cf_invalid_reason_counts
selected_drop
control_drop
selected_minus_control
contribution_effect_pair
contribution_effect_spearman
per_action_effect
per_probe_effect
selected_control_overlap
```

---

# 11. Predicate Target Builder

实现：

```text
fate_oia/datasets/aie_structured_evidence.py
configs/aie_scene_predicates.yaml
```

## 11.1 目标输出

```text
predicate_target             [B,32]
predicate_target_mask        [B,32]
predicate_counter_target     [B,32]
predicate_counter_mask       [B,32]
predicate_map_target         [B,32,3600]
predicate_map_mask           [B,32]
predicate_reliability        [B,32]
predicate_source_complete    [B,32]
source_counts
coverage
```

## 11.2 Fail-closed 规则

绝对禁止：

```text
traffic light存在 → green/red
traffic sign存在 → stop sign
car存在 → close与far同时为正
lane poly存在 → solid/turn/merge
drivable map存在 → 所有方向均可行驶
对象未出现 → 自动negative
```

## 11.3 可靠示例

### Traffic light

只有显式属性支持：

```text
color=green → traffic_light_green positive
color=red   → traffic_light_red positive
其他/无属性 → color predicates unknown
```

`traffic_light_visible` 可由 box/poly存在监督。

### Stop sign

仅显式类别/属性能确认 stop sign 时为正；一般 traffic sign 只能监督 `traffic_sign_visible`。

### Front vehicle close/far

基于：

```text
box bottom y
box height/area
front-center overlap
```

使用互斥阈值：

```text
close confidence >= high threshold
far confidence <= low threshold
中间区域 unknown
```

不得同时正。

### Lane/drivable

只有显式 lane style/type 支持：

```text
solid
dashed
turn lane
```

普通 lane poly只监督：

```text
lane boundary/location
```

### Road clear

只有 object source完整且 front corridor中明确无目标时，才允许 counter/clear监督。

## 11.4 Map rasterization

Box：

```text
按原图尺寸归一化到45×80
soft edge 1-patch Gaussian blur
```

Polyline：

```text
线宽2 patches
可选distance transform
```

Drivable：

```text
最近邻/area resize到45×80
保留方向区域
```

不得把完整 semantic segmentation设为正式必要输入。

## 11.5 Test leakage

模型 API：

```python
model(images, mechanism_scale=...)
```

不得接受：

```text
structured_records
BDD100K paths
action target
reason target
threshold
```

Structured evidence只在 trainer/loss 层使用。

---

# 12. Predicate Prediction Loss

Primary predicate head继续输出：

```text
predicate_logits
predicate_attention
```

损失：

\[
L_{pred-cls}
=
MaskedASL(
z_p,
y_p,
mask_p,
reliability_p
).
\]

Map loss：

\[
L_{pred-map}
=
0.5L_{KL}
+
0.5L_{Dice}.
\]

只有 `predicate_map_mask=1` 的条目参与。

Compactness 只作为极小辅助：

```text
0.0005
```

不得通过最小熵强迫 map退化成单patch。

---

# 13. Predicate Naming

实现：

```text
fate_oia/models/aie_predicate_naming.py
```

## 13.1 推理匹配质量

Predicted predicate map：

```text
P_ip [B,32,3600]
```

Evidence map：

```text
M_iak [B,4,4,3600]
```

空间一致性：

\[
q^{space}_{iakp}
=
SoftIoU(M_{iak},P_{ip}).
\]

低维兼容度：

\[
q^{compat}_{iakp}
=
\sigma(
\langle
W_e e_{iak},
k_p
\rangle
).
\]

Predicate presence：

\[
q^{presence}_{ip}
=
predicate\_prob_{ip}.
\]

基础命名质量：

\[
q_{iakp}^{base}
=
q^{space}
q^{compat}
q^{presence}.
\]

若当前有 CF effect：

\[
q_{iakp}
=
q_{iakp}^{base}
\cdot
clip(
norm(e_{iak}^{cf}),
0,1
).
\]

否则仅输出 base confidence，不用于 effect claim。

## 13.2 命名规则

```text
best predicate confidence >= 0.45
second-best margin >= 0.08
predicate presence >= 0.30
```

满足：

```text
name_id = best predicate
```

不满足：

```text
name_id = -1
name = unnamed_visual_evidence
```

不得设置 named/null softmax。

## 13.3 训练期 naming alignment

只在以下条件同时满足时计算：

```text
counterfactual event valid
target-signed contribution > 0
selected-minus-control > 0
BDD100K存在可靠positive predicate map
```

在可靠 positive predicates 中选择最佳 ground-truth map匹配。

\[
L_{name}
=
1-SoftIoU(M_{iak},G_{ip^*})
+
max(
0,
m_{name}
+
q_{wrong}
-
q_{correct}
).
\]

若无可靠 predicate：

```text
L_name = 0
不强制命名
```

Naming loss更新：

```text
AIE evidence probes
low-dimensional compatibility keys
```

不得更新：

```text
primary trunk
predicate head
Reason rereader
```

---

# 14. `AIEReasonRereader`

实现：

```text
fate_oia/models/aie_reason_rereader.py
```

## 14.1 输入

```text
reason_nodes_primary.detach()     [B,21,384]
patch_tokens_by_layer             [B,3,3600,384]
evidence_token.detach()           [B,4,4,384]
evidence_map.detach()             [B,4,4,3600]
bounded_contribution.detach()     [B,4,4]
predicate_attention.detach()      [B,32,3600]
predicate_probs.detach()          [B,32]
```

Reason loss不能更新 Action evidence或predicate head。

## 14.2 Action evidence attention

\[
\gamma_{irak}
=
softmax_{a,k}
\frac{
Q(h_{ir}^{primary})^\top
K(e_{iak})
}{
\sqrt d
}.
\]

Contribution-aware weight：

\[
\bar\gamma_{irak}
\propto
\gamma_{irak}
(
|\widetilde c_{iak}|+\epsilon
)^{0.5}.
\]

Action evidence prior：

\[
M_{ir}^{A}
=
\sum_{a,k}
\bar\gamma_{irak}
M_{iak}.
\]

## 14.3 Predicate prior

从 grammar 的 positive/contradictory matrix得到 reason-specific predicate weight。

\[
M_{ir}^{P}
=
normalize
\left[
\sum_p
\xi_{irp}^{+}P_{ip}
-
0.5
\sum_p
\xi_{irp}^{-}P_{ip}
\right].
\]

所有输入 detach，prior log-bias绝对值上限：

```text
1.5
```

## 14.4 Reason-private full-field reread

每个 Reason 使用自己的 multi-layer query：

\[
s_{irln}^{R}
=
\frac{
Q_r(h_{ir}^{primary})^\top
K_l(F_{iln})
}{
\sqrt d
}
+
\lambda_A^R\log(M_{ir}^{A}+\epsilon)
+
\lambda_P^R\log(M_{ir}^{P}+\epsilon).
\]

约束：

```text
lambda_action_reason <= 0.75
lambda_predicate_reason <= 0.75
```

Reason token：

\[
h_{ir}^{private}
=
\sum_{l,n}
A_{irln}^{R}
V_l^R(F_{iln}).
\]

再执行一层 21-token private self-attention。

## 14.5 Reason residual

\[
\delta z_{ir}^{R}
=
\kappa_R
\tanh(
Head_R^{private}(h_{ir}^{private})/\kappa_R
).
\]

默认：

```text
kappa_reason = 4.0
beta_reason starts at 0.10
beta_reason reaches 1.0 by 10% updates
```

推理：

\[
z_R^{final}
=
z_R^{primary}
+
\beta_R\delta z_R.
\]

训练：

\[
z_R^{final,train}
=
sg(z_R^{primary})
+
\beta_R\delta z_R.
\]

正式 Exp必须就是 `reason_logits_final`，不得存在更强 private-direct分支而未选中。

---

# 15. Evidence-Censored Reason Negative Weight

实现于：

```text
fate_oia/losses/aie_losses.py
configs/aie_reason_counter_evidence.yaml
```

BDD-OIA `reason=0` 不自动等于可靠负例。

## 15.1 Positive

\[
y_{ir}=1
\Rightarrow
w_{ir}=1.
\]

## 15.2 Reliable counter

根据：

```text
grammar contradictory predicate
predicate probability
predicate source completeness
region observability
```

构造 detached counter confidence：

\[
C_{ir}^{counter}\in[0,1].
\]

## 15.3 Zero label weight

\[
y_{ir}=0
\Rightarrow
w_{ir}^{neg}
=
0.25
+
0.75C_{ir}^{counter}.
\]

所以：

```text
无反证zero：weight=0.25
明确反证zero：weight接近1
```

不得：

```text
把zero伪标为positive
online pseudo-label
trainable reliability决定自己的loss
unknown absorbing state
transition matrix
```

## 15.4 Weighted ASL

扩展 `asymmetric_loss_with_logits(..., reduction="none")`：

```python
loss = asymmetric_loss_with_logits(..., reduction="none")
weighted = torch.where(target > 0.5, 1.0, neg_weight) * loss
return weighted.sum() / weight.sum().clamp_min(1.0)
```

---

# 16. `AIEOIAModel`

实现：

```text
fate_oia/models/aie_oia_model.py
```

## 16.1 API

```python
class AIEOIAModel(nn.Module):
    def encode_images(self, images: Tensor) -> dict[str, Any]:
        ...

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        action_scale: float,
        reason_scale: float,
        predicate_bias_enabled: bool = True,
        local_reread_enabled: bool = True,
    ) -> dict[str, Any]:
        ...

    def rerun_action_evidence_from_field(
        self,
        modified_field: dict[str, Any],
        fixed_primary: dict[str, Tensor],
        *,
        action_scale: float,
        predicate_bias_enabled: bool,
    ) -> dict[str, Tensor]:
        ...

    def forward(
        self,
        images: Tensor,
        *,
        action_scale: float = 1.0,
        reason_scale: float = 1.0,
    ) -> dict[str, Any]:
        ...
```

普通 forward不得接收 labels或BDD records。

## 16.2 正式输出

```text
action_logits_primary
action_logits_final
action_logits_final_train

reason_logits_primary
reason_logits_final
reason_logits_final_train

evidence_token
evidence_map
evidence_reference_point
evidence_sampling_offsets
evidence_layer_mixture

raw_contribution
bounded_contribution
action_delta
contribution_reconstruction_error

predicate_logits
predicate_probs
predicate_attention

name_id
name_confidence
name_margin
named_coverage

reason_action_evidence_attention
reason_action_prior
reason_predicate_prior
reason_private_attention
reason_delta
```

## 16.3 同一 field 的 ablation

必须支持，不重跑 DINO：

```text
primary_only
final
predicate_bias_off
local_reread_off
global_only
action_evidence_shuffle
wrong_action_evidence
reason_action_prior_off
reason_predicate_prior_off
all_reason_priors_off
```

---

# 17. Loss Registry

实现：

```text
fate_oia/losses/aie_loss_registry.py
```

每个 loss：

```text
注册一次
计算一次
加权一次
owner唯一
```

## 17.1 Primary CalAlign-core losses

保持 source raw foundation的主要直接目标：

\[
\begin{aligned}
L_{primary}=&
1.00L_{A-primary}\\
&+0.05L_{A-visual}\\
&+0.05L_{A-reason}\\
&+1.00L_{R-primary-partial}\\
&+0.08L_{R-primary-softF1}\\
&+0.12L_{predicate-cls}\\
&+0.05L_{predicate-reason-align}\\
&+0.0005L_{predicate-compactness}.
\end{aligned}
\]

不加入：

```text
matched pair
pair embedding
action combo CE
action combo drop/add
legacy calibration loss
trainable threshold loss
```

## 17.2 Final Action losses

\[
L_{AIE-A}
=
1.00L_{A-final-ASL}
+
0.03L_{A-final-softF1}
+
0.02L_{A-final-cardinality}.
\]

Cardinality 使用：

\[
(\sum_a\sigma(z_a)-\sum_ay_a)^2
\]

不实例化16 action-set classifier。

## 17.3 Final Reason losses

\[
L_{AIE-R}
=
1.00L_{R-final-weighted-ASL}
+
0.08L_{R-ranking}
+
0.04L_{R-softF1}.
\]

## 17.4 Counterfactual

实际权重：

```text
necessity             0.08
contribution_effect   0.04
set_sufficiency       0.02
```

仅在 valid CF event上激活。

## 17.5 Naming/probe regularization

```text
naming_alignment      0.02
duplicate_probe       0.005
delta_regularizer     0.005
```

Duplicate loss只在两个 probes均有显著 target-supportive contribution时计算。

## 17.6 总损失

\[
\boxed{
L
=
L_{primary}
+
L_{AIE-A}
+
L_{AIE-R}
+
L_{CF}
+
L_{name}
+
L_{probe-reg}
}
\]

一个 backward。

---

# 18. 严格参数所有权

| Owner | 参数 | 接受的loss |
|---|---|---|
| `primary` | ego、primary predicate head、trunk、predicate_reason | primary losses |
| `action_evidence` | multi-layer conditioner、16 probes、global/local inquiry、group self-attn | final Action、CF、naming、probe-reg |
| `action_contribution` | contribution heads、bounded residual参数 | final Action、CF |
| `reason_private` | Reason evidence attention、private reread、private self-attn、Reason delta head | final Reason |
| `posthoc_calibration` | 不在模型optimizer中 | train-calib fit only |
| `dino` | frozen | none |

硬梯度合同：

```text
final Reason loss
  → action_evidence grad = 0
  → action_contribution grad = 0
  → primary grad = 0
  → predicate head grad = 0

final Action loss
  → primary grad = 0
  → reason_private grad = 0

primary loss
  → action_evidence grad = 0
  → action_contribution grad = 0
  → reason_private grad = 0

predicate loss
  → action_evidence grad = 0

DINO grad = 0
```

## 18.1 Primary trajectory isolation test

构造两个模型：

```text
Model A：只训练primary losses
Model B：训练primary + AIE losses，但按owner firewall
```

同初始 state、同 batch、同 seed、同 optimizer超参，连续执行2个 optimizer updates。

必须满足：

```text
所有primary参数 max abs difference < 1e-7
```

这是防止重演 LENS “新增路径破坏强base”的核心合同。

---

# 19. Optimizer 与 scheduler

## 19.1 Optimizer groups

```yaml
primary:
  lr: 2.0e-4
  weight_decay: 0.05

action_evidence:
  lr: 1.5e-4
  weight_decay: 0.05

action_contribution:
  lr: 1.5e-4
  weight_decay: 0.05

reason_private:
  lr: 1.5e-4
  weight_decay: 0.05
```

不得把 generator 直接传入多个 group；所有参数先 `list(...)` 并做 exact-cover audit。

## 19.2 Update-based schedule

总 optimizer updates：

```text
ceil(num_micro_batches / accum) × epochs
```

Schedule：

```text
LR warm-up: first 5% updates
grounding multiplier: 0.25 → 1.0 in first 5%
action_scale: 0.10 → 1.0 in first 10%
reason_scale: 0.10 → 1.0 in first 10%
CF loss multiplier: 0 at 0–5%, linear 0→1 at 5–15%
```

之后所有机制保持开启。

Cosine floor：

```text
min_lr_ratio = 0.10
```

不允许最后若干轮 LR 接近 `1e-10`。

## 19.3 Gradient

```text
primary raw norm hard cap = 0.25
global clip = 1.0
```

日志同时记录 cap 前后 norm。

---

# 20. BF16 与运行模式

```python
torch.set_float32_matmul_precision("high")
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

BF16 不使用 FP16 GradScaler。

以下计算强制 FP32：

```text
softmax/entmax logits
counterfactual target-signed margins
contribution reconstruction
metric accumulation
threshold fitting
bootstrap/LCB
```

---

# 21. Train / Audit / Calibration subsets

内部协议优先性能，因此：

```text
train loader：全部16,082 train
train-calib：固定10% train subset，仅用于post-hoc threshold
train-audit：固定1024 train样本，用于no-grad诊断
```

train-calib/train-audit可以被主训练看见；manifest必须明确：

```text
internal_engineering_protocol=true
```

不得把它宣称为正式论文无偏协议。

保存：

```text
split_manifest.json
train_calib_ids.json
train_audit_ids.json
```

包括：

```text
seed
count
sample-id SHA256
overlap checks
```

test不用于 threshold fitting。

---

# 22. Post-hoc calibration

实现：

```text
fate_oia/utils/aie_calibration.py
```

不在模型中实例化 trainable threshold head。

每个 epoch：

1. 冻结模型；
2. 在 train-calib收集 primary/final raw logits；
3. 使用现有 `search_best_thresholds_for_f1` 拟合每标签阈值；
4. 对低支持标签做 group shrinkage；
5. 保存 threshold；
6. 模型 state hash校验前后相同；
7. 一次完整 test forward；
8. 计算 raw fixed 与 deploy threshold metrics。

Group shrinkage：

\[
\theta_l
=
\lambda_l\theta_l^{raw}
+
(1-\lambda_l)\theta_{group},
\]

\[
\lambda_l
=
\frac{n_l}{n_l+50}.
\]

不得：

```text
在test搜索threshold
把test oracle写回模型
threshold参与representation backward
```

---

# 23. Trainer

实现：

```text
fate_oia/engine/train_aie_oia.py
```

## 23.1 一条普通 micro-batch

1. direct image dataloader读取图像；
2. train-only structured predicate targets；
3. 一次 Frozen DINO；
4. primary CalAlign foundation；
5. 16 Action evidence probes；
6. contribution与 final Action；
7. Predicate naming diagnostic；
8. Reason evidence-conditioned reread；
9. final Reason；
10. primary、final、predicate losses；
11. 满足interval时增加CF losses；
12. 一个 backward；
13. accumulation；
14. optimizer step；
15. append-only日志。

## 23.2 Final partial accumulation window

epoch末如果：

```text
micro_batches % accumulation != 0
```

必须 flush 最后窗口，并按实际 micro-batch数重新缩放梯度。

不得丢失尾窗口。

## 23.3 CF event不改变普通batch的DINO次数

日志字段：

```text
dino_calls_ordinary_batch = 1
dino_calls_cf_event = 0 extra
```

## 23.4 No metric early stop

不因 epoch 0/1指标低自动停止。

只允许结构性停止：

```text
NaN/Inf
OOM
DINO重复调用
owner firewall失败
action logit runaway
显存持续增长
artifact/checkpoint损坏
dataloader确认卡死
```

---

# 24. 每100 optimizer updates必须输出

写入：

```text
loss_components.jsonl
owner_gradients.jsonl
runtime_components.jsonl
evidence_components.jsonl
```

字段至少包括：

```text
epoch
micro_step
optimizer_update
learning_rates
action_scale
reason_scale
grounding_scale
cf_scale

loss_total

loss_primary_action
loss_primary_action_visual
loss_primary_action_reason
loss_primary_reason_partial
loss_primary_reason_soft_f1
loss_predicate_cls
loss_predicate_map
loss_predicate_reason_align
loss_predicate_compactness

loss_final_action
loss_final_action_soft_f1
loss_final_action_cardinality

loss_final_reason
loss_final_reason_rank
loss_final_reason_soft_f1

loss_cf_necessity
loss_cf_effect
loss_cf_sufficiency
loss_name
loss_probe_duplicate
loss_delta

primary_action_logit_rms
final_action_logit_rms
action_delta_rms
primary_reason_logit_rms
final_reason_logit_rms
reason_delta_rms

raw_contribution_mean/std
bounded_contribution_mean/std
positive_contribution_rate
negative_contribution_rate
dominant_probe_ratio
probe_effective_count
probe_map_entropy
probe_pairwise_overlap

predicate_bias_strength_mean
predicate_compatibility_entropy
named_coverage
name_confidence_mean
name_margin_mean
unnamed_coverage

cf_valid_count
cf_invalid_count
cf_selected_drop
cf_control_drop
cf_selected_minus_control
cf_contribution_effect_correlation

counter_negative_weight_mean
counter_negative_weight_p10/p50/p90
reliable_negative_rate
weak_negative_rate

primary_grad_raw/capped
action_evidence_grad
action_contribution_grad
reason_private_grad
predicate_grad
dino_grad

data_time
dino_time
primary_time
evidence_global_time
evidence_local_time
reason_reread_time
counterfactual_time
backward_time

allocated_gb
reserved_gb
max_reserved_gb
```

任何核心字段不得用固定0 placeholder冒充。

---

# 25. 每个 epoch 的唯一 test forward

每个 epoch 只执行一次完整 test image encoding。

必须保存 primary/final logits：

```text
action_logits_primary_test.pt
action_logits_final_test.pt
reason_logits_primary_test.pt
reason_logits_final_test.pt
labels_action_test.pt
labels_reason_test.pt
file_names_test.json
```

Raw metrics：

```text
primary Action
final Action
primary Reason
final Reason
```

Deploy metrics：

```text
train-calib threshold
```

Best：

```text
best_selection_split = test
best_selection_metric = deploy_joint
deploy_joint = 0.5*Act_mF1 + 0.5*Exp_mF1
```

Manifest：

```json
{
  "eval_splits": ["test"],
  "best_selection_split": "test",
  "best_selection_metric": "deploy_joint",
  "internal_test_selected": true,
  "publication_eligible": false
}
```

---

# 26. 固定 audit-128 诊断

对固定前128个 test IDs，复用同一 encoded field输出：

```text
primary_only
final
predicate_bias_off
local_reread_off
global_only
action_evidence_shuffle
wrong_action_evidence
reason_action_prior_off
reason_predicate_prior_off
all_reason_priors_off
```

Counterfactual：

```text
selected substitution
matched control substitution
union sufficiency
wrong-probe substitution
wrong-action evidence
```

保存完整 tensor仅限 audit-128：

```text
evidence_map
contribution
predicate name/confidence
reason-evidence attention
selected/control masks
counterfactual logits/effects
```

禁止为 branch重新运行DINO。

---

# 27. Runtime profiler

实现：

```text
python -m fate_oia.engine.profile_aie_oia
```

真实路径必须包括：

```text
official DINO
360×640
3×3600 tokens
primary foundation
32 predicates
16 probes global inquiry
16 probes local reread
21 Reason reread
每4 updates一次CF的摊销
BF16
```

比较：

```text
A: batch=6, accum=5, workers=8, probe_chunk=16
B: batch=6, accum=5, workers=8, probe_chunk=8
C: batch=5, accum=6, workers=8, probe_chunk=16
D: batch=4, accum=8, workers=8, probe_chunk=16
```

每项：

```text
10 warm-up microbatches
30 measured microbatches
至少2个CF event
```

选择：

```text
reserved memory <45GB
无OOM
包含CF摊销后的samples/sec最高
data_time/step_time合理
```

差异小于3%时选显存更低者。

默认候选：

```text
batch=6
accum=5
workers=8
probe_chunk=16
```

必须由真实 profiler确认。

---

# 28. 推荐配置

```yaml
experiment:
  name: aie_oia_v1
  direct_image: true
  feature_cache_enabled: false
  token_compression: none
  eval_splits: [test]
  best_selection_split: test
  best_selection_metric: deploy_joint
  internal_test_selected: true
  publication_eligible: false

data:
  data_root: E:\sbw\BDD-OIA\data
  raw_root: E:\sbw\BDD-OIA
  bdd100k_root: E:\sbw\BDD100K
  image_height: 360
  image_width: 640
  patch_size: 8
  action_dim: 4
  reason_dim: 21
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  split_seed: 20260806
  train_calib_fraction: 0.10
  train_audit_count: 1024
  train_on_all_train: true

backbone:
  arch: vit_small
  patch_size: 8
  selected_layers: [3,7,11]
  pretrained_weights: ckp/reference/dino_deitsmall8_pretrain.pth
  checkpoint_key: teacher
  freeze_backbone: true
  no_grad_backbone: true

primary:
  dim: 384
  action_dim: 4
  reason_dim: 21
  scene_predicates: configs/aie_scene_predicates.yaml
  reason_grammar: configs/acpr_reason_predicate_grammar.yaml

evidence:
  probes_per_action: 4
  local_points_per_layer: 8
  max_offset: 0.25
  predicate_bias_max: 0.25
  action_kappa: 3.0
  action_scale_start: 0.10
  action_scale_ramp_ratio: 0.10
  probe_chunk_size: 16
  dominant_probe_max_ratio: 0.90

reason_private:
  action_prior_max: 0.75
  predicate_prior_max: 0.75
  reason_kappa: 4.0
  reason_scale_start: 0.10
  reason_scale_ramp_ratio: 0.10
  zero_negative_floor: 0.25

counterfactual:
  enabled: true
  interval_optimizer_updates: 4
  batch_fraction: 0.50
  max_actions_per_sample: 2
  max_atoms_per_event: 8
  topk_patches: 64
  max_control_overlap: 0.20
  necessity_margin: 0.05
  sufficiency_ratio: 0.50
  sufficiency_margin: 0.05
  start_ratio: 0.05
  full_ratio: 0.15
  rerun_dino: false

training:
  epochs: 20
  batch_size: 6
  gradient_accumulation_steps: 5
  precision: bf16
  optimizer: AdamW
  weight_decay: 0.05
  lr_primary: 2.0e-4
  lr_action_evidence: 1.5e-4
  lr_action_contribution: 1.5e-4
  lr_reason_private: 1.5e-4
  warmup_ratio: 0.05
  min_lr_ratio: 0.10
  primary_grad_cap: 0.25
  global_grad_clip: 1.0
  action_logit_norm_cap: 20.0
  no_metric_early_stop: true

loss_weights:
  primary_action: 1.00
  primary_action_visual: 0.05
  primary_action_reason: 0.05
  primary_reason_partial: 1.00
  primary_reason_soft_f1: 0.08
  predicate_cls: 0.12
  predicate_map: 0.06
  predicate_reason_align: 0.05
  predicate_compactness: 0.0005

  final_action: 1.00
  final_action_soft_f1: 0.03
  final_action_cardinality: 0.02

  final_reason: 1.00
  final_reason_rank: 0.08
  final_reason_soft_f1: 0.04

  cf_necessity: 0.08
  cf_effect: 0.04
  cf_sufficiency: 0.02
  naming: 0.02
  probe_duplicate: 0.005
  delta: 0.005

calibration:
  enabled: true
  source: train_calib
  grid_step: 0.01
  group_shrinkage_support: 50
  test_oracle_writeback: false

runtime:
  no_feature_cache: true
  require_no_token_compression: true
  test_every_epoch: true
  save_every_epoch: true
  fixed_test_audit_samples: 128
  print_every_optimizer_updates: 100
  max_reserved_memory_gb: 45.0
```

---

# 29. 实施任务顺序

## T01：建立 worktree 与远程 branch

完成第1、2节。

## T02：源行为快照

在真实图像保存：

```text
source DINO shapes
source primary logits
label nodes
label attention
predicate outputs
source hashes
```

## T03：实现 `AIECalAlignFoundation`

完成 state mapping与数值等价。

## T04：实现 conservative predicate builder

先输出：

```text
structured_source_inventory.json
predicate_coverage_report.json
```

再写规则。

## T05：实现 multi-layer conditioner

完成 shape、RMSNorm、K/V复用和无dense物化测试。

## T06：实现 global evidence probes

完成16 probes、layer mixture、full-field attention。

## T07：实现 predicate-bounded prior

完成低带宽接口与 detach/上限测试。

## T08：实现 deformable local reread

完成 `grid_sample`、offset/weight和full-field关联测试。

## T09：实现 contribution head

完成非零早期梯度、bounded delta、精确重构。

## T10：实现 counterfactual engine

完成 selected/control、same-region substitution、无DINO重跑。

## T11：实现 naming

完成质量准入、unnamed abstention、无null竞争。

## T12：实现 Reason private reread

完成Action evidence/predicate prior detach、full-field reread。

## T13：实现 evidence-censored Reason loss

完成positive/negative权重合同。

## T14：实现 loss registry与owner map

每个loss一次，每个参数一个owner。

## T15：实现 trainer

完成BF16、update schedule、accum尾窗、完整日志。

## T16：实现 calibration/evaluator

每epochtrain-calib fit、一次test。

## T17：实现 artifacts/resume

恢复：

```text
model
optimizer
scheduler/update count
RNG
split manifests
calibration
audit IDs
```

## T18：实现 audit/Skill

运行配套审查。

## T19：真实 profiler

选最快配置。

## T20：唯一4轮 pilot

不跑多seed、多结构。

## T21：pilot通过后20轮 full train

push当前HEAD并验证remote SHA后启动。

---

# 30. 必须新增测试

```text
tests/test_aie_source_head_contract.py
tests/test_aie_worktree_contract.py
tests/test_aie_forbidden_paths.py

tests/test_aie_foundation_equivalence.py
tests/test_aie_primary_trajectory_isolation.py
tests/test_aie_one_dino_call.py
tests/test_aie_dino_frozen.py

tests/test_aie_multilayer_conditioner.py
tests/test_aie_global_probe_shapes.py
tests/test_aie_groupwise_probe_attention.py
tests/test_aie_no_cross_action_probe_attention.py
tests/test_aie_predicate_low_bandwidth.py
tests/test_aie_predicate_bias_bound.py

tests/test_aie_deformable_reread.py
tests/test_aie_offset_bound.py
tests/test_aie_local_reread_changes_token.py
tests/test_aie_no_dense_qnd_materialization.py

tests/test_aie_contribution_nonzero_grad.py
tests/test_aie_contribution_reconstruction.py
tests/test_aie_action_base_detach.py
tests/test_aie_direction_preserving_cap.py

tests/test_aie_counterfactual_no_dino_rerun.py
tests/test_aie_same_region_control.py
tests/test_aie_control_mass_match.py
tests/test_aie_selected_control_overlap.py
tests/test_aie_target_signed_margin.py
tests/test_aie_effect_loss_direction.py
tests/test_aie_sufficiency_union.py

tests/test_aie_conservative_predicate_targets.py
tests/test_aie_no_green_from_presence.py
tests/test_aie_no_stop_from_sign.py
tests/test_aie_no_close_far_double_positive.py
tests/test_aie_no_lane_style_guess.py
tests/test_aie_test_forward_rgb_only.py

tests/test_aie_naming_abstention.py
tests/test_aie_no_null_competition.py
tests/test_aie_naming_quality_gate.py

tests/test_aie_reason_reread_full_field.py
tests/test_aie_reason_action_prior_detached.py
tests/test_aie_reason_predicate_prior_detached.py
tests/test_aie_formal_reason_is_final.py
tests/test_aie_reason_negative_weight.py

tests/test_aie_reason_to_action_firewall.py
tests/test_aie_action_to_reason_firewall.py
tests/test_aie_predicate_to_action_grad_firewall.py
tests/test_aie_owner_exact_cover.py
tests/test_aie_loss_terms_added_once.py

tests/test_aie_posthoc_calibration_no_model_mutation.py
tests/test_aie_test_not_used_for_calibration.py
tests/test_aie_same_forward_branch_metrics.py

tests/test_aie_runtime_memory_contract.py
tests/test_aie_accumulation_tail_flush.py
tests/test_aie_resume_exact.py
tests/test_aie_artifact_hash_contract.py
tests/test_aie_pilot_gate_recomputation.py
tests/test_aie_supervisor_protocol.py
```

原有基础回归测试仍须通过：

```text
test_bdd_oia_dataset
test_acpr_dino_field
test_acpr_label_trunk
test_acpr_model_forward
```

---

# 31. 预训练实施审查

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

必须生成：

```text
AIE_IMPLEMENTATION_REVIEW.json
AIE_RUNTIME_PROFILE.json
```

绑定：

```text
git_head
source_head
config_hash
schema_hash
source_tree_hash
split_seed
```

---

# 32. 唯一四轮 Pilot

## 32.1 数据

```text
train-main     4096
train-audit    1024
train-calib     512
test             512
epochs             4
seed        20260806
```

## 32.2 命令

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_aie_oia `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --output-dir .background_runs\aie_oia_v1_pilot_<HEAD> `
  --run-kind pilot `
  --epochs 4 `
  --max-train-samples 4096 `
  --max-audit-samples 1024 `
  --max-calib-samples 512 `
  --max-test-samples 512 `
  --device cuda
```

## 32.3 Gate A：Foundation 与 primary保护

```text
foundation equivalence <1e-6
primary loss从step0非零
primary owner有grad和parameter delta
primary trajectory isolation <1e-7
DINO grad=0
普通batch DINO calls=1
```

## 32.4 Gate B：Evidence 立即激活

前50 optimizer updates：

```text
action_evidence_grad > 0
action_contribution_grad > 0
raw contribution std > 1e-3
evidence map entropy有限
local reread token != global-only token
```

## 32.5 Gate C：Action 排序方向

最后两轮至少一轮：

\[
Act_{mAP}^{final}
\ge
Act_{mAP}^{primary}
+
0.003.
\]

并且：

\[
Act_{mF1}^{final}
\ge
Act_{mF1}^{primary}-0.002.
\]

## 32.6 Gate D：Reason 排序方向

最后两轮至少一轮：

\[
Exp_{mAP}^{final}
\ge
Exp_{mAP}^{primary}
+
0.003.
\]

并且：

\[
Exp_{mF1}^{final}
\ge
Exp_{mF1}^{primary}-0.003.
\]

## 32.7 Gate E：Counterfactual

```text
valid CF events > 0
selected-minus-control macro mean > 0
至少3/4 Action方向为正
contribution-effect Spearman > 0.30
selected/control overlap <=0.20
```

## 32.8 Gate F：Probe不坍缩

禁止：

```text
全部contribution=0
一个probe承担>90%贡献且覆盖>80%样本
四个map平均两两overlap>0.90
所有map均匀
所有map单patch
```

## 32.9 Gate G：Predicate命名诚实

Grounded audit子集：

```text
naming quality > matched random
reliable predicate deletion方向正确
5% < named coverage < 90%
unnamed coverage >0
```

## 32.10 Gate H：Reason firewall

```text
final Reason loss → Action evidence grad = 0
final Reason loss → primary grad = 0
final Action loss → Reason-private grad = 0
final Action loss → primary grad = 0
predicate loss → Action evidence grad = 0
```

## 32.11 Gate I：运行健康

```text
无NaN/Inf/OOM
reserved<45GB
无cache/compression
无[B,Q,N,D]大张量
accum尾窗被flush
日志字段完整
```

失败时：

```text
不得降低门槛后直接full train
不得增加一个新gate掩盖问题
将原始证据追加findings.md
```

---

# 33. Full Train

## 33.1 配置

```text
epochs       20
seed          20260806
precision     bf16
batch/accum   profiler结果
eval          test only
best          deploy_joint
```

## 33.2 启动前绑定

当前 HEAD 必须匹配：

```text
AIE_IMPLEMENTATION_REVIEW.json
AIE_PILOT_PASS.json
AIE_FULL_TRAIN_READY.json
AIE_RUNTIME_PROFILE.json
```

任何模型、loss、trainer、schema或config改动都使 pilot pass失效。

## 33.3 命令

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_aie_oia_foreground `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --output-dir E:\FATE_OIA_aie_oia_v1_full_<HEAD> `
  --run-kind full `
  --epochs 20 `
  --device cuda
```

## 33.4 Checkpoint

每轮：

```text
checkpoint_latest.pth
checkpoint_epoch_XXX.pth
```

Best：

```text
checkpoint_best_test_deploy_joint.pth
checkpoint_best_test_action_mF1.pth
checkpoint_best_test_reason_mF1.pth
checkpoint_best_test_action_mAP.pth
checkpoint_best_test_reason_mAP.pth
```

论文主结果只能使用同一个 joint-best checkpoint。

---

# 34. Git 提交顺序建议

```text
1. chore: create AIE-OIA V1 worktree and contracts
2. refactor: add CalAlign-compatible AIE foundation
3. feat: add action-induced global/local evidence probes
4. feat: add exact contribution and final action
5. feat: add matched counterfactual evidence training
6. feat: add conservative predicate grounding and naming
7. feat: add evidence-conditioned private reason reread
8. feat: add AIE losses and owner firewall
9. feat: add trainer evaluator calibration and artifacts
10. test: add AIE mechanism and protocol tests
11. docs: add AIE implementation plan and audit skill
12. audit: record implementation closure
```

每个 commit 后 push。

---

# 35. 最终禁止事项

Codex 不得：

```text
直接在acpr_calalign_v1_2 worktree修改
加载RunC/CalAlign历史task checkpoint作为teacher
蒸馏历史ckpt
生成/读取feature cache
token compression
增加视频输入
加入VLM/MLLM
加入graph/PMI/static co-occurrence delta
恢复HardPair/pair memory
让action-set成为final Action
让reason logits进入Action evidence
让predicate high-dimensional token进入Action value path
让BDD100K GT进入test forward
设置named/null softmax
通过route entropy强迫单一路由
把全部reason=0作为hard negative
用unknown/emission替代formal Reason
为counterfactual重跑DINO
用zero placeholder冒充loss
只定义类但formal forward不调用
输出tensor但不进入loss/backward
出现更强private branch但formal未选用
未通过pilot直接full train
```

---

# 36. Definition of Done

“能训练”不算完成。

必须证明：

```text
CalAlign primary raw logits真实恢复
primary从头到尾直接训练
AIE final loss不能改变primary轨迹
16 probes真实读取三层完整视觉场
local reread不是MLP占位
predicate只通过低带宽空间接口进入Action
每个evidence atom进入formal final Action
contribution精确重构final residual
selected deletion优于matched control
contribution与effect相关
predicate可可靠命名或诚实unnamed
Reason读取detach后的Action evidence
noisy Reason不能回写Action evidence
formal Reason就是高容量direct refined Reason
所有owner有非零grad和parameter delta
所有重要机制在日志中可独立判断
每个epoch只执行一次test image encoding
无cache、无compression、无DINO重复调用
```

唯一正式方法：

\[
\boxed{
\textbf{AIE-OIA V1}
=
\text{CalAlign primary}
+
\text{Action-induced visual evidence}
+
\text{effect-consistent contribution}
+
\text{quality-aware predicate naming}
+
\text{asymmetric Reason evidence sharing}
}
\]
