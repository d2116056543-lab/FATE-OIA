# LENS-OIA V1：Codex 完整代码级实施计划

## Latent Evidence under Noisy Supervision for Explainable Object-Induced Action

**日期：** 2026-08-05  
**仓库：** `d2116056543-lab/FATE-OIA`  
**唯一源分支：** `acpr_calalign_v1_2`  
**已核对源 HEAD：** `373aa49feac17372574fd7fb056c1d79c7c848fe`  
**目标分支：** `acpr_lens_oia_v1_direct_image`  
**目标 worktree：** `E:\sbw\FATE_Drive\fate_oia_acpr_lens_oia_v1_worktree`  
**正式任务：** 单帧 RGB、4 action、21 reason 的 BDD-OIA 多标签联合学习  
**硬件：** NVIDIA RTX 5880，48 GB  
**用户内部实验协议：** 每轮只评估 test，以 test deploy-joint 选择 best  
**论文边界：** 该协议必须在 manifest 中标注 `internal_test_selected=true`、`publication_eligible=false`

---

# 0. 文件地位与唯一方案

本文件是 Codex 实施 LENS-OIA V1 的唯一代码合同。不得把它解释为多个可选方案，也不得自行将 LENS 改写成 SAVE、METER、DEFT、BASIS、PU-only、threshold-only 或其他历史分支。

LENS-OIA 的唯一科学命题是：

> BDD-OIA 的 21 维 reason 标签不是无噪声视觉事实，而是潜在视觉谓词状态经过不完整、错误或单帧不可观察的标注过程后形成的观测。模型应以一套潜在视觉证据同时解释 action 与 benchmark reason，但 raw reason annotation 不得直接改写 final action。

统一概率分解：

\[
p(y^A,\widetilde y^R\mid x)
=
\sum_{\mathbf s}
p_\phi(\mathbf s\mid x)
p_\theta(y^A\mid x,\mathbf s)
p_\psi(\widetilde y^R\mid\mathbf s).
\]

其中每个 reason 的潜在状态为：

\[
s_{ir}\in\{+,-,?\},
\]

分别表示：

```text
+  图像中存在支持该谓词的可观察证据
-  图像中存在反对该谓词的可观察证据
?  当前单帧不可观察、证据不足或标注依赖历史/上下文
```

整个正式模型只有一条主线：

```text
RGB
 ↓
Frozen DINO layers 3/7/11，完整3600 patch tokens
 ↓
CalAlign-compatible 25-query foundation
 ↓
21个reason-aligned adaptive visual evidence maps
 ↓
present / counter / unknown latent posterior
 ├── clean observable state → CalAlign-compatible action fusion
 ├── action-conditioned factor-local full-field rereading
 └── ordered annotation emission → formal benchmark Exp
```

禁止另建：

```text
clean/private reason competing branches
utility predictor gate
online matched-control teacher
HardPair/pair memory
graph/PMI
完整21×21噪声转移矩阵
VLM/MLLM
蒸馏或历史checkpoint教师
feature cache
token compression
```

---

# 1. 开始任何远程任务前的强制规则

Codex 在执行 Git、读取远程数据、修改代码、测试、训练、评估或 push 前，必须读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

三者是唯一训练/实验状态 Markdown：

```text
task_plan.md  记录计划与约束
findings.md   记录发现、根因和审查结论
progress.md   记录按时间发生的实施、测试和训练进度
```

不得新建：

```text
implementation_status.md
audit_status.md
pilot_status.md
run_status.md
training_status.md
```

本实施计划和配套 Skill 属于方法规范文件，允许放入：

```text
docs/superpowers/plans/
docs/superpowers/specs/
.codex/skills/
```

但不能代替三个 canonical 状态文件。

---

# 2. 源分支验证与 worktree 合同

## 2.1 验证远程源 HEAD

```powershell
cd E:\sbw\FATE_Drive\fate_oia_worktree
git fetch origin --prune
git rev-parse origin/acpr_calalign_v1_2
```

必须输出：

```text
373aa49feac17372574fd7fb056c1d79c7c848fe
```

如果不一致：

```text
STOP
将实际SHA追加到findings.md
不得自行基于不同HEAD实施
```

## 2.2 禁止修改源 worktree

开始前记录：

```powershell
git -C E:\sbw\FATE_Drive\fate_oia_worktree status --porcelain --untracked-files=all
git -C E:\sbw\FATE_Drive\fate_oia_worktree rev-parse HEAD
git worktree list --porcelain
```

整个任务结束时再次核对。源 worktree 必须保持原状态。

## 2.3 新建目标 worktree

先检查目标 branch 和目录不存在：

```powershell
git show-ref --verify --quiet refs/heads/acpr_lens_oia_v1_direct_image
Test-Path E:\sbw\FATE_Drive\fate_oia_acpr_lens_oia_v1_worktree
```

若任一已存在，停止并记录，不能覆盖。

创建：

```powershell
git worktree add `
  -b acpr_lens_oia_v1_direct_image `
  E:\sbw\FATE_Drive\fate_oia_acpr_lens_oia_v1_worktree `
  origin/acpr_calalign_v1_2
```

验证：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_lens_oia_v1_worktree
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
```

必须满足：

```text
branch = acpr_lens_oia_v1_direct_image
HEAD   = 373aa49feac17372574fd7fb056c1d79c7c848fe
status = clean
```

## 2.4 立即同步 GitHub branch

```powershell
git push -u origin acpr_lens_oia_v1_direct_image
git ls-remote origin refs/heads/acpr_lens_oia_v1_direct_image
```

以后每个实现阶段：

```text
commit
push
核对 local HEAD == remote HEAD
```

---

# 3. 当前源结构的代码级事实

Codex 不得根据文件名猜测，必须先阅读并记录以下实际行为。

## 3.1 `ACPRDinoFieldExtractor`

源文件：

```text
fate_oia/models/acpr_dino_field.py
```

当前已经：

```text
输入固定360×640
Frozen DINO ViT-S/8
selected layers = 3/7/11
patch_tokens_by_layer = [B,3,3600,384]
cls_tokens_by_layer   = [B,3,384]
no_grad
```

LENS 直接复用，不重写 backbone。

## 3.2 `ACPRLabelTrunk`

源文件：

```text
fate_oia/models/acpr_label_trunk.py
```

当前已经包含：

```text
25个label queries
完整patch检索
entmax attention
25-token self-attention
4个action nodes
21个reason nodes
visual action head
reason_to_action
sample-dependent fusion gate
```

当前 source action：

\[
z_a^{Cal}
=
g_a z_a^{visual}
+
(1-g_a)W_{R\rightarrow A}z_R^{visual}.
\]

LENS 必须保留该基础，不能以新 action head 取代。

## 3.3 `ACPROIAModel`

源文件：

```text
fate_oia/models/acpr_oia_model.py
```

当前还实例化：

```text
32 scene predicates
predicate reason delta
pair memory
action-combo auxiliary
trainable threshold head
legacy calibration
```

LENS 正式模型不能直接继承整个 `ACPROIAModel`，否则 pair、action-set、trainable threshold 等无关路径会继续进入参数表和审查范围。

正确做法是新建 `LENSCalAlignFoundation`，只复用：

```text
ACPRDinoFieldExtractor
ACPREgoRegionEncoder
ACPRScenePredicateHead        仅作为CalAlign-compatible foundation context
ACPRLabelTrunk
ACPRPredicateReasoner         仅用于source reason control/equivalence
```

不得在 LENS 正式路径实例化：

```text
ACPRPairMemory
ACPRActionComboAux
ACPRThresholdHead
ACPRCalibrationHead
```

## 3.4 当前 weak predicate target 的已知错误风险

`acpr_predicate_targets.py` 当前会做出过强推断，例如：

```text
只看到traffic light就同时标记traffic_light_green
只看到traffic sign就同时标记stop_sign_present
一个front car可能同时触发close/far
普通lane poly可能被推断为solid/turn region
```

LENS 不得复用这些 target 逻辑。必须新建 fail-closed、source-complete-aware 的三状态 grounding builder。

## 3.5 当前 trainer 的可复用与必须删除部分

可复用：

```text
direct image dataset
AspectRatioLetterboxTransform
test-only epoch evaluation
artifact写入基础
train-calib split思想
batch/accum fallback思想
```

必须删除：

```text
pair mining与memory enqueue
epoch式HardPair schedule
trainable threshold optimizer group
action-set auxiliary
旧predicate_reason alignment loss
test oracle threshold用于任何正式选择
```

---

# 4. 目标目录结构

## 4.1 新增配置与 schema

```text
configs/fate_oia_train_360x640_lens_oia_v1.yaml
configs/lens_reason_state_schema.yaml
configs/lens_observability_groups.yaml
```

`lens_reason_state_schema.yaml` 必须包含每个 reason：

```yaml
id:
name:
mirror_id:
observability_group:
groundability: full | partial | latent
default_region:
support_sources:
counter_sources:
complete_source_required:
unknown_prior:
```

左/右镜像必须明确：

```text
9  ↔ 15
10 ↔ 16
11 ↔ 17
12 ↔ 18
13 ↔ 19
14 ↔ 20
```

Action 镜像：

```text
0 forward → 0
1 stop    → 1
2 left    → 3
3 right   → 2
```

## 4.2 新增模型

```text
fate_oia/models/lens_calalign_foundation.py
fate_oia/models/lens_adaptive_evidence.py
fate_oia/models/lens_latent_state.py
fate_oia/models/lens_annotation_emission.py
fate_oia/models/lens_action_reread.py
fate_oia/models/lens_oia_model.py
```

## 4.3 新增 grounding/data

```text
fate_oia/datasets/lens_structured_evidence.py
fate_oia/datasets/lens_splits.py
fate_oia/datasets/lens_mirror.py
```

## 4.4 新增损失

```text
fate_oia/losses/lens_action_losses.py
fate_oia/losses/lens_reason_losses.py
fate_oia/losses/lens_latent_losses.py
fate_oia/losses/lens_grounding_losses.py
fate_oia/losses/lens_loss_registry.py
```

## 4.5 新增训练与审查

```text
fate_oia/engine/train_lens_oia.py
fate_oia/engine/eval_lens_oia.py
fate_oia/engine/profile_lens_oia.py
fate_oia/engine/audit_lens_oia_implementation.py
fate_oia/engine/evaluate_lens_oia_pilot.py
fate_oia/engine/supervise_lens_oia_foreground.py
```

## 4.6 新增工具

```text
fate_oia/utils/lens_artifacts.py
fate_oia/utils/lens_calibration.py
fate_oia/utils/lens_metrics.py
fate_oia/utils/lens_contracts.py
fate_oia/utils/lens_hashes.py
```

## 4.7 新增脚本与文档

```text
scripts/FATE_OIA_lens_oia_v1_pilot.ps1
scripts/FATE_OIA_lens_oia_v1_foreground.ps1

.codex/skills/lens-oia-v1-implementation-audit/SKILL.md
docs/superpowers/specs/2026-08-05-lens-oia-v1-design.md
docs/superpowers/plans/2026-08-05-lens-oia-v1-implementation.md
```

---

# 5. `LENSCalAlignFoundation`

## 5.1 API

```python
class LENSCalAlignFoundation(nn.Module):
    def encode_images(self, images: Tensor) -> dict[str, Any]:
        ...

    def decode_field(self, field: dict[str, Any]) -> dict[str, Tensor]:
        ...

    def forward(self, images: Tensor) -> dict[str, Any]:
        ...
```

## 5.2 内部模块

```python
self.dino = ACPRDinoFieldExtractor(...)
self.ego = ACPREgoRegionEncoder(...)
self.scene_predicate = ACPRScenePredicateHead(...)
self.trunk = ACPRLabelTrunk(...)
self.source_predicate_reason = ACPRPredicateReasoner(...)
```

不得实例化 pair、action-set、threshold 或 calibration。

## 5.3 输出合同

```text
patch_tokens_by_layer       [B,3,3600,384]
cls_tokens_by_layer         [B,3,384]
source_scene_predicate_*    仅foundation control
label_nodes_source          [B,25,384]
label_attention_source      [B,25,3600]
action_nodes_source         [B,4,384]
reason_nodes_source         [B,21,384]
action_visual_source        [B,4]
reason_visual_source        [B,21]
action_reason_source        [B,4]
action_fusion_gate_source   [B,4]
action_logits_source        [B,4]
reason_logits_source        [B,21]
```

定义：

```text
action_logits_source = trunk["action_logits_direct"]
reason_logits_source =
    trunk["reason_logits_visual"] +
    source_predicate_reason["predicate_reason_delta"]
```

## 5.4 State-dict 映射

必须提供：

```python
def load_from_acpr_state_dict(
    self,
    acpr_state_dict: Mapping[str, Tensor],
    strict: bool = True,
) -> LoadResult:
    ...
```

映射：

```text
dino.*
ego.*
predicate_head.*      → scene_predicate.*
trunk.*
predicate_reason.*    → source_predicate_reason.*
```

用于数值等价测试，不用于加载历史训练 checkpoint 作为正式初始化。

正式训练仍然：

```text
从随机任务头开始
只加载官方DINO预训练权重
```

---

# 6. Adaptive Visual Evidence Pooling

实现：

```text
fate_oia/models/lens_adaptive_evidence.py
```

## 6.1 输入

```text
reason_nodes_source    [B,21,D]
patch_tokens_by_layer  [B,3,N,D]
soft_region_prior      Optional[B,21,N]
```

## 6.2 Label-specific layer fusion

每个 reason 有 3 层权重：

\[
\omega_{rl}
=
softmax_l(\lambda_{rl}).
\]

\[
F_{irn}
=
\sum_l
\omega_{rl}W_lF_{iln}.
\]

输出不得 materialize `[B,21,3600,384]`。实现方式：

```text
先投影三层patch
score阶段按einsum计算[B,21,3600]
value pooling按reason chunk完成
```

## 6.3 Score 与信噪比统计

\[
a_{irn}
=
\frac{
(W_qh_{ir})^\top
(W_kF_{irn})
}{
\sqrt d
}.
\]

计算：

```text
score_mean
score_std
topk_mean，k=32
topk_gap = topk_mean-score_mean
normalized_dispersion
```

\[
\chi_{ir}
=
\frac{topk\_mean-mean}
{std+\epsilon}.
\]

## 6.4 自适应温度

\[
\tau_{ir}
=
\tau_{\min}
+
(\tau_{\max}-\tau_{\min})
\sigma
\left(
MLP_\tau[
h_{ir},\chi_{ir},std_{ir}
]
\right).
\]

默认：

```text
tau_min = 0.35
tau_max = 2.00
```

## 6.5 Null/unknown spatial mass

预测 null logit：

\[
b_{ir\varnothing}
=
MLP_\varnothing
[
h_{ir},\chi_{ir},std_{ir}
].
\]

\[
[M_{ir1},...,M_{irN},M_{ir\varnothing}]
=
softmax
\left(
[
a_{ir}/\tau_{ir},
b_{ir\varnothing}
]
\right).
\]

不得：

```text
在3600 patches上强制总质量为1且没有null
hard top-k
hard ROI
将区域外设为-inf
```

Region prior只能作为有限 soft log-bias：

```text
abs(region_bias) <= 2.0
```

## 6.6 Evidence token

\[
e_{ir}
=
\frac{
\sum_nM_{irn}W_vF_{irn}
}{
1-M_{ir\varnothing}+\epsilon
}.
\]

输出：

```text
evidence_map               [B,21,N]
evidence_null_mass         [B,21]
evidence_token             [B,21,D]
evidence_temperature       [B,21]
evidence_snr               [B,21]
evidence_entropy           [B,21]
evidence_layer_weight      [21,3]
```

---

# 7. 可识别的三状态 posterior

实现：

```text
fate_oia/models/lens_latent_state.py
```

不直接使用自由 3-way softmax。为避免 present/counter/unknown 任意置换，使用可识别分解：

\[
q_{ir}^{obs}=1-u_{ir},
\]

\[
q_{ir}^{sign}
=
\sigma(z_{ir}^{support}).
\]

\[
v_{ir+}
=
(1-u_{ir})q_{ir}^{sign},
\]

\[
v_{ir-}
=
(1-u_{ir})(1-q_{ir}^{sign}),
\]

\[
v_{ir?}
=
u_{ir}.
\]

## 7.1 Support logit

\[
z_{ir}^{support}
=
z_{ir}^{source-visual}
+
\alpha_{state}(s)\Delta z_{ir}^{evidence}.
\]

`support_delta` 最后一层严格 zero-init。

输入：

```text
source reason node
evidence token
null mass
entropy
SNR
```

## 7.2 Unknown probability

\[
u_{ir}^{learned}
=
\sigma
\left(
MLP_u[
h_{ir},e_{ir},
M_{ir\varnothing},
H(M_{ir}),
\chi_{ir}
]
\right).
\]

\[
u_{ir}
=
\alpha_{unknown}(s)
u_{ir}^{learned}.
\]

因此 progress=0：

```text
unknown=0
support_logit=source reason visual logit
v+=sigmoid(source logit)
v-=1-v+
```

## 7.3 状态 token

\[
d_{ir}
=
e_{ir}
+
\sum_{t\in\{+,-,?\}}
v_{irt}U_{rt}.
\]

输出：

```text
state_prob                 [B,21,3] order=[positive,counter,unknown]
state_positive_prob        [B,21]
state_counter_prob         [B,21]
state_unknown_prob         [B,21]
state_observability        [B,21]
state_support_logit        [B,21]
state_token                [B,21,D]
```

不得对 state posterior 做：

```text
mean(dim=-1)
```

三状态的语义必须完整保留。

---

# 8. Ordered Annotation Emission

实现：

```text
fate_oia/models/lens_annotation_emission.py
```

## 8.1 有序参数化

状态顺序：

```text
counter / unknown / positive
```

\[
\eta_{r-}=b_r,
\]

\[
\eta_{r?}
=
b_r+softplus(u_r),
\]

\[
\eta_{r+}
=
b_r+softplus(u_r)+softplus(v_r).
\]

\[
T_{rs}
=
\sigma(\eta_{rs}).
\]

必须保证：

\[
T_{r+}>T_{r?}>T_{r-}.
\]

## 8.2 Group shrinkage

不使用 21×21 transition matrix。

按 schema 的 observability group 建立 group base，再加 bounded per-label delta：

```text
group 0：directly observable
group 1：partially observable
group 2：temporal/latent
```

每标签 delta scale 默认 0.25。

## 8.3 初始化

Trainer 从 train-main 频率初始化：

```text
T_minus   ≈ clamp(0.25*frequency, 0.005, 0.08)
T_unknown ≈ clamp(frequency, T_minus+0.05, 0.60)
T_plus    ≈ clamp(0.90+0.08*(1-frequency), 0.90, 0.995)
```

再转换到有序参数。

保存：

```text
emission_initialization.json
```

## 8.4 Identity ramp

为了 progress=0 恢复 CalAlign reason：

```text
T_identity(counter,unknown,positive) = [0.0,0.5,1.0]
```

\[
T^{eff}
=
(1-\alpha_{emission})T^{identity}
+
\alpha_{emission}T^{learned}.
\]

Latent benchmark probability：

\[
p_{ir}^{latent-exp}
=
\sum_s
v_{irs}T_{rs}^{eff}.
\]

\[
z_{ir}^{latent-exp}
=
logit
\left(
clip(p_{ir}^{latent-exp})
\right).
\]

Formal reason：

\[
z_{ir}^{formal}
=
(1-\alpha_{reason})
z_{ir}^{source}
+
\alpha_{reason}
z_{ir}^{latent-exp}.
\]

progress=0 必须直接返回 source tensor，不能依赖 `logit(sigmoid(z))` 的近似等价。

输出：

```text
emission_prob              [21,3]
emission_order_margin_1    [21]
emission_order_margin_2    [21]
reason_prob_latent         [B,21]
reason_logits_latent       [B,21]
reason_logits_formal       [B,21]
```

---

# 9. Conflict-discounted Generalized EM

实现：

```text
fate_oia/losses/lens_latent_losses.py
```

模型 forward 不接收 labels。Responsibility 在 loss 层计算。

## 9.1 Visual evidence

\[
m^V_{irs}=v_{irs}.
\]

## 9.2 Annotation likelihood

对 \(\widetilde y_{ir}\in\{0,1\}\)：

\[
m^R_{irs}
\propto
T_{rs}^{\widetilde y_{ir}}
(1-T_{rs})^{1-\widetilde y_{ir}}.
\]

按 state 归一化。

## 9.3 Action-state utility

Action rereader输出：

```text
action_logits_state_substitution [B,21,3,4]
```

计算逐样本逐状态 action BCE：

\[
\ell^A_{irs}
=
\frac14
\sum_a
BCEWithLogits
\left(
z_{ira}^{state},
y_{ia}^{A}
\right).
\]

\[
m^A_{irs}
=
softmax_s
\left(
-\lambda_A(s)\ell^A_{irs}
\right).
\]

\(\lambda_A\) 前15% updates从0连续增至0.5。

## 9.4 冲突折扣

\[
\kappa_{ir}
=
\frac{
JS(m^V_{ir},m^R_{ir})
}{
\log 2
}.
\]

\[
\bar m^R_{irs}
=
(1-\kappa_{ir})m^R_{irs}
+
\kappa_{ir}\mathbf 1[s=?].
\]

## 9.5 Detached responsibility

\[
\gamma_{irs}
=
normalize_s
\left[
m^V_{irs}
\bar m^R_{irs}
m^A_{irs}
\right].
\]

必须：

```python
gamma = gamma.detach()
```

## 9.6 Shared visual gradient attenuation

原始：

\[
w^{raw}_{ir}
=
(1-\kappa_{ir})
(\gamma_{ir+}+\gamma_{ir-}).
\]

为防止历史上的 hard gate/gradient starvation，使用非零 floor：

\[
w_{ir}^{share}
=
w_{\min}(\widetilde y_{ir})
+
[1-w_{\min}(\widetilde y_{ir})]
w_{ir}^{raw},
\]

其中：

```text
observed positive floor = 0.25
observed zero floor     = 0.05
```

数值不变、梯度缩放：

```python
v_safe = v.detach() + w_share[..., None] * (v - v.detach())
source_reason_safe = (
    source_reason.detach()
    + w_share * (source_reason - source_reason.detach())
)
```

Formal reason training logit：

```text
(1-alpha_reason)*source_reason_safe
+ alpha_reason*logit(sum(v_safe*T))
```

Emission `T` 接收完整 annotation gradient；Action head与Action reread不接收 reason annotation gradient。

## 9.7 Loss

\[
L_{state}
=
-\sum_{irs}
\gamma_{irs}\log(v_{irs}+\epsilon).
\]

\[
L_{emission}
=
-\sum_{irs}
\gamma_{irs}
\log
p_\psi(\widetilde y_{ir}\mid s).
\]

输出日志：

```text
conflict_mean/std/p10/p50/p90
share_weight_mean/min/max
gamma_positive/counter/unknown
annotation_likelihood_entropy
action_state_utility_entropy
```

---

# 10. Action base：保留完整强基础但去除annotation noise

## 10.1 Clean observable log-odds

\[
\ell_{ir}^{clean}
=
(1-v_{ir?})
\log
\frac{
v_{ir+}+\epsilon
}{
v_{ir-}+\epsilon
}.
\]

progress=0：

\[
\ell_{ir}^{clean}
=
z_{ir}^{source-visual}.
\]

## 10.2 CalAlign-compatible Action

\[
z_{ia}^{reason-latent}
=
W_{R\rightarrow A}
\ell_i^{clean}.
\]

\[
z_{ia}^{base}
=
g_{ia}^{source}z_{ia}^{visual-source}
+
(1-g_{ia}^{source})
z_{ia}^{reason-latent}.
\]

必须使用原 trunk 的：

```text
reason_to_action
action_fusion_gate
action_visual_logits
```

不得新增第二套平行 action base。

progress=0：

\[
z_A^{base}=z_A^{source}.
\]

---

# 11. Action-conditioned Factor-local Full-field Rereading

实现：

```text
fate_oia/models/lens_action_reread.py
```

## 11.1 Sparse factor selection

\[
\pi_{iar}
=
entmax_{r\cup\varnothing}
\left[
\frac{
(W_qh_{ia}^{A})^\top
(W_kd_{ir})
}{
\sqrt d
},
b_{a\varnothing}
\right].
\]

输出包括 null factor。

## 11.2 Action patch score

\[
s_{ian}^{A}
=
\frac{
(W_q^Ah_{ia}^{A})^\top
(W_k^AF_{in}^{detail})
}{
\sqrt d
}
+
\lambda_0
\log
(\epsilon+\alpha_{ian}^{source}).
\]

## 11.3 Factor-local reread

\[
\alpha_{iarn}
=
softmax_n
\left[
s_{ian}^{A}
+
\lambda_M
\log
(M_{irn}+\epsilon)
\right].
\]

\[
h_{iar}^{local}
=
\sum_n
\alpha_{iarn}W_v^AF_{in}^{detail}.
\]

实现必须支持：

```text
factor_chunk_size=21  完全vectorized
factor_chunk_size=7   低显存fallback
```

Profiler选择最快配置。

## 11.4 状态特定 contribution

\[
c_{iar}^{(+)}
=
\pi_{iar}
w_{ar}^{\top}
[
h_{iar}^{local},
e_{ir},
U_{r+}
],
\]

\[
c_{iar}^{(-)}
=
\pi_{iar}
w_{ar}^{\top}
[
h_{iar}^{local},
e_{ir},
U_{r-}
].
\]

Unknown 不产生 named Action contribution：

\[
c_{iar}^{(?)}=0.
\]

期望贡献：

\[
c_{iar}
=
v_{ir+}c_{iar}^{(+)}
+
v_{ir-}c_{iar}^{(-)}.
\]

## 11.5 Unnamed direct visual contribution

Null factor对完整 patch field读取：

\[
h_{ia\varnothing}
=
\sum_n
softmax(s_{ian}^{A})
W_vF_{in}^{detail}.
\]

\[
c_{ia\varnothing}
=
\pi_{ia\varnothing}
w_{a\varnothing}^{\top}
h_{ia\varnothing}.
\]

21类 ontology 外的证据只能计入 unnamed，不得伪造 reason 名称。

## 11.6 Bounded final action

\[
C_{ia}^{raw}
=
c_{ia\varnothing}
+
\sum_rc_{iar}.
\]

\[
\Delta z_{ia}^{bounded}
=
\kappa_a
\tanh
\left(
C_{ia}^{raw}/\kappa_a
\right).
\]

\[
z_{ia}^{final}
=
BoundDirection
\left[
z_{ia}^{base}
+
\alpha_{action}(s)
\Delta z_{ia}^{bounded}
\right].
\]

保留历史已验证的：

```text
direction-preserving action logit L2 cap = 20
foundation gradient cap = 0.25
```

## 11.7 精确加性 explanation

\[
scale_{ia}
=
\frac{
\Delta z_{ia}^{bounded}
}{
C_{ia}^{raw}+\epsilon
}.
\]

\[
\widetilde c_{iar}
=
scale_{ia}c_{iar},
\]

\[
\widetilde c_{ia\varnothing}
=
scale_{ia}c_{ia\varnothing}.
\]

必须满足：

\[
z_{ia}^{final}-z_{ia}^{base}
=
\alpha_{action}
\left[
\sum_r\widetilde c_{iar}
+
\widetilde c_{ia\varnothing}
\right].
\]

误差：

```text
fp32 < 1e-6
bf16 < 5e-4
```

## 11.8 State substitution

预先输出：

```text
action_logits_state_substitution [B,21,3,4]
```

不重跑 DINO。

对 reason r、state s：

```text
raw_total_variant =
    raw_total
    - expected_contribution_r
    + state_specific_contribution_r_s
```

再通过同一 tanh/cap 得到 Action variant。

---

# 12. Conservative Structured Evidence Builder

实现：

```text
fate_oia/datasets/lens_structured_evidence.py
```

## 12.1 输出

```text
state_target               [B,21,3]
state_mask                 [B,21]
map_target                 [B,21,3600]
map_mask                   [B,21]
source_reliability         [B,21]
source_id                  [B,21]
source_complete            [B,21]
coverage dictionary
```

## 12.2 Fail-closed 原则

任何规则不具备明确 source attribute 时：

```text
unknown
```

绝不能：

```text
traffic light出现 → green
traffic sign出现 → stop sign
car出现 → close与far同时成立
lane poly出现 → solid line
lane poly出现 → turn lane
记录中没出现对象 → hard absent
```

## 12.3 保守示例

```text
reason 0 traffic light green：
  support：明确color=green
  counter：明确color∈{red,yellow}
  其余unknown

reason 5 obstacle car：
  support：car/truck/bus box位于front corridor
  counter：只有完整对象源且front corridor明确无目标时才允许
  否则unknown

reason 9 no lane on left：
  support：完整drivable source明确左侧无drivable
  counter：完整drivable source明确左侧有drivable
  否则unknown

reason 11 solid line on left：
  support：显式lane attribute=solid
  counter：显式lane attribute=dashed
  否则unknown

reason 14/20 front car turning：
  单帧BDD100K无明确转向属性时不构造hard state target
```

## 12.4 Source inventory

实现前运行一次只读 inventory：

```text
categories
attributes
lane styles
traffic-light colors
drivable availability
per-reason possible support/counter counts
```

写：

```text
artifacts/lens/structured_source_inventory.json
```

这是 JSON artifact，不是 feature cache。

## 12.5 无 test leakage

Structured evidence仅由 trainer在 train batch构造并用于 loss。

模型 API：

```python
model(images, progress=...)
```

不得接受：

```text
structured_records
BDD100K path
reason target
action target
threshold
```

Test forward仅 RGB。

---

# 13. Mirror 与 weak-view consistency

实现：

```text
fate_oia/datasets/lens_mirror.py
```

每4个 optimizer updates，在当前 batch最多一半样本上构造 weak view。

## 13.1 允许变换

```text
brightness
contrast
轻量gamma
轻微feature noise
horizontal mirror
```

不得改变语义的增强使用相同标签。

## 13.2 Horizontal mirror 必须重映射

Action：

```text
[0,1,2,3] → [0,1,3,2]
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

同时重映射：

```text
reason target
state target
map target
evidence map
state posterior
Action labels
```

Map空间水平翻转。

## 13.3 一次 DINO

Original与paired view沿batch维拼接：

```text
one DINO call
```

不得分别调用两次 backbone。

---

# 14. Loss 合同

实现：

```text
fate_oia/losses/lens_loss_registry.py
```

每个 loss：

```text
注册一次
调用一次
权重一次
明确owner
```

## 14.1 Action

\[
\begin{aligned}
L_A=&
1.00L_{ASL}(z_A^{final},y_A)\\
&+0.35L_{ASL}(z_A^{base},y_A)\\
&+0.15L_{ASL}(z_A^{factor-aux},y_A)\\
&+0.03L_{SoftF1}^{A}
+0.02L_{card}^{A}.
\end{aligned}
\]

`factor-aux`：

```text
detach base + bounded factor residual
```

确保 factor route 从 step0有直接Action梯度。

## 14.2 Explanation

Formal training logit使用 conflict-scaled shared gradient。

\[
\begin{aligned}
L_E=&
1.00L_{ASL}(z_R^{formal-train},\widetilde y_R)\\
&+0.35L_{ASL}(z_R^{latent-train},\widetilde y_R)\\
&+0.08L_{rank}^{R}
+0.03L_{SoftF1}^{R}.
\end{aligned}
\]

第二项保证 latent annotation route 在 progress=0也有直接 ranking gradient。

## 14.3 Latent model

\[
L_L
=
0.20L_{state}
+
0.10L_{emission}
+
0.01L_{emission-prior}.
\]

Emission prior只约束其不在前几百step无界漂移，不将其固定为identity。

## 14.4 Grounding与一致性

\[
L_G
=
0.06L_{map-anchor}
+
0.05L_{state-anchor}
+
0.03L_{view-consistency}
+
0.02L_{unknown-prior}
+
0.01L_{route-sparsity}.
\]

## 14.5 总目标

\[
\boxed{
L=L_A+L_E+L_L+L_G
}
\]

只执行一次 backward。

---

# 15. 梯度所有权

| 参数组 | Action | Reason annotation | Latent responsibility | Grounding |
|---|---:|---:|---:|---:|
| Frozen DINO | 0 | 0 | 0 | 0 |
| CalAlign joint core | ✓ | conflict-scaled | ✓ | 0 |
| Foundation scene predicate | ✓/reason shared | conflict-scaled | ✓ | 0 |
| Adaptive evidence pool | ✓ | conflict-scaled | ✓ | ✓ |
| State head | ✓ | conflict-scaled | ✓ | ✓ |
| Ordered emission | 0 | ✓ | ✓ | 0 |
| Action factor reread | ✓ | 0 | 0 | 0 |
| Post-hoc calibration | 0 | 0 | 0 | 0 |

必须通过 autograd probe：

```text
reason-only annotation loss → action head grad == 0
reason-only annotation loss → action reread grad == 0
action-only loss → emission grad == 0
grounding-only loss → DINO grad == 0
action-only loss → evidence/state/action reread grad >0
low-conflict reason loss → evidence grad > high-conflict reason loss
```

---

# 16. `LENSOIAModel` 正式 API

```python
class LENSOIAModel(nn.Module):
    def encode_images(self, images: Tensor) -> dict[str, Any]:
        ...

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        progress: float,
        mechanism_ablation: str = "none",
        return_state_variants: bool = True,
    ) -> dict[str, Any]:
        ...

    def forward(
        self,
        images: Tensor,
        *,
        progress: float = 1.0,
    ) -> dict[str, Any]:
        ...
```

普通 forward不得接收 target或structured records。

## 16.1 必须输出的主要 tensor

```text
action_logits_source
reason_logits_source
action_logits_base
action_logits_factor_aux
action_logits_final

reason_logits_latent
reason_logits_formal
reason_prob_formal

evidence_map
evidence_null_mass
evidence_token
evidence_temperature
evidence_snr
evidence_entropy

state_prob
state_support_logit
state_unknown_prob
state_token

emission_prob
emission_order_margin

factor_selection
factor_local_attention
factor_contribution_state
factor_contribution_expected
factor_contribution_bounded
unnamed_contribution
contribution_reconstruction_error

action_logits_state_substitution
```

## 16.2 分支输出

同一 encoded field 必须能得到：

```text
source_calalign
lens_base
lens_final
factor_only
reread_off
latent_state_off
unknown_off
emission_identity
evidence_map_shuffle
wrong_factor
```

测试 branch 不得重新执行 DINO。

---

# 17. Trainer

实现：

```text
fate_oia/engine/train_lens_oia.py
```

## 17.1 Split

从 train split建立固定、互斥、multi-label iterative stratified split：

```text
train-main   90%
train-audit   5%
train-calib   5%
```

保存：

```text
split_manifest.json
train_main_ids.json
train_audit_ids.json
train_calib_ids.json
```

包括 SHA256。

Pilot在各固定split上截断：

```text
train-main 4096
train-audit 1024
train-calib 512
test 512
```

## 17.2 一条普通训练 batch

1. 直接读取图像。
2. 构造 train-only conservative structured targets。
3. 一次 DINO。
4. CalAlign foundation。
5. Adaptive evidence。
6. Three-state posterior。
7. Ordered emission。
8. Latent clean Action base。
9. Factor-local reread。
10. State-substitution Action。
11. Responsibility与conflict。
12. 单次 backward。
13. optimizer step/accum。
14. append-only日志。

## 17.3 BF16

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

不对 BF16 使用 FP16 GradScaler。

运行前：

```python
torch.set_float32_matmul_precision("high")
```

## 17.4 Scheduler

Update-based：

```text
LR warm-up：前5%
grounding multiplier：0.25→1，前5%
unknown/emission/reason-output/action-reread：0→1，前10%
action-state utility λA：0→0.5，前15%
```

之后不再按epoch开启或关闭任何模块。

Cosine floor：

```text
min_lr_ratio=0.10
```

不能在最后两轮降至近0。

## 17.5 Gradient cap

```text
foundation raw grad hard cap 0.25
global grad clip 1.0
action logit norm cap 20.0
```

日志同时记录 cap 前后 norm。

---

# 18. Calibration 与 test-only 选择

LENS 正式模型内不实例化 trainable threshold head。

每个 epoch：

1. 冻结模型。
2. 在 train-calib 收集 raw logits。
3. 拟合：
   - group-shrinkage per-label threshold；
   - 可选 group temperature。
4. 模型 state hash 在校准前后必须相同。
5. 固定 calibration 后，一次完整 test forward。
6. 计算 raw 与 deploy metrics。
7. 以 test deploy-joint 保存 best。

必须写入 manifest：

```json
{
  "eval_splits": ["test"],
  "best_selection_split": "test",
  "best_selection_metric": "deploy_joint",
  "internal_test_selected": true,
  "publication_eligible": false
}
```

禁止：

```text
test per-label threshold search写回参数
test oracle用于best
测试标签更新emission/state/threshold
```

---

# 19. 日志与 artifact 合同

## 19.1 每100 optimizer updates

写一行 `loss_components.jsonl`：

```text
loss_total

loss_action_final
loss_action_base
loss_action_factor_aux
loss_action_soft_f1
loss_action_cardinality

loss_reason_formal
loss_reason_latent_aux
loss_reason_rank
loss_reason_soft_f1

loss_state
loss_emission
loss_emission_prior
loss_map_anchor
loss_state_anchor
loss_view_consistency
loss_unknown_prior
loss_route_sparsity

progress_lr
progress_grounding
progress_unknown
progress_emission
progress_reason_output
progress_action_reread
lambda_action_state

state_positive_mean
state_counter_mean
state_unknown_mean
state_unknown_p10/p50/p90

emission_Tplus_mean
emission_Tunknown_mean
emission_Tminus_mean
emission_order_margin_min

conflict_mean/std/p90
share_weight_mean/min/max
gamma_positive/counter/unknown
action_state_utility_entropy

evidence_null_mean
evidence_entropy_mean
evidence_snr_mean
evidence_temperature_mean
evidence_layer_weights

factor_selection_entropy
factor_effective_count
factor_named_abs_mean
unnamed_abs_mean
unnamed_fraction
contribution_reconstruction_error

action_source_logit_rms
action_base_logit_rms
action_final_logit_rms
action_residual_rms
reason_source_logit_rms
reason_latent_logit_rms
reason_formal_logit_rms

foundation_grad_raw/capped
evidence_grad
state_grad
emission_grad
action_reread_grad
DINO_grad

data_time
dino_time
foundation_time
evidence_time
latent_time
action_reread_time
backward_time
allocated_gb
reserved_gb
```

## 19.2 每epoch

同一 test forward输出：

```text
source action/reason
LENS base/final action
latent/formal reason
raw/deploy metrics
per-label AP/F1/AUC
```

固定前128个 test IDs在相同 encoded field上完成：

```text
factor-off
reread-off
unknown-off
emission-identity
map-shuffle
wrong-factor
selected deletion
equal-mass control
state positive↔counter
```

不得为这些branch重新运行DINO。

## 19.3 Train-audit synthetic flip

每epoch一次：

```text
5% reason flips
10% reason flips
```

只用于诊断，不更新模型。

输出：

```text
flip_detection_AUROC
conflict_clean/flip
unknown_clean/flip
shared-gradient-change_LENS
shared-gradient-change_raw_BCE
gradient_robustness_ratio
```

## 19.4 大 tensor 保存限制

全 test 保存：

```text
logits
labels
file names
state概率摘要
factor contribution摘要
```

只对固定128 audit subset保存完整：

```text
evidence maps
factor selection
patch contribution
deletion variants
```

不得保存全 test `[B,4,21,3600]`。

---

# 20. Runtime profiler 与最快配置

实现：

```text
python -m fate_oia.engine.profile_lens_oia
```

真实路径必须包括：

```text
真实DINO
360×640
3×3600 tokens
foundation scene predicate
adaptive evidence
latent posterior/emission
factor-local reread
BF16
每4 update paired-view摊销
```

比较：

```text
A: batch=6, accum=5, workers=4, factor_chunk=21
B: batch=6, accum=5, workers=8, factor_chunk=21
C: batch=6, accum=5, workers=4, factor_chunk=7
D: batch=5, accum=6, workers=8, factor_chunk=21
```

每项：

```text
20 warm-up microbatches
50 measured microbatches
```

选择：

```text
reserved memory <45GB
无OOM
含paired-view摊销后的samples/sec最高
data_time/step_time不过高
```

吞吐差小于3%时，选显存更低者。

默认优先：

```text
batch=6
accum=5
workers=8
factor_chunk=21
```

但必须由实际 profiler 确认。

---

# 21. 配置文件建议

```yaml
experiment:
  name: lens_oia_v1
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
  split_seed: 20260805
  train_main_fraction: 0.90
  train_audit_fraction: 0.05
  train_calib_fraction: 0.05

backbone:
  arch: vit_small
  patch_size: 8
  selected_layers: [3,7,11]
  pretrained_weights: ckp/reference/dino_deitsmall8_pretrain.pth
  checkpoint_key: teacher
  freeze_backbone: true
  no_grad_backbone: true

training:
  epochs: 14
  batch_size: 6
  gradient_accumulation_steps: 5
  precision: bf16
  optimizer: AdamW
  weight_decay: 0.05
  lr_foundation: 8.0e-5
  lr_evidence: 1.5e-4
  lr_state: 1.5e-4
  lr_action_reread: 1.5e-4
  lr_emission: 5.0e-5
  warmup_ratio: 0.05
  mechanism_ramp_ratio: 0.10
  action_state_ramp_ratio: 0.15
  min_lr_ratio: 0.10
  foundation_grad_cap: 0.25
  global_grad_clip: 1.0
  action_logit_norm_cap: 20.0

evidence:
  tau_min: 0.35
  tau_max: 2.00
  topk: 32
  region_bias_abs_max: 2.0
  factor_chunk_size: 21

latent:
  positive_share_floor: 0.25
  zero_share_floor: 0.05
  action_state_lambda_max: 0.50
  emission_label_delta_scale: 0.25

paired_view:
  interval_optimizer_updates: 4
  max_fraction: 0.50
  mirror_probability: 0.25

loss_weights:
  action_final: 1.00
  action_base: 0.35
  action_factor_aux: 0.15
  action_soft_f1: 0.03
  action_cardinality: 0.02

  reason_formal: 1.00
  reason_latent_aux: 0.35
  reason_rank: 0.08
  reason_soft_f1: 0.03

  latent_state: 0.20
  annotation_emission: 0.10
  emission_prior: 0.01

  map_anchor: 0.06
  state_anchor: 0.05
  view_consistency: 0.03
  unknown_prior: 0.02
  route_sparsity: 0.01

calibration:
  enabled: true
  source: train_calib
  group_shrinkage: true
  fit_temperature: true
  fit_thresholds: true
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

# 22. 实施任务顺序

## T01：锁定源与建立 worktree

完成第1、2节。

## T02：记录 source behavior

对真实图像保存：

```text
source field shapes
source action/reason logits
source label nodes/attention
source fusion gate
source state dict hashes
```

## T03：实现 `LENSCalAlignFoundation`

完成 source module映射和数值回归。

## T04：实现 schema 与 conservative builder

先输出 structured source inventory，再实现 target。

## T05：实现 adaptive evidence

完成shape、null、temperature、SNR、无dense token materialization测试。

## T06：实现 identifiable latent state

完成 progress0 exact、state sum、unknown ramp和state embedding测试。

## T07：实现 ordered emission

完成顺序、初始化、identity ramp和频率初始化。

## T08：实现 action reread

完成factor selection、full-field reread、state variants、exact contribution。

## T09：实现 responsibility engine

完成annotation likelihood、JS conflict、unknown discount、gamma和share gradient。

## T10：实现 LENS model

完成无target forward API和全部branch。

## T11：实现 losses与owner registry

确认每项只加入一次。

## T12：实现 split、mirror和paired view

完成全部左右 permutation测试。

## T13：实现 trainer

BF16、update-based ramp、single backward、完整日志。

## T14：实现 calibration/evaluator

每epoch只test一次；branch复用field；test不写回。

## T15：实现 artifacts/resume

恢复：

```text
model
optimizer
scheduler/update count
split manifest
emission initialization
RNG
EMA统计
calibration
```

## T16：实现 audit与Skill

执行配套审查文件。

## T17：实际 profiler

选择最快配置。

## T18：唯一4轮 pilot

不运行多个seed或多个结构版本。

## T19：pilot通过后full 14轮

push当前HEAD，核对remote SHA，再启动。

---

# 23. 必须新增测试

```text
tests/test_lens_source_head_contract.py
tests/test_lens_worktree_contract.py
tests/test_lens_forbidden_paths.py

tests/test_lens_foundation_equivalence.py
tests/test_lens_one_dino_call.py
tests/test_lens_dino_frozen.py

tests/test_lens_adaptive_evidence_shapes.py
tests/test_lens_adaptive_temperature.py
tests/test_lens_null_mass.py
tests/test_lens_no_dense_factor_tokens.py

tests/test_lens_identifiable_state.py
tests/test_lens_progress_zero_state.py
tests/test_lens_unknown_ramp.py
tests/test_lens_state_not_mean_collapsed.py

tests/test_lens_ordered_emission.py
tests/test_lens_emission_initialization.py
tests/test_lens_reason_progress_zero_equivalence.py

tests/test_lens_responsibility_normalization.py
tests/test_lens_conflict_moves_mass_to_unknown.py
tests/test_lens_share_gradient_floor.py
tests/test_lens_synthetic_flip_detection.py

tests/test_lens_action_base_equivalence.py
tests/test_lens_factor_reread_full_field.py
tests/test_lens_factor_selection_has_null.py
tests/test_lens_state_substitution.py
tests/test_lens_contribution_reconstruction.py
tests/test_lens_unnamed_contribution.py
tests/test_lens_direction_preserving_cap.py

tests/test_lens_conservative_grounding.py
tests/test_lens_no_green_from_light_presence.py
tests/test_lens_no_stop_from_sign_presence.py
tests/test_lens_no_lane_style_guess.py
tests/test_lens_test_forward_rgb_only.py

tests/test_lens_mirror_action_permutation.py
tests/test_lens_mirror_reason_permutation.py
tests/test_lens_mirror_map_equivariance.py
tests/test_lens_paired_view_one_dino.py

tests/test_lens_reason_to_action_firewall.py
tests/test_lens_action_to_emission_firewall.py
tests/test_lens_grounding_to_dino_firewall.py
tests/test_lens_owner_exact_cover.py
tests/test_lens_loss_terms_added_once.py

tests/test_lens_posthoc_calibration_no_model_mutation.py
tests/test_lens_test_not_used_for_calibration.py
tests/test_lens_same_forward_branch_metrics.py

tests/test_lens_runtime_memory_contract.py
tests/test_lens_resume_exact.py
tests/test_lens_artifact_hash_contract.py
tests/test_lens_pilot_gate_recomputation.py
tests/test_lens_supervisor_protocol.py
```

所有原 CalAlign 基础回归测试仍须通过。

---

# 24. 预训练实现审查

运行：

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
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_lens_oia_implementation `
  --config configs\fate_oia_train_360x640_lens_oia_v1.yaml `
  --output-dir .background_runs\lens_oia_v1_preflight `
  --device cuda
```

必须产生：

```text
LENS_IMPLEMENTATION_REVIEW.json
LENS_RUNTIME_PROFILE.json
```

绑定当前：

```text
git_head
config_hash
source_tree_hash
schema_hash
split_seed
```

---

# 25. 唯一 Pilot

## 25.1 配置

```text
train-main     4096
train-audit    1024
train-calib     512
test             512
epochs             4
seed        20260805
```

## 25.2 命令

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_lens_oia `
  --config configs\fate_oia_train_360x640_lens_oia_v1.yaml `
  --output-dir .background_runs\lens_oia_v1_pilot_<HEAD> `
  --run-kind pilot `
  --epochs 4 `
  --max-train-main-samples 4096 `
  --max-train-audit-samples 1024 `
  --max-train-calib-samples 512 `
  --max-test-samples 512 `
  --device cuda
```

## 25.3 Pilot Gate A：基础与运行

```text
Action progress0 error <1e-6
Reason progress0 error <1e-6
普通batch DINO calls=1
DINO grad=0
无cache/compression
无NaN/Inf/OOM
reserved<45GB
```

## 25.4 Gate B：状态与emission不坍缩

```text
21/21 labels满足T+>T?>T-
minimum ordered margin >0.02
mean ordered margin >0.10
不能21类全部unknown
不能21类全部positive
有anchor的positive state概率 > matched counter
```

## 25.5 Gate C：Synthetic flip识别

```text
5%/10% flips conflict AUROC >=0.70
flip unknown mass > clean unknown mass
LENS共享视觉梯度变化 <= raw BCE变化的70%
```

## 25.6 Gate D：Action route

至少连续两个epoch：

```text
final Act_mAP >= source/base Act_mAP -0.002
final Act_mF1 >= source/base Act_mF1 -0.003
```

并且至少一个epoch：

```text
final Act_mAP >= base Act_mAP +0.001
```

机制：

```text
factor-off降低或显著改变target action
4/4 action factor分布不完全相同
effective factor count在2–8
unnamed contribution非恒0
contribution reconstruction通过
```

## 25.7 Gate E：Exp不再坍缩

```text
formal Exp_mAP >= source Exp_mAP -0.005
formal Exp_mF1 >= source Exp_mF1 -0.010
latent branch loss和gradient非零
不存在更强但未被formal选择的private branch
support足够的reason不能大面积F1=0
```

## 25.8 Gate F：Faithfulness

固定 audit subset：

```text
selected deletion effect > equal-mass control
target factor effect > wrong factor
state positive↔counter改变对应Action
95% LCB方向正确
```

## 25.9 Gate G：Owner与日志

```text
reason annotation → action params严格0梯度
action → emission严格0梯度
所有core owner grad>0且parameter delta>0
全部必要日志字段存在
```

Pilot失败：

```text
不得自动降低标准
不得直接full train
将原始证据追加findings.md
```

---

# 26. Full Train

## 26.1 配置

```text
epochs       14
seed          20260805
batch/accum   profiler结果
precision     bf16
eval          test only
best          deploy_joint
```

## 26.2 启动前要求

当前HEAD必须匹配：

```text
LENS_IMPLEMENTATION_REVIEW.json
LENS_PILOT_PASS.json
LENS_FULL_TRAIN_READY.json
LENS_RUNTIME_PROFILE.json
```

任一模型、loss、trainer、config、schema修改都会使 pilot 失效。

## 26.3 命令

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_lens_oia_foreground `
  --config configs\fate_oia_train_360x640_lens_oia_v1.yaml `
  --output-dir E:\FATE_OIA_lens_oia_v1_full_<HEAD> `
  --run-kind full `
  --epochs 14 `
  --device cuda
```

## 26.4 只允许结构性停止

```text
NaN/Inf
OOM
DINO call>1
action logits runaway
owner firewall失败
显存持续增长
artifact/checkpoint损坏
dataloader确认卡死
```

不因 epoch0/1 指标低而自动停止。

---

# 27. Git 提交建议

```text
1. chore: create LENS-OIA V1 worktree contract
2. feat: add CalAlign-compatible LENS foundation
3. feat: add adaptive latent visual evidence
4. feat: add ordered annotation emission and latent responsibility
5. feat: add action-conditioned factor rereading
6. feat: add conservative structured evidence and mirror mapping
7. feat: add LENS losses and gradient ownership
8. feat: add LENS trainer evaluator calibration and artifacts
9. test: add LENS mechanism and protocol tests
10. docs: add LENS plan and audit skill
11. audit: record implementation closure
```

每个 commit 后 push。

---

# 28. 最终禁止事项

Codex 不得：

```text
为了省事直接继承ACPROIAModel并保留dead pair/threshold模块
复用当前错误WeakPredicateTargetBuilder
将reason=0全部当hard negative
将unknown状态压成常数或mean
让raw reason label直接更新action head/reread
让Action loss更新annotation emission
用utility predictor决定factor
用online CF teacher作为主loss
添加HardPair/pair memory
使用graph/PMI
使用历史RunC checkpoint/cached logits
使用VLM/MLLM
生成或读取feature cache
压缩3600 patch tokens
在test拟合threshold/emission
重新编码DINO以计算branch
只实现类定义但formal forward未调用
输出loss key但不进入total/backward
用placeholder zero冒充机制
未通过pilot直接full train
```

---

# 29. 完成标准

“模型能跑”不算完成。

必须证明：

```text
CalAlign source真正恢复
adaptive evidence真正读取3600 patches
三状态posterior真实变化
unknown不坍缩
ordered emission真实学习且有序
conflict真正改变shared gradient
Action使用latent clean state而非benchmark annotation
factor-local reread真实进入final action
named+unnamed贡献严格重构final residual
formal Exp就是annotation marginal
structured grounding保守且train-only
mirror语义重映射正确
每个loss/owner真实有梯度和更新
同一test forward提供所有诊断
pilot可以据日志判断各组件效果
```

唯一正式方案：

\[
\boxed{
\text{LENS-OIA V1}
=
\text{CalAlign-compatible foundation}
+
\text{single latent visual predicate state}
+
\text{conflict-discounted ordered annotation emission}
+
\text{action-conditioned full-field rereading}
}
\]
