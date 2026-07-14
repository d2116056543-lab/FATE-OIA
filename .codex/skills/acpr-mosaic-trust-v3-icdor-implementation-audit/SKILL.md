---
name: acpr-mosaic-trust-v3-icdor-implementation-audit
description: >-
  Fail-closed code, graph, data-boundary, mechanism, artifact, runtime and pilot audit for
  ACPR-MOSAIC-TRUST v3 / IC-DOR on BDD-OIA. Use this skill before profiling, pilot,
  full training, resume, evaluation, checkpoint selection, or GitHub completion claims.
version: 1.0.0
---

# ACPR-MOSAIC-TRUST v3 / IC-DOR
## Codex 配套实现审查 Skill

**目标分支：** `acpr_mosaic_trust_v3_icdor_direct_image`  
**源分支：** `acpr_mosaic_ad_v1_direct_image`  
**正式模型：** `MOSAICTrustICDORModel`  
**正式 trainer：** `fate_oia.engine.train_acpr_mosaic_trust_icdor`  
**正式配置：** `configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml`  
**审查策略：** fail closed；任何证据缺失、哈希漂移、路径含糊或数值不满足，都视为失败。

---

# 0. 使用目的与非目标

本 Skill 的任务不是证明模型一定超过强基线，而是确保：

1. 上一轮确定的 IC-DOR 计算图被完整实现；
2. 关键组件不是空壳、日志装饰或永远关闭的死分支；
3. factor、action、reason、observation、calibration 的参数所有权严格成立；
4. “因素可识别性证书”“factor-target admission”“双观测 reason”“target-owned support/veto re-reading”都真实进入训练和评估；
5. 测试集、BDD100K geometry、reason labels、oracle thresholds 不进入测试时正式 forward；
6. full train 前，单次短 pilot 已输出足够的诊断信息，可以定位后续失败发生在视觉因素、目标路由、reason 噪声建模、action 安全还是 calibration；
7. 审查所依据的代码、配置、证书、edge map、runtime profile 和 Git HEAD 完全一致。

本 Skill 不接受以下“完成”标准：

```text
模型可以 import
模型可以跑一个 batch
loss 是 finite
pytest 只检查 shape
日志里出现了组件名字
配置文件里声明了某项功能
某个类被实例化但 forward 不使用
某个 gate 永远接近零
某个分支只在 smoke-only 代码里存在
```

只有动态计算图、梯度、输出敏感性、数据来源、artifact 和哈希都通过，才能写出最终 `REVIEW_PASS`。

---

# 1. 安装位置与调用约定

将本文件安装为：

```text
.codex/skills/acpr-mosaic-trust-v3-icdor-implementation-audit/SKILL.md
```

正式审查命令统一为：

```powershell
python -u -m fate_oia.engine.audit_acpr_mosaic_trust_icdor `
  --config configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml `
  --worktree_root . `
  --output_dir .review `
  --device cuda `
  --fail_closed
```

Profiler、pilot、full trainer 都必须检查当前有效的审查文件：

```text
.review/acpr_mosaic_trust_v3_icdor_REVIEW_PASS.json
```

无该文件、文件过期、哈希不一致或任何 gate 不是 `PASS`，都必须拒绝启动。

---

# 2. 审查开始前的强制上下文读取

远程执行任何代码前，Codex 必须读取：

```powershell
Get-Content E:\sbw\FATE_Drive\task_plan.md
Get-Content E:\sbw\FATE_Drive\findings.md
Get-Content E:\sbw\FATE_Drive\progress.md
```

然后记录：

```text
上下文读取时间
三个文件的 SHA256
源 worktree 路径
目标 worktree 路径
源分支完整 SHA
目标分支完整 SHA
计划文件 SHA256
本 Skill SHA256
resolved config SHA256
```

写入：

```text
.review/icdor_audit_context.json
```

缺少任何上下文文件或哈希，审查直接失败。

训练/实验状态只追加到上述三个 canonical Markdown；不得创建新的远程训练状态 Markdown。实现计划和本 Skill 是用户明确要求的规范文件，不是训练日志。

---

# 3. 工作树隔离与 Git 合同

## 3.1 必须满足

```text
source worktree:
E:\sbw\FATE_Drive\fate_oia_acpr_mosaic_ad_v1_worktree

source branch:
acpr_mosaic_ad_v1_direct_image

target worktree:
E:\sbw\FATE_Drive\fate_oia_acpr_mosaic_trust_v3_icdor_worktree

target branch:
acpr_mosaic_trust_v3_icdor_direct_image
```

审查必须确认：

1. source worktree 存在；
2. target worktree 存在；
3. 两者路径不同；
4. target branch 名称准确；
5. source worktree 在创建 target 后没有被修改；
6. `.git` worktree 关系有效；
7. target 初始 parent 等于记录的 source HEAD；
8. `.background_runs/`、checkpoint、logits、visual audit images 未被 Git 跟踪；
9. target 工作树在写 `REVIEW_PASS` 时 clean；
10. 当前 target HEAD 已推送到 `origin/acpr_mosaic_trust_v3_icdor_direct_image`。

若 GitHub push 因 TLS/凭证失败，可以继续本地代码审查，但不得生成“GitHub sync PASS”，也不得生成最终 `REVIEW_PASS`。必须写出：

```text
.review/ICDOR_GATE_GITHUB_SYNC_FAIL.json
```

## 3.2 必须生成

```text
.review/icdor_source_manifest.json
.review/icdor_git_manifest.json
.review/icdor_changed_files.txt
.review/icdor_git_diff_stat.txt
.review/icdor_tracked_large_files.txt
```

最终 Git manifest 至少包含：

```json
{
  "source_branch": "acpr_mosaic_ad_v1_direct_image",
  "source_head": "<full sha>",
  "target_branch": "acpr_mosaic_trust_v3_icdor_direct_image",
  "target_head": "<full sha>",
  "origin_target_head": "<full sha>",
  "worktree_clean": true,
  "source_worktree_unchanged": true,
  "push_verified": true
}
```

---

# 4. 必需文件完整性

以下文件必须存在且进入正式 import/forward/train 路径。

## 4.1 配置

```text
configs/mosaic_icdor_factor_candidates.yaml
configs/mosaic_icdor_action_routes.yaml
configs/mosaic_icdor_reason_routes.yaml
configs/mosaic_icdor_certificate_rules.yaml
configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml
```

## 4.2 模型

```text
fate_oia/models/mosaic_low_rank_rezero_adapter.py
fate_oia/models/mosaic_factor_certificate.py
fate_oia/models/mosaic_target_sparse_router.py
fate_oia/models/mosaic_masked_target_rereader.py
fate_oia/models/mosaic_icdor_action_decoder.py
fate_oia/models/mosaic_icdor_dual_reason_decoder.py
fate_oia/models/mosaic_icdor_observation_head.py
fate_oia/models/acpr_mosaic_trust_icdor_model.py
```

## 4.3 修改后的正式依赖

```text
fate_oia/models/mosaic_geometry_typed_attention.py
fate_oia/models/mosaic_observable_predicates.py
fate_oia/models/mosaic_selective_observation.py
fate_oia/models/mosaic_native_semantics.py
fate_oia/models/__init__.py
```

## 4.4 Loss / optimizer

```text
fate_oia/losses/mosaic_icdor_factor_losses.py
fate_oia/losses/mosaic_icdor_action_losses.py
fate_oia/losses/mosaic_icdor_reason_losses.py
fate_oia/losses/mosaic_icdor_transport_losses.py
fate_oia/optim/mosaic_action_pareto_admission.py
```

## 4.5 Engine

```text
fate_oia/engine/mosaic_icdor_schedule.py
fate_oia/engine/build_mosaic_factor_certificate.py
fate_oia/engine/build_mosaic_edge_admission.py
fate_oia/engine/mosaic_target_transfer_metrics.py
fate_oia/engine/train_acpr_mosaic_trust_icdor.py
fate_oia/engine/eval_acpr_mosaic_trust_icdor.py
fate_oia/engine/profile_acpr_mosaic_trust_icdor.py
fate_oia/engine/audit_acpr_mosaic_trust_icdor.py
fate_oia/engine/export_mosaic_trust_visual_audit.py
fate_oia/engine/build_mosaic_trust_ablation_table.py
```

## 4.6 Dataset / launcher / tests

```text
fate_oia/datasets/mosaic_icdor_split.py
scripts/FATE_OIA_acpr_mosaic_trust_v3_icdor_foreground.ps1

 tests/test_icdor_split.py
 tests/test_icdor_three_lane_adapters.py
 tests/test_icdor_factor_certificate.py
 tests/test_icdor_independent_prototypes.py
 tests/test_icdor_target_router.py
 tests/test_icdor_target_rereader.py
 tests/test_icdor_action_decoder.py
 tests/test_icdor_dual_reason_decoder.py
 tests/test_icdor_selective_observation.py
 tests/test_icdor_action_pareto.py
 tests/test_icdor_gradient_ownership.py
 tests/test_icdor_schedule.py
 tests/test_icdor_artifact_schema.py
 tests/test_icdor_no_test_leakage.py
 tests/test_icdor_runtime_profile.py
 tests/test_icdor_full_forward.py
```

仅存在文件不足以通过。审查器必须建立“配置 → 构造函数 → forward → loss → optimizer → artifact”的调用图，证明每个文件的核心对象被正式路径使用。

---

# 5. 禁止模式扫描

审查器必须对 Python、YAML、PowerShell 文件做静态扫描，并对命中项进行人工语义复核。以下内容若出现在正式 IC-DOR 路径，原则上直接失败。

## 5.1 禁止旧正式模块

```text
MOSAICSupportVetoComposer
旧 MOSAICActionDecoder
旧 MOSAICReasonDecoder
```

允许这些 legacy 类继续留在仓库，但：

```text
MOSAICTrustICDORModel 不得实例化
IC-DOR trainer 不得 import 后用于正式 forward
正式 config 不得选择 legacy mode
```

## 5.2 禁止旧信息路径

```text
reason_adapter -> factor extractor
reason_adapter -> action forward
reason logits -> action logits
reason labels -> action forward
observed reason posterior q -> action forward
propensity pi -> action forward
factor probability/q/rho -> direct action logit bias
factor/state mean -> action logit additive shortcut
action-set marginalization -> final action
```

## 5.3 禁止自我认证

不得存在：

```text
image -> factor tier classifier
factor head -> strong/weak/unidentifiable prediction
tier logits trained jointly with factor prediction
learned factor reliability used as its own loss weight without detached external evidence
```

Factor tier 必须来自 train-audit 的 certificate artifact。

## 5.4 禁止不合法先验与泄漏

```text
PMI
label co-occurrence matrix as prediction bias
training-set action/reason graph as strong final prior
test labels in threshold/certificate/admission
BDD100K geometry in test forward
reason labels in test forward
action labels in test forward
test oracle threshold written into model
scene token claimed as object/lane/drivable grounding
```

## 5.5 禁止缓存/压缩/历史模型依赖

```text
feature_cache = true
token_compression != none
cached logits as model input
RunC residual
ACPR-CalAlign checkpoint as teacher
MOSAIC-AD checkpoint resume into formal IC-DOR full run
old best checkpoint distillation
```

只允许加载公开 DINO 预训练权重和当前 IC-DOR 自身 resume checkpoint。

## 5.6 禁止样本级外部文本/VLM 路径

```text
GPT/Qwen/VLM generated scene description as input
text encoder executed in test forward
sample-specific caption cache
VLM pseudo labels injected into final action
```

任务级 prompt embedding 可离线初始化语义 query，但测试时不得运行 text encoder。

## 5.7 禁止伪算法命名

若函数命名为：

```text
ASL
softF1
nnPU
Pareto admission
multi-prototype
hard semantic mask
```

其数学实现必须与名称一致。普通 BCE 不得命名为 ASL；普通 loss 加权不得命名为 Pareto；prototype 先平均后匹配不得命名为 multi-prototype routing。

---

# 6. 编译、导入和全测试 Gate

必须执行：

```powershell
python -m compileall fate_oia tests
python -m pytest tests -q
```

至少额外执行 IC-DOR 定向测试：

```powershell
python -m pytest `
  tests/test_icdor_split.py `
  tests/test_icdor_three_lane_adapters.py `
  tests/test_icdor_factor_certificate.py `
  tests/test_icdor_independent_prototypes.py `
  tests/test_icdor_target_router.py `
  tests/test_icdor_target_rereader.py `
  tests/test_icdor_action_decoder.py `
  tests/test_icdor_dual_reason_decoder.py `
  tests/test_icdor_selective_observation.py `
  tests/test_icdor_action_pareto.py `
  tests/test_icdor_gradient_ownership.py `
  tests/test_icdor_schedule.py `
  tests/test_icdor_artifact_schema.py `
  tests/test_icdor_no_test_leakage.py `
  tests/test_icdor_runtime_profile.py `
  tests/test_icdor_full_forward.py -q
```

要求：

- 0 failed；
- 0 xfailed，除非在本 Skill 中预先登记且不是核心功能；
- 0 skipped 核心测试；
- 无 NaN/Inf warning；
- 无未初始化参数 warning；
- 无 silent fallback 到 legacy model；
- 无 CPU-only 替代 CUDA 路径冒充动态测试。

写入：

```text
.review/ICDOR_GATE_COMPILE_TEST_PASS.json
```

---

# 7. 数据 split 与无泄漏 Gate

## 7.1 train 内部三分法

`mosaic_icdor_split.py` 必须从原始 BDD-OIA train 构造：

```text
train_core
train_audit
train_calib
```

必须满足：

```text
train_core ∩ train_audit = ∅
train_core ∩ train_calib = ∅
train_audit ∩ train_calib = ∅
union == original train IDs
```

分配由固定 seed、文件名 hash 和多标签分布修正决定，不能每次随机漂移。

## 7.2 test 使用边界

Test 只能用于用户指定的每 epoch final evaluation 和 best selection，不得用于：

```text
factor certificate
factor tier
factor reliability
edge admission
route threshold
loss weight
calibration fitting
hyperparameter selection inside trainer
```

测试扰动：将 test labels 随机重排后，以下对象必须 bitwise 不变：

```text
model raw logits
factor certificate
edge admission map
train-calib thresholds
checkpoint weights
```

## 7.3 BDD100K geometry 边界

BDD100K box/polyline/drivable 数据只允许：

```text
train_core factor weak supervision
train_audit certificate evidence
visual audit
```

Test forward 在有/无 geometry metadata 时 logits 必须精确一致，最大绝对差：

```text
<= 1e-7 in fp32 audit
```

生成：

```text
.review/ICDOR_GATE_SPLIT_NO_LEAKAGE_PASS.json
```

---

# 8. 三通道 adapter 与参数所有权 Gate

## 8.1 对象独立性

正式模型必须有：

```python
self.factor_adapter
self.action_adapter
self.reason_adapter
```

三者必须：

- 是不同 Python 对象；
- 参数 storage 不共享；
- 参数名不重叠；
- 均为低秩 ReZero residual adapter；
- 包含 rank bottleneck、depthwise 3×3 local path、identity skip；
- ReZero scalar 初始接近 0，且有上界。

禁止使用三个完整 384×384 identity 1×1 convolution 冒充低秩 adapter。

## 8.2 Forward 可达性

动态 autograd/Jacobian probe 必须满足：


action logits：

```text
d(action_visual_logits) / d(action_adapter) != 0
d(action_visual_logits) / d(reason_adapter) == 0
d(action_visual_logits) / d(observation_head) == 0
```

reason logits：

```text
d(reason_visual_observed_logits) / d(reason_adapter) != 0
d(reason_visual_observed_logits) / d(action_adapter) == 0
```

factor outputs：

```text
d(factor_outputs) / d(factor_adapter) != 0
d(factor_outputs) / d(reason_adapter) == 0
```

## 8.3 Loss 梯度所有权

单独反传每个 loss，必须满足：

```text
L_action_base -> action adapter/action visual decoder: nonzero
L_action_base -> reason adapter/reason decoder: exactly zero
L_reason_obs  -> action adapter/action decoder: exactly zero
L_reason_latent -> action adapter/action decoder: exactly zero
L_reason_obs -> factor adapter/extractor: exactly zero
L_factor -> factor adapter/extractor: nonzero
L_factor -> action adapter/action decoder: exactly zero
L_action_route -> base action decoder: exactly zero
L_action_route -> router/rereader/gate: nonzero
L_observation -> observation head: nonzero
L_observation -> action adapter: exactly zero
```

“Exactly zero”审查在 fp32 单 batch下要求：

```text
sum_abs_grad <= 1e-12
```

非零路径要求：

```text
sum_abs_grad >= 1e-8
```

## 8.4 Optimizer group 唯一性

每个 parameter id 必须：

- 只属于一个 optimizer group；
- 不可同时属于 shared/action/reason/factor groups；
- frozen 参数不得进入 optimizer；
- factor certificate 冻结后 factor adapter/extractor/prototype 不再更新。

生成：

```text
.review/ICDOR_GATE_THREE_LANE_OWNERSHIP_PASS.json
.review/icdor_parameter_ownership.json
```

---

# 9. Direct-image、DINO、无 cache/压缩 Gate

正式模型输入必须为：

```text
images: [B, 3, 360, 640]
```

不能假装使用 temporal clip。

每个正式 batch：

- DINO 在当前 image batch 上真实执行；
- DINO 参数冻结；
- layer 3/7/11 token 真实提取；
- 不读取 feature cache；
- 不读取 token cache；
- token compression 为 `none`；
- 不减少 360×640 对应的正式 token 序列后再声称 full-token direct-image。

动态检查：随机改变输入图像，DINO tokens 与最终 logits 必须变化；随机改变一个未被模型读取的 cache 文件，输出不得变化。

生成：

```text
.review/ICDOR_GATE_DIRECT_IMAGE_DINO_PASS.json
```

---

# 10. Geometry-typed factor attention Gate

## 10.1 三类路径都必须真实执行

正式 factor ontology 至少包含：

```text
point/object
curve/lane
region/corridor
```

针对每类至少选 2 个 factor 做动态 forward/backward：

- sampler 被调用；
- 采样坐标有限且在合法范围；
- 输出不恒定；
- 对应参数梯度非零；
- 不回退到统一全局 mean pooling。

## 10.2 Curve path 强度

Curve/lane path 必须包含有序采样和小型 1D Transformer/sequence encoder：

```text
8–12 ordered curve points
arc-length positional encoding
at least one self-attention block
```

随机打乱 curve token 顺序时 curve factor 表示应改变；point factor 不应受该顺序测试影响。

## 10.3 Independent prototypes

每个 multi-prototype factor 必须先独立计算 token–prototype score：

```text
[B, F, K, N] 或等价的 memory-efficient分块计算
```

允许分块、gather、einsum，但不得构造：

```text
[B, F, N, D]
```

禁止：

```python
query = prototypes.mean(dim=prototype_dim)
# 然后再进行唯一一次视觉匹配
```

动态测试：只扰动 prototype k，必须改变该 prototype 的 assignment/score，而非只改变平均 query。

## 10.4 Prototype utilization

输出：

```text
prototype occupancy
prototype effective count
dominant rate
dead count
pairwise cosine
```

短 pilot 前基本 gate：

```text
dominant rate < 0.95 on smoke
no NaN occupancy
effective count > 1.0 for multi-prototype factors
```

正式 certificate gate 使用更严格阈值。

## 10.5 空间 prior 不得支配

必须实现三种模式：

```text
full
content_only
prior_only
```

空间 prior scale：

```text
init <= 0.05
hard max <= 0.20
prior dropout = 0.50 ± tolerance
```

审查器应比较 score 量级，确保视觉 content logit 不是比 prior 小一个数量级。

生成：

```text
.review/ICDOR_GATE_TYPED_FACTOR_ATTENTION_PASS.json
```

---

# 11. Presence / observability 语义 Gate

每个 factor 输出至少包括：

```text
factor_presence_prob p
factor_observability_prob v
factor_present_evidence = v * p
factor_absent_evidence = v * (1 - p)
factor_uncertainty
factor_mask
factor_feature
prototype_assignment
```

必须通过以下单元测试：

## 11.1 Unknown

当：

```text
v = 0
```

则：

```text
present_evidence = 0
absent_evidence = 0
```

不能把不可观察解释为 negative。

## 11.2 Visible-present

```text
v = 1, p = 1
present_evidence = 1
absent_evidence = 0
```

## 11.3 Visible-absent

```text
v = 1, p = 0
present_evidence = 0
absent_evidence = 1
```

## 11.4 Geometry missingness

- geometry source缺失：unknown；
- source完整且匹配：positive；
- source完整且未匹配：只有直接可观察、attribute-free factor可获得低可靠 weak negative；
- 不支持属性：unknown；
- zero reason label不得自动成为 factor negative。

## 11.5 Collapse gate

短 pilot 中：

```text
presence >0.95 且 visibility >0.95 的 factor 比例 <=20%
所有 factor presence/visibility std 有效
unknown rate 不得全 0
weak negative rate 不得全 0 或异常接近 1
```

生成：

```text
.review/ICDOR_GATE_PRESENCE_OBSERVABILITY_PASS.json
```

---

# 12. Factor Identifiability Certificate Gate

## 12.1 证书不是模型输出

审查必须确认：

- model.forward 不输出 tier logits；
- tier 不由当前 batch预测；
- tier 构建脚本只读 train_audit；
- certificate 统计和模型训练 loss 之间没有反向传播；
- certificate 生成后写磁盘并冻结为 buffer；
- resume 时 hash 必须一致。

## 12.2 证书字段

`factor_certificate.json` 每个 factor 必须包含：

```text
confirmed_positive
reliable_negative
weak_negative
unknown
geometry_valid
full
content_only
prior_only
query_shuffle_drop
image_shuffle_drop
grounding_minus_random
view_consistency
mirror_consistency
ece
effective_prototype_count
dominant_prototype_rate
dead_count
bootstrap_lcb95 for every thresholded statistic
tier
reasons
```

## 12.3 Tier 规则必须与配置一致

Certified：

```text
(pos >=32 and reliable_neg >=32) or geometry_valid >=200
LCB95(full-prior) >0.02
content_only >=0.70*full
LCB95(query_shuffle_drop) >0.01
LCB95(image_shuffle_drop) >0.01
geometry available -> LCB95(grounding-random) >0.02
or no geometry -> view/mirror consistency gate
effective prototype count >1.5
dominant rate <0.85
presence and visibility nonzero variance
```

Reason-only：有稳定视觉 signal，但缺乏负例、grounding或 target utility。

Abstained：全 unknown、prior shortcut、shuffle 无下降、无独立锚点、prototype collapse或单帧不可识别。

## 12.4 权限测试

```text
Certified -> reason candidate + action shadow candidate
Reason-only -> reason candidate only
Abstained -> no explicit action/reason factor edge
```

Abstained factor 不得偷偷通过 dense semantic pool、fallback mean或 unrestricted cross-attention进入正式 target。

## 12.5 生命周期

- epoch 0–4 前 certificate 不存在：action route final必须 off；
- epoch 4 结束：构建 certificate；
- certificate 生成后 factor adapter/extractor/prototype冻结；
- certificate hash进入 checkpoint和 artifact；
- 更改 certificate 文件后 resume 必须拒绝。

生成：

```text
.review/ICDOR_GATE_FACTOR_CERTIFICATE_PASS.json
```

---

# 13. Target ontology、极性和硬可达性 Gate

每条候选边必须明确四元信息：

```text
factor
polarity = present | absent
direction = support | veto
target = action | reason
```

必须区分：

```text
factor present/absent
```

与：

```text
target support/veto
```

例如：

```text
front obstacle present -> forward veto
front obstacle absent  -> forward support
red light present      -> stop support
left solid boundary present -> left veto
```

不允许 factor probability 的正负直接等同于 action direction。

配置审查必须确认：

- 不允许边被 hard mask；
- action/reason 路由只看各自 allow set；
- 无 PMI/co-occurrence 权重；
- 规则只定义候选和符号，不固定实例 prediction；
- left/right 镜像关系一致；
- action/reason label ID 与名称 mapping 一致。

动态 hard-mask 检查：对 disallowed factor 施加任意大 feature/logit，target router mass仍必须精确为 0。

生成：

```text
.review/ICDOR_GATE_ONTOLOGY_POLARITY_PASS.json
```

---

# 14. Sparse router + dustbin Gate

`MOSAICTargetSparseRouter` 必须：

1. 对 support/veto 分开路由；
2. 对每个 target 使用自己的 allow set；
3. 输入 factor feature、present/absent evidence、frozen admission mask；
4. 使用 sparsemax/entmax或等价可稀疏归一化；
5. 包含 dustbin；
6. disallowed mass 精确为 0；
7. 无可靠 factor 时允许 dustbin占主导；
8. 有高质量 factor 时 dustbin不是强制 1；
9. 输出逐边 routing mass、entropy、dustbin mass。

动态测试：

- 所有 factor evidence 置零：dustbin mass 应显著上升；
- 一个 allowed factor evidence 提高：该 edge mass 应提高；
- disallowed factor evidence 提高：target mass不变；
- factor certificate从 Certified改为 Reason-only：action router该 edge变 0；
- factor certificate改为 Abstained：action/reason显式 edge均为 0。

短 pilot gate：

```text
dustbin rate 不得全 0
dustbin rate 不得全 1
router entropy有限
至少部分 admitted candidate 有非零 shadow mass
```

生成：

```text
.review/ICDOR_GATE_TARGET_ROUTER_PASS.json
```

---

# 15. Masked target visual re-reading Gate

Factor 只能定义目标重新读取的位置/极性，不能直接写 final action logit。

## 15.1 正式输入

Action re-reader 必须读取：

```text
action_pyramid
support/veto target query
factor-derived soft masks
```

不得读取：

```text
reason adapter tokens
reason logits
reason labels
propensity/posterior
```

Reason re-reader 必须读取 reason_pyramid，而不是 action_pyramid。

## 15.2 Mask 真实生效

动态 probe：

- 原 mask；
- all-zero mask；
- all-one mask；
- shuffled factor mask；
- wrong-target mask；
- equal-area random mask。

re-reader node/logit必须产生可解释差异。若所有差异 `<1e-6`，判定为死模块。

## 15.3 Support / veto 符号

Support evidence：

```text
delta_support >= 0
```

Veto evidence：

```text
delta_veto >= 0
final = visual + support - veto
```

必须由 softplus或等价单调非负参数化保证，不靠 loss 软约束。

## 15.4 Route 强度

记录：

```text
RMS(route_delta) / RMS(visual_logits)
per-action ratio
p50/p95 route delta
support RMS
veto RMS
```

短 pilot 预期：

```text
shadow阶段 >0.005，避免死路
final admission阶段目标 0.02–0.15
绝不 >0.25
```

生成：

```text
.review/ICDOR_GATE_TARGET_REREADER_PASS.json
```

---

# 16. 强 action visual decoder Gate

`MOSAICICDORActionDecoder` 的 visual-only 路径必须独立可训练、可评估，不依赖 factor。

最低结构：

```text
4 action queries
2 layers category-specific sparse cross-attention
high/mid/context feature access
1 layer action-query self-attention
per-action logits
```

动态测试：

- factor outputs全部清零，visual logits仍非恒定；
- visual decoder对不同图像有区分；
- action query互换后 per-action输出互换/改变；
- visual-only branch能单独计算 ASL、cross-sample rank和cardinality loss；
- factor/reason loss不更新 visual decoder。

必须输出：

```text
action_visual_nodes
action_visual_attention
action_visual_logits
```

不得用旧 checkpoint、reason logits或 action-set final增强 visual-only branch。

生成：

```text
.review/ICDOR_GATE_ACTION_VISUAL_DECODER_PASS.json
```

---

# 17. Shadow route 与 factor-target edge admission Gate

## 17.1 Shadow 不得修改 final action

在 edge admission 完成前：

```text
action_final_logits == action_visual_logits exactly
action_shadow_logits == action_visual_logits + route_delta
```

fp32 审查：

```text
max_abs(final - visual) <=1e-7
RMS(shadow - visual) >0 when candidate factors active
```

## 17.2 Intervention 设计

每条 candidate edge 至少需要：

```text
64 valid samples
1 selected mask
8 disjoint matched-random masks
same layer distribution
same region/type
same token count
same mask area/compactness
same imputation function
selected ∩ random = ∅
```

必须保存 all-target effects，而非只保存目标 target。

## 17.3 指标

逐 edge 计算：

```text
SignedEffect
TET
TES
CCA
isolated AP delta
TP->FN
FP->TP
valid sample count
bootstrap 95% CI
```

## 17.4 Admission 条件

Action edge 只有同时满足：

```text
factor tier == Certified
valid samples >=64
LCB95(SignedEffect) >0
LCB95(TET) >0
LCB95(TES) >0
CCA >=0.60
isolated AP_t >= visual AP_t -0.002
```

Reason edge必须符合 reason allow map和 factor tier；Reason-only factor不能进入 action。

## 17.5 冻结与 hash

生成：

```text
edge_admission.json
edge_admission_sha256.txt
```

epoch 6后冻结；checkpoint/resume必须校验 hash。

若无 action edge通过：

```text
action final永久 visual-only
trainer可继续
必须记录 no_admitted_action_edges=true
不得自动放宽阈值
```

生成：

```text
.review/ICDOR_GATE_EDGE_ADMISSION_PASS.json
```

---

# 18. Action Pareto Admission Gate

必须实现 per-action output-level constraint，而不是仅记录 gradient cosine。

对 action t：

```text
c_t = L_route_t - L_visual_t - tolerance_t
c_t <=0
```

Dual update：

```text
mu_t = clamp(mu_t + dual_lr*c_t, min=0, max=dual_max)
```

审查：

- `mu_t` 为 4 维独立 buffer；
- checkpoint保存和恢复；
- 违反约束时 `mu_t` 上升；
- 满足约束时不应无界增长；
- route loss使用 detached visual base；
- base action decoder不接收 route loss梯度；
- per-action violation、dual、loss差都写 artifact。

动态 synthetic test：构造一个明确有害 route，必须触发正 constraint和 dual增长；构造有益 route，不应错误惩罚。

Pilot gate：

```text
violation rate <=0.20 for smoke
full eligibility target <=0.05
dual 不得 >20% steps贴住上界
```

生成：

```text
.review/ICDOR_GATE_ACTION_PARETO_PASS.json
```

---

# 19. Dual-observation reason decoder Gate

正式 reason 不能只输出一个 logit。至少必须区分：

```text
reason_visual_observed_logits
reason_latent_logits
reason_observation_model_prob
reason_observed_logits / prob
reason_deploy_logits
```

## 19.1 Direct observed visual path

- 21 reason queries；
- high/mid/context sparse visual decoding；
- reason-label self-attention；
- 直接优化 observed benchmark labels；
- 不依赖 factor也能工作；
- 不更新 factor/action参数。

这是吸收 PMT-S/强 explanation 路线的主通道，不能被 semantic-only分支替代。

## 19.2 Latent semantic reason path

- 每个 reason只读取 hard-allowed factors；
- disallowed semantic attention logit = `-inf`；
- disallowed mass = 0；
- factor-guided mask引导 reason_pyramid重新读取视觉；
- 可含一个低容量 escape token；
- escape usage有惩罚并被记录；
- Abstained factor不得显式进入。

## 19.3 Observation model

低容量 observation head只能读取 detached：

```text
factor observability
factor uncertainty
reason group/frequency metadata
action_visual_logits.detach() if enabled by config
```

不得读取：

```text
full DINO embedding
reason latent hidden with gradient
reason labels at test
factor raw trainable feature with gradient
action decoder parameters with gradient
```

## 19.4 官方 observed 输出

官方 Exp metrics必须使用 observed output，而不是 latent output。

实现必须明确：

```text
p_latent = sigmoid(z_latent)
p_annotation = pi*p_latent + epsilon*(1-p_latent)
p_direct = sigmoid(z_visual_observed)
p_final_observed = bounded_mixer(p_direct, p_annotation)
```

若采用 logit mixer，必须数值稳定并保留上述三种独立诊断。

## 19.5 双分支非退化

动态检查：

- visual-observed path off：输出明显变化；
- latent path off：输出明显变化；
- observation model off：输出明显变化但不应破坏 action；
- factor mask shuffle：latent reason与部分 final observed输出下降；
- reason labels shuffle不改变 test forward logits。

生成：

```text
.review/ICDOR_GATE_DUAL_REASON_PASS.json
```

---

# 20. Selective observation 与 posterior Gate

## 20.1 数学公式

对 reason r：

```text
p* = sigmoid(z_latent)
pi = P(Y_obs=1 | R*=1, restricted detached features)
epsilon = P(Y_obs=1 | R*=0), bounded small
p_obs_model = pi*p* + epsilon*(1-p*)
```

Observed zero posterior：

```text
q = p*(1-pi) / [p*(1-pi) + (1-p)*(1-epsilon)]
```

Observed positive：

```text
q = 1
```

必须有数值稳定 epsilon clamp。

## 20.2 单元数值检查

对手算输入比较，绝对误差：

```text
<=1e-6 fp32
```

## 20.3 Stop-gradient

```text
q detach before posterior ranking
pi inputs detach
observation loss does not update action/factor lane
```

## 20.4 Propensity 约束

- group-shared/低容量；
- bounded；
- 不得任意读取高容量图像 embedding；
- 记录 min/max/mean/std和 boundary saturation。

## 20.5 Synthetic hidden-positive recovery

从 train_audit 已知 positives中遮蔽：

```text
10%
30%
50%
```

比较 zero-as-negative baseline，至少写：

```text
AUPRC
ECE
recall@high-q
per-reason/group results
```

短 pilot gate：

```text
recovery AUPRC finite
posterior not all 0/1
```

full eligibility目标：

```text
AUPRC >= zero-as-negative +0.02
```

生成：

```text
.review/ICDOR_GATE_SELECTIVE_OBSERVATION_PASS.json
```

---

# 21. Posterior-weighted ranking queue Gate

必须替代旧 HardPair：

```text
reason weight = q_i * (1-q_j)
action pair uses true action labels
```

要求：

- queue 存 detached logits/posteriors/labels/sample IDs；
- optimizer step后更新；
- 同一 sample ID不得配对；
- observed-zero中间 q自然低权重；
- 无 pair时 loss为 graph-connected zero；
- 不因 `reason=0` 自动当 hard negative；
- queue coverage、有效 pair数、per-label pair数写 artifact。

动态检查：

- q=1与q=0 pair权重大；
- q≈0.5 pair权重较小；
- q detach；
- queue内容不保留旧 autograd graph；
- no NaN when a label has no valid pair。

生成：

```text
.review/ICDOR_GATE_POSTERIOR_RANKING_PASS.json
```

---

# 22. Loss 实现正确性 Gate

## 22.1 Action ASL

若命名为 ASL，必须具有：

```text
asymmetric positive/negative focusing
negative probability clipping if configured
masked reduction
per-action diagnostics
```

普通 BCE不得命名为 ASL。

## 22.2 Action rank

必须是同一 action label跨样本排序，不是同一图像内4 actions相互排序。

## 22.3 Reason observed loss

直接 observed visual path必须获得明确监督，不能只训练 observation model。

## 22.4 Latent / observation losses

- observation NLL使用 observed labels；
- latent/posterior consistency使用 detached q；
- target route consistency只作用于 allowed factors；
- escape penalty真实非零且可关闭消融。

## 22.5 Factor loss

- per-factor class balance；
- positive/reliable/weak-negative/unknown分开；
- unknown masked；
- geometry/view/flip/prototype损失各自可记录；
- 不允许一项总 factor loss掩盖 component。

## 22.6 Calibration soft-F1

若配置称 soft-F1，必须真正计算 differentiable per-label/macro F1 surrogate；不能用 BCE冒充。

生成：

```text
.review/ICDOR_GATE_LOSS_CORRECTNESS_PASS.json
```

---

# 23. Calibration Gate

正式 calibration：

```text
representation frozen
train_calib only
group threshold + bounded per-label delta
deploy_logits = raw_logits - theta
```

必须区分：

```text
raw
threshold_off
deploy_fixed
train_calib_teacher_diagnostic
test_oracle_diagnostic
```

Test oracle：

- 只在评估器中计算；
- 不写入 checkpoint；
- 不更新 threshold head；
- 不参与 best threshold fitting；
- 不参与 factor certificate/edge admission。

Threshold扰动测试：随机打乱 test labels，learned theta必须完全不变。

配置要求：

```text
feature cache false
token compression none
calibration source train_calib
```

生成：

```text
.review/ICDOR_GATE_CALIBRATION_PASS.json
```

---

# 24. Canonical schedule Gate

所有阶段切换必须来自一个 canonical schedule：

```text
fate_oia/engine/mosaic_icdor_schedule.py
```

禁止 trainer、model、loss文件中散落相互矛盾的 `if epoch >=...`。

必须覆盖：

```text
epoch 0–2  visual foundation
epoch 3–4  factor certification
epoch 5–6  dual reason + shadow action
epoch 7–8  safe action routing
epoch 9–10 joint ranking
epoch 11   consolidation
post pass  train-calib calibration
```

Schedule 输出必须包含：

```text
factor_trainable
action_route_shadow
action_route_final
reason_semantic_enabled
observation_enabled
posterior_ranking_enabled
certificate_required
edge_admission_required
factor_frozen
propensity_frozen
route_gate_cap
loss weights
```

测试每个边界 epoch，并确保 config 声明与实际 schedule 一致。

生成：

```text
.review/ICDOR_GATE_SCHEDULE_PASS.json
```

---

# 25. Artifact 完整性 Gate

用户要求后续不再为判断中间机制单独重跑，因此训练必须一次性保存充足信息。

## 25.1 根目录文件

```text
run_manifest.json
resolved_config.yaml
source_manifest.json
split_manifest.json
runtime_selection.json
factor_certificate.json
edge_admission.json
checkpoint_latest.pth
checkpoint_best_test_joint.pth
metrics_summary.jsonl
```

## 25.2 每 epoch

```text
epoch_XXX/metrics_summary.json
epoch_XXX/loss_components.jsonl
epoch_XXX/branch_metrics.json
epoch_XXX/per_label_metrics.json
epoch_XXX/factor_stats.jsonl
epoch_XXX/factor_certificate_snapshot.json
epoch_XXX/prototype_stats.jsonl
epoch_XXX/action_route_stats.jsonl
epoch_XXX/reason_dual_observation_stats.jsonl
epoch_XXX/target_transfer_stats.jsonl
epoch_XXX/target_transfer_summary.json
epoch_XXX/pareto_stats.jsonl
epoch_XXX/gradient_ownership.jsonl
epoch_XXX/calibration_stats.jsonl
epoch_XXX/runtime_stats.jsonl
epoch_XXX/failure_cases.jsonl
epoch_XXX/visual_audit_manifest.json
epoch_XXX/logits/
```

## 25.3 必须保存的 logits

```text
action_visual_logits.pt
action_shadow_logits.pt
action_final_logits.pt
action_deploy_logits.pt
reason_visual_observed_logits.pt
reason_latent_logits.pt
reason_observation_model_prob.pt
reason_observed_logits.pt
reason_deploy_logits.pt
action_labels.pt
reason_labels.pt
file_names.json
```

## 25.4 Branch metrics

Action：

```text
visual
shadow
final
support-only
veto-only
threshold-off
deploy
oracle diagnostic
per-action AP/AUC/F1
flip counts
```

Reason：

```text
visual-observed
latent-semantic
observation-model
final-observed
factor-route-off
factor-route-shuffled
threshold-off
deploy
oracle diagnostic
per-reason AP/F1
common/tail
```

## 25.5 Factor / route / PU / gradient

必须逐 factor、逐 edge、逐 reason、逐参数组保存，不能只给全局均值。

Artifact schema测试必须：

- 真实运行一个 epoch；
- 读取所有文件；
- 校验字段、shape、dtype、finite；
- 校验 sample count与 split；
- 校验 logits/labels/file names顺序一致；
- 校验 branch关系，例如 shadow = visual + route delta；
- 校验 deploy = raw - theta。

生成：

```text
.review/ICDOR_GATE_ARTIFACT_SCHEMA_PASS.json
```

---

# 26. 训练时自动诊断与科学停止 Gate

Trainer 必须在每个 epoch检查：

1. NaN/Inf；
2. factor all-on collapse；
3. certificate/hash；
4. edge admission数据来源；
5. gradient ownership；
6. route strength；
7. per-action AP non-regression；
8. Pareto violation；
9. disallowed reason semantic mass；
10. factor-shuffle sensitivity；
11. propensity saturation；
12. calibration leakage；
13. cache/compression；
14. DINO真实执行。

至少以下条件必须 fail closed：

```text
>20% factors p>0.95 and v>0.95
route strength two epochs <0.005
route strength >0.25
any action AP drop >0.01 vs visual branch
Pareto violation rate >0.20
disallowed reason semantic mass !=0
factor shuffle does not alter latent route when factors are active
>20% propensity values stuck at bounds
certificate/admission hash mismatch
```

停止时必须写：

```text
STOP_REASON.json
last_valid_checkpoint.pth
failure_snapshot/
```

不得只打印日志后继续训练。

生成：

```text
.review/ICDOR_GATE_SCIENTIFIC_STOP_PASS.json
```

---

# 27. Runtime profiler Gate

Profiler 必须真实执行完整 Phase-D/route forward-backward，而不是只 profile visual foundation。

候选：

```text
batch 8 / accum 4
batch 6 / accum 5
batch 5 / accum 6
batch 4 / accum 8
workers 4 / 2 / 0
```

每个候选：

```text
20 warmup
100 measured steps
bf16
direct image 360x640
DINO current batch forward
factor extraction
reason dual branch
action shadow/final route
backward + optimizer
```

记录：

```text
samples/s
step p50/p95
GPU allocated/reserved peak
CPU/data wait
NaN/OOM/stall/retry
```

选择：

1. reserved ≤43.5 GB；
2. 100 steps无故障；
3. samples/s最高；
4. 并列选更大 batch；
5. Windows worker异常退 2/0。

Runtime selection文件必须包含 current target HEAD、config hash、phase和 profiler code hash。任何代码变更后 profile失效。

生成：

```text
.review/ICDOR_GATE_RUNTIME_PROFILE_PASS.json
.review/mosaic_trust_icdor_runtime_selection.json
```

---

# 28. 单次 4-epoch pilot Gate

用户要求不做多 seed长 pilot。本 Skill只要求一次短 pilot，但必须覆盖全部关键阶段。

数据：

```text
train_core 2048
train_audit 512
train_calib 512
test 512
seed 20260713
```

Pilot epoch：

```text
0 visual foundation
1 forced factor certification
2 dual reason + shadow action
3 edge admission + safe route
```

## 28.1 只判断机制，不用 tiny F1证明性能

必须检查：

### 数值

```text
all losses finite
all logits finite
no NaN/Inf/OOM/stall
```

### Factor

```text
no all-on collapse
certificate generated
at least one Certified or Reason-only factor if data supports
all-unknown factors Abstained
prototype diagnostics finite
full/content/prior/shuffle fields complete
```

### Action

```text
visual branch nonconstant
shadow route nonzero
before admission final==visual
post admission route only uses admitted edges
route strength finite and bounded
support/veto signs correct
Pareto stats active
```

### Reason

```text
visual-observed nonconstant
latent nonconstant
observation model nonconstant
final observed nonconstant
factor-route shuffle effect logged
hard mask disallowed mass ==0
```

### Gradient ownership

全部 expected-zero/nonzero条件逐阶段通过。

### Artifacts

4个 epoch 的全部 artifact 完整。

Pilot 可以在没有 action edge通过时通过“实现审查”，前提是系统正确保持 visual-only final并明确记录；但此时 full train的论文 action-route主张不具备机制证据，必须在 `REVIEW_PASS` 中标注：

```text
action_route_admission_status = no_edges_admitted
```

若用户仍要求 full run，trainer可以运行，但不得声称 action route已被验证有效。

生成：

```text
.review/ICDOR_GATE_SHORT_PILOT_PASS.json
.review/icdor_short_pilot_summary.json
```

---

# 29. Test-only evaluation 与 best selection Gate

按用户本轮工程要求：

```text
每轮只评估 test
best split = test
best metric = deploy_fixed_joint
```

审查必须确认：

- official val 未被 trainer每轮评估；
- representation结束后在 train_calib拟合 threshold；
- test forward只输入图像；
- test oracle仅诊断；
- checkpoint_best_test_joint根据 deploy_fixed_joint更新；
- action/explanation/joint同一 checkpoint指标保存；
- raw mAP与deploy F1同时保存。

同时在 manifest 中写明：

```text
engineering_protocol_test_selected = true
paper_protocol_unbiased = false
```

不得把该协议误写成无偏论文测试。

生成：

```text
.review/ICDOR_GATE_EVALUATION_PROTOCOL_PASS.json
```

---

# 30. Full train 启动前最终审查

只有以下 gate 全部 PASS，且哈希一致，才能生成最终 pass：

```text
ICDOR_GATE_COMPILE_TEST_PASS
ICDOR_GATE_SPLIT_NO_LEAKAGE_PASS
ICDOR_GATE_THREE_LANE_OWNERSHIP_PASS
ICDOR_GATE_DIRECT_IMAGE_DINO_PASS
ICDOR_GATE_TYPED_FACTOR_ATTENTION_PASS
ICDOR_GATE_PRESENCE_OBSERVABILITY_PASS
ICDOR_GATE_FACTOR_CERTIFICATE_PASS
ICDOR_GATE_ONTOLOGY_POLARITY_PASS
ICDOR_GATE_TARGET_ROUTER_PASS
ICDOR_GATE_TARGET_REREADER_PASS
ICDOR_GATE_ACTION_VISUAL_DECODER_PASS
ICDOR_GATE_EDGE_ADMISSION_PASS
ICDOR_GATE_ACTION_PARETO_PASS
ICDOR_GATE_DUAL_REASON_PASS
ICDOR_GATE_SELECTIVE_OBSERVATION_PASS
ICDOR_GATE_POSTERIOR_RANKING_PASS
ICDOR_GATE_LOSS_CORRECTNESS_PASS
ICDOR_GATE_CALIBRATION_PASS
ICDOR_GATE_SCHEDULE_PASS
ICDOR_GATE_ARTIFACT_SCHEMA_PASS
ICDOR_GATE_SCIENTIFIC_STOP_PASS
ICDOR_GATE_RUNTIME_PROFILE_PASS
ICDOR_GATE_SHORT_PILOT_PASS
ICDOR_GATE_EVALUATION_PROTOCOL_PASS
ICDOR_GATE_GITHUB_SYNC_PASS
```

任何一个 gate：

```text
missing
FAIL
STALE
hash mismatch
```

都不得生成 `REVIEW_PASS`。

---

# 31. REVIEW_PASS JSON 合同

最终文件：

```text
.review/acpr_mosaic_trust_v3_icdor_REVIEW_PASS.json
```

至少包含：

```json
{
  "status": "PASS",
  "project": "ACPR-MOSAIC-TRUST-v3-IC-DOR",
  "source_branch": "acpr_mosaic_ad_v1_direct_image",
  "source_head": "<full sha>",
  "target_branch": "acpr_mosaic_trust_v3_icdor_direct_image",
  "target_head": "<full sha>",
  "origin_target_head": "<full sha>",
  "plan_sha256": "<sha256>",
  "skill_sha256": "<sha256>",
  "resolved_config_sha256": "<sha256>",
  "split_manifest_sha256": "<sha256>",
  "runtime_selection_sha256": "<sha256>",
  "factor_certificate_sha256": "<sha256>",
  "edge_admission_sha256": "<sha256 or explicit none>",
  "test_only_engineering_protocol": true,
  "paper_protocol_unbiased": false,
  "feature_cache": false,
  "token_compression": "none",
  "direct_image": true,
  "dino_frozen": true,
  "short_pilot_seed": 20260713,
  "short_pilot_epochs": 4,
  "action_route_admission_status": "edges_admitted|no_edges_admitted",
  "gates": {
    "compile_test": "PASS",
    "split_no_leakage": "PASS",
    "three_lane_ownership": "PASS",
    "direct_image_dino": "PASS",
    "typed_factor_attention": "PASS",
    "presence_observability": "PASS",
    "factor_certificate": "PASS",
    "ontology_polarity": "PASS",
    "target_router": "PASS",
    "target_rereader": "PASS",
    "action_visual_decoder": "PASS",
    "edge_admission": "PASS",
    "action_pareto": "PASS",
    "dual_reason": "PASS",
    "selective_observation": "PASS",
    "posterior_ranking": "PASS",
    "loss_correctness": "PASS",
    "calibration": "PASS",
    "schedule": "PASS",
    "artifact_schema": "PASS",
    "scientific_stop": "PASS",
    "runtime_profile": "PASS",
    "short_pilot": "PASS",
    "evaluation_protocol": "PASS",
    "github_sync": "PASS"
  },
  "created_at": "<ISO8601>"
}
```

审查器必须在写出后立即重新读取并校验：

- JSON 可解析；
- 所有 SHA 与当前磁盘/HEAD 一致；
- target worktree clean；
- origin target HEAD一致；
- 所有 gate 文件存在；
- gate artifacts均未早于 target HEAD实现时间；
- pilot使用的代码/config/runtime hash与当前一致。

---

# 32. Full trainer 的强制拒绝逻辑

`train_acpr_mosaic_trust_icdor.py --mode full` 在启动前必须执行：

```python
assert review_pass_exists
assert review_pass["status"] == "PASS"
assert review_pass_target_head == git_head
assert review_pass_config_hash == resolved_config_hash
assert review_pass_split_hash == current_split_hash
assert review_pass_runtime_hash == runtime_selection_hash
assert review_pass_certificate_hash == current_certificate_hash
assert review_pass_edge_hash == current_edge_hash_or_none
assert feature_cache is False
assert token_compression == "none"
assert direct_image is True
```

任一失败必须退出非零状态码，不能只 warning。

Resume 同样必须验证：

```text
checkpoint target head
certificate hash
edge admission hash
split hash
schedule version
optimizer ownership schema
```

---

# 33. 代码审查者必须回答的最终问题

生成最终 pass 前，审查报告必须用事实回答：

1. Factor、action、reason 是否来自三条独立 adapter？
2. Reason loss 是否能通过任何路径更新 action adapter/base decoder？
3. Observed reason loss 是否能更新 factor extractor？
4. Action route loss是否能更新 base action decoder？
5. 强/弱/不可识别是否由网络自我预测？
6. Certificate 是否只使用 train_audit？
7. All-unknown factor 是否被 Abstain？
8. Multi-prototype 是否在视觉匹配前独立存在？
9. Curve factor 是否使用有序 sequence modeling？
10. Prior-only 是否不能支配 full factor？
11. Factor 是否只通过 mask/routing引导 action visual re-read？
12. Support/veto 的符号是否由结构保证？
13. Shadow 阶段 final action是否精确等于 visual-only？
14. Edge admission 是否使用 TET/TES/CCA和AP非劣条件？
15. No-edge case是否安全回退 visual-only，而非放宽 gate？
16. Reason 是否同时具有 direct observed、latent semantic和observation model输出？
17. Official Exp metrics是否使用 observed输出？
18. Latent reason是否用于 factor faithfulness而不是 benchmark label替代？
19. Disallowed reason factor mass是否精确为0？
20. Selective observation posterior公式和detach是否正确？
21. HardPair是否完全从正式路径移除？
22. Calibration是否只使用train_calib？
23. Test labels/geometry是否不能改变正式 logits/threshold/certificate/admission？
24. 每epoch是否保存足以定位根因的branch、factor、route、PU、gradient、calibration、logits artifacts？
25. Full run是否使用当前reviewed immutable HEAD并已同步GitHub？

任何答案为“未知”“大概”“配置里有”“日志显示名字”，都不能通过。

---

# 34. 审查失败分类

失败报告必须区分：

## A. Implementation failure

例如：

- 文件缺失；
- 路径未调用；
- 梯度所有权错误；
- factor route直加logit；
- reason semantic hard mask缺失；
- artifact缺失。

## B. Mechanism inactivity

例如：

- route strength接近0；
- dustbin全1；
- latent reason恒定；
- prototype collapse；
- factor shuffle无效。

## C. Mechanism harmfulness

例如：

- action AP退化；
- TES<0；
- CCA低；
- Pareto violation持续；
- observation model降低Exp mAP。

## D. Identifiability failure

例如：

- factor全unknown；
- content不优于prior；
- propensity与latent reason不可分；
- 高 q observed-zero 人工 precision 不足。

## E. Protocol failure

例如：

- test参与证书/admission/calibration；
- GitHub HEAD 漂移；
- 旧 checkpoint 依赖；
- cache/compression 开启；
- 测试时读取 geometry/reason。

报告必须给出对应文件、行号、动态证据和修复条件，不能只输出 `FAIL`。

---

# 35. 最终可接受声明

在本 Skill 通过后，Codex可以声明：

```text
ACPR-MOSAIC-TRUST v3 / IC-DOR 的代码级计算图、数据边界、参数所有权、
factor certificate、target-owned support/veto re-reading、dual-observation reason、
selective observation、action Pareto admission、calibration、artifact 和运行协议已经实现并通过审查。
```

不得声明：

```text
一定超过强基线
已经达到顶刊
factor是真实因果原因
高q observed-zero必然是真实漏标
latent deletion等价于真实图像因果干预
工程test-selected结果是无偏论文结果
```

模型性能和论文主张必须由正式 full run、消融、多 seed和最终无偏协议另行验证。

---

# 36. 最终执行顺序

Codex 必须严格按以下顺序：

```text
1. 读取三个canonical MD
2. 验证source clean/HEAD
3. 新建target worktree/branch
4. 安装本Skill并记录hash
5. TDD实现T01–T30
6. compileall +全tests
7. 静态禁用模式审查
8. 动态三lane/梯度/forward审查
9. split/no-leakage审查
10. factor/certificate审查
11. router/rereader/action/reason/PU审查
12. artifact和scientific-stop审查
13. immutable HEAD推送GitHub
14. 当前HEAD运行runtime profiler
15. 当前HEAD运行一次4-epoch pilot
16. 重新运行全部fail-closed gates
17. 写REVIEW_PASS
18. full trainer验证REVIEW_PASS后启动
19. 每epoch只评估test并输出全部诊断
20. 代码与结果同步记录到task_plan/findings/progress
```

任何代码变更都会使以下 artifact 失效：

```text
runtime profile
short pilot
factor certificate
edge admission
all dynamic gates
REVIEW_PASS
```

变更后必须从对应阶段重新审查。
