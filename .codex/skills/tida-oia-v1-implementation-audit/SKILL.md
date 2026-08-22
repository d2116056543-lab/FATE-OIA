---
name: tida-oia-v1-implementation-audit
description: Fail-closed audit for TIDA-OIA video and Flow Credit implementation, data, mechanism, memory, Git, and training contracts.
---

# TIDA-OIA V1 Strict Implementation Audit Skill

## 2026-08-22 Flow Credit Addendum

This addendum is mandatory for branch `tida_oia_flow_credit_v1` and takes priority over conflicting V1 text.

- Required code: `tida_flow_transition_bank.py` and `tida_flow_credit_losses.py` must exist and be reached from formal model/trainer call graphs.
- The terminal predictor may use query identity and causal history but must not use current target-frame static/global features as predictor input.
- `terminal_no_history` remains an artifact row with loss weight exactly zero.
- The transition bank must expose signed velocity, acceleration, region velocity, persistence, reliability, and transition tokens without dense patch-time factor tensors.
- Both action and reason readers must consume transition tokens. Reason flow inputs must be detached from shared/action owners.
- Scale zero and direct `history_off` must reproduce image logits exactly.
- Training must compute same-image action/reason GT-margin credit against history-off, repeated-last, and alternating shuffle/reverse interventions.
- Positive reason labels have weight 1. Unknown negatives use PU weights. Test labels and test thresholds cannot update any parameter or schedule.
- Mechanism artifacts must report per-intervention action/reason GT-margin advantage, per-label values, flow route mass, transition reliability, and velocity direction diagnostics.
- A non-zero gate, route, or delta is not evidence of useful traffic flow. A flow-aware claim requires ordered-history target-margin and metric advantages.
- REVIEW_PASS proves implementation correctness only. It cannot claim metric improvement without real-video intervention evidence.

## 2026-08-21 用户指令覆盖条款

以下条款优先于本文后续冲突文字：

- 必须从 commit `af6f526fc48816a695d8b386390c160f0c11b311`、tree `9c885b803a34040be8d04baef81f60d6f567aa0a` 新建 branch `tida_oia_v1_video` 与独立 worktree；旧 VETRA worktree/branch 必须只读且前后 HEAD/tree/status 不变。
- 创建独立 TIDA worktree 是必须行为且不构成失败；未使用独立 TIDA worktree 才失败，代码为 `ISOLATED_WORKTREE_MISSING`。
- 所有 GitHub 比较与 push 仅针对 `refs/heads/tida_oia_v1_video`，禁止向旧 VETRA ref 写入。
- 正式官方 train 3115 条按 source-video group 精确划分为 train_core=2291、train_calib=312、train_audit=512；optimizer 只读 train_core labels；test=885。
- Memory profile 每个候选执行 10 个 warm-up optimizer updates 和 100 个 measured optimizer updates；micro-step timing 在 update 内汇总。
- query-level online attention aggregation 不属于被禁止的 cache/token compression；离线预计算、持久化 token 或进入 DINO 前的 token dropping 仍被禁止。
- PASS artifact 使用 design/implementation/mechanism/memory/full-ready 分阶段 required/nullable schema，以 design 和 implementation plan 的修订定义为准。


> 将本文件复制为：
>
> `.codex/skills/tida-oia-v1-implementation-audit/SKILL.md`

---

# Purpose

本 Skill 对 TIDA-OIA V1 执行 **fail-closed** 代码、机制、数据、显存、训练协议和 Git 审查。

审查目标不是确认：

```text
类已经定义
脚本可以启动
loss字典里有名字
模型能跑一个batch
```

而是确认：

> TIDA-OIA 的目标帧保护、历史查询读取、终端条件创新、谓词差分、Action视觉读出、Reason梯度防火墙、时序干预、每轮test评估、test选best、前台监督和无feature-cache合同，全部在正式运行路径中真实实现、被调用、产生正确梯度并写出可复核artifact。

`REVIEW_PASS` 不证明最终指标一定提高。它只证明待训练方法和讨论方案一致，没有空壳、旁路、错误owner或伪审计。

---

# 1. 审查状态机

状态只能按顺序产生：

```text
DESIGN_REVIEW_PASS
IMPLEMENTATION_REVIEW_PASS
MECHANISM_REVIEW_PASS
MEMORY_REVIEW_PASS
FULL_TRAIN_READY
TRAIN_COMPLETED
```

任何前置状态失败，后续状态不得写出。

## 1.1 DESIGN_REVIEW_PASS

证明：

```text
design spec与implementation plan一致
单一方法而非多套候选
公式、shape、owner、数据与训练协议无冲突
```

## 1.2 IMPLEMENTATION_REVIEW_PASS

证明：

```text
required files存在
formal import/call graph真实到达所有模块
无placeholder
静态公式和owner正确
全部测试通过
```

## 1.3 MECHANISM_REVIEW_PASS

证明：

```text
真实视频/RGB smoke通过
历史路径不是no-op
目标帧fallback精确
Reason firewall正确
时序干预产生方向性变化
```

## 1.4 MEMORY_REVIEW_PASS

证明：

```text
真实DINO
真实15帧
无feature cache
显存<=45GiB
候选配置完成稳定吞吐profile
```

## 1.5 FULL_TRAIN_READY

必须同时满足：

```text
前四项PASS
worktree clean
local HEAD == GitHub HEAD
review绑定当前HEAD/config/schema/manifest
foreground启动命令已验证
```

## 1.6 TRAIN_COMPLETED

证明：

```text
正式10轮自然完成
每轮test完整
checkpoint/artifact完整
Stage C完成
无中途代码/config变化
```

---

# 2. 强制上下文

执行任何 Git、代码、测试、训练、评估、进程管理或 push 前读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

再读取：

```text
docs/superpowers/specs/2026-08-21-tida-oia-v1-design.md
docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md
configs/fate_oia_train_tida_oia_v1_15f.yaml
configs/tida_predicate_roles.yaml
```

不得创建额外训练状态 Markdown。审查结论追加到 canonical 三文件。

若 plan/spec/skill存在冲突：

```text
停止编码；
列出冲突；
先修正文档；
重新运行DESIGN_REVIEW；
不得静默采用最容易实现的版本。
```

---

# 3. Git 与当前 worktree 审查

执行：

```powershell
git branch --show-current
git rev-parse --show-toplevel
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain --untracked-files=all
git ls-remote origin refs/heads/tida_oia_v1_video
git worktree list --porcelain
```

要求：

```text
source branch = vetra_from_scratch_staged_v1
implementation branch = tida_oia_v1_video
必须新建独立TIDA worktree
不切换目标开发分支
无force push/reset --hard/clean -fd证据
训练前worktree clean
local HEAD == remote branch HEAD
```

计划撰写时源HEAD：

```text
af6f526fc48816a695d8b386390c160f0c11b311
```

TIDA branch 的 base commit 必须固定为上述 `af6f526...`，不得随源分支前移；源分支若独立前移只记录为外部状态，不改变本次TIDA lineage。

失败代码：

```text
WRONG_BRANCH
ISOLATED_WORKTREE_MISSING
DESTRUCTIVE_GIT_OPERATION
DIRTY_FULL_TRAIN_TREE
REMOTE_HEAD_MISMATCH
UNBOUND_GIT_HEAD
```

---

# 4. Required files

必须存在：

```text
configs/fate_oia_train_tida_oia_v1_15f.yaml
configs/tida_predicate_roles.yaml

fate_oia/datasets/bdd_oia_video.py
fate_oia/datasets/tida_clip_manifest.py
fate_oia/transforms_video.py

fate_oia/models/tida_context_encoder.py
fate_oia/models/tida_terminal_query_reader.py
fate_oia/models/tida_temporal_encoder.py
fate_oia/models/tida_terminal_innovation.py
fate_oia/models/tida_predicate_differential.py
fate_oia/models/tida_action_reader.py
fate_oia/models/tida_reason_reader.py
fate_oia/models/tida_oia_model.py

fate_oia/losses/tida_losses.py
fate_oia/losses/tida_loss_registry.py

fate_oia/utils/tida_temporal_interventions.py
fate_oia/utils/tida_artifacts.py
fate_oia/utils/tida_contracts.py

fate_oia/explain/tida_dynamic_concepts.py

fate_oia/engine/build_tida_clip_manifest.py
fate_oia/engine/audit_tida_video_data.py
fate_oia/engine/train_tida_oia.py
fate_oia/engine/evaluate_tida_oia.py
fate_oia/engine/profile_tida_oia.py
fate_oia/engine/audit_tida_oia_implementation.py
fate_oia/engine/collect_tida_tta_outputs.py
fate_oia/engine/export_tida_deployment.py
fate_oia/engine/supervise_tida_oia_foreground.py

scripts/FATE_OIA_tida_oia_v1_foreground.ps1

docs/superpowers/specs/2026-08-21-tida-oia-v1-design.md
docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md
.codex/skills/tida-oia-v1-implementation-audit/SKILL.md
```

并保留旧单帧入口可运行。

测试 inventory 不得只用 wildcard。Audit 必须从 `docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md` 的 `Contract tests` 清单解析每个反引号包围的 `test_tida_*.py` 文件名，逐名验证存在、被 targeted pytest 收集、并在 JUnit/pytest report 中有执行结果。任何指定测试缺失或未收集均为 `REQUIRED_FILE_MISSING`。

失败：

```text
REQUIRED_FILE_MISSING
LEGACY_VETRA_ENTRYPOINT_BROKEN
```

---

# 5. Forbidden formal paths

TIDA formal import/call graph不得启用：

```text
second DINO/backbone
Video Swin
VLM
MLLM
BERT/text encoder
Grounding DINO
external detector
tracker
depth model
optical-flow network
BEV model
graph
PMI
static co-occurrence logit bias
HardPair
PairMemory
distillation
cached logits as train input
feature cache
token compression
persistent decoded-frame cache
test-label calibration
Reason label/token as Action Value
learned counterfactual license as main path
free latent traffic concept naming
```

旧文件可以存在，但正式TIDA路径不得import或调用。

失败代码：

```text
SECOND_BACKBONE_FOUND
VLM_OR_TEXT_PATH_FOUND
DETECTOR_TRACKER_FLOW_PATH_FOUND
GRAPH_OR_PAIR_PATH_FOUND
DISTILLATION_OR_CACHE_FOUND
REASON_TO_ACTION_VALUE_LEAK
TEST_LABEL_PARAMETER_FIT
FREE_CONCEPT_NAMING_FOUND
```

---

# 6. Config 合同

检查：

```text
num_frames = 15
history_seconds = 5.0
sampling = quadratic_multirate
target_resolution = 360x640
context_resolution = 192x344
patch_size = 8
DINO layers = [3,7,11]
history read order = [11,7,3]

action_dim = 4
reason_dim = 21

feature_cache_enabled = false
token_compression = none
persistent_frame_cache = false

epochs = 10
precision = bf16
no_metric_early_stop = true

eval_splits = [test]
best_selection_split = test
best_selection_metric = deploy_joint
internal_test_selected = true
publication_eligible = false
test_every_epoch = true

max_reserved_memory_gb = 45.0
```

失败：

```text
CONFIG_CONTRACT_MISMATCH
WRONG_VIDEO_LENGTH
WRONG_SAMPLING
WRONG_EVAL_PROTOCOL
PUBLICATION_FLAG_FALSELY_TRUE
CACHE_OR_COMPRESSION_ENABLED
```

---

# 7. 数据与clip审查

## 7.1 Manifest schema

逐行检查：

```text
official_split
partition
file_name
target_image_path
clip_path
source_video_id
duration_seconds
fps
num_frames
target_timestamp_seconds
target_frame_index
```

不得缺字段后用默认空字符串继续。

`official_split` 只能为 `train/test`；`partition` 只能为 `train_core/train_calib/train_audit/test`，且 `official_split=test` 当且仅当 `partition=test`。正式 manifest 禁止旧 `split` 字段，避免双重真源。Loader、optimizer 与 audit 都必须按 `partition` 过滤；不得用 `official_split` 代替 partition。

## 7.2 唯一映射与正式分区

要求：

```text
每个file_name至多1个clip
每个clip对应唯一target image
ambiguous count = 0
source_video_id跨split交集 = 0
正式总数 = 4000
train_core = 2291
train_calib = 312
train_audit = 512
test = 885
partition seed = 20260821
四个partition之间 normalized source ID / clip SHA / endpoint near-duplicate 零交集
optimizer batch 的 label provenance 只能来自 train_core
```

分区必须按 design 中固定的 SHA256 排序、exact subset-sum 和 lexicographic tie-break 重算，并比较 partition hash；不得接受 manifest 自报 split。

## 7.3 解码

时间采样/解码细节真实抽样至少：

```text
train 128 clips
test 128 clips
```

路径存在、标签维度、唯一映射、分区、source identity、clip SHA、末帧四指标和 near-duplicate 门禁必须全量检查4000条，不得抽样替代全量合同。

检查：

```text
15 timestamps严格单调
最后timestamp=0
frame_indices非降序
target valid
invalid历史有mask
```

## 7.4 采样公式

数值验证：

\[
t_i=-5(1-i/14)^2.
\]

test无jitter；train jitter不越过相邻边界。

## 7.5 同步增强

构造可识别标记帧，验证：

```text
15帧共享同一flip
15帧共享同一几何变换
target/context letterbox映射一致
```

## 7.6 末帧一致性

输出真实分布：

```text
exact hash rate
SSIM p10/p50/p90
PSNR p10/p50/p90
normalized MAE
64-bit pHash distance
```

逐样本硬门：

```text
SSIM >= 0.90
PSNR >= 20 dB
normalized MAE <= 0.08
pHash Hamming <= 16
```

全局仍要求 median SSIM >= 0.995。跨 partition 精确 clip SHA 冲突、normalized source stem 冲突直接失败。endpoint pHash Hamming <=4 且 duration diff <=0.1s 且 FPS diff <=0.5 的 pair 视为确定性 near-duplicate，必须 quarantine 并使正式4000条审查失败，不得静默继续。

失败代码：

```text
CLIP_MANIFEST_INVALID
AMBIGUOUS_CLIP_MAPPING
SOURCE_VIDEO_SPLIT_LEAKAGE
TIMESTAMP_ORDER_INVALID
TARGET_FRAME_INVALID
UNSYNCHRONIZED_AUGMENTATION
LAST_FRAME_MISMATCH
```

---

# 8. 目标帧数值等价审查

先在未修改的 source commit `af6f526fc48816a695d8b386390c160f0c11b311` 上，以同一 image checkpoint、16张固定真实test图、eval/fp32/no augmentation、action/reason scale=1 生成独立 golden oracle。必须保存输入、checkpoint、source hashes以及：

```text
DINO selected-layer cls/patch tensors
action_logits_primary / reason_logits_primary
predicate logits/probs/tokens/attention/layer weights
action_nodes_primary / reason_nodes_primary
action evidence token/map/reference/sampling offsets/sampling weights/layer mixture
bounded action contribution / final action
reason private evidence/route/delta/final reason
branch logits
```

TIDA fallback 除了比较A/B/C/D，还必须逐 tensor 与该 source-tree oracle 比较：fp32 max abs <1e-6，bf16 max abs <5e-4。Oracle artifact、输入清单和 raw tensor hashes 缺失时不得通过。

同一 image checkpoint、同一真实图像、eval mode：

```text
A: 原AIEOIAModel单帧forward
B: TIDA history_off forward
C: TIDA all-history-invalid forward
D: TIDA temporal scale=0 forward
```

比较：

```text
image Action logits
image Reason logits
primary Action/Reason
Predicate logits/probs/attention
Action evidence
Reason private output
```

要求：

```text
fp32 max abs <1e-6
bf16 max abs <5e-4
```

并检查冻结 image base state hash在训练前后完全一致。

失败：

```text
TARGET_FRAME_ACTION_MISMATCH
TARGET_FRAME_REASON_MISMATCH
TARGET_FRAME_PREDICATE_MISMATCH
IMAGE_BASE_MUTATED
FALLBACK_NOT_EXACT
```

---

# 9. DINO和动态grid审查

## 9.1 同一backbone对象

运行时断言：

```python
id(image_model.foundation.dino.backbone)
==
id(context_encoder.dino_extractor.backbone)
```

state_dict中只能出现一份DINO参数owner。

## 9.2 Shapes

目标：

```text
[B,3,360,640]
→ [B,3,3600,384]
grid=(45,80)
```

历史：

```text
[Bc,3,192,344]
→ [Bc,3,1032,384]
grid=(24,43)
```

## 9.3 Frozen/no-grad

```text
all DINO params requires_grad=False
DINO eval mode
DINO grad=0
output detached
```

## 9.4 Chunking

hook记录任一时刻完整历史patch tensor的shape。

禁止：

```text
[B,14,3,1032,384]
[B,15,3,3600,384]
```

允许：

```text
[B*chunk,3,1032,384]
```

完成query读取后必须释放patch引用。

失败：

```text
DUPLICATE_DINO_INSTANCE
WRONG_TARGET_DINO_SHAPE
WRONG_CONTEXT_DINO_SHAPE
DINO_TRAINABLE
CONTEXT_PATCHES_NOT_CHUNKED
HISTORY_PATCH_RETENTION
```

---

# 10. Query reader审查

输入/输出必须为：

```text
target Action queries       [B,4,384]
target Predicate queries    [B,32,384]
context patches             [Bc,3,1032,384]
history query tokens        [B,14,36,384]
history attention           [B,14,36,1032]
predicate region mass       [B,14,32,5]
```

检查：

1. layer ID映射明确；
2. 调用顺序确为11→7→3；
3. 每层独立Q/K/V与Norm；
4. 每层更新query；
5. 使用entmax，不是固定平均；
6. Action query来自target Action node；
7. Predicate query包含现有Predicate identity；
8. Reason logits/labels不进入该模块。

动态干预：

```text
shuffle layer 11 → early update变化
shuffle layer 7 → middle update变化
shuffle layer 3 → final update变化
change target Action query → 对应Action history token变化
```

失败：

```text
QUERY_READER_SHAPE_MISMATCH
LAYER_ORDER_WRONG
LAYERS_MEANED_BEFORE_READ
QUERY_NOT_TARGET_CONDITIONED
QUERY_READER_NO_GRAD
REASON_LEAK_INTO_HISTORY_READER
```

---

# 11. 时序编码器审查

检查：

```text
2 Transformer layers
4 heads
causal mask
continuous timestamp encoding
valid-frame key padding mask
```

动态测试：

```text
改变timestamp但不改变token → output变化
改变未来历史位置 → 更早位置causal state不变化
mask某帧 → 该帧不影响summary
全历史mask → summary为0/明确null
```

失败：

```text
TEMPORAL_ENCODER_PLACEHOLDER
NO_CAUSAL_MASK
DISCRETE_POSITION_ONLY
INVALID_FRAME_USED
TEMPORAL_ENCODER_NO_GRAD
```

---

# 12. 终端创新审查

## 12.1 单一共享预测器

静态对象身份检查：

```text
history和no-history forward使用同一个module object
参数只出现一次
```

## 12.2 无target leakage

使用forward hook/输入追踪确认：

```text
待预测的target Action/Predicate token
不进入predictor input
只作为stopgrad target
```

## 12.3 公式重算

从artifact重算：

\[
\rho=\operatorname{clip}((e_0-e_H)/(e_0+\epsilon),0,1)
\]

\[
\Xi=\rho LN(\hat E_H-\hat E_0)
\]

要求max abs <1e-6。

## 12.4 stop-gradient

```text
rho.requires_grad = false
terminal target requires_grad = false
Reason loss不能改变rho source
```

## 12.5 可辨识性

合成数据：

```text
历史完全决定target → eH < e0, rho高
历史纯噪声 → eH≈e0, rho低
全历史invalid → rho=0
repeated-last → rho低于有真实变化的历史
```

失败：

```text
TWO_PREDICTORS_FOUND
TERMINAL_TARGET_LEAKAGE
INNOVATION_FORMULA_MISMATCH
RHO_TRAINABLE
RHO_COLLAPSED_ALL_ZERO
RHO_COLLAPSED_ALL_ONE
INNOVATION_NO_GRAD
```

---

# 13. Predicate differential审查

## 13.1 Role exact cover

当前 Predicate names集合与role YAML：

```text
无缺失
无重复
无未知名字
每个Predicate恰好一个角色
```

## 13.2 时间导数

使用非均匀timestamp合成线性/二次轨迹，验证：

```text
线性轨迹一阶导数正确
线性轨迹二阶≈0
二次轨迹二阶方向正确
invalid dt被拒绝
```

## 13.3 公共运动去除

构造：

```text
所有静态和动态Predicate具有相同平移趋势
```

要求去除后动态相对速度接近0。

再给一个动态Predicate额外趋势，要求保留该相对趋势。

## 13.4 区域质量

验证 attention map × dynamic grid region mask 数值正确，不得使用固定45×80 reshape处理24×43 context。

## 13.5 人类概念

检查：

```text
dynamic concept函数无参数
无requires_grad tensor输出到loss
无Action/Reason标签输入
低置信度输出unknown
```

失败：

```text
PREDICATE_ROLE_COVERAGE_ERROR
TIME_DERIVATIVE_WRONG
COMMON_MOTION_NOT_REMOVED
REGION_MASS_GRID_WRONG
LEARNED_FAKE_TRAFFIC_CONCEPT
```

---

# 14. Action reader审查

## 14.1 Factor bank

要求：

```text
32 Predicate factors
4 Action innovation factors
1 null
总计37
```

## 14.2 Visual-only Value

autograd/input provenance必须证明 Value不含：

```text
Reason logits/token/label
文本
test annotation
```

## 14.3 Sparse route

```text
route shape [B,4,37]
每Action和为1
null存在
route非全均匀
4/4 Action有非null事件
```

前5% zero-scale 动态门：对 valid `rho_max>0` 样本重算

```text
H(pi) = -sum(pi*log(pi+eps))
m = 1 - pi_null
u_a = sum_f pi_af * normalize(k_f)
L_route = mean(H/log(37)) + mean(relu(0.05-m)) + mean_{a!=b} relu(cos(u_a,u_b)-0.90)
```

要求：最终 video Action/Reason logits 与 image logits fp32 max abs <1e-6；同时 `grad(predicate_differential projection)>0`、`grad(action key projection)>0`。若梯度来自 detached/fabricated probe 或只改日志则失败。

## 14.4 Bound

```text
abs(action_temporal_delta) <= 0.15 + 1e-6
```

## 14.5 Exact contribution

```text
sum(factor_contribution, factor_dim)
==
video_action - image_action
```

fp32误差 <1e-6。

## 14.6 Fallback

```text
all rho=0 → delta=0
null-only → delta=0
history invalid → delta=0
```

失败：

```text
WRONG_FACTOR_BANK
ACTION_VALUE_NOT_VISUAL
ACTION_ROUTE_COLLAPSE
ACTION_DELTA_OUT_OF_BOUND
ACTION_CONTRIBUTION_NOT_EXACT
ACTION_FALLBACK_BROKEN
TEMPORAL_ACTION_NO_GRAD
```

---

# 15. Reason firewall审查

分别构造：

```text
L_action only
L_reason only
L_innovation only
```

对 owner 求梯度。

`L_reason only`要求：

```text
grad(history_reader)=0
grad(temporal_encoder)=0
grad(innovation_predictor)=0
grad(predicate_differential shared path)=0
grad(temporal_action)=0
grad(image_model)=0
grad(temporal_reason)>0
```

`L_action only`：

```text
grad(temporal_action)>0
grad(temporal_reason)=0
```

Reason输入必须在代码中显式 `.detach()`，不能依赖optimizer不含参数来伪装防火墙。

失败：

```text
REASON_GRAD_LEAK_TO_SHARED
REASON_GRAD_LEAK_TO_ACTION
ACTION_GRAD_LEAK_TO_REASON
REASON_PRIVATE_NO_GRAD
DETACH_MISSING
```

---

# 16. Loss registry审查

## 16.1 Required terms

```text
terminal_hist
terminal_no_history
terminal_gain
temporal_order
repeated_last_contrast

action_asl
action_smooth_ap
action_base_protect
action_delta
action_route_sparse

reason_partial
reason_rank
reason_soft_f1
reason_delta
```

## 16.2 Exactly once

解析正式trainer，运行动态registry检查：

```text
每个configured term恰好add一次
无raw+budgeted重复
无同一loss通过两个名称重复
```

## 16.3 非零与梯度

为每项构造应激活的合成输入：

```text
loss finite
loss >0 when violation exists
correct owner grad >0
illegal owner grad=0
```

任何未计算项必须写：

```text
available=false
reason=<明确原因>
```

不能写固定0并让gate通过。

失败：

```text
LOSS_TERM_MISSING
LOSS_TERM_DUPLICATED
ZERO_PLACEHOLDER_LOSS
WRONG_LOSS_OWNER
NONFINITE_LOSS
```

---

# 17. 时序干预审查

所有干预必须保持：

```text
target_image hash不变
image Action/Reason logits不变
只改变历史压缩表示
```

检查：

```text
history_off
repeated_last
time_shuffle
time_reverse
selected_predicate_flatten
matched_predicate_flatten
wrong_action_route
```

至少128条真实test clip：

```text
real vs repeated输出非零差
real vs shuffle输出非零差
real vs reverse动态状态非零差
selected flatten平均drop > matched flatten平均drop
```

若数据本身低动态，应按高/低temporal novelty分层，而不是强迫全局大差值。

失败：

```text
INTERVENTION_CHANGES_TARGET_FRAME
INTERVENTION_NAME_ONLY
TIME_ORDER_UNUSED
SELECTED_NOT_STRONGER_THAN_MATCHED
TEMPORAL_PATH_NO_EFFECT
```

---

# 18. Static placeholder扫描

扫描TIDA-owned文件，禁止：

```text
pass
TODO作为正式逻辑
NotImplementedError
return torch.zeros(...)代替机制
{"available": true}占位
hard-coded review pass
hard-coded gate true
只写日志不参与forward
只实例化不调用
if False保护正式模块
loss_weight配置存在但trainer不读取
```

允许测试中的明确异常断言，但必须人工区分。

生成：

```text
TIDA_STATIC_PLACEHOLDER_SCAN.json
```

失败：

```text
PLACEHOLDER_IMPLEMENTATION
MODULE_INSTANTIATED_NOT_CALLED
CONFIG_VALUE_UNUSED
AUDIT_SELF_CERTIFICATION
```

---

# 19. 正式call graph审查

从：

```text
scripts/FATE_OIA_tida_oia_v1_foreground.ps1
```

追踪到：

```text
supervise_tida_oia_foreground
→ train_tida_oia
→ TIDAOIAModel.forward
→ context encoder
→ query reader
→ temporal encoder
→ innovation
→ differential
→ action reader
→ reason reader
→ loss registry
→ backward
→ evaluator
→ artifact/checkpoint
```

每条边必须通过：

```text
静态import
运行时hook call count
输出tensor被下游实际使用
```

失败：

```text
FORMAL_CALL_GRAPH_BROKEN
CORE_MODULE_BYPASSED
HIDDEN_STRONGER_BRANCH
FORMAL_OUTPUT_NOT_VIDEO
```

---

# 20. 显存与速度审查

候选：

```text
A: batch4/accum8/chunk2
B: batch3/accum10/chunk3
C: batch2/accum15/chunk5
D: batch1/accum30/chunk7
```

每个候选：

```text
10 warm-up steps
100 measured optimizer updates
至少2个temporal intervention events
真实视频decode
真实DINO
BF16
```

记录：

```text
peak allocated
peak reserved
steady reserved slope
samples/sec
decode fraction
target DINO fraction
context DINO fraction
query/temporal/backward fraction
```

要求：

```text
peak_reserved <=45.0GiB
steady growth <=0.25GiB/100 updates
无OOM
无NaN
```

选择最快安全候选。不得通过保留无用tensor凑显存。

失败：

```text
MEMORY_LIMIT_EXCEEDED
MEMORY_LEAK
PROFILE_NOT_REAL_VIDEO
PROFILE_OMITS_INTERVENTION
UNSAFE_BATCH_SELECTED
```

---

# 21. Trainer、checkpoint与resume

Checkpoint必须包含：

```text
model
optimizer
scheduler
EMA
epoch
global_update
best scores and source
Python RNG
NumPy RNG
Torch CPU RNG
CUDA RNG
sampler epoch/state
clip manifest hash
split hash
config hash
Git HEAD/tree
image checkpoint hash
predicate role hash
```

Exact resume测试：

```text
连续4 updates
vs
2 updates + save + resume + 2 updates
```

要求trainable参数、optimizer moments和scheduler一致。

正式run中代码/config/hash变化时拒绝resume。

失败：

```text
CHECKPOINT_SCHEMA_INCOMPLETE
RESUME_NOT_EXACT
RUN_IDENTITY_MISMATCH
IMAGE_CHECKPOINT_LINEAGE_BROKEN
```

---

# 22. 每轮test与best选择审查

Config和trainer必须证明：

```text
eval splits only [test]
test every epoch
best split = test
best metric = deploy_joint
internal_test_selected = true
publication_eligible = false
```

Test label只能：

```text
计算指标
选择best checkpoint
```

不得：

```text
拟合threshold
选择TTA weight
选择Reason beta
训练model
```

每轮必须写：

```text
image baseline metrics
video online metrics
video EMA metrics
raw/deploy
mF1/oF1/mAP
per-label
```

失败：

```text
TEST_NOT_EVALUATED_EVERY_EPOCH
BEST_NOT_TEST_SELECTED
TEST_LABEL_PARAMETER_LEAK
PUBLICATION_PROTOCOL_MISLABELED
INCOMPLETE_EPOCH_METRICS
```

---

# 23. 前台监督审查

静态扫描禁止：

```text
nohup
Start-Process
Win32_Process.Create
Scheduled Task
Start-Job
后台隐藏进程
```

PowerShell脚本必须直接调用 Python。

Python supervisor：

```text
subprocess/Popen继承当前console
stdout/stderr可见
父进程保持前台
无metric early stop
无patience stop
无target-reached stop
```

正式10轮完成前，不得因为指标弱主动终止。

结构性失败必须fail-closed并保存artifact，不能伪装完成。

检查：

```text
Ctrl-C传播
child exit code传播
异常写FAIL artifact
正常10轮写TRAIN_COMPLETED
```

失败：

```text
BACKGROUND_LAUNCH_FOUND
METRIC_EARLY_STOP_FOUND
DISCRETIONARY_KILL_FOUND
CHILD_FAILURE_SWALLOWED
FALSE_COMPLETION_MARKER
```

---

# 24. GitHub同步审查

训练前：

```text
worktree clean
local HEAD == remote HEAD
all review artifacts bind same HEAD
```

训练后代码未改变：

```text
final HEAD == training manifest HEAD
```

执行：

```powershell
git rev-parse HEAD
git status --porcelain --untracked-files=all
git ls-remote origin refs/heads/tida_oia_v1_video
```

失败：

```text
UNPUSHED_TRAINING_CODE
DIRTY_TRAINING_TREE
TRAINED_UNREVIEWED_HEAD
MIDRUN_CODE_CHANGE
```

---

# 25. 必须运行的命令

## 25.1 Compile

```powershell
E:\Anaconda\envs\sbw39\python.exe -m compileall fate_oia tests
```

## 25.2 Targeted tests

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest tests\test_tida_*.py -q
```

## 25.3 Regression

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest `
  tests\test_aie_*.py `
  tests\test_vetra_*.py `
  tests\test_acpr_dino_field.py `
  tests\test_acpr_ego_regions.py `
  tests\test_acpr_scene_predicate_head.py `
  -q
```

## 25.4 Data audit

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.audit_tida_video_data ...
```

## 25.5 Implementation audit

```powershell
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.audit_tida_oia_implementation ...
```

## 25.6 Real-video mechanism smoke

至少：

```text
128 train clips
128 test clips
100 optimizer updates
history/repeat/shuffle/reverse
```

## 25.7 Memory profile

执行A-D候选并生成选择artifact。

## 25.8 Full train

只有 `FULL_TRAIN_READY_TIDA_OIA_V1.json` 存在且绑定当前HEAD后，前台运行正式10轮。

---

# 26. 审查artifact schema

必须生成：

```text
.review/tida_oia_v1/
  design_review.json
  required_files.json
  forbidden_path_scan.json
  static_placeholder_scan.json
  call_graph.json
  data_audit.json
  target_equivalence.json
  dino_identity_and_shapes.json
  query_reader_audit.json
  temporal_encoder_audit.json
  innovation_audit.json
  predicate_differential_audit.json
  action_reader_audit.json
  reason_firewall_audit.json
  loss_registry_audit.json
  intervention_audit.json
  checkpoint_resume_audit.json
  foreground_supervisor_audit.json
  memory_profile.json
  git_binding.json
```

PASS文件：

```text
DESIGN_REVIEW_PASS_TIDA_OIA_V1.json
IMPLEMENTATION_REVIEW_PASS_TIDA_OIA_V1.json
MECHANISM_REVIEW_PASS_TIDA_OIA_V1.json
MEMORY_REVIEW_PASS_TIDA_OIA_V1.json
FULL_TRAIN_READY_TIDA_OIA_V1.json
```

所有PASS共享必填：`pass`、Git head/tree、base/source head/tree、plan/skill/spec hashes、commands/exit codes、gates、warnings、created_at。逐阶段 schema：

```text
DESIGN: config/manifest/image-checkpoint/tests/raw-tensor hash = null + not_yet_produced
IMPLEMENTATION: config/tests hash必填；manifest/image-checkpoint/raw-tensor = null + awaiting_mechanism
MECHANISM: manifest/image-checkpoint/golden-oracle/raw-tensor/data-audit/smoke hash必填
MEMORY: MECHANISM全部字段 + profile report hash + selected candidate必填
FULL_TRAIN_READY: 所有hash非null，嵌入全部前置PASS hash，worktree clean，local HEAD==remote TIDA branch HEAD
```

Skill hash在 DESIGN_REVIEW 后冻结；若 Skill 发生任何变化，旧 DESIGN PASS 立即失效，必须重新监督审查和生成新 hash。

不能只有：

```json
{"pass": true}
```

---

# 27. Codex执行规程

Codex必须按以下顺序工作：

1. 读取canonical三文件。
2. 读取本Skill、design和implementation plan。
3. 写出细化后的任务清单到canonical `task_plan.md`。
4. 执行DESIGN_REVIEW。
5. 先写失败测试，再实现代码。
6. 每完成一个核心层，运行对应测试。
7. 完成全量实现后执行两轮审查：
   - Review A：静态、公式、owner、call graph；
   - Review B：真实视频、梯度、干预、显存、resume。
8. 修复所有FAIL，不得把FAIL改成warning绕过。
9. Commit并push当前branch。
10. 重新绑定最终HEAD并运行全审查。
11. 生成FULL_TRAIN_READY。
12. 在当前前台会话启动10轮训练。
13. 全过程监督，不因弱指标停止。
14. 异常按结构性恢复规则处理。
15. 完成Stage C、Git核验和TRAIN_COMPLETED。
16. 将真实结果追加到canonical `findings.md`和`progress.md`。

Codex不得：

```text
先训练后补审查
用smoke指标当正式结果
修改gate让坏实现通过
用固定零填充artifact
用后台任务规避前台监督
因训练慢或暂时不提升而停掉健康run
```

---

# 28. 最终审查原则

最终批准语句只能是：

```text
APPROVED_FOR_FULL_TRAIN
```

或：

```text
REJECTED
```

不得使用：

```text
mostly implemented
probably works
available
looks correct
minor issue only
```

只有代码、真实运行和artifact共同支持时，才能批准。

最核心的五个硬门：

\[
\boxed{
\begin{aligned}
&1.\ \text{history off精确恢复强单帧模型};\\
&2.\ \text{历史创新由可测终端预测增益定义};\\
&3.\ \text{Action Value只来自视觉差分};\\
&4.\ \text{Reason梯度不能污染共享时序与Action};\\
&5.\ \text{真实历史优于重复、打乱、反转历史}.
\end{aligned}
}
\]
