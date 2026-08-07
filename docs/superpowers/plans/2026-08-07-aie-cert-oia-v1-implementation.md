# AIE-CERT-OIA V1：Codex 完整代码级实施计划

## Causally Anchored, Evidence-Conserved and Preference-Robust AIE for BDD-OIA

**日期：** 2026-08-07  
**仓库：** `d2116056543-lab/FATE-OIA`  
**唯一源分支：** `acpr_aie_oia_v1_direct_image`  
**已核对源 HEAD：** `8a324b94b1cd6b4a4377655a1bd426f7d854fec0`  
**目标分支：** `acpr_aie_cert_oia_v1_direct_image`  
**目标 worktree：** `E:\sbw\FATE_Drive\fate_oia_acpr_aie_cert_oia_v1_worktree`  
**任务：** BDD-OIA 单帧 RGB，4 维 Action + 21 维 Reason 多标签联合预测  
**硬件：** NVIDIA RTX 5880，48 GB 显存  
**正式协议：** direct image、Frozen DINO、无 feature cache、无 token compression、每轮仅评估 test、以 test deploy-joint 选主 best  
**论文边界：** `internal_test_selected=true`、`publication_eligible=false`  
**正式训练：** 16 epochs，单 seed，从官方 DINO 权重开始，不加载旧 AIE full checkpoint  
**功能预检：** 一次自动化实现审查 + 一次短 runtime profile + 一次 3-epoch pilot；不得做多随机种子搜索或大规模超参 sweep

---

# 0. 文件地位、执行目标与不可承诺边界

本文件是 Codex 将当前 AIE-OIA V1 改造成 AIE-CERT-OIA V1 的唯一代码合同。配套审查合同为：

```text
.codex/skills/aie-cert-oia-v1-implementation-audit/SKILL.md
```

Codex 不得把本文降级为“建议”、不得只实现可训练外壳、不得以旧 AIE 类名存在或测试通过代替新 formal 路径真实调用。所有核心组件必须同时出现在：

```text
formal model forward
formal loss
formal backward
formal evaluator
formal checkpoint/resume
formal train logs
formal epoch artifacts
implementation audit
```

本计划能够以 fail-closed 方式保证“代码功能完整实现并真实调用”，不能在训练前诚实保证某个随机优化过程必然达到指定数值。正式目标为同一 checkpoint：

```text
Act_mF1 >= 0.730
Exp_mF1 >= 0.380
```

当前 AIE 已经分别达到：

```text
Action best:      0.725682
Explanation best: 0.393320
```

因此本轮不是重建弱模型，而是解决 Action 与 Explanation 最优时相错位和后期 residual 漂移。任何结果声明必须基于正式 full run artifact，不能用 pilot、test oracle threshold 或离线 logit 混合作为主结果。

---

# 1. 唯一科学命题

AIE-CERT 的中心命题：

> **任何能够修改 Action 或 Reason 的 residual，都必须由同一个守恒 evidence atom 产生；该 atom 的 token、map、signed contribution 和 counterfactual effect 必须保持身份一致；residual 的允许强度由可验证干预效果和可靠视觉偏好决定，而不是由固定全强度 schedule 无限放大。**

统一数据流：

```text
RGB 360x640
  ↓
Frozen official DINO ViT-S/8, layers 3/7/11
  ↓ one call only
patch field [B,3,3600,384]
  ↓
CalAlign-compatible 25-query primary anchor
  ├── 4 primary Action
  └── 21 primary Reason
  ↓
4x4 Action evidence atoms
  ├── full-field visual inquiry
  ├── sparse arithmetic predicate prior
  ├── evidence-conditioned deformable reread
  ├── map-token co-transport
  └── same-region background centering
  ↓
bias-free signed contribution
  ↓
multi-control robust counterfactual certificate
  ↓
primal-dual effect/necessity/budget constraints
  ↓
final Action
  ↓ stop-gradient signed evidence
21 full-field Reason rereaders
  ├── supportive Action evidence
  ├── inhibitory Action evidence
  ├── supportive predicate evidence
  ├── contradictory predicate evidence
  └── sample-label-specific residual budget
  ↓
primary-referenced ECPO
  ↓
final Reason
  ↓
read-only predicate naming / honest abstention
```

Evidence atom：

\[
\mathcal E_{iak}
=
(\widetilde M_{iak},\widetilde e_{iak},c_{iak},d^{cert}_{iak},n_{iak}).
\]

要求：

1. `map` 与 `token` 在 probe interaction 后仍一一对应；
2. `contribution` 不含图像无关 bias；
3. selected deletion 相对多控制产生 robust certificate；
4. contribution 与 certificate 方向一致；
5. Reason 区分 support 与 inhibition，不能使用 `abs(contribution)`；
6. noisy Reason label 不得更新 clean predicate interface 或 Action evidence；
7. naming 只读 evidence，不得反向改变 Action；
8. residual 达到效果预算后不能继续无约束变大。

---

# 2. 开始远程任务前的强制规则

在任何 Git、文件修改、测试、训练、评估、进程管理或 push 前，Codex 必须读取：

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

这三份文件是唯一持久训练/实验状态 Markdown。不得新建：

```text
implementation_status.md
audit_status.md
pilot_status.md
run_status.md
training_status.md
```

方法规范允许放入：

```text
docs/superpowers/plans/
.codex/skills/
```

每次远程会话开始，在 `progress.md` 追加：

```text
AIE-CERT task start
timestamp
source branch/HEAD
target branch/HEAD
canonical files read=true
```

所有后续实施、审查、pilot、full run 和 GitHub 同步记录，按时间追加到三份 canonical 文件，不覆盖历史记录。

---

# 3. 新 worktree 与 GitHub branch 合同

## 3.1 核验源分支

在管理 worktree 执行：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_worktree

git fetch github --prune
git rev-parse github/acpr_aie_oia_v1_direct_image
git ls-remote github refs/heads/acpr_aie_oia_v1_direct_image
```

两处必须为：

```text
8a324b94b1cd6b4a4377655a1bd426f7d854fec0
```

若远程名不是 `github`，先执行：

```powershell
git remote -v
```

只允许使用实际指向 `d2116056543-lab/FATE-OIA` 的 remote；不得静默换用其他 fork。

## 3.2 记录并保护原 worktree

执行并保存输出到 `progress.md`：

```powershell
git -C E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree rev-parse HEAD
git -C E:\sbw\FATE_Drive\fate_oia_acpr_aie_oia_v1_worktree status --porcelain --untracked-files=all
git worktree list --porcelain
```

原 AIE worktree 在本任务中只读。不得：

```text
edit
reset
clean
checkout
commit
stash
delete
reuse as target
```

## 3.3 目标不存在检查

```powershell
git show-ref --verify --quiet refs/heads/acpr_aie_cert_oia_v1_direct_image
Test-Path E:\sbw\FATE_Drive\fate_oia_acpr_aie_cert_oia_v1_worktree
git ls-remote github refs/heads/acpr_aie_cert_oia_v1_direct_image
```

若本地 branch、目录或远程 branch 已存在：

```text
STOP
记录实际状态
不得删除或覆盖未知工作
```

## 3.4 新建 worktree

```powershell
git worktree add `
  -b acpr_aie_cert_oia_v1_direct_image `
  E:\sbw\FATE_Drive\fate_oia_acpr_aie_cert_oia_v1_worktree `
  github/acpr_aie_oia_v1_direct_image
```

验证：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_aie_cert_oia_v1_worktree
git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=all
```

必须：

```text
branch = acpr_aie_cert_oia_v1_direct_image
HEAD   = 8a324b94b1cd6b4a4377655a1bd426f7d854fec0
status = clean
```

## 3.5 立即创建 GitHub branch

```powershell
git push -u github acpr_aie_cert_oia_v1_direct_image
git ls-remote github refs/heads/acpr_aie_cert_oia_v1_direct_image
```

以后每个代码闭环：

```text
commit
push
ls-remote
local HEAD == GitHub HEAD
```

禁止提交：

```text
.background_runs/
.review/ runtime JSON
checkpoints
logits
datasets
large tensors
generated PNG batches
```

---

# 4. 当前 branch 的具体代码事实与必须修复的位置

Codex 开始改动前必须通读下列源文件，不得按本文摘要替代读代码：

```text
configs/fate_oia_train_360x640_aie_oia_v1.yaml

fate_oia/models/aie_calalign_foundation.py
fate_oia/models/aie_oia_model.py
fate_oia/models/aie_evidence_interface.py
fate_oia/models/aie_deformable_reread.py
fate_oia/models/aie_contribution_head.py
fate_oia/models/aie_reason_rereader.py
fate_oia/models/aie_predicate_naming.py

fate_oia/datasets/aie_structured_evidence.py

fate_oia/losses/aie_losses.py
fate_oia/losses/aie_loss_registry.py

fate_oia/utils/aie_counterfactual.py
fate_oia/utils/aie_calibration.py
fate_oia/utils/aie_metrics.py

fate_oia/engine/train_aie_oia.py
fate_oia/engine/audit_aie_oia_implementation.py
fate_oia/engine/profile_aie_oia.py
fate_oia/engine/supervise_aie_oia_foreground.py
```

已确认的具体缺陷与新代码对应关系：

| 当前位置 | 当前行为 | AIE-CERT 必改 |
|---|---|---|
| `AIEEvidenceInterface.forward` | predicate maps 先形成几何平均式 log prior | 改为 sparse arithmetic mixture 后再取 bounded log-density ratio |
| `AIEEvidenceInterface.forward` | group attention 只混合 token | map 与 token 使用同一 transport matrix 同步传输 |
| `AIEDeformableReread.forward` | offsets 主要由原 probe 生成 | offsets 由 probe + global token + map summary 共同生成 |
| `AIEContributionHead` | 存在 `[4,4]` contribution bias | 新 formal head 无任何 contribution bias |
| `AIEContributionHead` | residual 只有固定 scale/cap | 加 effect-based primal-dual budget |
| `AIEReasonRereader` | 使用 `abs(contribution)` | 分离 support / inhibition |
| `AIEReasonRereader` | contradictory predicate 被 `clamp_min(0)` 删除 | 正、反 predicate prior 分别保留 |
| `aie_losses.py` | 只有 Action delta 正则 | Reason 有显式 sample-label budget 与约束 |
| `reason_ranking_loss` | FIFO 无 age、无 primary reference | 改为 ECPO + age-bounded balanced queue |
| `AIEPredicateNaming` | 有独立 predicate keys | 与 Action predicate interface 使用同一 key bank |
| `AIEPredicateNaming` | naming loss可更新 evidence owner | naming 输入、共享 key 对 naming owner均 detach |
| `aie_counterfactual.py` | 单 matched control 为主 | 2 matched + wrong-probe + wrong-action，多控制证书 |
| `train_aie_oia.py` | 固定 CF loss weights | primal-dual constraints，按效果自调 |
| `train_aie_oia.py` | noisy Reason 可经 predicate alignment更新 predicate | clean predicate interface禁止 Reason gradient |
| `aie_structured_evidence.py` | source-complete粒度过粗 | 每 predicate / reason 单独可观察性与 verified counter |
| evaluator | full-scale residual一直使用 | 同时记录 residual budget、漂移和同场 ablation |

这些不是可选优化；任何一项缺失都不能生成 `REVIEW_PASS_AIE_CERT_OIA_V1.json`。

---

# 5. 新 formal 文件结构

保留旧 AIE 文件用于基线和回归测试，但新 formal import graph 不得调用旧 AIE evidence/contribution/reason/naming/counterfactual。

## 5.1 新增文件

同时修改根目录 `.gitignore`，加入：

```text
.review/
```

保证审查 artifact 不污染 code worktree；不得忽略任何 `fate_oia/`、`tests/`、`configs/`、`docs/superpowers/` 或 `.codex/skills/` 源文件。

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

tests/test_aie_cert_source_regression.py
tests/test_aie_cert_sparse_predicate.py
tests/test_aie_cert_atom_transport.py
tests/test_aie_cert_local_reread.py
tests/test_aie_cert_contribution.py
tests/test_aie_cert_counterfactual.py
tests/test_aie_cert_constraints.py
tests/test_aie_cert_reason_signed.py
tests/test_aie_cert_ecpo_queue.py
tests/test_aie_cert_naming.py
tests/test_aie_cert_owner_firewalls.py
tests/test_aie_cert_schedule.py
tests/test_aie_cert_eval_artifacts.py
tests/test_aie_cert_runtime_contract.py
tests/test_aie_cert_static_contracts.py

docs/superpowers/plans/2026-08-07-aie-cert-oia-v1-implementation.md
.codex/skills/aie-cert-oia-v1-implementation-audit/SKILL.md
```

## 5.2 允许复用

```text
fate_oia/models/acpr_dino_field.py
fate_oia/models/acpr_ego_regions.py
fate_oia/models/acpr_label_trunk.py
fate_oia/models/acpr_scene_predicate_head.py
fate_oia/models/acpr_predicate_reason.py
fate_oia/models/acpr_reason_grammar.py

fate_oia/datasets/bdd_oia_multitask.py
fate_oia/datasets/bdd100k_grounding.py
fate_oia/datasets/aie_splits.py

fate_oia/losses/asymmetric_loss.py
fate_oia/losses/acpr_losses.py

fate_oia/utils/aie_artifacts.py
fate_oia/utils/aie_hashes.py
fate_oia/utils/aie_calibration.py
fate_oia/utils/aie_metrics.py
```

复用不等于直接 import 旧 `AIEOIAModel`。正式入口必须构造：

```python
AIECertOIAModel
```

---

# 6. Formal API 与 tensor 合同

`AIECertOIAModel.decode_from_field()` 必须一次返回下列正式 tensor。所有 shape 在 batch `B` 下：

```text
patch_tokens_by_layer_raw            [B,3,3600,384]

action_logits_primary                [B,4]
reason_logits_primary                [B,21]
action_nodes_primary                 [B,4,384]
reason_nodes_primary                 [B,21,384]

predicate_logits_clean               [B,32]
predicate_probs_clean                [B,32]
predicate_attention_clean            [B,32,3600]
shared_predicate_keys                [32,64]

probe_queries                        [B,4,4,384]
global_visual_score                  [B,4,4,3600]
global_attention                     [B,4,4,3600]
global_token                         [B,4,4,384]
predicate_mixture                    [B,4,4,32]
predicate_mixture_map                [B,4,4,3600]
predicate_prior_available            [B,4,4]
predicate_prior_strength             [B,4,4]

atom_map_pre_transport               [B,4,4,3600]
atom_token_pre_transport             [B,4,4,384]
atom_transport_matrix                [B,4,4,4]
atom_transport_gamma                 scalar or [4]
atom_map                              [B,4,4,3600]
atom_token                            [B,4,4,384]

reference_point                      [B,4,4,2]
sampling_offsets                     [B,4,4,3,8,2]
sampling_weights                     [B,4,4,3,8]
local_token                          [B,4,4,384]

atom_region_mask                     [B,4,4,3600]
background_token                     [B,4,4,384]
centered_atom_token                  [B,4,4,384]

raw_contribution                     [B,4,4]
bounded_contribution                 [B,4,4]
action_delta                         [B,4]
action_logits_final                  [B,4]
action_logits_final_train            [B,4]
contribution_reconstruction_error    scalar

reason_action_support_prior          [B,21,3600]
reason_action_inhibit_prior          [B,21,3600]
reason_predicate_support_prior       [B,21,3600]
reason_predicate_counter_prior       [B,21,3600]
reason_private_attention             [B,21,3,3600]
reason_uncertainty                   [B,21]
reason_evidence_agreement            [B,21]
reason_budget                        [B,21]
reason_delta                         [B,21]
reason_logits_final                  [B,21]
reason_logits_final_train            [B,21]

name_id                              [B,4,4]
name_confidence                      [B,4,4]
name_margin                          [B,4,4]
name_quality                         [B,4,4,32]
named_coverage                       scalar
```

不得用日志时临时重新构造这些值。正式 forward 即应输出，便于训练、评估和审查使用同一结果。

---

# 7. Primary anchor 与 clean predicate gradient firewall

## 7.1 `AIECertCalAlignFoundation`

以当前 `AIECalAlignFoundation` 为数值基础，但新建文件，不修改旧文件。

构造模块仍仅包含：

```text
ACPRDinoFieldExtractor
ACPREgoRegionEncoder
ACPRScenePredicateHead
ACPRLabelTrunk
ACPRPredicateReasoner
```

forward 数值必须与源 AIE foundation 在相同 state 下完全一致，但梯度合同改变：

```python
predicates = predicate_head(patch, region_masks)
predicate_tokens_for_primary = predicates["predicate_tokens"].detach()
predicate_probs_for_primary = predicates["predicate_probs"].detach()

trunk = trunk(patch, predicate_tokens=predicate_tokens_for_primary)
predicate_reason = predicate_reason(
    reason_nodes,
    predicate_probs_for_primary,
    predicate_tokens_for_primary,
)
```

因此：

```text
primary Action/Reason loss -> primary_core
structured predicate loss  -> predicate_visual
Reason labels               -X-> predicate_visual
final Action/Reason         -X-> primary_core
```

forward detach 不改变数值，所以 progress-zero equivalence 必须仍满足：

```text
fp32 max abs < 1e-6
bf16 max abs < 5e-4
```

## 7.2 optimizer owners

精确 owner：

```text
primary_core:
  foundation.ego
  foundation.trunk
  foundation.predicate_reason

predicate_visual:
  foundation.predicate_head

action_evidence:
  shared_predicate_bank
  evidence_interface
  deformable_reread
  atom_transport

action_contribution:
  contribution_head

reason_private:
  reason_rereader

naming_readout:
  naming projection/readout only
```

DINO 不在 optimizer。

所有可训练参数必须被恰好一个 owner 覆盖，不得重复或遗漏。

---

# 8. Clean structured evidence

新建 `AIECertStructuredEvidenceBuilder`，可以复用当前 BDD100K index/parser，但不得复用其整行 source-complete 逻辑。

输出：

```text
predicate_target                 [B,32]
predicate_positive_mask          [B,32]
predicate_counter_mask           [B,32]
predicate_map_target             [B,32,3600]
predicate_map_mask               [B,32]
predicate_reliability            [B,32]
predicate_source_complete        [B,32]

reason_positive_support          [B,21]
reason_verified_counter          [B,21]
reason_counter_reliability       [B,21]
reason_observable_mask           [B,21]

source_counts
coverage
per_predicate_coverage
per_reason_counter_coverage
```

硬规则：

1. `predicate_source_complete[row,p]` 只表示该 predicate 所需的数据源完整；
2. object JSON 完整不能自动使 lane/drivable/traffic-light-color 等所有 predicate complete；
3. `reason_verified_counter` 只能来自：
   - 显式 contradictory predicate 正观测；
   - 该 predicate source complete；
   - reliability 达阈值；
4. 禁止用：
   - model contradiction score；
   - attention.max；
   - Reason=0 本身；
   - co-occurrence / PMI；
5. test forward 不读取 BDD100K structured evidence；structured evidence 仅用于 train supervision、train-calib诊断和固定 test audit 的 no-grad解释审计，不进入正式 test final logits。

---

# 9. Shared predicate bank 与 sparse arithmetic mixture

## 9.1 `entmax15`

在 `aie_cert_sparse.py` 实现自包含 FP32-stable `entmax15`。禁止为此新增未经固定版本验证的外部包。

合同：

```text
input arbitrary finite logits
output nonnegative
sum(last_dim)=1
can produce exact zeros
all-equal input -> uniform
gradient finite
bf16 caller internally casts fp32 and returns original dtype
```

## 9.2 shared key bank

`AIECertPredicateBank`：

```python
predicate_keys: nn.Parameter  # [32,64]
probe_projection: nn.Linear(384,64)
```

只存在这一份 predicate keys。

使用：

```text
Action evidence: shared keys, trainable through action_evidence owner
Naming: shared_keys.detach()
```

Naming 模块不得再定义自己的 `predicate_keys`。

## 9.3 sparse mixture

\[
u_{iakp}
=
\frac{W_q e_{iak}^{global}\cdot K_p}{\sqrt{64}}
+
\log(p_{ip}+\epsilon).
\]

\[
\pi_{iak}
=
\operatorname{entmax}_{1.5}(u_{iak}).
\]

availability：

```python
available = predicate_probs.max(-1) >= 0.30
```

若 unavailable：

```text
predicate_mixture = 0
predicate_prior_strength = 0
Action evidence退回纯visual
```

arithmetic mixture：

\[
P^{mix}_{iakn}
=
\sum_p \pi_{iakp}P_{ipn}.
\]

先算 arithmetic mixture，再算相对 uniform 的 bounded log-density ratio：

\[
B_{iakn}
=
\operatorname{clip}
\left[
\log(P^{mix}_{iakn}+\epsilon)+\log N,
-1.5,1.5
\right].
\]

禁止：

```python
sum(pi * log(predicate_map))
prod(predicate_map ** pi)
```

prior strength：

\[
0 \le \lambda_{iak}\le0.25.
\]

最终 map score：

\[
s_{iakn}=s^{visual}_{iakn}+\lambda_{iak}B_{iakn}.
\]

---

# 10. Global evidence 与 evidence-conditioned local reread

## 10.1 global inquiry

保持当前三层 full-field query/key/value 结构，但必须输出：

```text
global_visual_score
global_attention
global_token
layer_mixture
```

Action node 输入必须 detach，确保 final Action 不更新 primary。

## 10.2 local query

当前 local offsets 不能只由原 probe 决定。新实现：

\[
m_{iak}
=
\sum_n M^{pre}_{iakn}F^{mix}_{iakn}.
\]

\[
q^{local}_{iak}
=
LN
\left(
q^{probe}_{iak}
+
W_g e^{global}_{iak}
+
W_m m_{iak}
\right).
\]

`offset_head` 和 `weight_head` 仅接收 `q_local`。

输出合同：

```text
abs(offset) <= 0.25
sampling weights sum over layer*point = 1
local token changes when global map is shuffled
local token changes when field changes
```

禁止用：

```text
global token + MLP
top-k mean
second DINO forward
```

---

# 11. Evidence atom map-token co-transport

新建 `AIECertAtomTransport`。

输入：

```text
token_pre [B,4,4,384]
map_pre   [B,4,4,3600]
```

只在同一 Action 内交互：

```text
[B,4,4,D] -> [B*4,4,D]
```

使用 `nn.MultiheadAttention(..., need_weights=True, average_attn_weights=False)` 得到：

```text
A_heads [B*4,H,4,4]
A = mean(A_heads, dim=head)
```

移除 diagonal 后按行归一化：

\[
A^{off}_{kj}
=
\frac{A_{kj}(1-I_{kj})}
{\sum_jA_{kj}(1-I_{kj})+\epsilon}.
\]

\[
\gamma
=
0.25\sigma(g),\quad
\gamma_{init}=0.05.
\]

同一 matrix 同时传输：

\[
\widetilde e_k
=
LN(e_k+\gamma\sum_jA^{off}_{kj}e_j),
\]

\[
\widetilde M_k
=
Normalize(M_k+\gamma\sum_jA^{off}_{kj}M_j).
\]

正式 contribution、counterfactual、Reason、naming 必须全部使用：

```text
atom_token = token_post
atom_map   = map_post
```

不得任何组件回退到 pre-transport map/token。

---

# 12. Overlap ceiling，不再强制完全正交

删除旧 raw cosine duplicate objective。新损失：

\[
L_{overlap}
=
\operatorname{mean}
\left[
I(|c_k|>\tau_c)
I(|c_j|>\tau_c)
\max(0,\cos(M_k,M_j)-0.65)^2
\right].
\]

默认：

```text
overlap_ceiling = 0.65
contribution_threshold = 1e-3
weight = 0.005
```

含义：

```text
合理共享区域不罚
仅高度重复且都声称有贡献时惩罚
```

日志必须输出：

```text
mean overlap
p90 overlap
over-ceiling rate
effective probe count
```

不得再把 overlap 降到接近 0 作为成功门槛。

---

# 13. Same-region background centering 与 bias-free contribution

## 13.1 atom region

根据每个 atom 的 `reference_point`，从现有 ego regions 中确定 own region：

```text
upper_traffic_region
left_corridor
right_corridor
bottom_drivable_region
front_center
```

输出 `atom_region_mask [B,4,4,3600]`。

## 13.2 background token

在 own region 内排除 selected top-k evidence 后求背景；若剩余质量不足，退回完整 own region：

\[
e^{bg}_{iak}
=
\frac{\sum_nR_{iakn}(1-H_{iakn})F^{mix}_{iakn}}
{\sum_nR_{iakn}(1-H_{iakn})+\epsilon}.
\]

\[
\bar e_{iak}
=
LN(\widetilde e_{iak}-e^{bg}_{iak}).
\]

## 13.3 contribution head

新 `AIECertContributionHead` 只能包含：

```text
LayerNorm
weight [4,384]
```

禁止：

```text
bias
per-probe constant
class logit offset independent of image
```

\[
r_{iak}=w_a^\top\bar e_{iak}.
\]

\[
\Delta z_{ia}
=
s_A\kappa_A\tanh(\sum_kr_{iak}/\kappa_A).
\]

沿用 direction-preserving L2 cap 20。

精确分解：

\[
z_A^{final}-z_A^{primary}
=
\sum_k\widetilde c_{iak}.
\]

要求：

```text
reconstruction max abs < 1e-6 fp32
contribution changes after image/field shuffle
zero centered token -> zero raw contribution
state_dict无 contribution bias key
```

---

# 14. Multi-control robust counterfactual certificate

新建 `AIECertCounterfactualEngine`。每个 event 只复用同一 DINO field，不重新编码图像。

## 14.1 selected evidence

```text
target-signed contribution最大probe
top-k=64
selected map使用post-transport atom_map
selected region使用该atom自己的region
```

## 14.2 四类 controls

至少构造：

```text
matched_same_region_1
matched_same_region_2
wrong_probe_own_region
wrong_action_own_region
```

要求：

1. 两个 matched controls 使用不同确定性 seed；
2. selected-control overlap <= 0.20；
3. wrong-probe mask 来自同 Action 的其他 probe，并使用 wrong probe 自己的 region；
4. wrong-action mask 来自其他 Action，并使用 wrong Action atom 自己的 region；
5. 每个 control 与 selected patch mass一致；
6. 至少 3 个 valid controls，否则 event invalid；
7. invalid 原因逐项记录。

## 14.3 robust certificate

对 target-signed margin：

\[
d^{sel}=m(F)-m(F^{-sel}).
\]

\[
U^{ctl}
=
mean(d_1^{ctl},...,d_J^{ctl})
+
1.0\cdot std(d_1^{ctl},...,d_J^{ctl}).
\]

\[
d^{cert}=d^{sel}-U^{ctl}.
\]

\[
r^{cf}
=
\exp(-std(d^{ctl})/0.25).
\]

输出：

```text
selected_drop
control_drops [E,4]
control_mean
control_std
certificate
reliability
valid_mask
per-control validity
cases
```

禁止用 selected-minus-single-random 代替 certificate。

---

# 15. Primal-dual constraints

新建 `AIECertDualState`，dual variables 为 checkpointed buffers，不进入 AdamW。

```text
lambda_effect
lambda_necessity
lambda_action_budget
lambda_reason_budget

constraint_ema_*
```

默认：

```text
dual_lr = 0.01
dual_ema = 0.95
lambda_min = 0
lambda_max = 10
```

## 15.1 effect constraint

\[
g_{effect}
=
E[r^{cf}Huber(c^T,d^{cert})]-0.05.
\]

## 15.2 necessity constraint

\[
g_{necessity}
=
0.05-E[r^{cf}d^{cert}].
\]

## 15.3 Action residual budget

\[
g_A
=
E[\Delta z_A^2]
-
1.25E[r^{cf}ReLU(d^{cert})^2]
-
0.02.
\]

## 15.4 Reason residual budget

对 ECPO valid pairs 定义：

\[
gain_{ij}
=
(\Delta^{final}_{ij}-\Delta^{primary}_{ij}).
\]

\[
g_R
=
E[\Delta z_R^2]
-
1.0E[ReLU(gain_{ij})^2]
-
0.02.
\]

无 ECPO pair 的 update：

```text
不更新lambda_reason_budget
记录reason_constraint_available=false
```

primal loss：

\[
L_{constraint}
=
\sum_j\lambda_j[g_j]_+.
\]

每个 optimizer update 后：

```python
ema = beta * ema + (1-beta) * g.detach()
lambda = clamp(lambda + dual_lr * ema, 0, lambda_max)
```

dual 状态必须随 checkpoint 保存、resume 精确恢复；eval 不得更新。

---

# 16. Signed Reason reread

## 16.1 Action support / inhibition

Reason query 对 atom token 计算相关度。分别使用：

```text
relu(contribution)
relu(-contribution)
```

得到：

```text
reason_action_support_prior
reason_action_inhibit_prior
```

禁止出现：

```python
abs(contribution)
```

## 16.2 Predicate support / counter

使用 grammar：

```text
positive_predicate_mask
contradictory_predicate_mask
```

分别构造 arithmetic mixtures：

```text
reason_predicate_support_prior
reason_predicate_counter_prior
```

不得将 counter 路径 `clamp_min(0)` 后丢失。

## 16.3 full-field attention

\[
\begin{aligned}
s_{irln}=&q_{ir}^\top k_{iln}/\sqrt D\\
&+\lambda_{A+}B(M^{A+}_{irn})
-\lambda_{A-}B(M^{A-}_{irn})\\
&+\lambda_{P+}B(M^{P+}_{irn})
-\lambda_{P-}B(M^{P-}_{irn})\\
&+\log \pi_{irl}.
\end{aligned}
\]

所有 prior strength 有界：

```text
0 <= lambda <= 0.75
```

Reason 输入：

```text
primary reason nodes.detach()
atom token/map.detach()
contribution.detach()
predicate maps/probs.detach()
full field.detach()
```

所以 final Reason 不能更新 Action/predicate/primary。

---

# 17. Sample-label-specific Reason budget

primary uncertainty：

\[
u_{ir}=4p^0_{ir}(1-p^0_{ir}).
\]

证据 agreement 使用 support 和 counter 两对分布的 Bhattacharyya overlap：

\[
a_{ir}
=
\frac12
\left[
BC(M^{A+},M^{P+})
+
BC(M^{A-},M^{P-})
\right].
\]

任何一对缺失时，该对 contribution 为 0，不得伪造 neutral 1。

schedule 给出的当前 `reason_budget_max(t)`：

\[
b_{ir}
=
0.10+
(reason\_budget\_max(t)-0.10)u_{ir}a_{ir}.
\]

\[
\Delta z^R_{ir}
=
b_{ir}\kappa_R\tanh(\delta_{ir}/\kappa_R).
\]

默认：

```text
kappa_R = 4.0
budget_min = 0.10
budget_max_final = 0.60
```

日志：

```text
budget mean/p10/p50/p90
min-budget rate
max-budget rate
uncertainty mean
agreement mean
delta RMS/quantiles
delta-to-budget ratio
```

---

# 18. Evidence-Conditioned Concept Preference Optimization（ECPO）

## 18.1 verified pairs

正样本：

```text
reason_target == 1
```

负样本必须同时满足：

```text
reason_target == 0
reason_verified_counter == 1
reason_counter_reliability >= 0.50
reason_observable_mask == 1
```

禁止用：

```text
model contradiction
model confidence
Reason=0 alone
attention concentration
PMI/co-occurrence
```

## 18.2 primary-referenced preference

\[
\Delta^{final}_{ij}
=
z^{final}_{ir}-z^{final}_{jr}.
\]

\[
\Delta^{primary}_{ij}
=
sg(z^{primary}_{ir}-z^{primary}_{jr}).
\]

\[
L_{ECPO}
=
-w_{ij}\log\sigma
\left[
2.0
(\Delta^{final}_{ij}-\Delta^{primary}_{ij})
\right].
\]

pair weight：

```text
positive reliability
x verified counter reliability
x exp(-age/32)
```

## 18.3 queue

`AIECertPreferenceQueue`：

```text
capacity = 512 records total
max_age = 64 optimizer updates
age_tau = 32
per_label_positive_sample_cap = 8
per_label_negative_sample_cap = 8
```

保存 detached：

```text
primary_reason_logits
final_reason_logits
reason_target
verified_counter
counter_reliability
enqueue_update
sample_id
```

先使用当前 batch + queue 构 pair，完成 backward 后 enqueue。

要求：

```text
age >64不可采样
每标签单独平衡
queue checkpoint/resume
无pair返回可微zero
pair count/label/age全日志
```

旧 `reason_ranking_loss` 和旧无 age FIFO 不进入 formal loss。

---

# 19. Read-only predicate naming

`AIECertNaming` 不定义 predicate keys；接收：

```text
atom_token.detach()
atom_map.detach()
shared_predicate_keys.detach()
predicate_attention_clean.detach()
predicate_probs_clean.detach()
counterfactual certificate.detach()
```

quality：

```text
spatial soft-IoU
x shared-key compatibility
x predicate presence
x CF reliability/effect gate
```

训练使用 grounded predicate preference：

```text
correct quality > strongest wrong quality + margin
```

只有：

```text
naming projection/readout
```

获得 naming loss 梯度。

硬要求：

```text
naming loss -> action_evidence grad = 0
naming loss -> shared keys grad = 0
naming loss -> predicate_visual grad = 0
```

低置信时输出：

```text
name_id = -1
```

不得降低阈值制造 coverage。`named_coverage=0` 不阻止分数训练，但必须记录为论文解释边界。

---

# 20. Formal losses 与 owner

## 20.1 primary_core

```text
primary_action                 1.00
primary_action_visual          0.05
primary_action_reason          0.05
primary_reason_partial         1.00
primary_reason_soft_f1         0.08
```

## 20.2 predicate_visual

```text
predicate_cls                  0.12
predicate_map                  0.06
predicate_compactness          0.0005
```

删除 Reason-target-to-predicate alignment 的 predicate gradient。若保留 grammar alignment，只能：

```text
predicate_probs.detach()
owner = reason_private
```

## 20.3 action_evidence / contribution

```text
final_action                   1.00
final_action_soft_f1           0.03
final_action_cardinality       0.02
atom_overlap_ceiling           0.005
dual_effect                    dynamic
dual_necessity                 dynamic
dual_action_budget             dynamic
```

## 20.4 reason_private

```text
final_reason                   1.00
final_reason_soft_f1           0.04
ecpo                           0.08 * schedule
dual_reason_budget             dynamic
```

Reason negatives使用 external-only counter reliability；不得调用旧 model-based `compute_counter_confidence()`。

## 20.5 naming

```text
naming_preference              0.02 * schedule
```

所有 configured term 必须每个 forward 注册一次；inactive term写可微zero并记录 inactivity 原因。

---

# 21. 连续 schedule，不分裂成多阶段模型

所有组件在同一 formal model 中存在，不做 Stage A/B/C 换模型。只做连续 ramp：

| progress | 行为 |
|---:|---|
| `0 → 0.05` | LR warm-up；predicate grounding `0.25 → 1.0`；predicate spatial prior `0 → 1` |
| `0 → 0.10` | Action residual scale `0.10 → 1.0`；Reason max budget `0.10 → 0.60`；transport gamma cap `0.05 → 0.25` |
| `0.05 → 0.15` | CF event/loss scale `0 → 1`；ECPO scale `0 → 1` |
| `0.08 → 0.20` | dual update scale `0 → 1` |
| `0.10 → 0.20` | naming scale `0 → 1` |
| `0.05 → 1.00` | cosine LR decay，最终 ratio `0.05` |

实现函数：

```python
schedule_values(
    optimizer_update,
    schedule_total_updates,
    cfg,
) -> {
    lr_multiplier,
    grounding_scale,
    predicate_prior_scale,
    action_scale,
    reason_budget_max,
    transport_gamma_cap,
    cf_scale,
    ecpo_scale,
    dual_scale,
    naming_scale,
}
```

硬要求：

```text
无 epoch-based hard switch
resume后同一update得到完全相同schedule
pilot使用自己的总updates测试完整ramp
full使用16 epochs总updates
```

---

# 22. 正式 YAML 配置

必须写入下列主值：

```yaml
experiment:
  name: aie_cert_oia_v1
  source_branch: acpr_aie_oia_v1_direct_image
  source_head: 8a324b94b1cd6b4a4377655a1bd426f7d854fec0
  direct_image: true
  feature_cache_enabled: false
  token_compression: none
  eval_splits: [test]
  best_selection_split: test
  best_selection_metric: deploy_joint
  internal_test_selected: true
  publication_eligible: false

data:
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
  selected_layers: [3, 7, 11]
  freeze_backbone: true
  no_grad_backbone: true

predicate_interface:
  num_predicates: 32
  key_dim: 64
  presence_threshold: 0.30
  prior_strength_max: 0.25
  log_density_bound: 1.5

evidence:
  probes_per_action: 4
  local_points_per_layer: 8
  max_offset: 0.25
  probe_chunk_size: 16
  transport_heads: 4
  transport_gamma_init: 0.05
  transport_gamma_max: 0.25
  overlap_ceiling: 0.65
  action_kappa: 3.0
  action_scale_start: 0.10
  action_scale_max: 1.00
  action_logit_norm_cap: 20.0

counterfactual:
  enabled: true
  interval_optimizer_updates: 4
  batch_fraction: 0.50
  max_actions_per_sample: 2
  max_atoms_per_event: 8
  topk_patches: 64
  matched_control_count: 2
  min_valid_controls: 3
  max_control_overlap: 0.20
  control_std_multiplier: 1.0
  reliability_tau: 0.25
  rerun_dino: false

dual:
  lr: 0.01
  ema_decay: 0.95
  lambda_max: 10.0
  effect_tolerance: 0.05
  necessity_margin: 0.05
  action_budget_alpha: 1.25
  action_budget_epsilon: 0.02
  reason_budget_alpha: 1.00
  reason_budget_epsilon: 0.02

reason_private:
  action_prior_max: 0.75
  predicate_prior_max: 0.75
  reason_kappa: 4.0
  budget_min: 0.10
  budget_max: 0.60

ecpo:
  beta: 2.0
  queue_capacity: 512
  max_age_updates: 64
  age_tau: 32.0
  pairs_per_label: 8
  verified_counter_threshold: 0.50

training:
  epochs: 16
  batch_size: 6
  gradient_accumulation_steps: 5
  precision: bf16
  optimizer: AdamW
  fused_adamw_if_available: true
  torch_compile: false
  weight_decay: 0.05
  lr_primary_core: 0.00020
  lr_predicate_visual: 0.00010
  lr_action_evidence: 0.00010
  lr_action_contribution: 0.00008
  lr_reason_private: 0.00010
  lr_naming: 0.00005
  warmup_ratio: 0.05
  min_lr_ratio: 0.05
  primary_grad_cap: 0.25
  predicate_grad_cap: 0.25
  action_evidence_grad_cap: 0.50
  action_contribution_grad_cap: 0.50
  reason_private_grad_cap: 0.50
  naming_grad_cap: 0.25
  global_grad_clip: 1.0
  no_metric_early_stop: true

calibration:
  enabled: true
  source: train_calib
  grid_step: 0.01
  group_shrinkage_support: 50
  ema_decay: 0.80
  max_threshold_step: 0.05
  accept_min_joint_delta: -0.001
  accept_max_action_mf1_drop: 0.002
  test_oracle_writeback: false

runtime:
  test_every_epoch: true
  save_every_epoch: true
  fixed_test_audit_samples: 128
  print_every_optimizer_updates: 100
  max_reserved_memory_gb: 44.5
  memory_growth_tolerance_gb: 0.25
```

---

# 23. Calibration guard

复用 train-calib threshold search，但增加：

1. logit-space EMA；
2. 每标签单次 threshold step 限制；
3. train-calib guard。

流程：

```text
fit current candidate
clip candidate step
EMA with previous accepted threshold
compute train-calib raw metrics
compute train-calib candidate deploy metrics
accept only if:
  deploy_joint >= raw_joint - 0.001
  deploy_Act_mF1 >= raw_Act_mF1 - 0.002
else:
  retain previous accepted threshold
```

保存：

```text
candidate threshold
accepted threshold
accept/reject
reject reason
raw/deploy calib metrics
EMA state
```

test 永远不拟合 threshold。

---

# 24. 48 GB 最快 runtime profile

正式 config 的 `(batch=6, accum=5)` 是安全默认，不直接假设最快。preflight 必须在真实 RTX 5880 上按顺序 profile：

```text
(batch=8, accum=4, chunk=16)
(batch=7, accum=4, chunk=16)
(batch=6, accum=5, chunk=16)
(batch=5, accum=6, chunk=16)
(batch=6, accum=5, chunk=8)  # 仅内存fallback
```

每个 candidate：

```text
10 warmup microsteps
30 measured microsteps
至少2个真实multi-control CF events
official DINO
structured builder
formal losses
formal backward
optimizer step
```

有效条件：

```text
no OOM
reserved < 44.5 GB
steady growth <= 0.25 GB
all finite
CF valid events >0
DINO calls ordinary=1
DINO calls CF=0
```

选择：

```text
最大 event-adjusted samples/sec
若速度差 <=3%，选显存更低者
```

输出：

```text
.review/aie_cert_oia_v1/AIE_CERT_RUNTIME_PROFILE.json
```

supervisor 从 profile artifact 读取 batch/accum/workers，不接受人工口头覆盖。若环境中已有其他 GPU 进程占用超过 2 GB，profile 必须停止并记录，不得把共享占用误判为模型 OOM。

---

# 25. 单次自动化实现审查

preflight 脚本只需一个入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_aie_cert_oia_v1_preflight.ps1
```

内部按顺序：

```text
read canonical files proof
branch/worktree/source checks
py_compile/compileall
targeted pytest
static forbidden scan
real-DINO one-image equivalence/shape audit
gradient owner/firewall audit
three-update dynamic mechanism smoke
runtime profile
requirement matrix generation
REVIEW_PASS generation
```

任何 hard check失败：

```text
exit code != 0
remove/withhold REVIEW_PASS
full train blocked
```

不得用 mock-only PASS。

---

# 26. 唯一 3-epoch pilot

一次 pilot，单 seed，不做多 seed 或超参搜索：

```text
train samples: 4096
train-calib: 512
train-audit: 512
test: 512
epochs: 3
schedule按pilot总updates完整走连续ramp
test-only per epoch
```

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_aie_cert_oia_v1_pilot.ps1 `
  -Epochs 3 `
  -MaxTrainSamples 4096 `
  -MaxCalibSamples 512 `
  -MaxAuditSamples 512 `
  -MaxTestSamples 512
```

pilot 是功能与数值健康检查，不作为性能结果。

## 26.1 pilot hard gates

```text
all tensors/losses finite
DINO grad == 0
ordinary DINO calls == 1
CF DINO calls == 0

all owners have nonzero gradient/update
Reason loss cannot update predicate/action owners
Naming cannot update evidence/shared keys

predicate mixture active_rate >0
predicate fallback_rate <1
predicate effective count finite/nontrivial

local token/global token ratio in [0.05, 2.0]
transport token delta >0
transport map delta >0
co-transport uses identical matrix
contribution reconstruction <1e-6
contribution head has no bias

CF valid events >=64 aggregate
at least3 control types observed
certificate finite
certificate positive rate >=0.40
contribution-certificate correlation >0

dual finite
no lambda at max
at least one lambda changes when constraint violated

ECPO valid pairs >=100 aggregate
queue max age <=64
at least8 Reason labels obtain pairs

reason budget not all minimum
reason budget not all maximum
reason delta RMS <1.5 at pilot end

final Action mAP not below primary by >0.02
final Reason mAP not below primary by >0.02

naming raw quality nonzero
naming coverage may be zero but must be honestly reported
```

若 gate失败，先修代码/逻辑，再重复同一个 pilot；不得通过降低核心 gate、改 test 子集或换 seed制造 PASS。

---

# 27. 正式 16-epoch full run

只有以下 artifact 均绑定当前 HEAD/config/source tree 才能启动：

```text
REVIEW_PASS_AIE_CERT_OIA_V1.json
AIE_CERT_RUNTIME_PROFILE.json
AIE_CERT_PILOT_GATE.json pass=true
AIE_CERT_FULL_TRAIN_READY.json
clean worktree
local HEAD == GitHub HEAD
```

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_aie_cert_oia_v1_foreground.ps1 `
  -Epochs 16 `
  -Device cuda
```

正式训练：

```text
full train images
official DINO
from scratch
single seed 20260806
batch/accum from runtime profile
bf16
test-only at each epoch end
train-calib threshold only
no val
no cache
no compression
no metric early stop
foreground/supervised child output
```

保存 best：

```text
checkpoint_best_deploy_joint.pth
checkpoint_best_action_mf1.pth
checkpoint_best_action_map.pth
checkpoint_best_reason_mf1.pth
checkpoint_best_reason_map.pth
checkpoint_latest.pth
checkpoint_epoch_000..015.pth
```

主结果是 `checkpoint_best_deploy_joint.pth`。其他 best 只能作为诊断。

---

# 28. 训练时必须直接输出的诊断

## 28.1 每 100 optimizer updates

单行 JSON event `aie_cert_batch`，至少包含：

### identity/schedule

```text
epoch
micro_step
optimizer_update
progress
lr per owner
grounding_scale
predicate_prior_scale
action_scale
reason_budget_max
transport_gamma_cap
cf_scale
ecpo_scale
dual_scale
naming_scale
```

### loss

```text
every raw loss
every weighted loss
constraint residuals
dual penalty
total
```

### gradients/updates

```text
owner grad norm before clip
owner grad norm after clip
owner parameter update RMS
zero-grad rate
DINO grad max
```

### Action

```text
primary/final logit RMS
action delta RMS/p10/p50/p90/max
action cap rate
per-action contribution RMS
positive/negative contribution rate
reconstruction error
```

### evidence atom

```text
global/pre/post map entropy
mean/p90 overlap
over-ceiling rate
effective probe count
predicate active/fallback/effective count/top1 mass
predicate prior strength
transport gamma/offdiag mass
token transport delta
map transport delta
co-transport matrix discrepancy
local/global token RMS ratio
offset RMS/max/clamp rate
background token RMS
centered/raw token norm ratio
```

### counterfactual

```text
event count
valid count
invalid reasons
selected drop
matched control1/2 drop
wrong-probe drop
wrong-action drop
control mean/std
certificate mean/p10/p50/p90
certificate positive rate
reliability mean
per-action event/certificate
contribution-certificate Pearson/Spearman
```

### dual

```text
lambda effect/necessity/action/reason
constraint raw/EMA
update applied
max-clamp rate
```

### Reason/ECPO

```text
Action support/inhibit prior mass/entropy
predicate support/counter prior mass/entropy
primary uncertainty
evidence agreement
reason budget quantiles
reason delta quantiles/RMS
delta-to-budget ratio
ECPO current/queue pair count
per-label pair coverage
queue size/age quantiles
preference gain
```

### naming/structured/runtime

```text
naming eligible count
raw confidence/margin/IoU
coverage
grounded precision when defined
structured coverage by source/predicate/reason
data/encode/primary/evidence/local/transport/reason/CF/backward time
allocated/reserved/peak GB
samples/sec
```

## 28.2 每 epoch full test 一次

全 test 只做一次 DINO encoding，计算：

```text
primary raw
final raw
primary deploy
final deploy
per-label F1/AP/precision/recall
thresholds and calibration guard
joint
```

## 28.3 固定 128 test same-field audit

相同固定 sample IDs，每轮只在已编码 field 上 decode，不重跑 DINO：

```text
final
predicate_prior_off
local_reread_off
atom_transport_off
background_center_off
action_residual_off/primary
reason_action_prior_off
reason_predicate_prior_off
reason_signed_to_unsigned_legacy
reason_budget_off diagnostic
reason_delta_off/primary
```

输出每个 branch：

```text
Act_mF1/mAP
Exp_mF1/mAP
delta vs final
per-action AP/F1
per-reason AP/F1
wrong flips
recovered flips
```

这些是诊断，不参与 best selection。

---

# 29. 每 epoch artifact schema

根目录：

```text
run_manifest.json
config_resolved.yaml
implementation_fingerprint.json
split_manifest.json
metrics_summary.jsonl
loss_components.jsonl
mechanism_stats.jsonl
counterfactual_summary.jsonl
dual_state.jsonl
ecpo_summary.jsonl
runtime_stats.jsonl
best_checkpoints.json
```

每轮 `epoch_XXX/`：

```text
metrics_primary_raw.json
metrics_final_raw.json
metrics_primary_deploy.json
metrics_final_deploy.json
per_action_metrics.json
per_reason_metrics.json
calibration.json
mechanism_summary.json
predicate_mixture_stats.json
atom_transport_stats.json
contribution_stats.json
counterfactual_certificate.json
dual_constraints.json
reason_budget_stats.json
ecpo_stats.json
naming_stats.json
structured_coverage.json
branch_audit_128.json
test_outputs.pt
fixed_audit_outputs.pt
checkpoint_pre_eval.pth
```

`test_outputs.pt` 至少保存：

```text
file_names
targets
primary/final logits
thresholds
action delta
reason delta
reason budget
```

大 map/token 只保存固定 128 audit，不能保存全 test 造成磁盘爆炸。

---

# 30. 直接可判定组件价值的规则

每轮 artifact 必须自动生成 `component_diagnosis.json`：

| 组件 | 有效判据 | 无效/有害判据 |
|---|---|---|
| predicate prior | off 后 Action/CF下降且mix非坍缩 | off不变或更好，fallback/单predicate饱和 |
| local reread | off 后Action/CF下降，local/global ratio合理 | off不变；offset全0/全边界 |
| atom transport | off 后Action/CF下降，map/token均变化 | 只token变；off更好；gamma饱和 |
| background center | off 后CF specificity变差 | off更好；centered token近0 |
| contribution | final优于primary且cert相关 | residual大但cert无关 |
| CF/dual | certificate正、lambda稳定、late delta受控 | cert负/高方差、lambda饱和 |
| signed Reason | unsigned legacy更差 | unsigned更好或support/inhibit相同 |
| ECPO | pair覆盖非零、Exp mAP提升 | pair少/陈旧、final ranking下降 |
| dynamic budget | late delta稳定且budget有分布 | 全min/全max、delta仍接近2 |
| naming | precision/coverage达到门槛 | coverage 0或precision低，只能称unnamed |

该 JSON 只做基于已记录数据的规则诊断，不允许凭单个 loss 值宣称组件有效。

---

# 31. Stop / failure 规则

立即停止并记录：

```text
NaN/Inf
DINO grad !=0
ordinary DINO calls !=1
CF reruns DINO
map/token transport matrix不一致
contribution bias出现
reconstruction error >1e-5持续
reason->predicate/action gradient非零
naming->evidence gradient非零
queue age >64被采样
dual lambda NaN或全部max
reserved >44.5 GB
steady memory growth >0.25 GB
test用于threshold拟合
val被评估/选best
cache/compression启用
worktree/HEAD/config hash变化
```

不因单轮弱指标停止正式 16 epoch full run；保存 best并继续，除非发生上述实现/运行异常。

---

# 32. Requirement Matrix

审查器必须生成：

```text
.review/aie_cert_oia_v1/AIE_CERT_REQUIREMENT_MATRIX.json
```

至少包含以下 requirement IDs：

```text
C01 isolated_worktree
C02 source_head_exact
C03 direct_image_frozen_dino
C04 primary_forward_equivalence
C05 primary_final_gradient_isolation
C06 reason_to_predicate_action_firewall
C07 shared_predicate_key_identity
C08 sparse_arithmetic_predicate_mixture
C09 visual_fallback
C10 global_inquiry
C11 evidence_conditioned_local_reread
C12 map_token_cotransport
C13 overlap_ceiling
C14 same_region_background_center
C15 bias_free_exact_contribution
C16 multi_control_counterfactual
C17 robust_certificate
C18 primal_dual_constraints
C19 signed_action_reason_priors
C20 signed_predicate_reason_priors
C21 dynamic_reason_budget
C22 ecpo_primary_reference
C23 queue_age_balance_resume
C24 readonly_naming
C25 continuous_schedule
C26 full_train_diagnostics
C27 single_dino_test_eval
C28 calibration_guard
C29 checkpoint_resume
C30 runtime_profile
C31 pilot_gate
C32 github_sync
```

每项必须记录：

```text
implementation symbol
source file
static test
dynamic test
runtime artifact key
status
evidence
```

任何 status 非 PASS，不得生成 REVIEW_PASS。

---

# 33. 建议 commit 顺序

```text
Commit 1: Add AIE-CERT plan, skill, config skeleton and isolated entrypoints
Commit 2: Implement clean foundation, structured evidence and sparse predicate bank
Commit 3: Implement evidence-conditioned reread, atom co-transport and bias-free contribution
Commit 4: Implement multi-control certificate and primal-dual constraints
Commit 5: Implement signed Reason reread, dynamic budget, ECPO and read-only naming
Commit 6: Implement trainer, evaluator, diagnostics, artifacts and resume
Commit 7: Add targeted tests, static contracts, dynamic audit, profile and scripts
Commit 8: Fix all audit findings; generate code-only clean REVIEW_PASS-bound HEAD
```

每个 commit 后 push 并核对 GitHub HEAD。full train 前最后一次代码 commit 后：

```text
worktree clean
local HEAD == GitHub HEAD
review/profile/pilot全部重新绑定最终HEAD
```

---

# 34. Codex 最终执行顺序

Codex 必须严格按以下顺序，不得跳步：

```text
1. 读取canonical文件
2. 核验源HEAD和原worktree只读状态
3. 新建目标worktree/branch并push
4. 将本计划和Skill放入目标worktree
5. 通读当前AIE源代码
6. 建立Requirement Matrix骨架
7. TDD实现所有新formal模块
8. 完成trainer/evaluator/artifact/resume
9. 运行一键preflight
10. 对所有REVIEW_FAIL逐项修复
11. commit/push最终code-only HEAD
12. 在最终HEAD重新preflight和runtime profile
13. 运行唯一3-epoch pilot
14. pilot pass后生成FULL_TRAIN_READY
15. 运行16-epoch full
16. 验证所有epoch/checkpoint/artifact
17. 写GOAL_COMPLETED_AIE_CERT_OIA_V1.json
18. 将最终结果和边界追加到canonical文件
```

---

# 35. 最终验收

实现完成不等于文件存在。Codex 只有在以下全部成立时才可报告“代码功能完整”：

```text
formal import graph只进入AIE-CERT模块
32项Requirement Matrix全部PASS
真实DINO audit通过
所有gradient firewall通过
map-token co-transport动态通过
contribution无bias且精确重建
4-control certificate动态通过
dual状态更新/保存/resume通过
signed Reason priors动态不同
ECPO真实pair和age queue激活
naming read-only通过
single-DINO evaluator通过
训练日志和epoch artifact完整
runtime profile通过
pilot通过
GitHub同步
```

数值目标只有 full run 后才能判定。正式报告必须同时给出：

```text
best deploy-joint checkpoint
Act_mF1/oF1/mAP
Exp_mF1/oF1/mAP
joint
primary-vs-final delta
late-epoch drift
counterfactual certificate
component diagnosis
naming coverage/precision boundary
```

不得仅报告最好单项而隐藏同一 checkpoint 的另一任务指标。
