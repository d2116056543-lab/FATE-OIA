# Codex_CALI-Flow++_CurrentBranch_Implementation_Plan_20260627

适用仓库：`https://github.com/d2116056543-lab/FATE-OIA`  
目标分支：`acpr_interactflow_pp_v1`  
目标 worktree：`E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree`  
目标方法：`CALI-Flow++: Calibrated Contribution-Aligned Latent Interaction Flow for PSI`  
正式入口：`python -m fate_oia.engine.train_acpr_interactflow_psi`  
正式配置：`configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml`  
重要用户约束：

```text
1. 不新建 worktree；直接在当前 acpr_interactflow_pp_v1 worktree 上改。
2. 不生成 feature cache / token cache / logit cache；直接图片训练。
3. 每轮结束只评估 test；best 也用 test 选择。
4. RTX 5880 / 48GB，目标实际 peak reserved 42–45GiB，硬上限 46GiB。
5. 当前 branch 可复用，但不为复用而保留不完整逻辑；必要时推翻重写。
6. 代码必须先通过严格审查 skill，再允许 full train。
7. 代码、配置、测试、脚本必须同步到 GitHub branch `acpr_interactflow_pp_v1`；不得提交 `.background_runs`、checkpoint、logits、dataset、大文件。
```

---

## 0. 本计划的定位

这个文件不是论文文字方案，而是给 Codex 直接执行的代码级合同。Codex 必须把 CALI-Flow++ 落成可运行、可审计、可训练的代码，不允许只创建文件、只写 placeholder、只在日志里声明功能存在。

CALI-Flow++ 的主链路必须是：

```text
15-frame observed clip
→ event-aware visual evidence budget
→ dynamic action-inducing predicate trajectories
→ traffic interaction state grammar
→ response-lag aligned state factors
→ benefit-gated exact decision ledger
→ contribution-grounded Exp29 weak explanation
→ train-only calibrated deploy path
→ intervention-verified model-level counterfactual dependence
```

不能退化为：

```text
DINO token + BERT text + generic multi-label head + several auxiliary losses
```

---

## 1. 当前 branch 事实与必须修复的结构冲突

当前 branch 已有 formal namespace、engine、configs、tests 和前期 audit/preflight 框架。Codex 不能从空目录实现，也不能照旧保留不完整逻辑。必须针对下面冲突修正：

### 1.1 已存在但必须强化的目录

当前 GitHub branch 已有：

```text
fate_oia/acpr_interactflow/
fate_oia/engine/train_acpr_interactflow_psi.py
fate_oia/engine/eval_acpr_interactflow_psi.py
fate_oia/engine/audit_acpr_interactflow.py
fate_oia/engine/profile_acpr_interactflow.py
fate_oia/engine/run_acpr_interactflow_preflight.py
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml
tests/acpr_interactflow/
scripts/FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1
```

Codex 可以复用这些入口，但必须替换内部逻辑以满足本计划和审查 skill。

### 1.2 当前必须修复的核心问题

1. `visual_encoder.py` 当前强制把 DINO 输入插值到 `360×640` 并固定 `45×80=3600` patch token。正式 YAML 写的是 `320×576`，这会导致配置与实际计算不一致。必须改成：
   - DINO high-res size 由 YAML 决定；
   - grid size 根据 patch size 动态计算；
   - artifact 记录实际 `dino_input_h/w`、`grid_h/w`、`anchor_count`、`dino_chunk_size`；
   - 不允许 hard-code 360×640 / 45×80。

2. `Exp29Head` 当前只是 `factor_tokens + predicate_tokens` 的 label-query cross-attention。必须改成 contribution-grounded Exp29：
   - 读取 `ledger.gated_state_contributions`；
   - 读取 normalized signed/magnitude contribution；
   - 读取 cluster→state prior 与 reliability；
   - 输出 `cluster_attention_to_factors [B,29,F]`；
   - `ExpCal` 作为主 fixed-threshold deploy path，`ExpRaw` 作为 ranking/audit path。

3. `losses/acpr_interactflow_losses.py` 当前 `predicate_nnpu` 实际监督 `output.exp29_logits`。必须拆成：
   - `predicate_pu_loss`: 监督 dynamic predicate trajectory / bag logits；
   - `exp29_pu_loss`: 监督 Exp29 weak clusters；
   - 日志字段不能再让 predicate path 与 Exp29 path 混名。

4. `non_degradation_hinge_loss` 当前使用 hard CE。必须改成 soft-target KL：
   - PSI loader 已有 `action_soft_target`；
   - `L_safe = ReLU(KL_soft(final) - stopgrad(KL_soft(global)) + margin)`；
   - 不允许把 majority action CE 当作 action 主监督。

5. `DecisionLedgerHead` 当前默认 `num_actions=4`。PSI formal action_dim 必须为 3：
   - 默认值改为 3；
   - constructor 加 `assert num_actions == 3`；
   - 所有 tests 覆盖独立实例化。

6. 当前 contribution alignment 是 action-component-level JS，不是 Exp29 attention ↔ factor contribution alignment。必须重写为：
   - `q_f = normalize(sum_c |α_f Δz_f,c|)`；
   - `JS(A_exp[k,f], q_f)`；
   - 只在 positive/reliable Exp29 行上计算；
   - unknown all-zero 行不参与 alignment。

7. 当前 full train 曾经出现 epoch0 Exp fixed-threshold F1=0，但 Exp_mAP 非零。必须通过代码保证：
   - all-zero unknown 不作为 29 个 hard negative；
   - calibrated logits 进入 train loss 和 primary eval；
   - 训练时用 train-only positive-rate / cardinality calibration；
   - eval 输出 raw/cal/diagnostic 三视图。

8. 训练慢的根因是 DINO/model forward，不是 DataLoader。必须新增 timing instrumentation：
   - data_gap / H2D / DINO / motion / predicate / interaction / ledger_exp / backward / optimizer / eval / artifact_write；
   - 每 200 step 和每 epoch 写入；
   - profile 用真实 direct-image + full losses。

---

## 2. 代码修改总览

Codex 必须在当前 worktree 直接修改或新增以下文件。允许重写已有模块，但必须保留正式入口名。

### 2.1 必须修改

```text
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml
configs/acpr_interactflow_predicates.yaml
configs/acpr_interactflow_state_grammar.yaml
configs/acpr_interactflow_text_rules.yaml

fate_oia/acpr_interactflow/types.py
fate_oia/acpr_interactflow/model.py
fate_oia/acpr_interactflow/visual_encoder.py
fate_oia/acpr_interactflow/motion_path.py
fate_oia/acpr_interactflow/predicate_transfer.py
fate_oia/acpr_interactflow/dynamic_predicate_field.py
fate_oia/acpr_interactflow/interaction_flow.py
fate_oia/acpr_interactflow/response_lag.py
fate_oia/acpr_interactflow/decision_ledger.py
fate_oia/acpr_interactflow/exp29_head.py
fate_oia/acpr_interactflow/cluster_semantics.py
fate_oia/acpr_interactflow/interventions.py
fate_oia/acpr_interactflow/psi_metrics.py
fate_oia/acpr_interactflow/artifacts.py

fate_oia/losses/acpr_interactflow_losses.py

fate_oia/engine/train_acpr_interactflow_psi.py
fate_oia/engine/eval_acpr_interactflow_psi.py
fate_oia/engine/profile_acpr_interactflow.py
fate_oia/engine/run_acpr_interactflow_preflight.py
fate_oia/engine/audit_acpr_interactflow.py
fate_oia/engine/supervise_acpr_interactflow_foreground.py
fate_oia/engine/export_acpr_interactflow_visuals.py
fate_oia/engine/build_acpr_interactflow_atlas.py

fate_oia/explain/acpr_interactflow_renderer.py
fate_oia/explain/acpr_interactflow_atlas.py
fate_oia/explain/acpr_interactflow_faithfulness.py

scripts/FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1
```

### 2.2 必须新增

```text
fate_oia/acpr_interactflow/timing.py
fate_oia/acpr_interactflow/calibrated_exp29.py
fate_oia/acpr_interactflow/traffic_event_budget.py
fate_oia/acpr_interactflow/reliability.py
fate_oia/engine/audit_califlowpp_current_branch.py

tests/acpr_interactflow/test_califlowpp_visual_budget.py
tests/acpr_interactflow/test_califlowpp_exp29_ledger_grounding.py
tests/acpr_interactflow/test_califlowpp_exp29_calibration.py
tests/acpr_interactflow/test_califlowpp_predicate_pu_split.py
tests/acpr_interactflow/test_califlowpp_soft_kl_safety.py
tests/acpr_interactflow/test_califlowpp_benefit_gate_advantage.py
tests/acpr_interactflow/test_califlowpp_config_runtime_consumption.py
tests/acpr_interactflow/test_califlowpp_timing_profile.py
tests/acpr_interactflow/test_califlowpp_no_cache_test_only_best.py

docs/runbooks/Codex_CALI_FlowPP_CurrentBranch_Implementation_Plan.md
.codex/skills/cali-flowpp-current-branch-audit/SKILL.md
```

如仓库政策不允许提交 `.codex/skills`，则复制 skill 到用户级：

```text
C:\Users\Lenovo\.codex\skills\cali-flowpp-current-branch-audit\SKILL.md
```

并在仓库内仅保留 `docs/runbooks/...` 说明。

---

## 3. 数据与标签层

### 3.1 Loader 合同

`PSIDAMO11902Dataset` 必须输出：

```python
frames: Tensor[B,15,3,H,W]
action_soft_target: Tensor[B,3]
action_hard: Tensor[B]
exp29_target: Tensor[B,29]
exp29_mask: Tensor[B,29]
sample_weight: Tensor[B]
paper_effective_weight: Tensor[B]
input_frame_indices: Tensor[B,15]
target_frame_index: Tensor[B]
video_ids: list[str]
sample_ids: list[str]
raw_explanation_text: list[str]
raw_reasoning_text: list[str]
frame_paths: list[list[str]]
target_frame_path: list[str]  # metadata only; never image-loaded for model input
```

### 3.2 Exp29 mask 必须强制改为三值语义

在 loader 或 batch-building 中执行：

```python
positive_mask = exp29_target > 0.5
known_reliable_negative_mask = parse_reliable_negatives_from_text_rules(...)
row_has_positive = positive_mask.any(dim=1, keepdim=True)

# all-zero rows 默认 unknown，不允许当 29 个 hard negatives
exp29_mask = positive_mask | known_reliable_negative_mask

# 若当前数据包已有原始 exp29_mask，则只能与上述 reliable mask 取交/并时保留 unknown 语义，
# 禁止 all-zero row -> mask全1。
```

审查必须检查：

```text
test all-zero row: exp29_target.sum()==0 → exp29_mask.sum()==0 unless reliable_negative_mask has true entries
```

### 3.3 Split 与 formal source

正式训练只使用：

```text
train: 8873
test: 2417
```

Val 只允许 data audit，不允许 formal epoch eval 或 best selection。`target_frame` 只允许作为 label alignment 元数据，不允许读取 target image 进模型。

---

## 4. 视觉证据预算层

### 4.1 VisualEncoder 重写目标

`InteractVisualEncoder` 应支持两个模式：

```yaml
model:
  visual_encoder:
    mode: fixed_anchor              # formal default
    anchor_frames: [0,3,6,9,12,14]
    dino_input_height: 320
    dino_input_width: 576
    patch_size: 8
    dino_chunk_size: 6
    selected_layers: [3,7,11]
    event_budget_enabled: false     # ablation / after-profile optional
```

并提供可审计输出：

```python
InteractVisualOutput(
    anchor_indices: list[int],
    anchor_tokens: Tensor[B,A,D],
    patch_tokens_by_layer: Tensor[B,A,L,N,D],
    cls_tokens: Tensor[B,A,L,D],
    fast_motion_tokens: Tensor[B,15,D],
    lowres_motion_maps: Tensor[B,15,Hm,Wm,D],
    eventness: Tensor[B,15],
    grid_hw: tuple[int,int],
    stats: dict
)
```

### 4.2 不允许硬编码

禁止：

```python
if flat.shape[-2:] != (360, 640): ...
reshape(..., 3600, ...)
grid_hw=(45,80)
```

必须改成：

```python
dino_h = cfg.model.visual_encoder.dino_input_height
dino_w = cfg.model.visual_encoder.dino_input_width
grid_h = dino_h // patch_size
grid_w = dino_w // patch_size
num_patches = grid_h * grid_w
```

### 4.3 Event-aware budget

实现 `traffic_event_budget.py`，但 formal default 先不启用，作为 profile 后可开关：

```python
event_t = w1*frame_delta_t + w2*corridor_change_t + w3*lowres_conflict_t + w4*motion_uncertainty_t
anchors = {0, 14} ∪ topK(event_t)
```

要求：

```text
- deterministic
- 只选 observed frames
- 不缓存 feature/token/logit
- train/eval 同策略
- 每个 batch 写 selected anchors 统计
```

---

## 5. 动态谓词层

### 5.1 Predicate ontology

总谓词数：

```text
OIA base: 32
PSI-specific: 16
total: 48
```

PSI-specific 必须至少包含：

```text
pedestrian_waiting
pedestrian_approaching_ego_path
pedestrian_entering_ego_lane
pedestrian_crossing_ego_path
pedestrian_moving_away
pedestrian_group
pedestrian_on_curb
pedestrian_in_roadway
pedestrian_looking_towards_ego
crosswalk_conflict
side_occlusion_risk
ego_path_clear
vehicle_yielding_ahead
vehicle_passing_pedestrian
traffic_signal_constraint
intersection_constraint
```

### 5.2 Predicate transfer

`predicate_transfer.py` 必须输出完整 transfer report：

```json
{
  "source_loaded": true,
  "source_checkpoint_path": "...",
  "source_checkpoint_sha256": "...",
  "source_tensor_key": "predicate_head.predicate_queries",
  "source_shape": [32,384],
  "mapped_shape": [48,384],
  "oia_name_order_verified": true,
  "loaded_predicate_names": [...32 names...],
  "text_embedding_source": "transformers_frozen",
  "text_encoder_model": "E:/sbw/hf_cache/...",
  "fallback_used": false
}
```

禁止：

```text
hash pseudo embedding
BoW fallback as formal result
source checkpoint unresolved
OIA 32 name order mismatch
```

### 5.3 DynamicPredicateField

`DynamicPredicateField.forward()` 必须从 high-res anchor tokens + lowres all15 motion 得到：

```python
predicate_logits_trajectory: Tensor[B,15,48]
predicate_probs_trajectory: Tensor[B,15,48]
predicate_tokens_trajectory: Tensor[B,15,48,D]
predicate_evidence_maps: Tensor[B,A,48,Hg,Wg]
predicate_confidence: Tensor[B,15,48]
predicate_centroid: Tensor[B,15,48,2]
predicate_relative_motion: Tensor[B,14,48,2]
predicate_corridor_mass: Tensor[B,15,48,4]
predicate_temporal_stats: dict
transfer_gate: Tensor[48]
```

非 anchor 帧不能简单复制平均 token，必须用 lowres motion + temporal GRU/TCN 更新：

```python
h_t,k = GRU(h_t-1,k, [anchor_or_proxy_evidence, motion_t, centroid_delta, corridor_mass])
```

---

## 6. 交通流状态层

### 6.1 State grammar

`configs/acpr_interactflow_state_grammar.yaml` 必须定义：

```text
Regime: clear_to_go, caution_required, yielding_required, stop_required
Phase: waiting, approaching, entering, crossing, leaving, uncertain
Source: pedestrian_conflict, crosswalk_context, traffic_signal, front_vehicle_yielding, side_occlusion, intersection_constraint
Corridor: left_sidewalk_zone, center_ego_path, right_sidewalk_zone, crosswalk_zone
```

决策 factor 默认：

```text
F = regime + phase + source = 16
```

Corridor 是 support token，不直接作为 final action factor，除非 config 明确启用 ablation。

### 6.2 InteractionFlowReasoner

当前 `InteractionFlowReasoner` 主要从 `[B,P,D]` predicate token attend 得到 factor，必须改为从 `[B,T,P,D]` trajectory 建模：

```python
factor_tokens_t = entmax_attend(
    factor_query,
    predicate_tokens_trajectory[t],
    predicate_probs_trajectory[t],
    corridor_mass[t],
    motion_tokens[t],
    grammar_priors
)  # [B,T,F,D]
```

输出：

```python
factor_tokens: Tensor[B,15,F,D]
factor_logits: Tensor[B,15,F]
factor_probs: Tensor[B,15,F]
factor_to_predicate: Tensor[B,15,F,P]
factor_to_corridor: Tensor[B,15,F,4]
factor_evidence_maps: Tensor[B,A,F,Hg,Wg]
lineage: dict/list
state_group_logits: Tensor[B,G]
```

### 6.3 Weak state targets

构造 detached weak targets：

```text
pedestrian_entering_ego_lane / crossing_ego_path + center_ego_path high
  → yielding_required / stop_required / pedestrian_conflict

pedestrian_waiting + ego_path_clear
  → clear_to_go / waiting

approaching + side_occlusion
  → caution_required / uncertain / side_occlusion

traffic_signal_constraint
  → traffic_signal source

vehicle_yielding_ahead
  → front_vehicle_yielding
```

这只是 weak regularizer，不是 hard state label。审查必须验证 state branch 不是 zero placeholder，也不是 action majority 的简单 one-hot 映射。

---

## 7. Response-lag 层

`response_lag.py` 必须改成 per-factor lag：

```python
lag_weights: Tensor[B,F,5]
lag_aligned_tokens: Tensor[B,F,D]
```

公式：

```python
λ_f,l = softmax_l(q_decision^T W_l s_{t=14-l,f})
s_tilde_f = Σ_l λ_f,l s_{14-l,f}
```

要求：

```text
- 只使用 observed frames 0..14；
- lags=[0,1,2,3,4]；
- lag_disabled intervention 将 λ 强制为 lag0 或 uniform，并且必须改变 downstream action prob；
- temporal_reverse 重新跑 visual/motion/predicate/flow/lag，而不是只翻转显示 tensor；
- synthetic delayed-event test 必须能恢复已知 lag。
```

---

## 8. Exact Decision Ledger 层

### 8.1 结构

`DecisionLedgerHead` 必须为 3 类 action：

```python
assert num_actions == 3
```

输入：

```python
global_visual_context: Tensor[B,D]
motion_context: Tensor[B,D]
predicate_context: Tensor[B,D]
lag_aligned_factor_tokens: Tensor[B,F,D]
factor_confidence: Tensor[B,F]
```

输出：

```python
global_logits: Tensor[B,3]
raw_state_contributions: Tensor[B,F,3]
benefit_gate: Tensor[B,F,1] or [B,F,3]
gated_state_contributions: Tensor[B,F,3]
flow_delta_logits: Tensor[B,3]
calibration_delta: Tensor[B,3]
final_logits: Tensor[B,3]
identity_error: scalar
benefit_target: Tensor[B,F,1] optional
contribution_attention: Tensor[B,F]
```

强制 identity：

```python
final_logits = global_logits + gated_state_contributions.sum(dim=1) + calibration_delta
```

### 8.2 Benefit target

实现 detached advantage：

```python
kl_global = soft_kl(global_logits, action_soft_target)
kl_candidate_f = soft_kl(global_logits + raw_state_contributions[:,f], action_soft_target)
adv_f = kl_global - kl_candidate_f
benefit_target_f = sigmoid(adv_f / temperature).detach()
L_benefit = BCEWithLogits(gate_logit_f, benefit_target_f)
```

注意：

```text
- 不允许 gate 永远开/关；
- gate_mean 必须被记录；
- gate_target_mean 与 gate_mean 都要输出；
- benefit target 不反传到 raw contribution。
```

### 8.3 Non-degradation

必须用 soft KL：

```python
L_safe = relu(KL_soft(final, y_soft) - stopgrad(KL_soft(global, y_soft)) + margin)
```

禁止：

```python
F.cross_entropy(final_logits, action_majority)
```

作为 safety hinge 主逻辑。

---

## 9. Exp29 弱文本解释层

### 9.1 Cluster semantics

`cluster_semantics.py` / `reliability.py` 必须从 Exp29 embedding json/pkl 建立：

```python
cluster_id
medoid_text
top_phrases
bert_embedding
support_count
phrase_concentration
action_prior: Tensor[29,3]
state_prior: Tensor[29,F]
predicate_prior: Tensor[29,P]
cluster_reliability: Tensor[29]
```

Reliability 公式可以实现为可审计启发式：

```python
r_k = sigmoid(
    a*log_support_count
  + b*phrase_concentration
  + c*text_predicate_agreement
  + d*action_compatibility
  - e*contradiction_rate
)
```

若字段缺失，必须写 explicit unavailable reason，不允许 silently set all reliability=1 without report。

### 9.2 Exp29Head forward

替换当前 head：

```python
def forward(
    factor_tokens_lag: Tensor[B,F,D],
    predicate_tokens_summary: Tensor[B,P,D],
    gated_state_contributions: Tensor[B,F,3],
    global_decision_hidden: Tensor[B,D],
    action_logits: Tensor[B,3],
    exp29_mask: Tensor[B,29] | None = None,
) -> Exp29Output:
```

核心：

```python
contrib_mag_f = normalize(sum_c abs(gated_state_contributions[f,c]))
score_kf = q_k^T W_s s_f
         + β * log(contrib_mag_f + eps)
         + log(state_prior_kf + eps)
         + sample_reliability_bias
A_kf = softmax(score_kf, dim=factor)

label_token_k = Σ_f A_kf * concat/project(s_f, gated_contribution_f, predicate_support_k, global_hidden)
raw_logit_k = MLP(label_token_k)
cal_logit_k = raw_logit_k - θ_k + δ_k(x)
```

Output：

```python
Exp29Output(
  logits_raw,
  logits_calibrated,
  probs_raw,
  probs_calibrated,
  cluster_attention_to_factors,
  cluster_reliability,
  cluster_to_state_prior,
  stats
)
```

### 9.3 Calibrated deploy path

Primary fixed-threshold eval 使用：

```python
probs_calibrated = sigmoid(logits_calibrated)
pred = probs_calibrated >= 0.5
```

同时输出：

```text
ExpRaw_mF1/oF1/mAP
ExpCal_mF1/oF1/mAP
ExpDiag_threshold_sweep_mF1  # diagnostic only
```

不得把 test-selected threshold 作为正式主结果。

### 9.4 Positive-rate/cardinality calibration

训练阶段增加 train-only batch/EMA loss：

```python
π_k_train = train split positive rate from known positive mask only
L_rate = Σ_k |mean_batch(sigmoid(logit_cal_k)) - clip(π_k_train, π_min, π_max)|
L_card = |mean_batch(sum_k p_cal_k) - mean_known_cardinality|
```

推荐：

```text
π_min = 0.03
π_max = 0.35
rate_loss_weight = 0.04
cardinality_loss_weight = 0.02
```

只用 train split 统计，不允许 test leakage。

---

## 10. Loss 函数重写

`compute_interactflow_losses()` 必须输出每项：

```text
raw_value
weight
weighted_value
finite
gradient_target_modules
```

### 10.1 Default weights

建议正式配置：

```yaml
loss:
  action_final_soft_kl: 1.00
  action_global_soft_kl: 0.40
  ledger_residual_soft_kl: 0.15
  non_degradation_soft_kl_hinge: 0.08
  benefit_gate_advantage_bce: 0.04

  predicate_pu: 0.08
  predicate_structural_weak: 0.03
  interaction_state_semantic: 0.04
  response_lag_sharpness: 0.010
  response_lag_temporal_consistency: 0.020
  temporal_consistency: 0.020

  exp29_raw_asl: 0.12
  exp29_calibrated_asl: 0.25
  exp29_pu: 0.04
  exp29_soft_f1: 0.08
  exp29_positive_rate: 0.04
  exp29_cardinality: 0.02
  exp29_pairwise_rank: 0.04
  exp29_ledger_alignment_js: 0.06

  gate_prior_noncollapse: 0.004
  factor_sparsity: 0.001
```

### 10.2 Loss ramp

避免 epoch0 gate/Exp29 全压缩：

```yaml
loss_schedule:
  ramp_epochs: 3
  start_scale:
    exp29_ledger_alignment_js: 0.30
    benefit_gate_advantage_bce: 0.30
    response_lag_sharpness: 0.50
  full_scale_epoch: 3
```

Action/global/ExpCal ASL/soft-F1 从 epoch0 生效。

### 10.3 禁止项

审查中硬卡：

```text
all-zero Exp29 as negative BCE
predicate_nnpu supervising exp29 logits
hard CE as sole action supervision
contribution magnitude minimized as residual
state probability mean-to-zero loss
pattern logit square-to-zero
test-threshold leakage
```

---

## 11. Model forward 总流程

`ACPRInteractFlowPPModel.forward(batch_or_frames, epoch, intervention=None)` 必须执行：

```python
visual = self.visual(frames, mode=config.visual_encoder.mode)
motion = self.motion(visual.fast_motion_tokens, visual.lowres_motion_maps)

predicates = self.predicates(
    visual.patch_tokens_by_layer,
    visual.anchor_indices,
    motion.tokens,
    raw_text=batch.raw_text if training weak labels needed else None
)

flow = self.flow(
    predicate_tokens_trajectory=predicates.tokens_trajectory,
    predicate_probs_trajectory=predicates.probs_trajectory,
    predicate_confidence=predicates.confidence,
    predicate_corridor_mass=predicates.corridor_mass,
    motion_tokens=motion.tokens,
    grammar=self.grammar,
    intervention=...
)

lag = self.response_lag(flow.factor_tokens_trajectory, motion.tokens, disabled=...)
ledger = self.ledger(
    global_visual_context=visual.global_token,
    motion_context=motion.global_token,
    predicate_context=predicates.global_context,
    lag_aligned_factor_tokens=lag.tokens,
    factor_confidence=flow.factor_confidence,
    action_soft_target=batch.action_soft_target if training else None
)

exp29 = self.exp29(
    factor_tokens_lag=lag.tokens,
    predicate_tokens_summary=predicates.summary_tokens,
    gated_state_contributions=ledger.gated_state_contributions,
    global_decision_hidden=ledger.global_hidden,
    action_logits=ledger.final_logits,
)
```

Intervention 规则：

```text
evidence_off       → visual evidence map/tokens affected; recompute predicate onward
predicate_off      → zero selected predicates; recompute flow/lag/ledger/exp29
factor_off         → zero selected factor tokens; recompute lag/ledger/exp29
lag_disabled       → recompute ledger/exp29 with disabled lag
temporal_reverse   → reverse observed frames; rerun full model
global_only        → bypass gated_state_contributions but keep global branch
```

不得只改展示 tensor。

---

## 12. 训练和超参数

### 12.1 Formal run

正式实验：

```text
epochs = 30
eval_splits = [test]
best_selector = test joint
metric early stop = false
cache = disabled
precision = BF16 autocast
optimizer = AdamW
scheduler = warmup 5% + cosine, min_lr_ratio=0.10
gradient_clip_norm = 1.0
```

### 12.2 推荐显存配置

RTX 5880 / 48GB：

```yaml
training_profile_selection:
  target_peak_reserved_gib: [42, 45]
  hard_peak_reserved_gib: 46
  primary:
    batch_size: 6
    gradient_accumulation_steps: 5
    effective_batch: 30
    dino_chunk_size: 6
    image_size: [320, 576]
    anchors: [0,3,6,9,12,14]
  preferred_if_profile_passes:
    batch_size: 8
    gradient_accumulation_steps: 4
    effective_batch: 32
    dino_chunk_size: 8
  fallback_1:
    batch_size: 5
    gradient_accumulation_steps: 6
    effective_batch: 30
    dino_chunk_size: 5
  fallback_2:
    batch_size: 4
    gradient_accumulation_steps: 8
    effective_batch: 32
    dino_chunk_size: 4
```

正式脚本默认先 profile，选择 highest stable samples/sec under 46GiB。不要为了“占满显存”加入 dummy tensor。若 batch=6 + chunk=6 peak 低于 40GiB，可以尝试 batch=8 + chunk=8；若 batch=8 稳定并 under 46GiB，采用 batch=8/accum=4。若 batch=8 OOM 或 step time 更差，回到 batch=6/accum=5。

### 12.3 Optimizer groups

```yaml
learning_rates:
  dino_backbone: 0.0          # frozen
  dino_adapter: 1.0e-5
  motion_path: 1.0e-4
  predicate_transfer: 8.0e-5
  dynamic_predicate: 1.0e-4
  interaction_flow: 1.0e-4
  response_lag: 7.5e-5
  decision_ledger: 1.0e-4
  exp29_head: 1.2e-4
  exp29_calibration: 5.0e-5
  cluster_reliability: 5.0e-5

weight_decay:
  default: 0.01
  dino_adapter: 0.05
  bias_norm_gate_calibration: 0.0
```

理由：
- PSI/SGDCL 类强基线使用小 batch 和较高 LR 的 SGD 方案；本分支是 frozen DINO + 多层 head/ledger 的 AdamW，主干冻结后 head LR 采用 `7.5e-5–1.2e-4` 更稳。
- 解释校准头只做边界移动，LR 不应高于主解释 head。
- benefit gate / calibration / norm / bias 不做 weight decay。

### 12.4 训练命令

Codex 完成实现、审查通过、review pass 刷新后，正式命令：

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1 `
  -Config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  -Epochs 30 `
  -BatchSize 6 `
  -GradAccum 5 `
  -DinoChunkSize 6 `
  -Device cuda `
  -RequireReviewPass
```

若 profile 证明 batch=8 稳定：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1 `
  -Config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  -Epochs 30 `
  -BatchSize 8 `
  -GradAccum 4 `
  -DinoChunkSize 8 `
  -Device cuda `
  -RequireReviewPass
```

---

## 13. Timing instrumentation

新增 `fate_oia/acpr_interactflow/timing.py`：

```python
class StepTimer:
    sections = [
      "data_gap", "h2d", "visual_dino", "visual_motion",
      "predicate", "interaction_flow", "response_lag",
      "decision_ledger", "exp29", "loss", "backward",
      "optimizer", "eval_forward", "artifact_write"
    ]
```

训练每 200 step 写：

```text
timing_train_steps.jsonl
```

每轮 eval 写：

```text
timing_eval_epoch.json
```

审查必须能看到：

```text
dino_time_fraction
forward_time_fraction
artifact_write_time
data_time_fraction
samples_per_second
peak_reserved_gib
```

---

## 14. Evaluation / artifacts

### 14.1 每轮 test eval

每个 epoch 必须写：

```text
epoch_XXX/action_metrics.json
epoch_XXX/exp29_metrics.json
epoch_XXX/exp29_raw_metrics.json
epoch_XXX/exp29_calibrated_metrics.json
epoch_XXX/exp29_diagnostic_threshold_sweep.json
epoch_XXX/joint_metrics.json
epoch_XXX/loss_components.jsonl
epoch_XXX/timing_epoch.json
epoch_XXX/gradient_norms.json
epoch_XXX/predicate_stats.json
epoch_XXX/cluster_reliability_stats.json
epoch_XXX/nnpu_calibration.json
epoch_XXX/interaction_state_stats.json
epoch_XXX/response_lag_stats.json
epoch_XXX/decision_ledger_stats.json
epoch_XXX/exp29_ledger_alignment_stats.json
epoch_XXX/lightweight_interaction_influence.json
epoch_XXX/predictions_action.jsonl
epoch_XXX/predictions_exp29.jsonl
epoch_XXX/fixed_case_intermediate_outputs.jsonl
```

Run root：

```text
run_manifest.json
config_resolved.yaml
git_provenance.json
psi_dataset_contract.json
damo_metric_parity.json
oia_transfer_report.json
optimizer_groups.json
throughput_profile.json
timing_summary.json
checkpoint_latest.pth
checkpoint_best_action.pth
checkpoint_best_exp.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
metrics_summary.jsonl
core_metrics_summary.jsonl
innovation_intermediate_metrics.jsonl
supervisor_live_status.json
supervisor_decisions.jsonl
run_complete.json
```

### 14.2 Best selection

```python
joint = 0.60*Act_mAcc + 0.25*Stop_F1 + 0.15*ExpCal_mF1
```

Tie-breakers：

```text
Act_mAcc
Stop_F1
ExpCal_mF1
lower action_soft_KL
```

同时保存：

```text
checkpoint_best_action.pth    by Act_mAcc
checkpoint_best_exp.pth       by ExpCal_mF1
checkpoint_best_joint.pth     by joint
checkpoint_best_test.pth      alias of best_joint
```

### 14.3 Exp29 reporting

主报告字段：

```text
Exp_mF1 = ExpCal_mF1
Exp_oF1 = ExpCal_oF1
Exp_mAP = ExpCal_mAP or raw ranking AP if calibrated is monotonic-only; must record both
```

必须同时输出：

```text
ExpRaw_mF1 / ExpRaw_oF1 / ExpRaw_mAP
ExpCal_mF1 / ExpCal_oF1 / ExpCal_mAP
ExpDiag_best_global_threshold
ExpDiag_best_per_label_threshold  # diagnostic only
```

---

## 15. Intervention / faithfulness

### 15.1 Lightweight every epoch

固定 256 test samples：

```text
full
global_only
top_state_off
factor_off
predicate_off
lag_disabled
temporal_reverse
```

输出：

```text
lightweight_interaction_influence.json
```

### 15.2 Full best checkpoint audit

对 `checkpoint_best_test.pth` 做 full test：

```text
global_only
regime_off
phase_off
source_off
individual_factor_off
predicate_off
evidence_tube_off
equal_mass_random_evidence_off, 5 seeds
temporal_reverse
temporal_shuffle
lag_disabled
last_frame_only
prefix_5
prefix_10
prefix_15
```

审查必须确认每个 intervention 是从 earliest affected layer 重新计算，不是改输出 logits 或显示 JSON。

---

## 16. Visualization

必须 tensor-linked：

```text
decision_ledger.json
decision_ledger.png
decision_waterfall.png
case_source.json
report.html
atlas.json
atlas.html
```

每个 case 包含：

```text
15-frame strip
selected DINO anchors
predicate evidence tubes
traffic-flow state ribbons
response-lag panel
exact decision waterfall
Exp29 cluster attention vs exact factor contribution
counterfactual twin: full/state-off/evidence-off/random/reverse
sample_id/video_id/frame_indices/checkpoint_sha/config_hash
```

禁止：

```text
manual boxes
fabricated effects
placeholder HTML
constant dummy tensor
```

---

## 17. 审查、测试、commit、push

### 17.1 本地实现后必须执行

```powershell
cd E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree

E:\Anaconda\envs\sbw39\python.exe -m py_compile `
  fate_oia\acpr_interactflow\*.py `
  fate_oia\engine\train_acpr_interactflow_psi.py `
  fate_oia\engine\eval_acpr_interactflow_psi.py `
  fate_oia\engine\audit_acpr_interactflow.py `
  fate_oia\engine\audit_califlowpp_current_branch.py `
  fate_oia\losses\acpr_interactflow_losses.py

E:\Anaconda\envs\sbw39\python.exe -m pytest tests\acpr_interactflow -q

E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_califlowpp_current_branch `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir .background_runs\cali_flowpp_current_branch_preflight `
  --device cuda
```

### 17.2 Preflight

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.run_acpr_interactflow_preflight `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir .background_runs\cali_flowpp_current_branch_preflight `
  --batch_size 6 `
  --gradient_accumulation_steps 5 `
  --dino_chunk_size 6 `
  --device cuda `
  --profile_batches 100
```

若 batch=8 profile pass，则审查记录 selected batch=8，否则 selected batch=6。

### 17.3 Review pass

只有 audit 脚本可以写：

```text
.background_runs\cali_flowpp_current_branch_preflight\REVIEW_PASS_CALI_FLOWPP_CURRENT_BRANCH.txt
```

必须绑定：

```json
{
  "pass": true,
  "git_head": "...",
  "github_remote_head": "...",
  "config_sha256": "...",
  "plan_sha256": "...",
  "audit_skill_sha256": "...",
  "selected_batch_size": 6,
  "selected_grad_accum": 5,
  "selected_dino_chunk_size": 6,
  "feature_cache_enabled": false,
  "token_cache_enabled": false,
  "logit_cache_enabled": false,
  "eval_splits": ["test"]
}
```

### 17.4 Commit / Push

```powershell
git status --short
git add configs fate_oia tests scripts docs .codex
git commit -m "Implement CALI-Flow++ ledger-grounded PSI interact flow"
git push github acpr_interactflow_pp_v1:acpr_interactflow_pp_v1
git ls-remote github refs/heads/acpr_interactflow_pp_v1
```

Review pass 必须在 commit/push 后重新生成，否则 stale。

---

## 18. Formal training 运行

正式训练前 supervisor 必须检查：

```text
worktree clean
local HEAD == GitHub branch HEAD
review pass git_head == local HEAD
cache flags false
eval_splits == [test]
selected batch profile exists
```

训练命令见第 12.4 节。

训练期间每 200 step stdout 至少打印：

```text
epoch step/total lr loss_total
Act train KL rolling if available
identity_error
predicate_positive_rate
ExpCal_pred_positive_rate@0.5
flow_delta_abs_mean
gate_mean / benefit_gate_mean
lag_argmax_mean
data_gap / dino_time / backward_time
gpu_reserved_gib
```

---

## 19. 验收信号

第一个 corrected epoch 后必须满足：

```text
identity_error == 0 or <1e-6
ExpCal_pred_positive_rate@0.5 > 0
ExpCal_mF1 > 0
ExpRaw_mAP not collapsed vs diagnostic baseline
predicate_positive_rate not constant zero
flow_delta_abs_mean > 0
benefit_gate_mean not saturated 0/1
lag_disabled changes action probabilities
state_off changes action probabilities
Exp attention ↔ ledger contribution rank correlation > random
timing fields non-missing
```

若不满足，不允许继续盲跑 30 epoch。必须停止、诊断、加 regression test、修复、重新 audit。

---

## 20. 完成定义

### 20.1 Implementation complete

只有满足下面全部条件才算代码实现完成：

```text
formal namespace exists
all config fields consumed or explicitly rejected
current branch clean and GitHub synced
direct-image no-cache proof exists
test-only eval and test-best enforced
DINO input size obeys config
OIA 32 transfer verified
BERT frozen text path verified
dynamic predicate trajectory active
predicate PU and Exp29 PU split
traffic-flow state grammar active
per-factor response lag active
exact decision ledger identity active
soft-target KL safety active
benefit advantage gate active
Exp29 reads ledger contributions
ExpRaw/ExpCal/ExpDiag three-view active
all-zero unknown mask active
positive-rate/cardinality calibration active
timing profile active
interventions recompute downstream
visualization tensor-linked
preflight A-K or new audit gates all pass
review pass bound to pushed HEAD
```

### 20.2 Experiment complete

```text
30 epochs completed
test eval after every epoch
best action/exp/joint/test checkpoints saved
best-test full intervention audit completed
Dynamic Interaction Decision Ledger cases exported
Atlas generated
run_complete.json written
psi_task_plan.md / psi_findings.md / psi_progress.md updated only for PSI
GitHub branch HEAD verified equal local HEAD
```

---

## 21. Hard failures

任何一项出现即禁止训练或禁止继续训练：

```text
new worktree created despite user instruction
feature/token/logit cache enabled
val eval or val best used in formal run
target_frame image enters model input
DINO size/grid hard-coded against config
all-zero Exp29 row treated as 29 negatives
ExpCal exists but not used in primary loss/eval
predicate_nnpu still supervises Exp29 logits
hard CE is action primary or non-degradation primary
DecisionLedgerHead default remains 4
ledger identity not exact
benefit gate has no detached advantage target
state path disconnected from decision
lag path configured but unused
Exp29 head does not read ledger contribution
contribution alignment not Exp attention ↔ ledger factor contribution
intervention only changes display/logit after the fact
timing profile missing DINO/forward/backward split
review pass stale
worktree dirty
local/GitHub SHA mismatch
```
