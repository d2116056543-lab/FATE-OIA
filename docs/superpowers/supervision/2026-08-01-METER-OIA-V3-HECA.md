# METER-OIA V3 / HECA 双代理监督记录

## 原始请求与边界

- 请求：严格依据 `METER_OIA_V2_TESA_Code_Audit_and_Result_Diagnosis_20260801.md` 与 `METER_OIA_V3_HECA_Final_Implementation_and_TrainingPlan_20260801.md`，在 V2/TESA 的干净提交 `10f3a277eae5cffa43602bb2eb9c1b209e86de5e` 上实现 V3/HECA。
- 质量标准：不仅可训练，还必须完整实现、真实调用、无占位、无部分实现、无逻辑冲突。
- 当前阶段：只做实现、测试、审计；未获全部 gate 证据前不启动 full train。
- 隔离分支：`acpr_meter_oia_v3_heca_direct_image`。
- 禁止项：graph/PMI/co-occurrence 强先验、RunC/cached-logit、test 外部 VLM、token compression、test threshold leakage、reason logits 控制 action、expert/router/selector/meta 路径。

## 适用技能

- `brainstorming`：用户给出的最终计划视为已批准设计，不再自由改方案。
- `executing-plans`：按冻结计划分批实现并设审查点。
- `test-driven-development`：先写 RED tests，再改生产代码。
- `dual-agent-supervision`：同强度监督代理审查计划与最终实现。
- `subagent-driven-development`：监督代理只审查，不直接污染主实现上下文。
- `using-git-worktrees`：V3 与 V2 隔离。
- `verification-before-completion`：完成声明前必须有新鲜测试证据。

## 功能覆盖矩阵

| ID | 计划功能 | 代码落点 | 必须证明的调用链/不变量 | 测试或 artifact | 状态 |
|---|---|---|---|---|---|
| F01 | V2 干净基座与禁用项 | worktree/config/audit | HEAD 固定；无 RunC/cache/compression/test leakage | clean-head audit | 已建立，待审计 |
| F02 | rank-16 shared/action-private/reason-private zero-init adapters | model/trunk | label self-attn 后接入；progress=0 与 V2 严格等价 | progress-zero、full-reason-init tests | 待实现 |
| F03 | typed factor semantic/spatial/state embeddings | `meter_signed_factors.py`, schema | 语义、空间、状态均参与 measurement；unknown 不作负类 | typed forward tests | 待实现 |
| F04 | 离线 ontology prototype | schema/loader/artifact | 训练与 test 只加载 `.pt`，不运行外部文本塔 | ontology identity test | 待实现 |
| F05 | softmax-to-entmax anchor + null + 5% exploration schedule | factor measurement | 20% updates 后 exploration 为 0；无 dense 假实现 | anchor loss/test/artifact | 待实现 |
| F06 | factor-specific observability tau | typed targets/artifacts | train_main beta-binomial group shrinkage alpha=20，clamp [0.05,0.95] | factor-specific-tau test + two artifacts | 待实现 |
| F07 | reliability 定义 | measurement | observability*(1-null)*(1-normalized state entropy) | per-factor metrics | 待实现 |
| F08 | normalized anchor loss | grounding losses | NLL/log(valid)+Dice；mask/unknown 正确 | normalized-anchor test | 待实现 |
| F09 | 删除 hard compatible-actions | schema/action credit | `action_owned` scalar；14/20=0、factor1=.5；其他 factor 可影响全部 action | no-hard-mask/left/green tests | 待实现 |
| F10 | learned soft entmax action allocation | semantic action | 不用 PMI；allocation 真实进入 contribution | state/action ablation metrics | 待实现 |
| F11 | selective gradient bridge | action credit/measurement | action->anchor=0；action->state ratio 1%-10%；action credit grad>0 | 3 个 autograd tests | 待实现 |
| F12 | state-conditioned value | signed factors/action | state effect embedding 同时改变 route/value；state-off 必须重算 | state-conditioned/state-off tests | 待实现 |
| F13 | 删除 admission gate | model/trainer/schema | 不再筛样本或产生 admission loss/gradient/artifact | admission-absent test | 待实现 |
| F14 | bounded additive final action | semantic action | EMA visual RMS kappa clamp [.1,1]；pilot gate 后 correction fraction=.25 | ablation/logit diagnostics | 待实现 |
| F15 | noisy reason global + private correction | reason decoder | global predictor 完整保留；zero-init rank16 private；14/20 correction=0 | init/progress tests | 待实现 |
| F16 | positive reason weight invariant | reason losses | observed positive 全局权重恒为1，不乘 factor reliability | global-positive test | 待实现 |
| F17 | noisy-zero trust + cross-view consistency | reason losses/trainer | EMA reason prob*positive-state*reliability*consistency；weight [.1,1]；仅审计通过后 PU soft-positive | noisy-zero/view tests | 待实现 |
| F18 | detached evidence reason correction | reason decoder | measurement detach；groundable only；RMS 8%-20%，max .5 | branch ablation/artifacts | 待实现 |
| F19 | action loss 精确配方 | action losses/trainer/config | 仅计划列出的 9 项；round-robin corruption；删除旧冲突项 | loss wiring/unit tests | 待实现 |
| F20 | reason loss 精确配方 | reason losses/trainer/config | 6 项权重准确，PU private | loss wiring tests | 待实现 |
| F21 | measurement loss 精确配方 | grounding losses/trainer/config | 6 项权重准确，measurement 不更新 foundation | loss/gradient tests | 待实现 |
| F22 | excess-risk shared balance | trainer | 只作用 shared adapter/trunk；action [.45,.70]、reason [.30,.55] | noisy-task test | 待实现 |
| F23 | adaptive foundation grad cap/LR schedule | trainer/resume state | cap=clip(2*EMA[g],.25,.75)；20% schedule；RMS/EMA guard；emergency cap=20 | resume/schedule tests | 待实现 |
| F24 | exact sample flow/gradient ownership | model/trainer | one DINO call；action/reason/measurement 所有权符合计划 | same-forward + autograd gates | 待实现 |
| F25 | config 固定值 | HECA YAML | 14 epochs、batch6/accum5、BF16、LR/loss/schedule 完整一致 | config audit | 待实现 |
| F26 | 每轮完整指标与 cheap counterfactual | eval/artifacts | action/reason/measurement/optimization；同一 DINO field 的 factor-off/state-uniform/reason-correction-off | schema validation | 待实现 |
| F27 | 21 项计划测试 | `tests/test_heca_*.py` | 每项先 RED 后 GREEN | targeted pytest | 待实现 |
| F28 | pilot gates A-G | audit/pilot scripts | 严格按阈值；不以弱早期 F1 随意停 | gate artifacts | 待实现 |
| F29 | 正式协议 | supervisor/train | pilot 全过后 fresh 14 epochs；publication eligibility 标记正确 | protocol tests/manifest | 待实现 |
| F30 | T01-T18 交付顺序 | implementation log | 不跳步，不用 REVIEW_PASS 替代代码审查 | final audit | 待实现 |
| F31 | B0-B5/Full 独立训练消融 | ablation manifests/scripts | pilot 后独立运行；不属于每轮 cheap diagnostics，也不属于 T13 实现步骤 | per-run manifest | 待实现 |

## 用户计划忠实度矩阵

| 阶段 | 要求 | 预期证据 | 状态 |
|---|---|---|---|
| T01 | freeze current V2 result | pinned HEAD/result manifest | 完成 |
| T02 | create V3 worktree | local/remote worktree + branch | 完成 |
| T03 | remove hard action-factor mask | schema/model tests | 待执行 |
| T04 | implement factor-specific tau | target builder + artifact + test | 待执行 |
| T05 | complete reason initialization | progress-zero/full-init tests | 待执行 |
| T06 | implement state-conditioned value | value/state-off tests | 待执行 |
| T07 | implement selective 5% gradient bridge | three autograd tests | 待执行 |
| T08 | remove admission gate | source scan + absence test | 待执行 |
| T09 | simplify action losses | exact loss wiring test | 待执行 |
| T10 | implement robust noisy-zero reason loss | positive invariant/noisy-zero tests | 待执行 |
| T11 | implement evidence-label consistency | mirror/light consistency test | 待执行 |
| T12 | shared/private adapters + excess-risk balance | adapter/gradient tests | 待执行 |
| T13 | add same-forward diagnostics | 单次 DINO field 的 cheap branch artifact tests | 待执行 |
| T14 | run unit/autograd tests | 21 HECA tests + regressions | 待执行 |
| T15 | real-DINO profile | one-call/memory/runtime artifact | 未执行 |
| T16 | 4-epoch pilot | 4096/1024/512/512 manifest | 未授权启动 |
| T17 | adversarial mechanism review | gates A-G + supervisor verdict | 待执行 |
| T18 | fresh 14-epoch full run | formal protocol/artifacts | 不在当前实现阶段启动 |

## 第一轮监督审查

- 状态：`CHANGES_REQUIRED`，未获实现许可。
- 审查员：Avicenna（同强度继承模型）。
- 主要阻断：bridge 分母、measurement 梯度边界、state-off 重算、pilot correction 时序、foundation LR、excess-risk 状态、PU gate、loss 唯一计入、cheap ablation 与 B0-B5 区分、Gate 阈值、泄漏边界、legacy 路径、artifact schema。

## 第二版不可简化执行契约

以下条款是对原计划中未写出实现细节处的确定化，不改变原方法目标。

### 梯度所有权

- `anchor_bridge = anchor.detach()`；action loss 对 anchor query、anchor key/value、anchor map 的梯度必须逐元素为 0。
- `state_bridge = state.detach() + 0.05*(state-state.detach())`，`global_bridge` 同理。
- bridge ratio 定义为：`||dL_action/d(raw_state_or_global_projection_output)|| / ||dL_action/d(credit_state_or_global_input)||`。两者在同一 forward、同一标量 action loss 上用 `autograd.grad` 测量；分别报告 state/global 和合并 L2 ratio，目标 `[0.01,0.10]`，理论中心约 0.05。不得用不同参数组梯度作分母。
- measurement loss 对 frozen DINO、CalAlign label trunk、shared adapter、action-private adapter、reason-private adapter 的梯度必须为 0；只更新 measurement query/projection/state/observability 参数。
- action loss 只允许 5% bridge 进入 measurement 的 state/global projection输出，不允许沿它们的输入 token 回流 DINO/CalAlign foundation。
- reason/PU loss 对 action credit 参数梯度为 0；PU 对 measurement 参数梯度为 0。

### State Ablation 与贡献守恒

- `state_uniform`：对每个 factor 仅在 valid states 上替换为均匀分布，随后重新计算 state bridge、state-conditioned factor key、soft allocation、每状态 value、factor value、contribution、bounded delta 和 final action。
- `state_off` 是 `state_uniform` 的兼容诊断别名，不得把 delta 直接清零、缩放或复用原 forward 的 value/contribution。
- anchor/global token 保持同一样本同一次 DINO encode 的值，确保差异只来自 state。
- 必须验证 `sum_r(action_factor_contribution)==pre_tanh_credit_sum`，且 bounded delta 严格等于 `kappa*tanh(sum/kappa)`。

### Schedule、Pilot 与 Resume

- `r_credit(progress)=clamp((progress-0.05)/0.15,0,1)`，按 optimizer update 计数，不按 dataloader batch。
- 4-epoch mechanism pilot 使用 `correction_fraction=0.20`；只有 Gate C 通过后，fresh full train 才使用 `0.25`。full train 不从 pilot checkpoint resume。
- foundation optimizer base LR 为 `1e-4`。scheduler multiplier：`0→0.5`（0%-5% updates）；`0.5→1.0`（5%-20%）仅当 action logit RMS<8 且 foundation grad EMA<5，否则保持0.5；20% 后保持已获准的倍率并进入 cosine decay。
- resume 必须恢复 optimizer update、scheduler multiplier/hold 状态、credit corruption phase、visual RMS EMA、foundation grad EMA、task loss floors、PU gate history；恢复前后下一 update 完全一致。
- pilot/fresh full 的 checkpoint 与 manifest 必须记录实际 correction fraction，禁止兼容 fallback 静默改值。

### Excess-Risk Balance

- 仅 action/reason 两个 shared loss 参与，measurement 排除。
- 每任务 floor 在首个 finite optimizer update 初始化；之后以 `ema_t=0.99*ema_t+0.01*L_t(detach)` 更新，`floor_t=min(floor_t,ema_t)`，epsilon=`1e-6`。
- `e_t=(L_t-floor_t)/(abs(floor_t)+eps)`，temperature=`1.0`；softmax 后只 clamp action weight 到 `[0.45,0.70]`，reason weight=`1-action`，因此二者严格和为1且 reason 自动处于 `[0.30,0.55]`。
- floor、EMA、weights、temperature 均写 artifact/checkpoint 并在 resume 精确恢复。

### PU Train-Audit Gate

- 最早 epoch 1 后检查；每个 label 至少 20 observed positives。
- gate 只读取 `train_audit`，禁止 test/val。每轮对每个 label 的 observed positives 以固定 `seed=20260801+label_id` 做分层抽样，隐藏 20%（至少4个、且保留至少16个 observed positives）；这些被隐藏样本只用于 audit target，不进入该轮训练 loss。真实 observed-zero 只作为未标记候选，不被假定为负类。
- audit score 使用当前 noisy-zero trust；在 `hidden positive + 等量按 action exact-vector 分层抽取的 observed-zero` 集合计算 AUPRC 与 prevalence，要求 `AUPRC-prevalence>=0.02`、cross-view consistency 中位数 `>=0.50`，连续两次 audit 通过。拆分索引与 SHA256 必须写 artifact，保证可复现且不被训练读取。
- 未通过时仅使用 noisy-zero reweight，不产生 soft positive；通过后 label-private PU lambda 在接下来的 `max(100, ceil(0.02*total_optimizer_updates))` 个 optimizer updates 从0线性 ramp 到最大0.10。
- artifact 逐 label 写 `sample_count/positive_count/prevalence/auprc/lift/view_consistency/pass_streak/active/lambda`。

### Reason Loss 的统一 Unknown 语义

- 所有 reason loss 共用同一个 `reason_supervision` 结构：`positive_mask=y_obs==1`、`unknown_mask=y_obs==0`、`negative_weight=w^-`、`soft_positive_weight`（仅 PU gate 后非零）。
- robust-ASL global/final：positive 项权重恒1；observed-zero negative 项乘 `w^-`；soft-positive 只来自已激活 PU label。
- reason rank：正例对 observed-zero 的 pair 权重使用对应 `w^-`，unknown 不得默认权重1。
- reason SoftF1：TP/FN 只用 observed positive 与已激活 soft positive；FP 项对 observed-zero 乘 `w^-`。
- correction-sign：observed positive 使用正向 margin；observed-zero 只以 `w^-` 参与 trusted-negative margin；unknown/低 trust 不得变成硬负类。
- 五条路径必须读取同一对象实例/同一 tensor SHA256，artifact 记录 consumer list，禁止各 loss 私自重算 mask/weight。

### Cross-View 的 One-DINO Contract

- 一个需要 consistency 的 micro-batch 先生成 paired view（mirror 或 light），将 `original` 与 `view` 在 batch 维拼为 `[2B,3,360,640]`，只调用一次 DINO extractor，随后按 batch 维拆分 field。
- 不需要 consistency 的 micro-batch仍只调用一次 `[B,...]`。任何 micro-batch `encode_call_count` 必须严格为1。
- mirror 同步置换 action/reason/factor/anchor/state targets；light view 不置换 label。consistency loss 只比较拆分后的同一调用 features/logits。

### Loss 唯一调用

- trainer 建立唯一 `loss_registry`，计划中的 action 9项、reason 6项、measurement 6项和 PU-private 每项只能注册一次、加权一次。
- 动态检查 `total_loss == sum(weighted_unique_terms)`；静态扫描和 runtime call counter 同时禁止 TwoWay BCE、anti-monopoly、near-boundary、admission loss，以及同一 batch 多于一个 identity corruption。
- identity corruption phase 为 `optimizer_update % 3`，不是 micro-batch step；accumulation 内相位固定。

### Ablation 与 Gate 精确定义

- 每轮 cheap same-forward ablation：`factor_off`、`state_uniform`、`reason_correction_off`，复用同一次 DINO field；不得把 B0-B5/Full 独立训练消融塞进每轮。
- B0-B5/Full 是 pilot/full 后独立实验矩阵，单独 manifest。
- Gate C 的 `factor-off must lower final` 定义为同一连续两 epoch 内 `factor_off Act_mAP <= final Act_mAP-0.001`。
- Gate D 原文解释为 `global Exp_mAP >= CalAlign Exp_mAP-0.003`；其他两项保持原阈值。
- Gate G 使用 `emergency_cap_rate<=0.01`；失败停止规则中的 `>5% updates` 是运行中强制停止上限，不与 pilot pass 门槛混用。

### 泄漏与 Legacy 路径

- test forward 的函数签名不接收 BDD100K weak targets、action/reason labels、threshold oracle 或样本文本；动态测试改变这些外部数据不得改变 logits。
- ontology text encoder 只能由离线导出命令调用；runtime 只加载带 SHA256、encoder id、prompt 和维度的 `.pt` manifest。
- teacher、LR、PU gate、schedule 只读 train_main/train_audit/train_calib 的许可数据；任何 test metric 不进入更新路径。
- V3 active config/model/trainer/eval/artifacts/checkpoint strict loader 全部不得出现或兼容 `compatible_actions`、`action_evidence_admission*`、legacy admission loss/state；V2 文件可保留历史实现，但 V3 import graph 必须不可达，且 strict V3 checkpoint 不接受这些 keys。

### 冻结 Artifact Schema

- `ontology_prototype_manifest.json`：encoder_id、prompt列表、tensor paths、shape、dtype、SHA256、offline_only。
- 原计划三件套均必须保留：`factor_source_statistics.json`（train_main per-factor/group counts/source coverage）、`factor_observability_tau.pt`（21维部署 tensor）、`factor_observability_tau_metadata.json`（source split、alpha=20、group tau、factor tau、clip、两个前置 artifact SHA256）；metadata 只补充，不替代前两者。
- `heca_gradient_ownership.jsonl`：update、各 owner grad norm、anchor exact-zero、state/global numerator/denominator/ratio、violations。
- `heca_loss_wiring.json`：term、owner、weight、call_count、weighted_value、总和误差、forbidden term scan。
- `heca_component_call_counters.json`：DINO、measurement、action credit、reason global/correction、各 ablation call counts。
- `heca_contribution_conservation.jsonl`：per-action sum、bounded delta、reconstruction error、kappa。
- `heca_schedule_state.json`：update/progress、LR multiplier/hold、credit ramp/fraction、EMA/floors/caps、corruption phase、PU gate state。
- `heca_ablation_manifest.json`：shared DINO field id、mode、recomputed nodes、forbidden shortcuts、metric keys。
- `HECA_GATE_A.json` 至 `HECA_GATE_G.json`：输入 artifact SHA256、阈值、逐项 measured/pass、overall pass、git HEAD。

### 新增防伪测试

- `test_heca_measurement_foundation_grad_zero.py`
- `test_heca_action_reason_independence.py`
- `test_heca_unknown_state_not_negative.py`
- `test_heca_tau_train_main_only.py`
- `test_heca_state_contribution_conservation.py`
- `test_heca_credit_ramp_and_pilot_fraction.py`
- `test_heca_loss_terms_added_once.py`
- `test_heca_no_runtime_text_encoder.py`
- `test_heca_no_weak_target_test_forward.py`
- `test_heca_ablation_reuses_single_dino_forward.py`
- `test_heca_resume_restores_floor_ema_caps_and_corruption_phase.py`
- `test_heca_legacy_paths_unreachable.py`

## 第二轮监督审查

- 状态：`CHANGES_REQUIRED`；其余第一轮 blocker 已通过，只剩 PU 构造、reason unknown 统一、cross-view one-DINO、tau 三件套、独立消融拆分。
- 实现许可：未批准。

## 基线验证

- 远端环境：`E:\Anaconda\envs\sbw39\python.exe`，PyTorch `2.5.1+cu121`。
- 基线相关测试：19 passed，1 failed。
- 既存失败：`test_delta_pairwise_ranking_rewards_sample_discrimination`，当前 good loss=`0.0032`，旧测试要求 `<1e-6`。HECA T09 将删除/重写旧 action loss wiring；此失败不得被隐藏，最终必须由新 loss 契约和回归测试共同转为 GREEN。

## 第三轮监督审查

- 状态：`APPROVED`。
- 审查结论：PU 构造、reason unknown 语义、one-DINO cross-view、tau 三件套、cheap/独立消融边界及原 T01-T18 顺序均已无歧义。
- 实现许可：允许开始 P0/RED tests 和 T03-T13 生产实现；此批准不代表功能已完成。
- 后续硬门槛：T14 前消除既存回归失败；T14-T17、Gate A-G、clean-HEAD 审查未通过前不得启动 full train。
