# 双代理监督日志：MOSAIC-TRUST v4 CREDO

**日期：** 2026-07-17
**任务：** 按 `MOSAIC_TRUST_v4_CREDO_RootCause_and_Full_ModificationPlan_20260717.md` 完整实现代码功能，先审查和修复，暂不启动 full train。
**状态：** 计划中 / 待监督审查
**基线：** `ce05e6235ffa8c8998e055941fcad2a91922a0b7`
**执行端：** 主会话
**监督端：** 待创建

## 1. 原始请求与约束

- 以 v4 CREDO 计划为源，不能只做“能训练”的外壳；每个功能必须真实实现并进入调用链。
- 不改变计划未涉及的逻辑。
- 先修复 hard certificate deadlock、语义错误、fine evidence、route ownership 和训练冲突，再做 revised pilot；没有验证证据不得启动 full train。
- 保留 direct visual action/reason 主路径、test-only 评估、train-calib threshold 边界、无 cache/无 token compression。

## 2. 适用 Skill

- `dual-agent-supervision`：建立覆盖矩阵、监督审查、执行合规和证据记录。
- `writing-plans`：保存逐任务实现与验证计划。
- `test-driven-development`：先写失败测试，再写生产代码。
- `verification-before-completion`：完成前必须有新鲜 compile/test/audit 证据。
- `using-git-worktrees`：保持基线和实验修改隔离。

## 3. 基线核验

本地 clone 和远端 worktree 均已核验为 clean，HEAD 为 `ce05e6235ffa8c8998e055941fcad2a91922a0b7`，远端 branch 为 `acpr_mosaic_trust_v3_icdor_direct_image`。旧 pilot 的 route 全部 abstained、certified=0，不能作为新代码的训练阻断条件。

## 4. 功能覆盖矩阵

| 编号 | 必须实现的功能 | 计划实现位置 | 验证方法 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | hard certificate 不再阻断 learning access | adaptive schedule / model route | route-off forward 仍产生 shadow/latent/factor loss；RED test | 待实现 |
| 2 | 连续视觉可信度 cV，不能读 reason labels | new credibility module / epoch EMA | reason shuffle 不改变 cV；caps/n_eff test | 待实现 |
| 3 | observable ontology 使用真实 BDD100K attributes | grounding + factor YAML | attribute, reliable-negative tests | 待实现 |
| 4 | no-lane 使用 absence polarity | reason routes / loss | polarity RED test | 待实现 |
| 5 | audit_visual/target 互斥 factor-aware split | sampler / artifacts | index disjointness + source coverage test | 待实现 |
| 6 | typed point/curve/region fine splat | new splat module | fine != coarse and coordinate shuffle tests | 待实现 |
| 7 | factor-seeded local re-reader | new rereader + model | offsets <= .08，shuffle changes target | 待实现 |
| 8 | observed visual reason direct primary | reason path/loss | visual/off/shuffle metrics and route output test | 待实现 |
| 9 | latent/annotation route epoch0 active | schedule/model | latent nonzero without cert; PU failure isolation | 待实现 |
| 10 | bounded reason residual/non-regression | model/reason audit | alpha init/cap and AP guard test | 待实现 |
| 11 | action shadow epoch0, final visual before admission | action/model/admission | exact equality <=1e-7 and shadow ratio tests | 待实现 |
| 12 | per-action partial edge admission | edge audit/schedule | one action admitted without all-four test | 待实现 |
| 13 | no HardPair duplication; matched-control budget | loss/trainer | loss ownership and budget test | 待实现 |
| 14 | owner-specific gradient firewall/clipping | trainer/optimizer | gradient forbidden-zero and clip tests | 待实现 |
| 15 | train-calib only threshold update | calibration/schedule | test isolation scan and artifact test | 待实现 |
| 16 | Regime A/B/C schedule | adaptive schedule | transition/state policy tests | 待实现 |
| 17 | batch-local DINO reuse and speed path | trainer/feature flow | call-count/perf artifact test | 待实现 |
| 18 | complete diagnostics/artifacts | trainer/audit/export | strict schema validation | 待实现 |

**覆盖结论：** 当前尚未允许执行；上述 18 项均需要源码和测试证据。

## 5. 用户计划保真矩阵

| 计划要求 | 是否保留 | 证据 |
| --- | --- | --- |
| 先修语义与 certificate deadlock，不盲训 | 保留 | 本次先只实现/测试 |
| cV 独立于 reason label，reason 只进入 semantic compatibility | 保留 | credibility API 与 route 调用审计 |
| audit_visual / audit_target 互斥 5%+5% | 保留 | split artifact + test |
| typed fine evidence + seeded rereader | 保留 | splat/rereader tests + forward artifact |
| observed reason direct primary，bounded annotation residual | 保留 | reason branch keys/loss tests |
| action shadow epoch0，visual-only final 到 partial admission | 保留 | exact equality/admission tests |
| no HardPair、owner gradient firewall、batch-local reuse | 保留 | loss/gradient/call-count tests |
| 6 epoch revised pilot 后才决定 full train | 保留 | pilot manifest 和 gate summary |

## 6. 监督审查

**监督端是否已创建：** 否，下一步创建并发送本矩阵与 v4 计划摘要。
**执行是否获准：** 否，必须先完成监督 review。

## 7. 当前风险

- 现有 `FOUNDATION` 通过 certificate 才打开 latent/shadow，存在确定性 deadlock。
- 现有 factor certificate/edge admission 将部署准入与学习访问混用。
- 现有 rereader 使用 coarse masks，尚未消费 sampling coordinates/features/attention。
- ontology 中有 traffic color、physical near、no-lane presence 等语义风险。
- 当前模型/调度器/训练器的 owner gradient 与 policy weight 是否真实消费，需逐行核验。

## 8. 执行交接

待监督端审查通过后，按 `docs/superpowers/plans/2026-07-17-mosaic-trust-v4-credo.md` 逐任务实现；每轮修订都回写本日志，禁止静默删减计划。
## 9. 本轮对抗式 plan/code coverage review

审查范围：`C:\Users\WLJTXY\Downloads\MOSAIC_TRUST_v4_CREDO_RootCause_and_Full_ModificationPlan_20260717.md`；监督基线：`ce05e6235ffa8c8998e055941fcad2a91922a0b7`。本轮不修改生产代码、不启动 smoke、不启动 full train。

### 状态与基线边界

- 远端 `E:\sbw\FATE_Drive\fate_oia_acpr_mosaic_trust_v3_icdor_worktree` HEAD 为 `ce05e6235ffa8c8998e055941fcad2a91922a0b7`，远端工作树 clean。
- 本地工作树存在已修改跟踪文件和未跟踪 v4 草稿模块/测试；这些内容未进入远端 HEAD，不能当作已实现或已验证证据。
- `review_status = changes_required`；未生成 REVIEW_PASS，不批准 full train。

### 覆盖矩阵

| 功能 | 代码/调用链证据 | 结论 |
| --- | --- | --- |
| hard certificate 与 learning access 解耦 | 旧 router 的 shadow 仍要求 `certified` tier；旧 schedule/edge readiness 仍是前置 | RED |
| continuous credibility | 远端没有 v4 cV 调用闭环；本地草稿未形成上一 epoch artifact/EMA 闭环 | RED |
| ontology 与 BDD100K grounding | 旧 builder 主要按 box/category；attribute、可靠负例、unknown、polarity 未完整进入 target | RED |
| no-lane absence polarity | reason 9/15 仍可能由 lane/drivable presence 作为正证据 | RED |
| typed fine transport | 远端 downstream 仍消费 coarse mask；本地 splat 草稿未同步且不满足完整 type-specific 语义 | RED |
| seeded rereader | 旧 rereader 主要接收 coarse factor mask；typed coordinate/feature/attention 未形成远端闭环 | RED |
| reason direct/latent/annotation | latent route 受旧 certificate/route mask 控制；observed/latent loss 由旧 phase 条件控制 | RED |
| action shadow/final admission | shadow active 受 certified tier 限制；partial per-action admission 未被旧 schedule/edge builder 完整证明 | RED |
| loss ownership/gradient firewall | trainer 使用 global `clip_grad_norm_`；三个 visual pyramid 被归为同一 owner；无 v4 per-owner clip | RED |
| Regime A/B/C schedule | 远端仍是旧 12 epoch/旧 adaptive state，并存在 fail-closed certificate/edge 逻辑 | RED |
| audit/artifacts | 旧 audit/artifact validator 检查 certificate/edge/adaptive 旧协议，不检查 v4 cV/fine/owner/reuse schema | RED |
| speed/reuse | 远端训练主链没有 v4 单 batch DINO reuse contract；本地 helper 未跟踪 | RED |

### 具体缺口

1. `fate_oia/models/mosaic_target_sparse_router.py` 的 `_active_edge_mask()` 先要求 `certificate_tier == certified`，所以全 abstained 时 shadow route 仍完全 inactive，正是旧 ce05 hard certificate deadlock。
2. `fate_oia/engine/mosaic_icdor_schedule.py` 与 `mosaic_icdor_adaptive_schedule.py` 将 latent、factor loss、edge/certificate readiness 绑定；FOUNDATION readiness 失败会进入 `failed_closed`，与 v4 epoch0 shadow/latent learning 及“PU 失败不关闭其它 route”冲突。
3. `fate_oia/models/acpr_mosaic_trust_icdor_model.py` 的本地修改虽出现 cV、typed 参数和 shadow/final 分离迹象，但未同步远端；并且没有证明 cV、target utility、fine transport、semantic compatibility、owner diagnostics 从 trainer 写入 artifacts。
4. `fate_oia/models/mosaic_observable_predicates.py` 的远端主输出仍是 coarse mask；`sampling_coordinates`、`sampled_features`、`sample_attention` 没有闭环进入 v4 fine transport。未跟踪 splat 草稿也不能证明 curve tangent anisotropic、region bilinear 和 fine/coarse 混合已实现。
5. `fate_oia/models/mosaic_masked_target_rereader.py` 的旧契约未完整实现 typed seeded local reread、offset cap、reason target route 和 shuffle sensitivity。
6. `fate_oia/datasets/mosaic_icdor_grounding.py` 未完整解析 `trafficLightColor`、lane direction/style/types、area type、occluded/truncated，也未把 strong/reliable-negative/weak-negative/unknown 可靠分离。
7. `configs/mosaic_icdor_factor_candidates.yaml` / reason routes 仍有 red/green image-only、front-near 物理语义、indicator image-only、no-lane presence 等会污染 cV/route/reason 的语义错误。
8. `fate_oia/engine/train_acpr_mosaic_trust_icdor.py` 的 loss ownership、policy weight resolver、per-owner gradient firewall、三路 visual owner 和 v4 matched-control 归属均未形成闭合调用链；现有 global clip 不能替代它们。
9. `fate_oia/engine/audit_acpr_mosaic_trust_icdor.py` 仍是旧 IC-DOR gate，不会硬卡 continuous access/deployment separation、typed fine transport、partial action admission、Regime A/B/C、DINO call count 和 v4 artifact schema。

### 实际验证证据

- 远端旧 schedule/model/train protocol 测试为 `48 passed`，但只证明旧 hard-certificate 行为，不是 v4 合规证据。
- 远端 typed/observable/action/audit collectors 定向测试为 `11 failed, 24 passed, 19 warnings`；已出现 sample budget/shape、小维度 head、bf16 dtype、输出 schema 失败。
- 本地新增 v4 tests 不能作为通过证据：本地 Python 缺少 torch；收集阶段还暴露缺失的 `BatchLocalDinoFieldReuse`、`clip_icdor_owner_gradients`、`resolve_icdor_policy_weights` 及 grounding attribute/corridor API。
- 本地 v4 模块/测试为 untracked，远端 clean 且无对应提交；不能写 REVIEW_PASS。

### 计划自身需要先明确的逻辑边界

1. `learning_route` 与 `deployment_route` 必须分离：epoch0 可学习 shadow/latent，但 admission 前 final action 必须严格等于 visual action。
2. cV 不能读 reason labels；cV 只能用 visual content、visibility/uncertainty、source completeness、stability、effective sample count，reason label 只能进独立 semantic compatibility/target utility audit。
3. `audit_visual`/`audit_target` 的 5%+5% 需固定分母、索引、source coverage 和 bootstrap confidence。
4. 必须固定 fine mask 的计算位置、梯度边界、fine-off/coarse-off/fine+coarse 三种 ablation，以及 point/curve/region kernel 规则。
5. 计划称不使用 HardPair，但旧 trainer 仍有 queue ranking；必须明确 matched-control 是否保留、预算相对哪个主损失、旧 HardPair 是否完全移除，防止重复加权。
6. 当前 config 的 action `gate_max=0.15` 与 v4 的 shadow cap `.05` 不一致，不能只在审查文字中声明一致。

### 必须优先 RED 的测试

1. `test_continuous_credibility_no_hard_deadlock.py`：全 abstained 时 cV、latent、reason、action shadow 仍有有限非零学习路径。
2. `test_shadow_route_active_without_discrete_certificate.py`：无 certified tier 时 shadow 非零；admission 前 final 与 visual `atol=1e-7` 相等。
3. `test_partial_action_admission.py`：只 admission 一个 action 时仅该 action 可变化。
4. `test_reason_anchor_cannot_certify_visual_factor.py`、`test_no_lane_uses_absence_polarity.py`、`test_bdd100k_attribute_grounding.py`、`test_corridor_occupancy_has_positive_and_negative.py`：防止语义泄漏和错误正标签。
5. `test_typed_coordinates_change_target_output.py`、`test_fine_mask_not_equal_coarse_upsample.py`：坐标/type-specific fine transport 必须真实影响 target output。
6. `test_reason_residual_nonregression.py`、`test_pu_failure_does_not_disable_other_routes.py`：bounded annotation residual 和 PU route isolation。
7. `test_per_owner_gradient_clip.py`、`test_policy_weights_consumed.py`、`test_batch_local_dino_reuse.py`：owner clip、权重实际消费、单 batch DINO 调用次数。

### 可执行修复顺序与禁止条件

0. 冻结远端基线，不把本地未跟踪草稿当实现，不启动训练。
1. 让 RED tests 在远端环境可收集并失败于正确断言。
2. 重写 ontology/grounding 的 attributes、source completeness、polarity、strong/reliable-negative/weak-negative/unknown。
3. 实现并接入上一 epoch artifact/EMA 的 dual-source continuous credibility，reason label 不得直接进入 cV。
4. 实现 typed point/curve/region fine transport，并真实接入 observable -> model -> action shadow/reason latent。
5. 接入 seeded rereader、bounded offsets、support/veto、coordinate/attention shuffle 动态验证。
6. 重构 reason direct/latent/annotation 与 PU loss ownership；direct visual 为主，annotation 只能 bounded residual。
7. 重构 per-action shadow/final admission，移除 all-four master gate。
8. 实现 owner-specific gradient firewall/clipping、policy weight resolver、matched-control 唯一归属和 forbidden-zero assertions。
9. 重写 Regime A/B/C、audit/artifact schema、batch-local DINO reuse 和 runtime profile。
10. 依次完成 py_compile、v4 targeted pytest、dynamic forward、strict artifact validation、runtime profile、最多 6 epoch factor-aware pilot；全部 gate 通过且 clean HEAD 绑定后，才允许 full train。

### 监督结论

`review_status = changes_required`。

当前禁止 full train、REVIEW_PASS、把本地草稿推送为已验证实现、以旧 `48 passed` 宣称 v4 合规。本轮没有修改生产代码、没有启动训练；只追加监督日志。下一次复审必须提供远端 clean pushed HEAD、v4 RED/GREEN 结果、dynamic forward/gradient 证据、strict artifact validation、runtime profile 和不超过 6 epoch 的 pilot manifest，之后才能重新判断 `approved`。
