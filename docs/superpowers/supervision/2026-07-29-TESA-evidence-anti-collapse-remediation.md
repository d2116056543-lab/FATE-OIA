# 双代理监督日志：TESA Evidence Anti-Collapse 修复

**日期：** 2026-07-29  
**任务：** 修复 TESA pilot 暴露的 typed state、action transport、reason correction、patch deletion、factor coverage 与 protocol artifact 问题；验证创新机制后启动后台 clean full train。  
**状态：** 监督批准，进入 TDD 执行  
**主执行端：** Codex 主会话  
**监督端：** 019fae26-be33-7c53-a02b-179acedd810d

## 1. 原始请求

用户要求严格区分训练样本较少、训练随机性、审计口径错误和模型/训练逻辑错误，结合当前代码、BDD-OIA/BDD100K 数据与顶会论文和官方 GitHub 进行修复。不能机械地因为某个 Gate 未过就判定方案失败，也不能通过放宽 Gate、伪造 coverage 或强制 epsilon 来制造通过。修复后要保证大部分创新点真实有效，再后台启动 full train。

用户已批准推荐方案 2：TESA evidence anti-collapse 修复。

## 2. 适用的 Skill

- `brainstorming`：已完成根因分析、三方案比较并获得用户对方案 2 的明确批准。
- `systematic-debugging`：先复现并定位确定性根因，再修改。
- `test-driven-development`：每项行为变化先写 RED test。
- `dual-agent-supervision`：用户要求严格覆盖、对抗审查与避免部分实现。
- `executing-plans`：按已批准设计分阶段执行并在关键节点验证。
- `verification-before-completion`：在宣称完成或启动 full train 前提供新鲜验证证据。

## 3. 初始计划

1. 用 RED tests 固化 protocol alias/test count、null 校准与 absent supervision、typed-state target、action transport 稀释、per-action identity、真正分层 patch audit 和 Gate 可识别性。
2. 修复 `run_manifest.json` 的 split schema，不修改科学指标。
3. 修复 typed evidence：null score 相对 patch partition 校准；仅对可靠 visible-present / observable-absent 样本监督 null；补可靠可推导的 factor state，未知保持 unknown。
4. 修复 action transport：cap 按有效稀疏 support 分配；保留可学习 action×factor compatibility；加入 source-aware anti-monopoly 和 per-action identity/ranking，禁止人工硬编码 action→factor 因果。
5. 延后 softmax→entmax 稀疏化，避免早期不可逆单 factor 塌缩。
6. 修复 reason correction 的 full/partial/latent 分层监督与统计，不通过简单扩大 delta 过 Gate。
7. 重写 patch audit：source-stratified action×eligible-factor 队列、局部几何匹配 control；分别报告 model-top faithfulness 与 stratified execution coverage。
8. 修复 pilot Gate 中数学不可能的判据，同时保留非塌缩、排名增益和 faithfulness 的科学要求。
9. 运行 py_compile、targeted pytest、完整 TESA/ACPR 回归测试、implementation audit。
10. 用现有 pilot checkpoint 运行一次短程机制验证，检查 null、state、factor diversity、action ranking、reason gain、deletion direction。
11. 只有大部分机制方向真实转正且无协议/运行异常，才从 clean initialization 后台启动 full train；采用已验证安全的 `batch=4, grad_accum=8, num_workers=4, bf16`，记录 PID、日志和首轮状态。
12. 所有 split 名称由单一 protocol schema 产生，writer 与 validator 使用同一映射；显式记录 test count。
13. grounding/action/reason/PU 权重只从 resolved config 进入一次，新增精确求和测试；implementation audit 必须调用 trainer 的完整 loss 图。
14. PU admission 与 training 使用同一个 score 函数和同一 stop-gradient 语义；保留 non-negative risk 边界。
15. state corruption 定义为“固定 reliability、只破坏 state identity”的隔离实验，并在 artifact 中写明；不伪称重新估计可观察性。
16. 使用远端真实 BDD100K 记录审计 lane `poly2d` 解析、anchor/state 有效计数和 source coverage。
17. 新增逐 factor 的 `factor_absent_valid`：
    - object-presence/actor-presence 仅在对应 BDD100K detection frame 已匹配且 `source_complete=true` 时允许 absent；
    - lane/boundary absent 仅在 lane frame 已匹配且 `lane_source_complete=true` 时允许；
    - drivable/corridor 仅在 drivable map 成功读取时允许判断几何可用性；
    - traffic-light color 缺失只表示 state unknown，不能当 red/green；无 traffic-light instance 且 detection source complete 时只允许监督 null；
    - 任一 source incomplete 时 absent/null/state 均 mask，不转成负类。
18. factor 2 occupancy 规则固定为：用成功读取的 drivable map 构造中心前向 corridor；只使用完整 detection frame 中 car/bus/truck/person/rider/bike/obstacle 的 `box2d` 与 corridor 的重叠。存在重叠为 occupied；source complete、drivable 有效且无重叠为 clear；否则 unknown。禁止仅凭 drivable anchor 推 clear。
19. patch audit artifact 固定输出 `eligible_factor_coverage`、`requested_factor_coverage`、`executed_factor_coverage`、`model_top_factor_coverage`。source-stratified coverage 不进入 model-top selected-vs-control 均值。control 必须匹配 patch 数、左右区域、纵向道路深度、有效区域，且与 selected 不重叠。
20. 2-epoch 准入标准固定：
    - present/absent/unknown 三组 null 方向正确，unknown loss/grad 为 0；
    - action final Act_mAP 不低于 visual 超过 0.002，Act_mF1 不低于 0.005，且至少 3/4 action target identity effect 为正；
    - reason final aggregate Exp_mAP gain > 0，Exp_mF1 下降不超过 0.003；full-groundable 分层方向为正；
    - deletion model-top faithfulness 使用至少 128 unique 样本并输出 bootstrap 95% CI；coverage 只计成功执行的 eligible intervention；
    - loss/gradient 全 finite，协议/artifact 无缺字段。
21. runtime/artifact schema 由共享 dataclass/常量定义，writer、validator、evaluator 共用；覆盖 runtime、typed evidence、patch audit 和 split counts。
22. 完整 trainer gradient ownership 矩阵：
    - reason/PU 对 foundation、typed factor、action transport 梯度严格为 0；
    - action 对 reason decoder 梯度严格为 0；
    - grounding 只更新 typed factor；
    - mirror、identity、dense 分别对 trainer total 产生非零可追踪增量；
    - disabled 分支梯度严格为 0。
23. mirror 独立验证：paired mirror 每个 batch pair 只进行一次联合 DINO encode；mirror loss 进入 total；非 mirror batch 该项为 0；不与 grounding 权重重复计入。
24. 所有 resolved loss key 必须恰好消费一次，无 dead key、无同名代码硬编码；reason 的 full/partial/latent 分层从 factor schema 读取，evaluator 禁止硬编码 `GROUNDABLE`。
25. mirror loss 与梯度所有权拆分：
    - anchor/state/observability/discrimination 和 mirror-anchor/state 仅更新 typed factor；
    - mirror-action 仅更新 foundation/action transport；
    - mirror-reason 仅更新 reason decoder；
    - 每项通过 detach 隔离非目标模块，并验证非目标梯度严格为 0。
26. 2-epoch 非平凡机制标准：
    - action correction RMS 每个 action 位于 `[0.02, 0.20]`，两轮平均 Act_mAP delta `>0`，任一轮不得低于 `-0.002`，至少 3/4 action identity target effect `>0.001`；
    - reason Exp_mAP gain 使用两轮 paired mean `>=0.001`，且 paired bootstrap 95% CI/逐样本差异一并报告；
    - deletion selected-minus-control mean `>0`；95% CI 包含 0 时记为 inconclusive，不计作有效；
    - 近零正数、仅单轮正向、inconclusive 均不能算通过。
27. 最终准入公式：
    - 确定性门槛必须全部通过：protocol/schema、unknown mask、source completeness、finite、gradient ownership、无 test 泄漏、artifact validation；
    - typed-state eligible factor 固定为 `positive_count>=20` 且 `negative_count>=20`；
    - typed-state pass 固定要求：eligible factor 数 `>=6`、macro `(AUPRC-prevalence)>=0.02`、至少 `2/3` eligible factors 的 `AUPRC>prevalence`；
    - factor 2 防垄断使用 `usage_excess = observed_usage_share - source_eligible_opportunity_share`，要求 `usage_excess<=0.15`；禁止用绝对 usage share 惩罚真实较高的 drivable source coverage；
    - 不得用 factor 2 单独代表 typed-state 成功；
    - action transport 必须通过；
    - typed state/null、reason correction、deletion faithfulness 三类至少两类通过；inconclusive 不算通过；
    - 只有上述组合成立才允许 full train。

## 4. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现步骤 | 预期改动位置或产物 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 严格区分小样本、随机性、代码错误、审计错误 | 必须 | 根因分类并用测试固化 | 本日志、测试、诊断 JSON | 对照四轮趋势和静态公式 | 已分析 |
| 2 | factor state 质量修复 | 必须 | 补可靠 state target、unknown 语义和分层指标 | dataset target、grounding loss、typed artifact | target 单测、真实 train-audit AUPRC/coverage | 规划中 |
| 3 | null 不塌缩且不造假 | 必须 | partition-relative null + selective absent loss | typed head、grounding loss | present/absent/unknown RED tests；真实 null 分布 | 规划中 |
| 4 | action transport 超越或至少有效改变 visual ranking | 必须 | 修 cap、anti-monopoly、per-action ranking/identity | action transport、action loss | RMS、mAP delta、per-action identity、factor usage | 规划中 |
| 5 | reason correction 有稳定增益 | 必须 | 分层可靠 correction 与 identity | reason loss/eval | full/partial/latent AP delta，F1 不异常 | 规划中 |
| 6 | deletion gap 与 factor coverage 可信 | 必须 | 真正 source-stratified audit、matched control | tesa diagnostics/artifacts | model-top 与 stratified 两套统计；128 unique | 规划中 |
| 7 | protocol artifact 字段匹配 | 必须 | 修 split aliases/test count | run manifest/evaluator | protocol regression test | 规划中 |
| 8 | 不通过放宽 Gate 或伪造值过审 | 必须 | Gate 只修数学不可能项，保留科学门槛 | evaluator/tests | 对伪造 epsilon/uniform coverage 的负测试 | 规划中 |
| 9 | 结合论文但不照搬 | 必须 | 采用 availability-weighted competition、local affinity audit | 设计和实现注释 | 代码审查确认无 Slot/MoE/graph 主路径 | 规划中 |
| 10 | 大部分创新点有效后启动后台 full train | 必须 | 短程机制验证后 clean full | run dir、PID、日志、manifest | 首轮训练与 eval artifact，进程存活 | 规划中 |
| 11 | loss 权重无硬编码/双重计入 | 必须 | resolved config 单一来源、精确求和 | grounding/action/reason loss、trainer | exact-sum RED test、完整 trainer loss audit | 规划中 |
| 12 | PU admission/training score 一致且非负 | 必须 | 共用 PU score API | PU losses、trainer、audit | 数值一致、inactive zero、risk non-negative | 规划中 |
| 13 | state corruption 语义明确 | 必须 | 固定 confidence 的 identity-only corruption | model diagnostics、artifact | corruption contract test | 规划中 |
| 14 | lane poly2d 真实链路有效 | 必须 | 真实远端数据统计 | typed target audit JSON | parsed/anchor/state/source counts 非伪造 | 规划中 |
| 15 | implementation audit 覆盖完整 trainer loss | 必须 | audit 调用 `_compute_losses` 或公开等价入口 | implementation audit JSON | grounding/identity/dense/mirror/PU 均有 finite grad | 规划中 |
| 16 | absent/null 不把未知当负类 | 必须 | `factor_absent_valid` 与 source completeness | typed targets、dataset/index | object/lane/color/drivable 逐类 RED tests | 规划中 |
| 17 | factor 2 occupancy 有字段级定义 | 必须 | drivable corridor × 完整 detection box overlap | typed targets | clear/occupied/unknown 三用例 | 规划中 |
| 18 | anti-monopoly 不强迫无关 factor | 必须 | source-aware compatibility regularizer | transport/loss | 单 eligible penalty=0、no-source 排除、compatibility 有 action grad | 规划中 |
| 19 | patch audit 四套 coverage 与 matched control | 必须 | shared artifact schema、stratified queue | diagnostics/artifacts | schema/不重叠/空间匹配/coverage 分离 tests | 规划中 |
| 20 | runtime/typed/patch schema 单一来源 | 必须 | 共享 dataclass/常量 | artifacts/writer/validator/evaluator | writer→validator 集成测试 | 规划中 |
| 21 | 梯度所有权与 mirror 行为正确 | 必须 | 完整 trainer loss probe | audit JSON/tests | ownership matrix、single encode、total delta | 规划中 |
| 22 | reason groundability 无硬编码 | 必须 | schema-derived full/partial/latent | schema/evaluator | schema mutation test | 规划中 |
| 23 | 2-epoch 准入标准可复现 | 必须 | 固定阈值、样本量、CI | pilot evaluator | 预注册 evaluator tests | 规划中 |
| 24 | mirror 各分量梯度所有权无冲突 | 必须 | 拆分 mirror anchor/state/action/reason | grounding loss/trainer/audit | 目标模块非零、非目标严格零 | 规划中 |
| 25 | 机制增益必须非平凡 | 必须 | RMS、两轮均值、bootstrap、最小 identity effect | pilot evaluator | near-zero/inconclusive 负测试 | 规划中 |
| 26 | 大部分创新点有效有组合公式 | 必须 | deterministic-all + action-required + 3选2 | pilot evaluator | 组合真值表测试 | 规划中 |
| 27 | state 质量不被 factor2 或无标签项劫持 | 必须 | eligible sample floor、macro excess、usage share | typed audit | degenerate/unlabeled 排除测试 | 规划中 |

**覆盖结论：** 所有必须项均有明确实现步骤与验证方法，尚未进入生产代码修改。

## 5. 用户计划保真矩阵

| 编号 | 用户原计划项 | 必须遵循/可选 | 对应执行步骤 | 保留情况 | 偏离原因 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 采用方案 2 | 必须遵循 | 全部步骤 | 原样保留 | 无 | 覆盖矩阵与代码 diff | 规划中 |
| 2 | 真实解决 factor/state/action/reason/deletion/coverage/protocol | 必须遵循 | 步骤 1-8 | 细化 | 无 | 单测、真实机制指标 | 规划中 |
| 3 | 不能刻板按旧 Gate 判失败 | 必须遵循 | 步骤 8 | 细化为“修不可能项、不降低科学要求” | 无 | evaluator RED tests | 规划中 |
| 4 | 结合 GitHub 与顶会解决 | 必须遵循 | 步骤 3-7 | 原样保留 | 不引入不适配的大模块 | 文献映射与代码审查 | 规划中 |
| 5 | 保证大部分创新点有效 | 必须遵循 | 步骤 10 | 细化为短程真实机制验证 | 不能凭静态代码保证效果 | 真实指标方向 | 规划中 |
| 6 | 后台 full train | 必须遵循 | 步骤 11 | 后置到机制验证之后 | 避免重复无效 full run | PID、GPU、首轮 artifact | 规划中 |

**保真结论：** 用户要求全部保留；仅将 full train 放在真实机制验证之后，符合用户明确的“解决好了再启动”条件。

## 6. 监督审查

**是否已发送给监督端：** 是  
**监督端 agent id：** 019fae26-be33-7c53-a02b-179acedd810d  
**发送内容摘要：** 原始请求、方案 2、适用 skills、初始计划、功能覆盖矩阵、保真矩阵，以及 null/state/anti-monopoly/patch audit/短程验证五类高风险问题。

**第一轮审查结果：**

- 必须统一 protocol schema，并做 writer→validator 集成测试。
- null 仅在 reliable present/observable absent 上监督，unknown mask；禁止 epsilon floor。
- state target 不得由 drivable anchor 单独推断 road-clear。
- action×factor compatibility 必须可学习，anti-monopoly 只能作用于有 source 的 factor。
- loss 权重必须取消硬编码双源，action specificity 必须能在总损失中精确追踪。
- PU admission/training score 必须共用 API；保留 non-negative risk。
- state corruption 必须明确是 identity-only 或重算 reliability，不能语义含混。
- patch audit 的 coverage 与 model-top faithfulness 必须分离，并改善 control 匹配。
- lane poly2d 必须用真实远端记录验证。
- implementation audit 必须覆盖完整 trainer loss 图。
- 监督日志必须可按 UTF-8 读取。

**是否允许进入执行：** 否，需修订后复审。

**第二轮审查结果：**

- 需要逐 factor 定义 source completeness 与 `factor_absent_valid`。
- 需要写死 factor 2 occupancy 几何和允许字段。
- anti-monopoly 需要防强迫无关 factor 的负测试。
- 需要写死 2-epoch 准入标准。
- patch audit 必须有四套 coverage 和精确 control 匹配。
- runtime/typed/patch artifact 需共用 schema。
- trainer audit 需验证梯度所有权矩阵。
- mirror 训练需独立覆盖。
- loss key 需恰好消费一次。
- reason groundability 需 schema-derived。

**第二轮是否允许进入执行：** 否，需纳入后第三轮复审。

**第三轮审查结果：**

- mirror 梯度所有权需按 anchor/state/action/reason 拆分。
- 机制准入必须排除近零正数和单轮偶然值。
- state 质量和“大部分创新点有效”需预注册组合公式。

**第三轮是否允许进入执行：** 否，需纳入后第四轮复审。

**第四轮审查结果：**

- typed-state eligible 样本下限、macro excess、通过比例和 factor 2 opportunity-adjusted usage 需要数值阈值。
- reason 两轮增益需设置非零数值下限。

**第四轮是否允许进入执行：** 否，需补数值后第五轮复审。

**第五轮审查结果：**

- `review_status = approved`
- 剩余必须修改项：无。
- 允许开始 RED tests；不代表允许 full train。
- 必须保留 RED→GREEN 证据、真实 lane audit、完整回归、implementation audit 和 2-epoch 机制验证。

**第五轮是否允许进入执行：** 是，仅授权 TDD/代码执行。

## 7. 计划修订

**监督结果是否已传回执行端：** 是  
**已采纳：**

- 以上 11 项全部纳入步骤 12-16 和功能覆盖矩阵 11-15。
- factor 2 的 state 只在 drivable 与可靠 occupancy 同时成立时监督；仅 drivable 时保持 unknown。
- patch audit 同时保存 source-stratified coverage 与 model-top faithfulness，二者不互相替代。
- 短程验证固定为 2 epoch、4096 main/1024 audit；不要求每个稀有标签机械通过旧 Gate。
- 第二轮 10 项全部纳入步骤 17-24 和覆盖项 16-23。
- 第三轮 3 项全部纳入步骤 25-27 和覆盖项 24-27。

**未采纳及理由：**

- “监督日志乱码”不是文件编码事实：文件由 `apply_patch` 以 UTF-8 写入，后续将用显式 UTF-8 解码测试验证；若验证失败再修复，不凭终端默认编码判断。

**修订后计划：**

1. 先为监督提出的 11 项逐项编写 RED tests 并确认按预期失败。
2. 修 protocol、null/state targets、transport、loss/PU/corruption contracts、patch audit 与完整 audit。
3. 完成真实 lane 数据审计、回归测试和 2-epoch 机制验证。
4. 满足允许条件后 clean full train。

## 8. 复审轮次

| 轮次 | 发送给监督端的修订内容 | 监督端结论 | 是否允许执行 | 剩余问题 |
| --- | --- | --- | --- | --- |
| 1 | 初始计划与双矩阵 | 需要修改 | 否 | 11 项硬化要求 |
| 2 | 纳入 protocol 单一来源、null 三语义、unknown state、action×factor、loss/PU 单一来源、corruption contract、双轨 patch audit、真实 lane、完整 trainer audit | 需要修改 | 否 | absent/source、factor2 geometry、artifact/gradient/mirror 需写死 |
| 3 | 新增步骤17-24与覆盖项16-23，写死 source completeness、occupancy、四 coverage、CI、梯度所有权和 mirror | 需要修改 | 否 | mirror ownership、非平凡增益、组合准入 |
| 4 | 新增步骤25-27与覆盖项24-27，拆分 mirror 梯度、写死非平凡阈值和 deterministic+mechanism 组合公式 | 需要修改 | 否 | typed-state/reason 数值阈值 |
| 5 | 写死 state 20/20 样本、6 factors、macro excess 0.02、2/3 比例、factor2 excess 0.15、reason gain 0.001 | 已通过 | 是 | 无 |

**最新监督状态：** 已通过，允许进入 RED tests

## 9. 执行交接

**是否已发送给执行端：** 是  
**执行端类型或 agent id：** Codex 主会话  
**交接内容：** 已批准的步骤 1-27、覆盖项 1-27、第五轮数值准入标准和 RED→GREEN 执行边界。

## 10. 执行合规检查

**执行端是否照做：** 未开始  
**所有必须功能是否完整实现：** 未开始  
**所有必须遵循的用户计划项是否已保留：** 规划已保留，待验证

## 11. 验证证据

待执行后记录。

## 12. 最终判断

**是否可以报告完成：** 否  
**理由：** 监督审查、TDD、实现、真实机制验证和 full train 启动均未完成。

## 13. 方案 2 执行证据（2026-07-29）

### 13.1 RED -> GREEN

- typed-state/null RED：缺少 `null_partition_calibration_loss`，随后完成 source-complete 三态 target、factor 2 occupancy、partition-calibrated null 与 unknown zero-gradient。
- action transport RED：缺少 action-by-factor compatibility、live sparse support cap、source-aware anti-monopoly、near-boundary ranking 与 per-action identity。
- protocol/artifact RED：writer/evaluator split schema 不一致，patch source coverage 与 model-top coverage 混淆，control 未同时匹配 side/depth/validity。
- trainer integration RED：PU audit/training score 语义不一致、grounding YAML 权重未传入、完整 trainer 梯度图未被 audit。

### 13.2 真实远端验证

- 新 anti-collapse targeted suite：`22 passed`。
- 完整 METER/TESA regression：`111 passed`，仅有既存 `TypedStorage` deprecation warning。
- implementation audit：`pass=true`。
- action loss gradient ownership：reason branch `0.0`。
- reason loss gradient firewall：foundation/factor/action 均 `0.0`，reason branch 非零。
- PU active gradient：仅 reason-private 非零。
- grounding gradient：仅 typed-factor 非零。
- mirror gradient：factor/action/reason 均非零，符合 paired consistency 设计。
- full trainer gradient：foundation/factor/action/reason 均非零。
- null present/absent 方向正确，unknown 梯度严格为零。

### 13.3 尚未完成

- clean-HEAD audit 尚未执行。
- 两轮 4096-main/1024-audit 机制诊断尚未运行。
- full train 尚未获准启动。

## 14. 最终集成反审查修复

监督端在首轮集成后发现实现虽然通过测试，但仍有可造成假阳性或死路的真实问题。已逐项修复：

- runtime subset 由 `build_runtime_subset_counts()` 统一生成，writer/evaluator 不再各写一套 schema。
- 最终 admission 绑定 implementation audit、组件级梯度归属、unknown mask、source completeness、no-test-leakage 和严格两轮配对。
- patch audit 改用带真实 BDD100K grounding 的 `factor_audit_loader`；action 由模型预测自主选择，不读 action GT；删除因素必须同时满足 source-eligible 与 schema-allowed。
- deletion effect 直接定义为 `clean_logit - deleted_logit`，移除按贡献符号翻转造成的假阳性；置信区间按 `sample_id` 聚类 bootstrap。
- null 表示“局部证据缺失”，不再把 clear/occupied 等 signed negative state 错标成 factor absent。
- action source availability 只由 predicted observability `>0.05` 判断；高熵初期 reliability 仅以 `0.10` 下限连续调制，不再把全部 action route 硬关为零。
- mirror audit 拆成 anchor/state/action/reason 四个独立 objective，分别验证梯度所有权，避免总 loss 有梯度掩盖跨分支污染。

## 15. 最新远端证据

- 真实 source audit（1024 条）：BDD100K label/drivable coverage `973/1024=0.9502`。
- 修复 drivable 文件优先级后，factor 2 anchor/state 可用 `926` 条，其中 clear `641`、occupied `285`。
- factor 10 positive/negative `163/734`；factor 16 positive/negative `282/631`。
- 达到每类 positive/negative 均不少于 20 的 typed-state factors 共 `10` 个。
- semantic segmentation 仅 `56/1024=0.0547`，继续只作非正式诊断，不作为训练依赖。
- 完整 METER/TESA regression：`114 passed`。
- py_compile：25 个当前改动 Python 文件全部通过。
- implementation audit：`pass=true`。
- action gradient：foundation/factor/action 非零，reason 严格为零。
- reason gradient：仅 reason-private 非零。
- mirror component gradient：anchor/state 仅 factor；action 为 foundation/factor/action；reason 仅 reason-private，全部通过。

**当前许可边界：** 允许 clean-HEAD audit 与严格两轮机制诊断；两轮 admission 未通过前仍禁止 full train。
