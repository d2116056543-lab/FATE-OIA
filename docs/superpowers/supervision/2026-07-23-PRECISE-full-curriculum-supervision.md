# 双代理监督日志：PRECISE Full-Run Curriculum

**日期：** 2026-07-23  
**任务：** 取消独立 3-epoch pilot，在正式 full run 内采用分阶段组件启用，并直接启动完整训练。  
**状态：** 规划中  
**主执行端：** 当前会话  
**监督端：** `019f8dd0-8dce-7f53-a199-4e413cf10c70`（Archimedes）

## 1. 原始请求

用户确认不再执行独立 3-epoch pilot，要求根据模型成熟度安排组件启用时序，找到最合理配置后直接启动完整训练，并保证创新组件最终完整参与且发挥正向作用。

## 2. 适用的 Skill

- `brainstorming`：行为和训练协议发生变化，已先完成设计并取得用户确认。
- `writing-plans`：将已确认设计拆为 TDD 实现步骤。
- `test-driven-development`：组件时序、scheduler 和恢复行为必须先有 RED 测试。
- `dual-agent-supervision`：用户长期要求严格覆盖、反复审查和防止空实现。
- `verification-before-completion`：启动 full train 前必须有新鲜验证证据。

## 3. 初始计划

1. 新增纯函数 curriculum 合同。
2. 在 final-logit 边界应用实际缩放，不只缩放 loss。
3. 实现 owner-local optimizer/scheduler 时钟。
4. 将完整激活状态写入所有运行 artifact 与 checkpoint。
5. 保留既有数据、模型、评价和无泄漏约束。
6. 通过回归、audit、监督复核和 clean-HEAD 检查后启动 full run。

## 4. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现步骤 | 预期改动 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 不跑独立 3-epoch pilot | 必须 | Task 4/5 | supervisor/full protocol | 无 pilot artifact 时仅显式用户覆盖可启动 | 已规划 |
| 2 | 正式训练首轮不能盲目全开 | 必须 | Task 1/2 | curriculum + model | epoch 0-1 delta 精确为零 | 已规划 |
| 3 | 2-3 轮后逐步打开较多组件 | 必须 | Task 1 | YAML/curriculum | epoch 表逐项断言 | 已规划 |
| 4 | 表征成熟后才启用高风险功能 | 必须 | Task 1/2/3 | exchange/intervention schedule | 前向与 loss 双重缩放 | 已规划 |
| 5 | 创新点最终全部启用 | 必须 | Task 1 | epoch 6-11 full stage | 所有 scale=1.0 | 已规划 |
| 6 | 直接启动完整训练 | 必须 | Task 5 | full supervisor | clean-HEAD 后真实进程和首批日志 | 未开始 |
| 7 | 未启用组件不能提前消耗 LR schedule | 必须 | Task 3 | owner optimizer clocks | inactive owner step 不变 | 已规划 |
| 8 | 能审计每个组件实际是否启用 | 必须 | Task 4 | artifacts/checkpoint | schema 与数值检查 | 已规划 |
| 9 | 不改变既有 no-cache/no-compression/test-only 等边界 | 必须 | Task 4/5 | audit | forbidden/leakage 回归 | 已规划 |

**覆盖结论：** 所有必须项均有明确实现步骤和验证方式。

## 5. 用户计划保真矩阵

| 编号 | 用户原计划项 | 必须遵循 | 对应步骤 | 保留情况 | 偏离原因 | 验证方法 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 不再进行 3-epoch pilot | 是 | Task 4/5 | 原样保留 | 无 | full override artifact | 已规划 |
| 2 | 找到最合适配置直接 full train | 是 | Task 1-5 | 细化 | 无法无实验数学保证“全局最优”，采用可复现风险最小配置 | 固定时序与全回归 | 已规划 |
| 3 | 组件不能过早打开 | 是 | Task 1-3 | 原样保留 | 无 | epoch 0-5 断言 | 已规划 |
| 4 | 模型能力足够后全部发挥 | 是 | Task 1/2 | 原样保留 | 无 | epoch 6-11 全开 | 已规划 |

**保真结论：** 用户明确要求已保留；“最好”被严格解释为基于当前证据的可复现风险最小配置，而非无法验证的全局最优声明。

## 6. 监督审查

**是否已发送给监督端：** 是  
**监督端 agent id：** `019f8dd0-8dce-7f53-a199-4e413cf10c70`  
**发送内容摘要：** 原始请求、固定时序、覆盖矩阵、保真矩阵、设计、实现计划与当前代码路径。

**审查结果：**

- 必须拆分 reread/exchange 和 direct-reason/reason-latent owner。
- owner scheduler 总步数必须按真实 active epochs 计算。
- threshold 0.5 必须作用于 deploy equation。
- zero-scale anchor 必须明确定义为现有 refined-head direct anchor。
- partial-scale 测试必须检查每个 delta 边界，不能假设全模型线性。
- CLI override 不足够，必须生成 hash-bound curriculum-ready gate。
- audit 必须识别早期 owner 不应 step，并禁止旧 full gate误授权。
- artifact schema必须写死。
- foreground-only 与 SSH-survival 冲突，必须只保留前者。

**是否允许进入执行：** 否，状态 `changes_required`。

## 7. 计划修订

**监督结果是否已传回执行端：** 是

**已采纳：**

- 拆分 `reread_adapter`、`exchange_adapter`、`reason_latent`。
- 写死 12/10/8 active-epoch scheduler totals。
- 明确 threshold deploy 缩放。
- 明确 refined-direct anchor。
- 改为逐边界 raw/effective delta 验证。
- 新增 `PRECISE_OIA_V1_FULL_CURRICULUM_READY.json`。
- audit 和 artifact schema curriculum-aware。
- 启动协议只保留 foreground-only。
- latent evidence 参数归入始终启用的 `evidence_core`，latent messaging 延后。
- inactive owner 每个 accumulation 边界清梯度，optimizer state 在首次激活前为空。
- prelaunch audit 与 runtime assertions 分离，epoch 6 未真实 step 即协议报错。
- curriculum 使用规范化 JSON hash，并增加三个阶段边界 resume 测试。

**修订后计划：** 见
`docs/superpowers/plans/2026-07-23-precise-full-curriculum-plan.md`。

## 8. 复审轮次

| 轮次 | 修订内容 | 监督结论 | 允许执行 | 剩余问题 |
| --- | --- | --- | --- | --- |
| 1 | 初始 curriculum 设计 | changes_required | 否 | owner、threshold、gate、artifact 语义不足 |
| 2 | 已吸收第一轮全部硬要求 | changes_required | 否 | latent owner、inactive 梯度生命周期、运行期断言不足 |
| 3 | 已吸收第二轮全部硬要求 | approved | 是，允许进入 TDD | 无计划级阻断项 |

**最新监督状态：** 已通过，允许进入 TDD；full train 仍须等待实现复审、全回归、clean-HEAD audit 和新 gate。

## 9. 执行交接

尚未交接。

## 10. 执行合规检查

尚未执行。

## 11. 验证证据

当前仅完成远端代码与配置核查：

- HEAD `28820ff1ca79de63a85f660efdf884736e4a0b30`
- 当前多数机制从 epoch 0 进入 final logits。
- 当前仅有辅助 loss 的 10% update warm-up。
- 当前只有 threshold 具有 `start_epoch=1`。

## 12. 最终判断

**是否可以报告完成：** 否  
**理由：** 尚未完成监督审查、TDD、实现、验证和 full train 启动。
## Implementation review resolution

- First implementation review: `changes_required` with six P1 findings.
- Resolved inactive intervention overhead by skipping packed deletion forwards at activation `0` while preserving the complete zero-valued artifact schema.
- Replaced legacy pilot authority for this approved run with hash-bound `PRECISE_OIA_V1_FULL_CURRICULUM_READY.json`; both repo and user skill copies are identical.
- Added trainer-side readiness verification so direct CLI invocation cannot bypass HEAD/config/source/skill/curriculum/runtime checks.
- Fixed owner-local cosine scheduler indexing so the final active update consumes the configured minimum LR.
- Added runtime assertions for inactive optimizer state, owner-local step counts, scheduler clocks, and epoch-6 all-on activation.
- Threshold deploy remains `raw - scale * theta`; scale `0` is inactive, scale `0.5` can fit on train-calib but cannot mutate the teacher, and scale `1` permits teacher update.
- Resume now fails closed on missing lifecycle fields and validates canonical curriculum state, optimizer state, owner step counts, and scheduler clocks at the `1->2`, `3->4`, and `5->6` boundaries.
- Fixed the observed-reason firewall to use the split owner names `reread_adapter`, `exchange_adapter`, and `reason_latent`.
- Independent re-review verdict: `approved_for_full_tests`, no unresolved P0/P1.
- Verification: `136 passed` for all PRECISE tests, `53 passed` for all ACPR regression tests, compileall passed, and `git diff --check` passed.
- Commit remains contingent on preserving this exact verified source state. Full training remains contingent on a fresh clean-HEAD runtime profile, preflight audit, and `FULL_CURRICULUM_READY`.
