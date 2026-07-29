# 双代理监督日志：METER-OIA V2 / TESA

**日期：** 2026-07-29  
**任务：** 基于 METER-OIA V1 clean pilot 实现 V2/TESA，完整覆盖用户计划，禁止部分实现、占位实现和逻辑冲突。  
**状态：** 规划与监督审查中  
**主执行端：** 当前 Codex 主会话  
**监督端：** 尚未创建

## 1. 原始请求

- 在 V1 代码基础上实现计划中的 V2/TESA。
- 功能必须完整，不得仅做到可训练。
- 不得错误实现、遗漏实现或引入内部逻辑冲突。
- 控制 token 消耗，只做与计划覆盖和验证直接相关的工作。

## 2. 适用的 Skill

- `brainstorming`：将用户已批准的唯一方案冻结成可验证设计，不重新发明方案。
- `using-git-worktrees`：从 clean pilot SHA 建立隔离 V2 工作树。
- `dual-agent-supervision`：建立覆盖矩阵并执行计划保真审查。
- `test-driven-development`：先写失败测试，再实现结构替换。
- `executing-plans`：按计划检查点分批执行。
- `verification-before-completion`：完成前必须提供新鲜验证证据。

## 3. 初始计划

1. 从 `00954c9` 新建 V2 分支和 worktree。
2. 建立 F01-F30 功能覆盖矩阵。
3. 先补 23 个 TESA RED tests。
4. 实现 typed factor、additive action、global reason correction。
5. 实现 dense intervention、PU、calibration、sequential evaluator。
6. 完成 owner/firewall/artifact/runtime 验证。
7. 通过 4-epoch pilot gates A-H 后才允许 full train。

## 4. 功能覆盖矩阵

完整矩阵位于：

`docs/superpowers/plans/2026-07-29-meter-oia-v2-tesa-implementation.md`

**覆盖结论：** 用户计划中的结构、公式、所有权、训练协议、诊断、测试与 gate 均已有明确实现位置和验证方法。

## 5. 用户计划保真矩阵

| 计划段 | 必须遵循 | 执行映射 | 保留情况 | 验证 |
|---|---|---|---|---|
| 6-10 typed evidence/state | 是 | F03-F10 | 原样细化 | schema/head/loss 动态测试 |
| 11 additive action | 是 | F11-F13 | 原样保留公式 | 精确重构与 corruption AP |
| 12 reason correction | 是 | F14-F17 | 原样保留 | global equivalence/firewall |
| 13 PU/calibration | 是 | F22-F23 | 原样保留边界 | train-audit/private-only/hash |
| 14 dense intervention | 是 | F19-F21 | 原样保留 | dense coverage/unique IDs |
| 15 meta audit-only | 是 | F18 | 原样保留 | optimizer exclusion |
| 16-18 flow/config | 是 | F02/F26/F29 | 原样保留 | config/runtime tests |
| 19 pilot gates A-H | 是 | F30 | 原样保留 | strict gate report |
| 20-21 full/diagnostics | 是 | F25/F28/F30 | 原样保留 | artifact schema |
| 22-24 code/tests/order | 是 | 全部 | 原样保留 | coverage matrix 状态 |

**保真结论：** 当前没有简化、替换或跳过用户计划项。

## 6. 监督审查

**是否已发送给监督端：** 否  
**审查结果：** 等待监督端审查  
**是否允许进入执行：** 否

## 7. 计划修订

**监督结果是否已传回执行端：** 否  
**已采纳：** 等待审查  
**未采纳及理由：** 无

## 8. 复审轮次

| 轮次 | 发送内容 | 监督结论 | 允许执行 | 剩余问题 |
|---|---|---|---|---|
| 1 | 待发送 | 待审查 | 否 | 待审查 |

**最新监督状态：** 待审查

## 9. 执行交接

**是否已发送给执行端：** 否  
**交接内容：** 监督批准后填写。

## 10. 执行合规检查

**执行端是否照做：** 尚未执行  
**所有必须功能是否完整实现：** 尚未执行  
**所有必须遵循计划项是否保留：** 当前规划阶段全部保留

## 11. 验证证据

- 远端 V2 worktree 已从 `00954c976244e5721ff1d25cae0fae820867f927` 创建。
- V1 脏 worktree 未被复制或修改。
- 用户计划 SHA256 已记录。
