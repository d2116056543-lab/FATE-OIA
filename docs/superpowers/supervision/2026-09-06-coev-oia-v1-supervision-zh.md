# COEV-OIA v1 中文监督日志

## 任务边界

- 当前分支：`tida_site_v20`
- 起点：`f33d7d5a52848bf52f41b844f865588ab49f1672`
- 目标 worktree：`E:\sbw\FATE_Drive\fate_oia_tida_site_v20_worktree`
- 不新建 worktree/branch，不覆盖历史 TIDA，不加载旧任务 checkpoint。
- 本日志是实现审查证据，不替代 `task_plan.md/findings.md/progress.md` 三份唯一实验状态记录。

## 监督状态

| 阶段 | 状态 | 证据/问题 |
|---|---|---|
| 基线身份 | PASS | 远端 branch/HEAD 与计划一致；tracked tree clean，历史未跟踪文件保留 |
| 需求展开 | IN_PROGRESS | 已建立 R01-R40 四层覆盖矩阵 |
| 数学实现 | NOT_RUN | 等待 RED tests 与实现 |
| 架构实现 | NOT_RUN | 等待正式 factory call graph |
| 运行审查 | NOT_RUN | 不接受 mock/shape-only 证据 |
| Git/ready | NOT_RUN | 必须 clean HEAD 后重新绑定 |

## 当前反方结论

仓库中不存在任何 COEV 生产文件，因此目前不能声称功能存在。旧 TIDA 引擎会加载冻结图像基线、EMA、calibrator 和历史 builder，与本计划冲突；新正式入口必须独立实现并通过 import/call trace 证明未调用旧 builder。

## 待检查的高风险项

1. DINO 上四块不能被外层 `no_grad` 或 `detach` 误冻结。
2. 25-query 后期回读不能退化为逐帧 25-token summary。
3. observer 的分类读取必须 detach，而 grounding loss 必须保留梯度。
4. 匹配 dustbin、IRLS 退化回退、actual PTS 和 gap 分段必须是真实现。
5. 238 因素读出必须逐样本精确重构 action/reason logits。
6. 正式 trainer 必须实际调用全部 loss，optimizer 参数恰好覆盖一次。
7. preflight 必须采集正式 factory，而不是单独 demo。

后续每轮监督在本文件追加源码位置、测试名、实测值与未解决风险。
