# TIDA-OIA V1 监督日志

## 原始请求与约束

- 按用户提供的 TIDA 实现计划和严格审查 Skill 完整实现视频方案，不能以“能跑”为完成标准。
- 必须新建 worktree，旧 VETRA 代码和实验不被覆盖。
- 正式数据仅使用三批已核验视频：3115 train、885 test，共4000条。
- 代码、真实视频运行和 hash 绑定 artifact 共同证明功能，不允许固定零、空壳、仅实例化未调用或审计自证。

## 计划冲突处理

原计划第1.2节和原审查第3节禁止新 worktree；用户本轮明确要求新 worktree。最新用户指令优先，因此执行分支改为 `tida_oia_v1_video`，源 HEAD/tree 保持 `af6f526`/`9c885b8`。所有原计划中写死 `vetra_from_scratch_staged_v1` 的 Git 审查目标相应改为新分支。其余方法、数据、训练和审查合同不变。

## 覆盖与忠实度门

完整覆盖矩阵位于 `docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md`。每个 must-have 均映射到实现面和动态验证证据。当前状态：设计与计划已整理，监督审查进行中，尚未开始核心代码实现。

## 监督审查轮次

### Round 1

- 发送内容：原始计划、严格 Skill、最新 worktree 覆盖指令、数据事实、覆盖矩阵要求。
- 监督结果：`CHANGES_REQUIRED`。指出 worktree/Git 目标、L0 重复权重、贡献精确和、DINO 双注册、独立 golden oracle、query 聚合与禁压缩语义、profile 单位、test 白名单、跨 split 近重复、前5% differential 梯度、缺失 artifacts/contracts、逐样本末帧门槛和 repeated-last 定义等问题。
- 是否传达给执行侧：已传达。
- 执行侧处理：全部接受；已补入 design 与 implementation plan。L0 仅加权一次；贡献使用解析缩放和数值残差修正；DINO 采用非注册 weakref；加入源树 oracle、近重复检测、phase-aware 梯度矩阵、100-update profile、明确 test 白名单和完整 completion schema。

### Round 2

- 发送内容：修订后的 design、implementation plan 与 Round 1 逐项处置。
- 监督结果：`CHANGES_REQUIRED`。要求 Skill 编码前安装；明确 train_core/calib/audit；给出 route_sparse 公式与 gradient path；量化末帧/近重复门槛；展开 source oracle 张量；分阶段 PASS nullable schema；固定 base commit/Git 命令；展开 required files/tests。
- 是否传达给执行侧：已传达。
- 执行侧处理：全部接受并补入 design/plan。正式划分为2291/312/512且 source-group exactly disjoint；补入 route 公式；逐样本 SSIM/PSNR/NMAE/pHash 门槛和近重复算法；golden oracle 完整张量清单；阶段 schema；base commit和仅push新ref命令；完整生产/测试文件清单。

### Round 3

- 发送内容：第二次修订文档、编码前 Skill 安装结果与逐项处置。
- 监督结果：`CHANGES_REQUIRED`。Skill 正文仍残留旧 worktree failure、抽样数据门、共同路径等价、缺少前5% route 动态门、统一PASS schema、Skill后改漂移和 wildcard 测试 inventory；partition 还缺确定性 tie-break。
- 是否传达给执行侧：已传达。
- 执行侧处理：全部接受。Skill 正文已逐节改写；partition seed固定20260821并规定SHA排序、exact subset-sum、lexicographic tie-break；required tests由plan逐名解析；Skill hash在DESIGN后冻结。

### Round 4

- 发送内容：第三次修订文档和已消除正文冲突的冻结 Skill。
- 监督结果：`CHANGES_REQUIRED`。仅剩 route non-null 项缺 mean、旧 HEAD 可前移句冲突、manifest 缺 official_split/partition 字段。
- 是否传达给执行侧：已传达。
- 执行侧处理：全部接受。route 三项均归约为标量；TIDA base固定；manifest明确定义 official_split 与 partition 及一致性约束。

### Round 5

- 发送内容：最终冻结 design/plan/Skill。
- 监督结果：`CHANGES_REQUIRED`。唯一缺口是 Skill 仍保留旧 split 必填，可能与 partition 双重真源冲突。
- 是否传达给执行侧：已传达。
- 执行侧处理：删除旧 split；正式 manifest 禁止该字段，所有过滤只认 partition。

### Round 6

- 发送内容：消除最后一处 manifest schema 双重真源后的冻结文档。
- 监督结果：`APPROVED`。
- 是否传达给执行侧：已传达。
- 执行侧处理：design、implementation plan 与 Skill 冻结；允许进入 RED tests 和实现阶段。任何后续 Skill 变更会使本批准失效并重新复审。

## 最终验证证据

尚未生成。完成时必须记录 compile、targeted/regression pytest、4000条真实数据审计、真实DINO/视频 smoke、机制干预、显存 profile、exact resume、Git SHA、review pass 和正式训练结果。
