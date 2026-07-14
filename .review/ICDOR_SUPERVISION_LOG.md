# IC-DOR 双重监督记录

- 用户合同 SHA256：298AF9DD5D2AA2203DCF0A74D24792149D99752B08B1FE6898D360491159A806。
- 约束：仅实现与对抗式审查；本轮禁止训练、pilot、profile 或生成 REVIEW_PASS。
- 执行侧：主代理负责远端真实 worktree 的覆盖矩阵、TDD、代码与验证。
- 监督侧：同强度只读代理 Linnaeus（019f5a47-10d5-7a80-a480-7553d43c5c30）负责独立审计源路径与计划矛盾；不得改动文件。
- 远端命名修正：origin 指向 SNNA，故上游一致性校验 github/acpr_mosaic_ad_v1_direct_image；不改变模型、数据或训练协议。

## 覆盖门

icdor_feature_coverage_matrix.json 中每个 must-have 都必须有实现位置、真实调用路径和动态验证。任一项为占位、未调用、无法验证或与禁止路径冲突，状态保持 changes_required，不得进入 pilot/full training。

## 当前监督状态

- 初始计划：待审查。
- 代码执行：未开始。
- 训练：未启动。
- REVIEW_PASS：禁止生成。
# 2026-07-14 第三轮对抗审查与纠正

## 第四轮最终复审

- 冻结对象：clean committed HEAD `84efbaa9ed6c2ef01681362a5858db0da62d7acb`。
- 监督裁决：`APPROVED`。
- 结论：未发现仍未实现、未调用或占位的计划代码功能；T01-T26、T28 代码通过。
- T27 runtime profile、T29 pilot/REVIEW_PASS、T30 full run 是未执行 Gate，不属于代码缺失，也未被伪报为完成。
- 执行端已遵循第三轮全部修订；真实 DINO audit 和 82 项测试作为独立证据，不替代逐项源码审查。

- 监督结论：第三轮返回 `CHANGES_REQUIRED`，未允许用测试通过替代功能覆盖。
- 已传回执行端并采纳：clean commit tree/contract manifest 双重绑定；resume 先恢复后初始化；checkpoint 保存 Python/Torch/CUDA RNG；visual matched-random 使用同 factor 等质量空间 roll；strict schema 检查 mask 实体、transfer 和梯度 firewall；`--fail_closed` 真实失败；full CLI 不可绕过 REVIEW_PASS。
- RED：新增回归测试初次运行出现 3 failures，暴露 tree 未绑定与 resume 顺序错误。
- GREEN：修正后定向 26 passed；完整 IC-DOR suite 曾达到 81 passed，clean commit 前还需 fresh 全量复验。
- 训练状态：未启动 profiler、pilot 或 full train；未生成 REVIEW_PASS。

## 计划与最新粘贴文本冲突矩阵

1. 计划要求每 epoch 只评 test、按 test deploy-fixed joint 选 best；最新文本要求 validation/train-audit 选 representation、test 最终一次。按用户明确裁决采用计划；该结果仅是内部工程上界，不能当论文无偏 test。
2. 计划要求每 epoch 后做 train-calib threshold pass；最新文本要求 representation 全部训练结束并冻结后再 calibration。按计划实现，test 标签不进入阈值学习。
3. 最新文本新增冻结 text encoder、3-5 prompts、prompt centroid 初始化和正负 prompt 噪声证据；原计划未要求，记录为计划外新增建议，不擅自引入。
4. 最新文本要求人工检查 200 个 observed-zero 高 q 样本且 precision>=0.80；代码只可导出 top-200，不能伪造人工 precision。该人工科学 Gate 未完成，阻止 PU 真值恢复主张。
5. 计划运行环境写 `sbw39`，远端实际可用为 `damo39`；仅替换解释器，manifest 保存真实命令。
6. 计划示例 remote 为 `origin`，远端 `origin` 指向 SNNA，FATE-OIA GitHub remote 名为 `github`；source manifest 固定实际 remote alias/HEAD。
7. 计划的 test-best 与最新文本的 paper-test-once 无法同时满足；实现保留计划工程协议并明确论文边界。

## 当前边界

- 所有第三轮审查指出的代码缺口已修补，等待 clean commit 后第四轮最终审查。
- runtime profile、4-epoch pilot、REVIEW_PASS、full train 是未执行 Gate，不得声称已完成。
