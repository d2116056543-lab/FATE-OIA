# 双代理监督日志：METER-OIA V2 / TESA 最终补全

**日期：** 2026-07-29  
**任务：** 在不启动训练的前提下，逐项补全 V2-TESA 计划，禁止部分实现、占位实现和假 REVIEW_PASS。  
**状态：** 代码补丁已提交；最新远端复验和 CUDA gate 仍阻塞。  
**主执行端：** 当前 Codex 主会话  
**监督端：** `Lorentz`（`019fad4f-64dc-70a3-b79c-9c2f3d897e62`）

## 1. 功能覆盖矩阵

| 编号 | 必须功能 | 实现位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- |
| 1 | dense necessity 与 specificity 恰好加入一次 | `train_acpr_meter_oia.py` | RED test + trainer source + smoke loss | 已实现，待最新 smoke 复验 |
| 2 | paired mirror 进入正式训练且 ordinary batch 只编码一次 | trainer/grounding loss/config | one-call test + trainer smoke | 已实现，最新 crash 已修，待复验 |
| 3 | `objects[].poly2d` lane 与显式 lane state | typed target builder | object-poly2d/unknown tests | 已验证 |
| 4 | progress=0 action/reason/label-node 等价 | implementation audit | 动态 forward | 已验证 |
| 5 | exact additive action 公式 | semantic action/audit | 动态重构误差 | 已验证 |
| 6 | active PU 仅进入 reason-private | implementation audit | 动态反传 ownership | 已验证 |
| 7 | Gate C/D/G 检查真实 identity/reason/PU | pilot evaluator | strict gate fixture + real pilot | 已实现，真实 pilot 未运行 |
| 8 | strict artifact schema | artifact validator | valid/invalid fixture + smoke artifact | 已实现，最新 smoke 待复验 |
| 9 | real-DINO runtime/profile/pilot | 远端 CUDA | `nvidia-smi`、real-DINO gate | 阻塞 |

## 2. 本轮确认并修复的缺口

1. dense intervention 总损失此前只加入 necessity，specificity 虽被计算但未进入训练。现改为 trainer 仅加入一次 `dense["total"]`，并保持 action loss 不重复加入。
2. mirror equivariance 此前只有随机镜像增强和 epoch audit，没有成对训练约束。现通过“普通 batch + 一张水平镜像”拼接后只调用一次 `encode_images`，每 8 个 microstep 启用 paired mirror loss。
3. BDD100K lane `poly2d` 位于 `frames[].objects` 时此前未进入 typed target。现纳入 lane rasterization；solid/dashed/turn 只在显式属性存在时标注，未知保持 invalid。
4. implementation audit 仍使用旧 aggregate-tanh action 公式。现按逐 factor bounded contribution 的直接求和公式重构，并增加 progress=0 label-node 等价检查。
5. PU audit 此前只验证 lambda=0 时关闭。现增加 active lambda 动态反传，要求梯度只进入 reason-private decoder。
6. pilot Gate C/D/G 此前没有硬检查 target-vs-wrong identity、grounded reason identity drop 和逐标签 PU LCB。现已加入。
7. epoch artifact validator 此前只检查文件存在和张量 shape。现增加 typed mechanism、train-calib calibration、runtime profile schema。

## 3. TDD 与验证证据

- RED tests 在远端确认 4 个预期失败：dense specificity 未接入、无 paired mirror trainer helper、objects poly2d lane 未接入、artifact schema 未硬卡。
- 修复后 targeted tests：`16 passed, 1 warning`。
- 修复后全量回归（HEAD `60d339c`）：`172 passed, 1 warning`。
- CPU 动态 audit（HEAD `bcf1194`）：
  - `progress_zero_action_error=0`
  - `progress_zero_reason_error=0`
  - `progress_zero_label_node_error=0`
  - `additive_error=4.765070116263814e-09`
  - action loss 到 reason gradient = `0`
  - reason loss到 foundation/factor/action gradient = `0`
  - active PU 仅 reason gradient 非零
  - audit `pass=true`

## 4. 端到端 smoke 发现

mock trainer 首 batch在 mirror loss 中因遍历输出字典并切片 0 维 tensor 崩溃。已新增回归测试，并在 HEAD `c9da51d` 将 mirror 输入限定为四个必要 batched tensor。

该最新修复尚未远端复验，因为随后 SSH 22 端口超时。

## 5. 外部阻塞

- 远端 `nvidia-smi` 无法与 NVIDIA driver 通信，CUDA/real-DINO/profile/pilot/full train 均不可执行。
- 最新连接检查显示 SSH 22 端口超时，因此 HEAD `c9da51d` 尚未同步回远端、尚未完成 full suite 与端到端 smoke 复验。

## 6. 监督状态

**监督结果是否已传回执行端：** 是。  
**执行端是否采纳：** 已采纳当前已确认缺口；等待监督端对 HEAD `c9da51d` 的最终复审。  
**最新状态：** 不允许生成 REVIEW_PASS，不允许 full train。

## 7. 后续硬门槛

1. SSH 恢复后同步 HEAD `c9da51d`。
2. 重跑全量 pytest、CPU mock trainer smoke、动态 implementation audit。
3. 修复任何新失败，直到 mock trainer 能完整写出并通过 strict artifact validation。
4. NVIDIA driver 恢复后运行 real-DINO runtime profile。
5. 运行严格 pilot A-H；全部通过后才允许生成 REVIEW_PASS 和启动 full train。

## 8. 最终判断

当前不能报告“100% 动态验证完成”。代码级缺口已继续补全，已有单元测试和 CPU 动态 audit 证据，但最新 mirror crash 修复尚未远端复验，real-DINO 与 pilot 也因 NVIDIA driver 不可用而阻塞。
