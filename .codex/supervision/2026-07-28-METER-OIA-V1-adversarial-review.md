# 双代理监督日志：METER-OIA V1 对抗式实现复核

**日期：** 2026-07-28  
**状态：** 修复完成，等待 clean-HEAD real-DINO profile 与最终 audit  
**主执行端：** 当前 Codex 主会话  
**监督端：** Helmholtz，agent `019fa82b-3269-7c01-90fa-823900e1bf53`

## 1. 用户原始要求

根据 `Codex_METER_OIA_V1_ImplementationPlan_20260728.md` 继续对抗性审查并完整实现代码。验收标准不是“可以训练”，而是计划功能全部进入正式 forward、loss、optimizer、evaluation、artifact、resume 和 supervisor 调用链；不得遗漏、占位、错误实现或留下逻辑冲突。本轮禁止启动 pilot/full train。

## 2. 固定边界

- worktree：`E:\sbw\FATE_Drive\fate_oia_acpr_meter_oia_v1_worktree`
- branch：`acpr_meter_oia_v1_direct_image`
- base：`acpr_calalign_v1_2@373aa49feac17372574fd7fb056c1d79c7c848fe`
- 保留 frozen DINO、direct image、360x640、完整 3600 patches。
- 禁止 cache、compression、pair memory、hard pair、action-set final、trainable threshold/calibration representation path、graph/PMI/co-occurrence、历史 checkpoint distillation。
- 本轮不启动训练。

## 3. 监督审查结论

监督端结论为 `changes_required`，不允许 pilot/full train。关键发现：

1. grounding evidence objective 会推动 support/counter score 相等，且 mirror loss 缺失；
2. meta utility 将同一个全局 utility 复制给四个 factor，没有逐 factor 虚拟更新；
3. calibration 只有 prevalence matching，没有 temperature、global/group shrinkage、回退和 mAP/RMS 约束；
4. full train 可只持有 PRE_PILOT_READY 而绕过 pilot；
5. PU audit 没有在 deliberately hidden positives 上计算 eligibility，PU 又被重复乘 lambda；
6. reason zero 权重没有 observability；
7. counterfactual 没有累计质量选择、严格 non-selected neighbor replacement 和 target-specific wrong factor；
8. private reason decoder 缺 reason self-attention、可学习多层融合和 step-0 candidate 监督；
9. optimizer/runtime 缺 no-decay、effective-batch LR scaling、TF32、真实 profiler 和真实 owner delta；
10. source hash、micro-step resume、standalone evaluator、failure/evidence case artifacts 不完整。

## 4. 已采纳并完成的修复

- grounding 改为 source-confidence 加权的 map NLL、presence/signed-direction margin、mirror balance 和 compactness。
- meta utility 改为逐 factor 行 mask、逐 factor candidate、逐 factor utility/EMA/omega。
- PU 使用完全 detached 的 private decode；trainer 不再二次乘 lambda。
- PU eligibility 改为 deliberately hidden positive 对 originally observed-zero 的 AUPRC 比较。
- reason negative weight 使用 `0.1 + 0.3 * observability * (1-evidence)`。
- private reason 新增一层 reason self-attention、三层可学习 router，并让 candidate 从 step 0 接受主 reason loss。
- counterfactual 使用累计 evidence mass、同 sector/相近纵向/低响应 control、排除 selected patch 的 3x3/5x5 neighbor replacement、target-specific factor/wrong factor。
- grounding 和 counterfactual 使用独立 5%/10% update ramps。
- calibration 搜索 temperature、global、group、group-shrinkage、per-label；保持 mAP，不超过 threshold RMS 限制，退化时回退 group/global/raw。
- metrics 增加 per-label AUC 和 mAUC；evaluator 增加计划要求的 branch aliases 与 selector visual/semantic 隔离。
- optimizer 按 owner 分组，norm/bias/embedding no decay，LR 按 effective batch / 32 缩放，启用 TF32。
- runtime log 增加真实 parameter delta、owner optimizer step count、zero-gradient rate 和分段时间。
- profiler 改为真实图像、真实 DINO、5 warmup + 20 measured optimizer updates，并单测 counterfactual/meta/calibration 事件并按频率摊销。
- standalone evaluator 真实加载 checkpoint/test dataset 并写 evaluation summary。
- failure/evidence cases 改为真实样本记录，不再写说明文字占位。
- checkpoint source hash 改为当前实现 HEAD；增加 deterministic epoch loader、mid-epoch checkpoint 和 micro-step resume。
- pilot 固定规模恢复为 `4096/1024/512/512`。
- trainer、PowerShell、supervisor 三层都要求 full 模式持有 `METER_OIA_V1_FULL_TRAIN_READY.json`。
- 3-epoch pilot 只有通过 action/reason/meta/evidence/显存/GitHub HEAD 全部门槛才生成 FULL_TRAIN_READY。
- audit 增加上述语义的动态和源码 hard checks，real-DINO 动态检查使用真实 backbone。

## 5. 对抗测试与复核轮次

| 轮次 | 证据 | 结论 |
| --- | --- | --- |
| RED 1 | 7 个新增语义测试全部失败 | 证明 optimizer、decoder、PU、calibration、AUC、full gate 均为真实缺口 |
| 修复回归 1 | targeted `32 passed` | 第一批语义修复成立 |
| 全量回归 1 | METER `64 passed` | METER 旧测试与新增测试兼容 |
| 全量回归 2 | 全仓 `154 passed` | ACPR/FATE 旧路径未被破坏 |
| Resume 加固后 | 全仓 `155 passed` | deterministic next-update/micro-step 合同成立 |
| Mock dynamic audit | 所有 functional/contract/dynamic checks 为 true | 仅因 real-DINO profile 尚未执行而正确拒绝 PASS |

## 6. 计划保真与冲突

- 未替换研究主线，所有修复都属于原 METER 计划合同的落实。
- calibration 的 temperature 会使 deploy 公式成为 `logits / temperature - theta`；旧测试只接受 `logits - theta`，与本计划“搜索 threshold 和 temperature”冲突。已按当前 METER 计划更新旧测试，同时保留 raw/deploy 分离和 mAP 不变性。
- 监督端建议恢复完整 pilot 样本规模，已采纳。
- 未采纳任何放宽 gate、跳过 pilot 或直接启动 full train 的做法。

## 7. 剩余硬门槛

1. 在提交后的 clean HEAD 上重跑全仓 tests。
2. 执行严格 real-DINO/real-data runtime profile。
3. 用 clean HEAD 和 profile 重跑 real-DINO implementation audit。
4. REVIEW_PASS/PRE_PILOT_READY 必须绑定当前 clean HEAD。
5. 更新 canonical `task_plan.md`、`findings.md`、`progress.md`，提交并推送；核验 GitHub HEAD。

## 8. 当前判定

- **代码语义审查：** 已完成监督端提出的修复，mock dynamic audit 全部通过。
- **可报告最终完成：** 尚不可；缺 clean-HEAD real-DINO profile/audit 证据。
- **允许 pilot：** 尚不可。
- **允许 full train：** 不可；必须先完成严格 3-epoch pilot 并生成 FULL_TRAIN_READY。

## 9. Final evidence update

- Final clean HEAD: `8e1c066bf026767bd83ee2210b69fa193f6fc966`; GitHub branch HEAD matches.
- Full repository verification: `156 passed`; one unrelated existing TypedStorage deprecation warning.
- Real-DINO runtime profile completed with 5 warmup and 20 measured optimizer updates per accepted comparison. Selected batch 6, accumulation 5, workers 4, prefetch 2; peak reserved 42.3652 GB; event-adjusted 10.2270 samples/s.
- Final real-DINO audit: `pass=true`, `missing_items=[]`, `warnings=[]`; all functional, contract, and dynamic checks passed.
- `REVIEW_PASS_METER_OIA_V1.txt` and `METER_OIA_V1_PRE_PILOT_READY.json` were generated.
- Supervisor verdict: implementation and pre-pilot audit closure are complete. Pilot/full training were not started. Full training still requires a strict pilot-generated `FULL_TRAIN_READY`.

## 10. Pilot 失效后的根因修复复审

**复审日期：** 2026-07-29  
**监督端：** `019fa909-9501-7a13-9abf-a37411fbf6ee`  
**复审状态：** 仅批准进入新的 real-DINO smoke/pilot，不批准直接 full train。

严格 pilot 证明视觉主干能够学习，但创新链存在真实失活：support/counter null 为 0、二者 cosine 约 0.99，semantic contribution RMS 为 0，semantic AP 跨 epoch 恒定，selector 接近 0.99 visual，meta utility/omega 为 0，counterfactual 仅覆盖 2 个 action 和 1 个 factor。

执行端按 TDD 修复以下根因：

- signed factor 将 3600 patch 分布与独立可学习 null mass 做联合归一，继续满足 `patch_map.sum(-1) + null_mass = 1`。
- support/counter query embedding 使用幅度更强且严格相反的初始化，保留 `q+=H+e+`、`q-=H+e-` 合同。
- semantic action 的 21-factor 分布与独立可学习 null mass 联合归一，避免 entmax 训练后退化为 null-only bias。
- 保留 softmax 到 entmax 的前 10% progress 过渡、完整 additive semantic expert、selector、meta utility 和 counterfactual 数据流。

监督端代码级结论：

- 修复保持 METER 计划的归一化、稀疏化和 additive contribution 语义。
- 新 maps/null 真实进入 reliability、semantic action、reason local、meta utility 和 counterfactual。
- 暂不修改 counterfactual selection；先观察上游贡献恢复后是否自然达到 4 actions / 12 factors，避免为过 gate 伪造覆盖。
- 若 null、map 分离、semantic contribution 已恢复而 CF 仍长期低覆盖，才允许将其判为独立采样偏置并按 TDD 修复。

**执行端采纳情况：** 全部采纳。  
**新鲜验证：** 全仓 `164 passed`，另有 1 个与本任务无关的 TypedStorage 弃用警告。  
**下一门槛：** clean commit/push -> real-DINO audit/readiness -> 真实 smoke/pilot 机制数值验证 -> full train。
