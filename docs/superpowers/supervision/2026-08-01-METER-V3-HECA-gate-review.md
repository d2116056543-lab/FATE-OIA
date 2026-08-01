# 双代理监督日志：METER-OIA V3 HECA 训练前代码复核与 Gate

**日期：** 2026-08-01  
**任务：** 用户要求继续代码级审查；只有计划功能全部真实实现、调用且逻辑无冲突时，启动 HECA Gate，使项目进入最终训练准备阶段。  
**状态：** 修订中  
**主执行端：** Codex 主会话  
**监督端：** 待分配

## 1. 原始请求

“继续进行代码级别审查 如果代码功能和计划完全一致那你就开始进行gate。我要的是能进入最终训练的阶段而不是停在前面……代码完美实现计划，一丝不漏，不要错误实现或者遗漏实现，而且保证代码逻辑通顺没有冲突；注意 token 消耗。”

计划来源：`C:\Users\WLJTXY\Downloads\METER_OIA_V2_Audit_and_V3_HECA_Package_20260801\METER_OIA_V3_HECA_Final_Implementation_and_TrainingPlan_20260801.md`。

## 2. 适用 Skill

- `dual-agent-supervision`：对计划保真、功能全覆盖和 Gate 前审查建立独立监督链。
- `executing-plans`：依照 V3 HECA 的 T01--T18 顺序复核并执行 Gate 阶段。
- `requesting-code-review`：在开始真实 Gate 前进行独立代码审查。
- `verification-before-completion`：只以新鲜的远端测试、审计和 Gate artifact 作为结论依据。

## 3. 初始计划

1. 对当前 clean HEAD 建立功能覆盖/计划保真矩阵，并检查远端工作树与 GitHub 的一致性。
2. 独立审查当前实现的模块调用链、梯度所有权、loss 接线、调度、artifact 和 Gate 防绕过逻辑。
3. 若发现 P1/P2 缺口，先补测试和代码，重新审查；无缺口后运行 implementation audit 与真实 4-epoch pilot Gate A--G。
4. Gate 结果全部通过才允许正式 fresh 14-epoch training；任一 Gate 不通过则记录证据并停在 Gate，不伪造 REVIEW_PASS。

## 4. 功能覆盖矩阵

| 编号 | 用户计划必须项 | 实现/调用位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- |
| 1 | frozen DINO、3600 token、无 cache/compression/RunC | `meter_oia_model.py`、config、audit | config/forbidden scan/dynamic forward | 已验证，待本轮复核 |
| 2 | rank-16 shared/action/reason private adapters 与严格梯度归属 | `meter_meta_adapters.py`、trainer、HECA optimizer | autograd tests、gradient ownership artifact | 已验证，待本轮复核 |
| 3 | typed factor anchor/state/observability/reliability 与 factor tau | signed factors、typed targets、static artifact engine | unit tests、tau/source artifact、Gate B | 已实现，待 Gate |
| 4 | 无 hard compatibility/admission；all-action soft allocation、state-conditioned exact credit | schema、semantic action、model decode | static scan、contribution tests、Gate C/F | 已实现，待 Gate |
| 5 | 5% selective bridge；anchor action grad 为零 | signed factors、optimizer | gradient tests、Gate E | 已实现，待 Gate |
| 6 | CalAlign-anchored global reason、detached correction、PU private-only | reason decoder/losses/trainer | reason/PU tests、Gate D/E | 已实现，待 Gate |
| 7 | action/reason/measurement losses、excess-risk balance、adaptive cap/LR、round-robin corruption | loss files、optimizer、trainer | registry/schedule tests、Gate G | 已实现，待本轮复核 |
| 8 | same-forward diagnostics、每轮 branch/typed/runtime artifacts、128 patch audit | eval/artifact/trainer | schema tests、Gate F | 已实现，待 Gate |
| 9 | 4-epoch A--G pilot，clean HEAD、raw-evidence recomputation | pilot evaluator/trainer/supervisor | pilot protocol tests、真实 Gate run | 已实现，待执行 |
| 10 | 14-epoch fresh final run 仅在 Gate 后允许 | foreground supervisor/config | readiness protocol tests | 已实现，尚未授权启动 |

**覆盖结论：** 每个必须项均映射到实现与验证；当前未验证项均是需要真实 pilot 才能判定的机制效果，不以静态审计替代。

## 5. 用户计划保真矩阵

| 计划项 | 保留情况 | 证明方式 | 当前状态 |
| --- | --- | --- | --- |
| T01--T02 固化 V2 与独立 V3 worktree | 已保留 | branch/worktree/HEAD | 已验证 |
| T03--T12 HECA 结构与优化改造 | 已保留，无合并替换 | source audit + targeted tests | 待独立复核 |
| T13 diagnostics | 已保留 | branch/typed/runtime artifacts | 待真实 Gate |
| T14 unit/autograd tests | 已保留 | `test_heca_*.py` | 已验证，待 fresh run |
| T15 real-DINO profile | 已保留 | Gate G runtime artifact | 待真实 Gate |
| T16 4-epoch pilot | 已保留为严格前置条件 | A--G evaluator/recomputation | 待执行 |
| T17 adversarial mechanism review | 已保留 | 本监督日志与独立审查 | 进行中 |
| T18 fresh 14-epoch full run | 未提前执行 | supervisor readiness | Gate 后执行 |

**保真结论：** 没有简化、跳过或用静态 PASS 替代 T16/T17；本轮顺序保持为复核 -> Gate -> 最终训练准备。

## 6. 监督审查

**是否已发送给监督端：** 待发送  
**审查结果：** 第 1 轮 `CHANGES_REQUIRED`。发现当前 corruption mode 使用 `optimizer_step`，在 grad accumulation=5 时连续五个 batch 使用同一扰动；初步修订为 `epoch * len(loader) + micro_step` 后，监督端指出 OOM fallback 改变 loader 长度时无法保持 resume exactness。

## 7. 计划修订

**监督结论是否已传回执行端：** 是。

**已采纳：**

- 使用独立、持久化的全局 `corruption_microbatch_index`，不从 `optimizer_step` 或当前 loader 长度推导。
- 每个 micro-batch 恰好执行一种 identity corruption，按 `schema -> cross_sample -> state` 循环。
- 把该计数保存/恢复到 `HECAScheduleState`，使 OOM fallback 或 epoch resume 不改变下一 batch 的 phase。
- 增加累积=5、尾部 batch、resume exactness 和 optimizer/scheduler 不变的回归测试。

**未采纳：** 无。

## 8. 复审轮次

| 轮次 | 结论 | 是否允许执行 Gate | 剩余问题 |
| --- | --- | --- | --- |
| 1 | CHANGES_REQUIRED：corruption phase 不可在 OOM fallback 后精确恢复 | 否 | 持久化 micro-batch counter 与回归测试 |
| 2 | 待复审 | 否 | 等待修复验证 |

## 9. 执行交接

Gate 启动前必须写明监督批准、当前 HEAD 和远端 clean 状态。修订后的执行交接为：先以 RED test 证明缺少持久化 counter，再改动 `HECAScheduleState`、trainer 与 checkpoint state，跑完整 HECA tests/audit，复审批准后才启动 Gate。

## 10. 执行合规检查

待 Gate 完成后检查所有 must-have、计划保真和任何偏离。

## 11. 验证证据

已记录：远端 pre-Gate implementation audit 通过，`test_heca_*.py` 在初步微批轮换修复后为 51 passed；后续必须以持久化 counter 修订后的新鲜验证替代该结果。

## 12. 最终判断

当前不得声明可进入 full train；需要独立审查批准并完成真实 A--G Gate。

## 13. Round 2 Resolution (2026-08-01)

- The earlier loader-length-derived phase was replaced with the persisted
  `HECAScheduleState.corruption_microbatch_index`. It is incremented after
  every micro-batch backward call, before the gradient-accumulation update
  branch, and is included automatically in checkpoint state.
- `_compute_losses` now requires `corruption_step` explicitly. The former
  fallback to `optimizer_step` was removed from the signature and all three
  real call sites (trainer, dynamic audit, and smoke test) pass the value.
- Regression coverage now checks the first ten micro-batches, a final short
  accumulation window, checkpoint round-trip with a changed batch/accum
  configuration, explicit caller wiring, and the absence of an implicit
  default. The remote HECA suite result is `53 passed`.
- A fresh remote implementation audit on the modified tree passed source,
  compile, pytest, protocol, forbidden-pattern, and dynamic forward checks.
  It correctly reports that pilot Gates A--G are still unevaluated.
- Independent reviewer `Kepler` returned `APPROVED` after the fallback removal.

| Round | Result | Gate authorized | Remaining prerequisite |
| --- | --- | --- | --- |
| 2 | APPROVED | Yes, begin real pilot Gate A--G | Commit current fix; rerun audit on clean HEAD; run the four-epoch pilot |

## 14. Offline Export Remediation (2026-08-01)

- The first Gate launch exposed a real preflight defect: the one-time ontology
  exporter was hard-coded to an uncached MiniLM model and attempted an online
  Hugging Face download. This violated the plan's offline frozen-text-tower
  contract, so no training epoch was allowed to start.
- Replaced the exporter default with the permitted project-local frozen
  `bert-base-uncased` snapshot, made `local_files_only=True` mandatory, and
  forced the PyTorch backend to avoid the remote TensorFlow/protobuf import
  incompatibility. The BERT snapshot and generated tensors remain ignored
  artifacts; runtime METER model code continues to load only static tensors.
- A review found one further bypass: the pilot accepted a caller-overridable
  text encoder path. The override was removed. The script now uses only the
  YAML-matched BERT path and the audit verifies both the literal path and the
  `--encoder_id` wiring.
- Evidence after this remediation: real remote BERT load passed; static 21x768
  factor and 21x3x768 state prototypes were generated with hashes; 55 HECA
  tests passed; clean-head audit passed with `offline_ontology_export=true`;
  independent reviewer `Kepler` returned `APPROVED`.

## 15. Readout And Admission Repair (2026-08-01)

- The prior real-data pilot is stale, not a full-training pass: Gates B/C/D
  failed for evidence-supported numerical reasons.
- Factor state values are now factor-specific (`V_r`) and action-conditioned
  (`U_a q_ia`). A zero-initialized state effect preserves the visual action
  anchor at initialization while receiving the first update gradient.
- The action non-regression term uses detached adaptive visual margins, and
  PU reason correction balances observed-positive and trusted-unknown groups.
- Gate B now evaluates only train-audit factors with real positive and
  negative support, comparing AUC to chance and AP to prevalence.
- The formal supervisor and direct full-run validator now reject tracked edits
  and untracked runtime source under `fate_oia/`, `configs/`, or `scripts/`.
- Remote verification before the next clean-HEAD audit/pilot: `py_compile`
  passed and the full current HECA suite passed (`80 passed`).
