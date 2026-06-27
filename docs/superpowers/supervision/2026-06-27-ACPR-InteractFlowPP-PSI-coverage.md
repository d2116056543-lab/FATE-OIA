# 双代理监督日志：ACPR-InteractFlow++ V1 PSI 代码覆盖审查

**日期：** 2026-06-27  
**任务：** 根据用户提供的 ACPR-InteractFlow++ V1 PSI/DAMO 11902 package，继续核对并补齐代码结构，确保功能细节尽量与计划一致；在训练前不得只满足“能跑”，必须覆盖关键功能和 gate。  
**状态：** 执行中，基础 compile/pytest/audit 已通过；full train gate 未完成，禁止训练。  
**主执行端：** 主会话  
**监督端：** 同强度主会话自监督；未创建低强度子代理。

## 1. 原始请求

用户要求继续远端任务，基于 PSI 数据集和提供的 package 文件进行代码核对与推进。明确要求：

- 不急着训练。
- 根据用户文件检查代码结构。
- 不仅要代码跑通，更重要的是补全功能和细节。
- 全部覆盖计划内容。
- 一定要扣细节，跟计划文件一致。

## 2. 适用的 Skill

- `dual-agent-supervision`：用于防止计划遗漏、半成品实现、只跑通不覆盖功能。
- `brainstorming`：本轮涉及功能补齐和行为修改，使用用户 package 作为既定设计来源；未重新设计方案。

## 3. 初始计划

1. 读取用户 package 和远端 worktree 当前状态。
2. 对照计划与 checklist 检查核心结构：PSI dataset、15 observed frames、target frame exclusion、DINO direct image、predicate field、interaction flow、decision ledger、Exp29、loss、train/eval/audit。
3. 只补会导致功能偏离的代码，不启动训练。
4. 跑 py_compile、targeted pytest、基础 audit。
5. 保持 REVIEW_PASS 不存在，直到 real-DINO smoke、throughput/memory、A-K gates 真实执行。

## 4. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 使用远端 PSI 11902 public reconstruction，不接错数据集 | 必须 | `configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml`, `PSIDAMO11902Dataset` | dataset test 读取 train/test pkl、15 frame、Exp29 mask | 已验证 |
| 2 | 输入 15 observed frames，不能输入 target frame | 必须 | `psi_damo_dataset.py`, `visual_encoder.py` | dataset test + manifest/audit | 已验证 |
| 3 | Formal train 不使用 mock DINO | 必须 | `train_acpr_interactflow_psi.py`, `_build_model` | audit 静态检查 `use_mock_dino=True` 不在 formal trainer | 已验证 |
| 4 | 不使用 cache/compression/BDD-OIA legacy trainer | 必须 | config/train/audit | forbidden pattern scan | 已验证基础项 |
| 5 | Fast motion 使用 15 observed-frame tokens | 必须 | `visual_encoder.py`, `motion_path.py`, `model.py` | pytest forward + audit `motion_uses_full15_fast_tokens` | 已验证 |
| 6 | Predicate field 输出 trajectory/evidence geometry | 必须 | `dynamic_predicate_field.py`, `types.py` | pytest shape + audit shape | 已验证 |
| 7 | Predicate transfer 不能用 hash-byte pseudo embedding | 必须 | `predicate_transfer.py` | audit 禁止 `hashlib`，检查 `build_bow_text_embeddings` | 已验证 |
| 8 | Interaction flow 使用 entmax factor assignment | 必须 | `interaction_flow.py` | source + forward test | 已验证基础项 |
| 9 | Decision ledger 要精确 additive decomposition | 必须 | `decision_ledger.py` | `identity_error < 1e-5` + audit `ledger_exact_contribution_chain` | 已验证 |
| 10 | PSI action stop 指标必须对应 stop_car 类 | 必须 | `psi_metrics.py` | test 检查 `Act_stopF1 == per_class_f1[2]` | 已验证 |
| 11 | Loss 名称和 config 权重一致 | 必须 | `acpr_interactflow_losses.py` | loss test + train uses `cfg["loss"]` | 已验证基础项 |
| 12 | Train artifact 写中间量 | 必须 | `train_acpr_interactflow_psi.py` | 代码检查；未跑正式 epoch | 部分验证 |
| 13 | Eval 必须保存 ledger 中间输出，visual export 不能是 placeholder | 必须 | `eval_acpr_interactflow_psi.py`, `export_acpr_interactflow_visuals.py` | audit 检查 middle tensors + decision_ledger JSON/PNG/HTML | 已验证基础项 |
| 14 | A-K gates 全部完成后才能写 REVIEW_PASS | 必须 | `audit_acpr_interactflow.py`, `run_acpr_interactflow_preflight.py` | audit gate evidence false 时 REVIEW_PASS 不存在 | 已验证 |
| 15 | full train 前必须 real-DINO smoke/profile/mechanism/intervention/visual gates | 必须 | preflight engine | 尚未运行，因用户要求先不训练且远端可能有其他任务 | 阻塞 |

**覆盖结论：** 基础代码结构和关键张量流已补齐到可进入 gate 阶段；full train gate 尚未完成，不能报告“可训练完成”。

## 5. 用户计划保真矩阵

| 编号 | 用户计划项 | 必须遵循/可选 | 保留情况 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 基于用户 package 完整实现 InteractFlow++ V1 | 必须遵循 | 保留；仍有未跑的 gate | compile/pytest/audit/gates | 部分完成 |
| 2 | 不急着训练，先审查结构和细节 | 必须遵循 | 原样保留 | 无训练进程、无 REVIEW_PASS | 已验证 |
| 3 | 不仅能跑，还要功能性一致 | 必须遵循 | 已加硬 audit 和测试 | 新增 predicate geometry/ledger/transfer/motion checks | 已验证基础项 |
| 4 | 使用远端构造 PSI 数据集 | 必须遵循 | 保留 | dataset smoke/test | 已验证 |
| 5 | 训练前必须 gate 通过 | 必须遵循 | 保留 | REVIEW_PASS 不存在 | 已验证 |

## 6. 监督审查

**审查结果：** 需要继续 gate，不允许 full train。  
**必须修改项已采纳：**

- 修正 `model.py`：motion path 使用 `visual.fast_motion_tokens`。
- 修正 `model.py`/`decision_ledger.py`：ledger 使用 `factor_tokens + flow_edges`，不再用 `flow.mean` 伪 delta。
- 修正 `dynamic_predicate_field.py`：补全 predicate trajectory、evidence maps、centroids、corridor mass、confidence、transfer gate。
- 修正 `predicate_transfer.py`：去除 `hashlib` 伪 embedding，改为 ontology vocabulary BoW prototype。
- 修正 `psi_metrics.py`：`Act_stopF1` 改为 stop_car 类索引 2，并补 DAMO-style aliases。
- 修正 `train_acpr_interactflow_psi.py`：显式 optimizer param groups，修 benefit gate artifact。
- 修正 `acpr_interactflow_losses.py`：`action_global_soft_kl` 使用 `ledger.global_logits`。
- 修正 `eval_acpr_interactflow_psi.py`：保存 global/visual/motion/predicate/flow/calibration/ledger gates 中间 tensor。
- 重写 `export_acpr_interactflow_visuals.py`：从真实 eval tensor 生成 per-case `decision_ledger.json`、`decision_ledger.png`、`report.html`，缺失 tensor 时失败。
- 加硬 `audit_acpr_interactflow.py` 和模型测试。

## 7. 执行合规检查

**执行端是否照做：** 部分。  
**所有必须功能是否完整实现：** 部分。  
**所有必须遵循计划项是否保留：** 部分。

**证据：**

- `py_compile`：远端通过。
- `pytest tests\acpr_interactflow -q`：远端 `6 passed, 1 warning`。
- `audit_acpr_interactflow`：远端基础 audit `pass=true`。
- `gate_evidence.real_dino_smoke=false`。
- `gate_evidence.throughput_memory_profile=false`。
- `gate_evidence.all_A_to_K_gates_true=false`。
- `REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt` 不存在。
- InteractFlow 训练进程数为 0。

**偏离/未完成：**

- 尚未执行 real-DINO tiny smoke。
- 尚未执行 throughput/memory profile。
- 尚未执行 128-sample mechanism fit。
- 尚未执行 temporal/lag/intervention/visual artifact gates。
- 尚未 code-only commit/push。

## 8. 验证证据

**运行的命令摘要：**

- `E:\Anaconda\envs\sbw39\python.exe -m py_compile ...`
- `E:\Anaconda\envs\sbw39\python.exe -m pytest tests\acpr_interactflow -q`
- `E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_interactflow --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml --output_dir .background_runs\acpr_interactflow_pp_v1_preflight --device cpu`

**输出摘要：**

- pytest：`6 passed, 1 warning`
- audit：`pass=true`
- audit functional checks：dataset/direct-image/test-only/15-frame motion/text-transfer/predicate-geometry/ledger/state-bank 均为 true
- audit functional checks 额外验证：eval ledger middle outputs 与 visual ledger export 均为 true
- gate evidence：real smoke/profile/gate summary 仍为 false

## 9. 最终判断

**是否可以报告完成：** 否。  
**理由：**

- 代码结构已推进到“基础审查通过、可进入 preflight gates”的状态。
- 但用户计划要求 full train 前必须 gate 全过；当前 A-K gates 未真实执行。
- 因此当前只可报告“代码补齐与基础验证阶段完成”，不能报告“可启动正式训练”。
