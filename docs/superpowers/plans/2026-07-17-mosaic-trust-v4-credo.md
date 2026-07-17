# MOSAIC-TRUST v4 CREDO 实现计划

> **For agentic workers:** 按任务逐项执行；每个功能必须有源码调用链和可运行验证证据，禁止以文件存在或单次 compile 代替功能验证。

**目标：** 在 `ce05e6235ffa8c8998e055941fcad2a91922a0b7` 基线上修复 IC-DOR 的 certificate deadlock，加入连续视觉可信度、类型化细粒度证据传输、reason/action 路由隔离和 owner 梯度约束，在不改无关逻辑的前提下完成可验证的 CREDO revised pilot。

**边界：** 本轮先实现和验证代码，不启动 full train。只有 revised pilot 的机制门槛和 artifact 校验通过，才允许提出 full train。

## 任务 1：锁定基线与配置

- [ ] 验证本地和远端均为 clean HEAD `ce05e6235ffa8c8998e055941fcad2a91922a0b7`。
- [ ] 复制并冻结旧 pilot 结果，不将旧 hard certificate 作为新训练 license。
- [ ] 扩展主 YAML、factor candidates、reason routes、action routes、certificate rules，记录 CREDO 的 source/role/polarity/negative policy、audit split、fine transport、schedule、owner lr/clip 和 revised pilot 参数。

## 任务 2：真实 observable ontology 与 grounding

- [ ] 重写 `mosaic_icdor_grounding.py`，解析 `trafficLightColor`、`laneDirection`、`laneStyle`、`laneTypes`、`areaType`、`occluded`、`truncated`，输出 source completeness、可靠负样本和 corridor 正负控制。
- [ ] 修改 observable predicate routes：删除 physical-near、indicator、turn-permission 等不可观测视觉正标签；front risk 仅作为 observable proxy。
- [ ] no-lane reasons 使用 absence polarity `v*(1-p)`，禁止 lane presence 作为 reason 9/15 的正证据。
- [ ] 增加 factor-aware audit sampling，audit_visual 与 audit_target 互斥且各 5%，不再取前 N 条。

## 任务 3：连续可信度与证据传输

- [ ] 新增 `mosaic_continuous_credibility.py`，计算与 reason labels 解耦的 `cV`、bootstrap/n_eff、content/prior、query/image shuffle、grounding/random、stability，落实未知/无可靠负样本/observed positive 不抬高 cV 的 caps，并支持 previous-epoch stopgrad EMA。
- [ ] 新增 `mosaic_typed_evidence_splat.py`，按 point/curve/region 真实 splat，输出 fine mask、坐标和诊断，并支持 fine/coarse ablation。
- [ ] 新增 `mosaic_factor_seeded_rereader.py`，读取 factor 坐标、sample features、attention，以 `tanh(offset)*0.08` 做局部 re-read，不退化为 coarse mask-only。
- [ ] 修改 observable predicate head 和 model，透传 factor coords/features/attention/fine masks，保证同一 DINO field 在 full/content/prior/shuffle 分支复用。

## 任务 4：reason/action 路由与非回归

- [ ] reason 保持 direct observed-reason visual 为 primary；latent route 从 epoch 0 激活；annotation residual 为 `visual + alpha*tanh(annotation-visual)`，alpha 初始化 `.05`、cap `.25`，不得使用 50/50 mixer。
- [ ] reason observed ASL=1.0、每标签 balanced normalization；queue ranking 只有正负覆盖存在时启用；隐藏 PU 失败只关闭 PU residual，不关闭其它路由。
- [ ] action 从 epoch 0 计算 shadow，但 final action 在 edge admission 前严格等于 visual-only；shadow delta bounded，gate init `.02`、cap `.05`，支持/veto 与 fine re-read 实际进入 shadow。
- [ ] edge admission 改为 per-action partial admission，必须同时检查 cV、TET/TES LCB、CCA、AP non-regression；不再要求四个 action 全部 certified。
- [ ] 把 old discrete certificate 限定为部署 admission，不能阻断学习访问、latent、reason 或 shadow 训练。

## 任务 5：损失、参数 ownership 与 schedule

- [ ] 按 CREDO 权重接入 action visual/rank/cardinality/shadow/matched-control、reason visual/obs-rank/annotationNLL/posterior-rank/semantic/escape、factor source-balanced/visibility/geometry/selective/view/prototype loss。
- [ ] 删除 HardPair 作为本路线训练项，不能与新 matched-control 重复计入。
- [ ] 实现 action/reason/factor/route/latent-threshold owner 分组、per-owner clip 和 forbidden gradient checks：`grad_theta_A L_R=0`、`grad_theta_R L_A=0`、`grad_theta_F L_R_obs=0`。
- [ ] 重写 schedule 为 Regime A epoch 0 active + visual final、Regime B partial admission、Regime C freeze/low LR；transition 由连续机制指标驱动，不由 all-four/15-reason master gate 驱动。
- [ ] train-calib threshold 独立，test 只评估，不参与 teacher/LR/route admission。

## 任务 6：测试与验证

- [ ] 先写并运行 CREDO 计划列出的 15 个 RED tests，再实现最小代码使其 GREEN。
- [ ] 对全部 MOSAIC/ACPR 回归测试运行 py_compile 和 targeted pytest。
- [ ] 动态验证 model forward、final action visual-only 等价、reason residual non-regression、fine coordinate shuffle effect、policy weights consumed、batch-local DINO reuse、per-owner clipping。
- [ ] 运行 revised pilot：单 seed、train_core=4096、audit_visual=512、audit_target=512、train_calib=512、test=512、6 epochs；保留完整机制 artifacts，不生成 cache，不读取 test labels/BDD100K 作为 forward。
- [ ] revised pilot 后按 CREDO gates 判定是否允许 full train；未满足则只记录阻塞，不启动 full train。

