# LENS-OIA V1 实施与对抗审查矩阵

## 已批准的设计边界

- 唯一规范：用户提供的 `Codex_LENS_OIA_V1_ImplementationPlan_20260805.md`。
- 实施基线：`acpr_calalign_v1_2 @ 373aa49feac17372574fd7fb056c1d79c7c848fe`。
- 远端 Git 别名差异：该机器的 FATE-OIA 远端名为 `github`；`origin` 指向无关的 SNNA 仓库。所有 LENS Git 操作使用 `github`，但基线提交、代码与协议不变。
- 未经用户批准，不启动 pilot 或 full train；先完成 RED tests、实现、compile、pytest、真实 DINO smoke 与审查。

## 覆盖矩阵

| 计划合同 | 实现位置 | 对抗验证 |
| --- | --- | --- |
| 冻结 DINO 3/7/11、一次 3600 patch 编码 | `lens_calalign_foundation.py`, `lens_oia_model.py` | 形状/无梯度/一次调用测试 |
| 21 个自适应 evidence map、null mass、有限 soft prior | `lens_adaptive_evidence.py` | 归一化、梯度、shuffle、无 dense materialization |
| 可识别 present/counter/unknown posterior | `lens_latent_state.py` | progress=0 精确恢复与状态顺序测试 |
| 有序 emission 与 group shrinkage | `lens_annotation_emission.py` | 严格排序、identity ramp、频率初始化测试 |
| CalAlign-compatible clean action base | `lens_oia_model.py` | progress=0 source 等价、annotation-to-action firewall |
| action-conditioned full-field reread 与可加贡献 | `lens_action_reread.py` | null factor、chunk、守恒、state substitution 测试 |
| fail-closed train-only grounding | `lens_structured_evidence.py` | source-complete/unknown 与无 test input 测试 |
| mirror/weak-view consistency | `lens_mirror.py` | action/reason/map 的左右置换与单次 DINO 测试 |
| 单次 backward、唯一 owner/loss registry | `lens_loss_registry.py`, trainer | loss/optimizer/autograd owner probes |
| train-calib deployment calibration/test-only | trainer/eval/calibration | 不可变 model hash、无 val/oracle leakage |
| artifacts/profile/pilot gates | engine/utils | schema、hash、runtime、raw gate recomputation |

## 审查方式

当前环境没有可调用的同级独立审查代理。为避免伪造“双代理”结论，采用独立的第二遍对抗审查：先以源码禁用路径、公式、形状、梯度归属和训练调用链反证实现；再运行测试与真实 DINO smoke。任何未能由源码和 artifact 共同证明的条目都保持未完成。

## 当前状态

- [x] 完整阅读用户计划与审查 Skill。
- [x] 远端 canonical 记录已读取。
- [x] 锁定并验证指定 source HEAD。
- [x] 创建并推送空 LENS 分支锚点。
- [ ] RED tests。
- [ ] 模型/数据/损失/训练实现。
- [ ] compile、pytest、审查、真实 DINO smoke、pilot gate。
