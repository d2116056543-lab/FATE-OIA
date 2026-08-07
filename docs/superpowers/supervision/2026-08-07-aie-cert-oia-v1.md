# AIE-CERT-OIA V1 双重监督记录

## 范围

- 基线：`acpr_aie_oia_v1_direct_image@8a324b94b1cd6b4a4377655a1bd426f7d854fec0`
- 目标：`acpr_aie_cert_oia_v1_direct_image`
- 原 worktree 只读；新实现不得导入旧 AIE evidence/contribution/reason/naming/counterfactual 主件。
- 本记录由当前主任务同时承担执行与审查职责；没有伪称存在独立子代理。

## 功能覆盖矩阵

| 功能 | 真实入口 | 动态验证 | 状态 |
|---|---|---|---|
| predicate-primary gradient firewall | `AIECertCalAlignFoundation.decode_field` | owner firewall test | 待远端 |
| FP32 entmax15 | `aie_cert_sparse.entmax15` | sparse test | 待远端 |
| shared predicate bank | `AIECertPredicateBank` | model forward/audit | 待远端 |
| arithmetic mixture + visual fallback | `AIECertPredicateBank.forward` | predicate diagnostics | 待远端 |
| evidence-conditioned deformable reread | `AIECertDeformableReread` | local reread test | 待远端 |
| map-token co-transport | `AIECertAtomTransport` | transport test | 待远端 |
| overlap ceiling | `atom_overlap_ceiling_loss` | loss test/pilot | 待远端 |
| same-region background centering | `AIECertEvidenceInterface.forward` | forward artifact | 待远端 |
| bias-free exact contribution | `AIECertContributionHead` | reconstruction test | 待远端 |
| four-control robust certificate | `run_counterfactual` + `AIECertCounterfactualEngine` | CF test/pilot | 待远端 |
| checkpointed primal-dual | `AIECertDualState` | constraint/resume test | 待远端 |
| signed reason priors | `AIECertReasonRereader` | signed-prior test | 待远端 |
| sample-label dynamic reason budget | `AIECertReasonRereader.forward` | pilot diagnostics | 待远端 |
| external-only ECPO | `build_ecpo` | queue/pair artifacts | 待远端 |
| age-bounded preference queue | `AIECertPreferenceQueue` | queue test | 待远端 |
| read-only naming | `AIECertNaming` | naming firewall test | 待远端 |
| update-based continuous schedule | `schedule_values` | schedule test/resume | 待远端 |
| guarded train-calib calibration | `AIECertCalibrationGuard` | calibration artifacts | 待远端 |

## 计划一致性矩阵

- C01-C32 的最终状态由 `.review/aie_cert_oia_v1/AIE_CERT_REQUIREMENT_MATRIX.json` 生成，不在源码中硬编码 PASS。
- `REVIEW_PASS` 只允许由 real-DINO 动态审查在当前 clean HEAD 上生成。
- `PILOT_PASS` 只允许来自单次 4096/512/512 三轮 pilot 的真实 artifact。
- gate 是安全性和机制验证，不替代真实 Act/Exp 数值；full train 的最终目标仍是同 checkpoint `Act_mF1>=0.730` 且 `Exp_mF1>=0.380`。

## 审查发现

1. 旧 AIE evidence 使用 geometric predicate log mixture，新实现必须改为 arithmetic mixture 后 bounded log-density ratio。
2. 旧 reason 路径使用 `abs(contribution)` 且 counter prior 被截断，新实现必须分别保留 support/inhibition 与 support/counter。
3. 旧 single-control counterfactual 不能形成稳健证书，新实现必须至少三个有效 controls 并扣除 control mean+std。
4. final residual loss 必须锚定 detached primary，避免解释分支损坏强主干。

## 批准状态

当前仅批准进入远端静态/单元验证；在真实 DINO、runtime profile 和 3-epoch pilot 通过前，不批准 full train。
