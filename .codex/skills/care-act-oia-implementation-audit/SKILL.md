---
name: care-act-oia-implementation-audit
description: Audit CARE-ACT OIA V1 implementation for direct action evidence, action-set consistency, test-only fairness, and foreground goal-mode training.
---

# CARE-ACT OIA Implementation Audit

Use this skill before launching CARE-ACT OIA V1 training.

Required command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_care_act_oia_implementation --config configs\fate_oia_train_360x640_care_act_oia_v1.yaml --output_dir .background_runs\care_act_oia_v1_preflight --device cuda
```

The audit must write:

```text
.background_runs\care_act_oia_v1_preflight\REVIEW_PASS_CARE_ACT_OIA_V1.txt
```

Training is forbidden without the review pass file.
