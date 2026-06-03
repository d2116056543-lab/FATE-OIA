---
name: psr-train-oia-implementation-audit
description: Audit PSR-Train OIA V1 end-to-end specialist routing implementation before foreground training.
---

# PSR-Train OIA V1 Audit

Use this skill before launching PSR-Train OIA V1 training.

Required checks:

- The training script must not read old RunC/CARE logits.
- The training script must not resume old task checkpoints.
- The only initialization path is DINO pretrained weights.
- Evaluation must be test-only, and best checkpoint selection must use test metrics.
- The model must contain action, explanation, calibration, and router components in one module.
- Router and calibration parameters must receive gradients in a synthetic backward test.
- Warmup must output `final_action=A_action` and `final_reason=E_reason`.
- Supervisor must run foreground commands and must not contain Start-Process, Start-Job, nohup, or hidden-window launchers.

Canonical command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_psr_train_oia_implementation `
  --config configs\fate_oia_train_360x640_psr_train_oia_v1.yaml `
  --output_dir .background_runs\psr_train_oia_v1_preflight
```

Passing artifact:

```text
REVIEW_PASS_PSR_TRAIN_OIA_V1.txt
```
