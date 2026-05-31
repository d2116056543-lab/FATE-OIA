# TRACE ActionPrimary Implementation Audit

Use this skill before launching TRACE-OIA ActionPrimary V2 training.

Required checks:
- The active config is `trace_oia_action_primary_v2_direct_image`.
- Feature cache is disabled and cache build is skipped.
- Evaluation and best-checkpoint selection are test-only.
- `TraceOIAModel` exposes action candidates, bounded `action_bias`, `reason_alpha`, and `safe_ensemble_r2a_logit`.
- `reason_alpha` and `action_bias` are included in optimizer parameter groups.
- Action-primary loss returns split action/reason/evidence diagnostics.
- Foreground supervisor streams child output and supports batch fallbacks `4/8`, `3/11`, `2/16`.
- The active PowerShell launcher and supervisor must not use background launch commands.

Canonical review command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_trace_action_primary_implementation
```

Training must require the resulting pass file:

```powershell
scripts\FATE_OIA_trace_oia_v1_foreground.ps1 -RequireReviewPass
```
