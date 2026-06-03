# psr-oia-implementation-audit

Audit PSR-OIA V2 implementation before any goal-mode run. The audit must verify:

- worktree branch is `care_moe_oia_v1_direct_image`
- configs are `psr_oia_v2_registry` and `psr_oia_v2_router`
- feature cache is disabled and evaluation is test-only
- action/reason specialists are discovered or reported as missing
- logits are strictly aligned by file names and labels
- static/dynamic/learned routers are not plain average placeholders
- Pareto safety falls back to the action specialist when candidate action is worse
- unreliable evidence sets routing reliability to zero
- calibration-only gains are not reported as ranking/AP gains
- foreground supervisor contains no detached/background launch pattern
- `GOAL_COMPLETED_PSR_OIA_V2.json` is written only after Stage 0-5 complete

Durable experiment status must be appended only to:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`
