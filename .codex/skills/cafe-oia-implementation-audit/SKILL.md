
# cafe-oia-implementation-audit skill

## Purpose

This skill exists to prevent CAFE-OIA implementation drift. It must be used together with `dual-agent-supervision` and `superpowers` before any CAFE-OIA training run.

The review is code-level and executable. Textual agreement is not enough.

## Scope

Use this skill for FATE-OIA tasks involving:
- CAFE-OIA
- counterfactual evidence
- object/lane/drivable grounding
- causal direct effect
- calibration/threshold tuning
- foreground training supervisors
- any run on branch `clean_cafe_oia_v1`

## Required agents

Agent A = Implementer
- writes the exact file-level plan;
- patches code only after reading the three FATE context files;
- records every changed file;
- writes self-checks.

Agent B = Reviewer/Auditor
- runs static and dynamic checks;
- blocks training unless all core functions are real;
- must issue `REVIEW_PASS_CAFE_V2` only after executable evidence exists.

## Required context read

Before anything under `E:\sbw\FATE_Drive`, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

Do not create extra durable training Markdown files. Only append to those three files.

## Hard implementation checks

The following must be true before training.

### 1. Config is real

Fail if:
- `--config` is accepted but YAML is not parsed;
- YAML values are not reflected in `config_resolved.yaml`;
- CLI overrides cannot be distinguished from defaults;
- `config_version` is not `cafe_oia_v2_evidence_fixed`.

Required artifacts:
- `config_resolved.yaml`
- `config_resolved.json`
- run manifest with config path and CLI overrides.

### 2. Real evidence exists

Fail if:
- evidence counts are all fallback;
- object evidence count is zero on real audit samples;
- grounding cache key hit rate is below 70% on train/test;
- lane/drivable are claimed when their counts are zero.

Required artifacts:
- `evidence_audit_real_split.json`
- `evidence_stats.jsonl` with object/lane/drivable/fallback counts.

### 3. Counterfactual is not a proxy

Fail if code contains:
- `target_deleted_reason = reason_logits - torch.relu`
- fake `context_only_reason = base["reason_logits"]` without a re-forward path;
- evidence-only logits formed purely by multiplying residuals;
- `cf_is_proxy=true`.

Required dynamic evidence:
- target-deleted/context-only/evidence-only/replaced logits are produced by actual forward calls with modified evidence units;
- `cf_valid_count > 0`;
- `cf_real_evidence_count > 0`;
- `cf_loss_nonzero_rate > 0` on a smoke batch with real evidence;
- `direct_effect_mean` finite and recorded.

### 4. Calibration is not a placeholder

Fail if:
- calibrated metrics are a direct copy of raw metrics;
- calibration params contain only a note/placeholder;
- no classwise bias/temperature or threshold vector is written.

Required artifacts:
- `calibration_params_val.json`
- `calibration_params_test_diagnostic.json`
- `metrics_val_calibrated.json`
- `metrics_test_calibrated_diagnostic.json`.

### 5. Rollback means restore

Fail if:
- plateau logic only decays LR but does not restore model/optimizer best state.

Required dynamic test:
- toy model weights degrade; scheduler restores previous best weights and decays LR.

### 6. Foreground only

Fail if scripts contain:
- `Start-Process`
- `Start-Job`
- `Win32_Process`
- `Invoke-WmiMethod`
- `nohup`
- hidden/detached command execution.

The supervisor must stream stdout/stderr and wait for training completion.

## Required review output

Write:

`.background_runs/cafe_oia_v2_preflight/review_report.json`

It must include:
- `passed`
- `git_head`
- `dirty_status`
- `checks`
- `failures`
- `required_next_action`

Only if all checks pass, write:

`.background_runs/cafe_oia_v2_preflight/REVIEW_PASS_CAFE_V2.txt`

The file must include:
- commit hash;
- test commands run;
- evidence audit file path;
- counterfactual smoke file path;
- calibration smoke file path;
- no-placeholder scan file path.

## Forbidden claims

Do not claim:
- "counterfactual implemented" without actual intervention forward passes;
- "object/lane/drivable evidence implemented" if real counts are zero;
- "calibrated result" if calibration is placeholder;
- "rollback" if best checkpoint restore is absent;
- "training started" if `REVIEW_PASS_CAFE_V2` is absent.

## Training rule

After review pass, one foreground run is allowed:

- 24 epochs;
- batch 2;
- grad accumulation 16;
- effective batch 32;
- no metric early stop;
- hard stop only for NaN/OOM/artifact/review failures.

