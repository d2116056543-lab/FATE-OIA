# METER-OIA V3-HECA GlobalCap Full-Run Contract

## Identity

This document identifies the code path used by the 14-epoch full-data run
started on 2026-08-02. It is intentionally separate from METER V2-TESA and
from HECA pilot protocols.

| Field | Value |
| --- | --- |
| GitHub branch | `acpr_meter_oia_v3_heca_globalcap_full` |
| Executed remote HEAD | `a6dd96a171998b38e7d654ba44907ef64589cb27` |
| Run directory | `.background_runs/meter_oia_v3_heca_globalcap_full_from_scratch_10b7cf7` |
| Run kind | `full`, `numeric_qualified=true`, from scratch |
| Data protocol | direct image; full train main; test-only epoch evaluation |
| DINO | frozen ViT-S/8, layers `(3, 7, 11)`, exactly one normal DINO encode per batch |
| Safety exclusions | no feature cache, no token compression, no action-set final prediction, no test-derived threshold fitting |

The GitHub branch is a code-equivalent synchronization of the executed source.
The remote and GitHub commit IDs differ because the synchronization commit also
contains branch documentation. The run manifest records the executed source,
configuration, and schema hashes.

## Exact Invocation And Schedule

```powershell
python -m fate_oia.engine.train_acpr_meter_oia `
  --config configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml `
  --output_dir .background_runs/meter_oia_v3_heca_globalcap_full_from_scratch_10b7cf7 `
  --epochs 14 --batch_size 4 --gradient_accumulation_steps 8 `
  --num_workers 4 --device cuda --test_only --no_feature_cache `
  --require_no_token_compression --run_kind full --numeric_qualified `
  --seed 20260801
```

The effective batch is `32`. HECA credit uses a 5% warm-up and reaches full
strength by 20% of optimizer updates. `numeric_qualified=true` selects the
conservative `0.20` action-correction fraction for a full run that did not use
an arbitrary gate score as the primary selection criterion.

## Forward Contract

```text
image
  -> frozen DINO dense field
  -> CalAlign foundation action/reason anchors
  -> typed factor measurement (anchor, state, reliability, observability)
  -> HECA shared/private label adapters
  -> action credit transport -> final action logits
  -> private reason decoder -> final reason logits
```

The corresponding implementation entry points are:

| Stage | Code | Enforced behavior |
| --- | --- | --- |
| Image/foundation | `fate_oia/models/meter_oia_model.py::METEROIAModel` | Produces frozen-DINO field and CalAlign anchors once per ordinary batch. |
| Typed measurement | `fate_oia/models/meter_signed_factors.py::TypedEvidenceStateHead` | Reads factor-specific patch anchors and states. Weak geometry is training/audit supervision, not a test-logit feature. |
| Action transport | `fate_oia/models/meter_semantic_action.py::StateConditionedActionCredit` | `action_final = action_visual + bounded_evidence_delta`; action set is never the final prediction. |
| Reason transport | `fate_oia/models/meter_reason_decoder.py::METERPrivateReasonDecoder` | `reason_final = reason_global + ramp * grounded_centered_evidence_delta`; `reason_global` is a CalAlign anchor plus a `<=0.05` sample-specific residual. |
| Gradient firewall | `fate_oia/models/meter_meta_adapters.py` and `fate_oia/optim/heca_optimization.py` | Action and reason have private owners; reason output has no action-logit residual route. |
| Loss wiring | `fate_oia/engine/train_acpr_meter_oia.py::_compute_losses` | Every registered term is added once with an explicit owner. |
| Evaluation | `fate_oia/engine/eval_acpr_meter_oia.py` | Computes final, visual/global branches and same-forward diagnostic modes. |

## Output Equations And Boundaries

```text
action_logits_final = action_logits_visual + ramp * bounded(action_credit)
reason_logits_global = detach(reason_logits_calalign) + bounded(global_delta, 0.05)
reason_logits_final = reason_logits_global + ramp * grounded_centered_evidence_delta
```

`factor_off`, `state_uniform`, and `reason_correction_off` are evaluation-only
same-forward ablations. They reuse the DINO field instead of encoding an image
again. The training loop also records gradient ownership, contribution
conservation, loss wiring, factor route statistics, PU state, and calibration
provenance.

## Calibration And PU

Calibration is fit using `train_calib` only. The deployed metrics apply that
fixed calibration to test outputs; test thresholds are never written into the
model.

The private PU loss is intentionally fail-closed. A reason label is activated
only after sufficient positives, the hidden-positive AUPRC lift threshold, the
cross-view consistency threshold, and the required consecutive pass streak.
Until then its lambda is zero. This protects action/reason ranking from noisy
unobserved reason labels, but means a completed run can legitimately contain no
active PU contribution.

## Result And Artifact Boundaries

The runtime directory and checkpoints are deliberately ignored by Git. The
authoritative numerical history is appended to the three canonical project
records:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

For result interpretation, use the run's `metrics_summary.jsonl`,
`loss_components.jsonl`, `heca_gradient_ownership.jsonl`,
`heca_contribution_conservation.jsonl`, `patch_audit_cumulative.json`, and
`best_metrics.json`. This document describes code identity and behavior only;
it does not replace those canonical experiment records.
