from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from fate_oia.utils.tida_object_intent_metrics import (
    combine_object_intent_utility_policies,
    fit_object_intent_utility_policy_oof,
)


def _label_f1(logits: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid() >= thresholds[None]
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu()]


def _fit_policy(
    rows: dict[str, torch.Tensor], thresholds: torch.Tensor,
    common: dict, selected_rate_cap: float,
) -> dict[str, torch.Tensor]:
    directional = fit_object_intent_utility_policy_oof(
        rows["pre_object_intent_action"],
        rows["object_intent_action_candidate"],
        rows["object_intent_action_directional_utility_gate"],
        rows["action_target"], thresholds,
        max_selected_rate=selected_rate_cap, **common,
    )
    risk = fit_object_intent_utility_policy_oof(
        rows["pre_object_intent_action"],
        rows["object_intent_action_candidate"],
        rows["object_intent_action_risk_utility_gate"],
        rows["action_target"], thresholds,
        max_selected_rate=selected_rate_cap, **common,
    )
    return combine_object_intent_utility_policies(directional, risk)


def _apply_policy(
    rows: dict[str, torch.Tensor], policy: dict[str, torch.Tensor], cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    utility = torch.where(
        policy["utility_source"][None].bool(),
        rows["object_intent_action_risk_utility_gate"],
        rows["object_intent_action_directional_utility_gate"],
    )
    selected = utility >= policy["cutoff"][None]
    delta = (policy["scale"][None] * rows["object_intent_action_candidate"]).clamp(
        -float(cap), float(cap)
    )
    deployed = (
        rows["pre_object_intent_action"]
        + policy["gate"][None] * selected.float() * delta
    )
    return deployed, selected, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--selected-rate-caps", default="0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    deployment = config["deployment"]
    rows = torch.load(args.policy_rows, map_location="cpu", weights_only=False)
    epoch = args.epoch_dir
    test = {
        key: torch.load(epoch / f"{name}_test.pt", map_location="cpu", weights_only=False)
        for key, name in {
            "base": "pre_object_intent_action",
            "candidate": "object_intent_action_candidate",
            "directional": "object_intent_action_directional_utility_gate",
            "risk": "object_intent_action_risk_utility_gate",
            "target": "action_target",
        }.items()
    }
    thresholds = torch.tensor(
        deployment["locked_image_thresholds"][:4], dtype=test["base"].dtype
    )
    common = dict(
        scales=tuple(deployment["object_intent_action_utility_scales"]),
        cutoffs=tuple(deployment["object_intent_utility_cutoffs"]),
        folds=int(deployment["object_intent_utility_oof_folds"]),
        min_oof_gain=float(deployment["object_intent_utility_min_oof_gain"]),
        min_selected_benefit_rate=float(
            deployment["object_intent_utility_min_selected_benefit_rate"]
        ),
        min_positive_fold_fraction=float(
            deployment["object_intent_utility_min_positive_fold_fraction"]
        ),
        min_non_degrading_fold_fraction=float(
            deployment["object_intent_utility_min_non_degrading_fold_fraction"]
        ),
        fold_degradation_tolerance=float(
            deployment["object_intent_utility_fold_degradation_tolerance"]
        ),
        min_nll_improvement=float(deployment["object_intent_utility_min_nll_improvement"]),
        min_brier_improvement=float(
            deployment["object_intent_utility_min_brier_improvement"]
        ),
        quantile_coverages=tuple(deployment["object_intent_utility_quantile_coverages"]),
        cap=float(config["model"]["object_intent_action_cap"]),
        allow_proper_score_tie=False,
    )
    base_f1 = _label_f1(test["base"], test["target"], thresholds)
    reports = []
    for selected_rate_cap in [float(value) for value in args.selected_rate_caps.split(",")]:
        policy = _fit_policy(rows, thresholds, common, selected_rate_cap)
        test_policy_rows = {
            "pre_object_intent_action": test["base"],
            "object_intent_action_candidate": test["candidate"],
            "object_intent_action_directional_utility_gate": test["directional"],
            "object_intent_action_risk_utility_gate": test["risk"],
        }
        deployed, selected, delta = _apply_policy(
            test_policy_rows, policy, float(common["cap"])
        )
        deployed_f1 = _label_f1(deployed, test["target"], thresholds)
        sign = 2.0 * test["target"].float() - 1.0
        base_correct = sign * (test["base"] - torch.logit(thresholds)[None]) >= 0
        deploy_correct = sign * (deployed - torch.logit(thresholds)[None]) >= 0
        beneficial_selected = ((sign * delta) > 0) & selected
        selected_benefit_rate = (
            beneficial_selected.sum(0).float() / selected.sum(0).clamp_min(1)
        )
        reports.append({
            "max_selected_rate": selected_rate_cap,
            "train_oof_gain_sum": float(policy["oof_gain"].sum()),
            "train_oof_gain": _tensor_list(policy["oof_gain"]),
            "gate": _tensor_list(policy["gate"]),
            "scale": _tensor_list(policy["scale"]),
            "cutoff": _tensor_list(policy["cutoff"]),
            "source": [int(value) for value in policy["utility_source"]],
            "test_selected_rate": _tensor_list(selected.float().mean(0)),
            "test_selected_benefit_rate": _tensor_list(selected_benefit_rate),
            "test_per_action_f1": _tensor_list(deployed_f1),
            "test_macro_f1": float(deployed_f1.mean()),
            "test_gain": float(deployed_f1.mean() - base_f1.mean()),
            "test_fp_fn_to_correct": int((~base_correct & deploy_correct).sum()),
            "test_correct_to_wrong": int((base_correct & ~deploy_correct).sum()),
        })
    selected_report = max(
        reports, key=lambda report: (report["train_oof_gain_sum"], -report["max_selected_rate"])
    )
    cohort_sizes = rows.get("_policy_cohort_sizes", {})
    calib_count = int(cohort_sizes.get("train_calib", 0))
    audit_count = int(cohort_sizes.get("train_audit", 0))
    core_count = int(cohort_sizes.get("train_core", 0))
    cohort_report = None
    if calib_count > 0 and audit_count > 0 and core_count > 0:
        tensor_keys = (
            "pre_object_intent_action", "object_intent_action_candidate",
            "object_intent_action_directional_utility_gate",
            "object_intent_action_risk_utility_gate", "action_target",
        )
        audit_slice = slice(calib_count, calib_count + audit_count)
        core_slice = slice(calib_count + audit_count, calib_count + audit_count + core_count)
        fit_indices = torch.cat((
            torch.arange(0, calib_count),
            torch.arange(core_slice.start, core_slice.stop),
        ))
        fit_rows = {key: rows[key][fit_indices] for key in tensor_keys}
        audit_rows = {key: rows[key][audit_slice] for key in tensor_keys}
        audit_base_f1 = _label_f1(
            audit_rows["pre_object_intent_action"], audit_rows["action_target"], thresholds
        )
        audit_candidates = []
        for selected_rate_cap in [float(value) for value in args.selected_rate_caps.split(",")]:
            fit_policy = _fit_policy(fit_rows, thresholds, common, selected_rate_cap)
            audit_deployed, _, _ = _apply_policy(
                audit_rows, fit_policy, float(common["cap"])
            )
            audit_f1 = _label_f1(audit_deployed, audit_rows["action_target"], thresholds)
            audit_candidates.append({
                "max_selected_rate": selected_rate_cap,
                "audit_per_action_f1": _tensor_list(audit_f1),
                "audit_per_action_gain": _tensor_list(audit_f1 - audit_base_f1),
            })
        selected_caps = []
        for action in range(4):
            best = max(
                audit_candidates,
                key=lambda report: (
                    report["audit_per_action_f1"][action],
                    -report["max_selected_rate"],
                ),
            )
            selected_caps.append(float(best["max_selected_rate"]))
        report_by_cap = {report["max_selected_rate"]: report for report in reports}
        final_policy = {
            key: torch.tensor([
                report_by_cap[selected_caps[action]][key][action]
                for action in range(4)
            ])
            for key in ("gate", "scale", "cutoff")
        }
        final_policy["utility_source"] = torch.tensor([
            report_by_cap[selected_caps[action]]["source"][action]
            for action in range(4)
        ], dtype=torch.long)
        cohort_deployed, cohort_selected, cohort_delta = _apply_policy(
            test_policy_rows, final_policy, float(common["cap"])
        )
        cohort_f1 = _label_f1(cohort_deployed, test["target"], thresholds)
        cohort_report = {
            "selection_rule": "fit train_calib+train_core; select cap per action on train_audit; refit all train cohorts",
            "selected_caps": selected_caps,
            "audit_candidates": audit_candidates,
            "final_policy": {
                "gate": _tensor_list(final_policy["gate"]),
                "scale": _tensor_list(final_policy["scale"]),
                "cutoff": _tensor_list(final_policy["cutoff"]),
                "source": [int(value) for value in final_policy["utility_source"]],
            },
            "test_selected_rate": _tensor_list(cohort_selected.float().mean(0)),
            "test_per_action_f1": _tensor_list(cohort_f1),
            "test_macro_f1": float(cohort_f1.mean()),
            "test_gain": float(cohort_f1.mean() - base_f1.mean()),
        }
    payload = {
        "selection_rule": "max train-only OOF gain; test read once after selection",
        "base_test_macro_f1": float(base_f1.mean()),
        "base_test_per_action_f1": _tensor_list(base_f1),
        "selected_max_rate": selected_report["max_selected_rate"],
        "selected_report": selected_report,
        "cohort_aware_report": cohort_report,
        "all_diagnostics": reports,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
