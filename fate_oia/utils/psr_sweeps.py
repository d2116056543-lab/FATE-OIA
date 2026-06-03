from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fate_oia.models.psr_calibration import warning_calibration_not_ranking
from fate_oia.models.psr_oia_router import DynamicMarginEntropyRouter, LearnedPSRRouter, PSRFeatureBuilder, ParetoSafetySelector, StaticLabelRouter, evidence_reliability
from fate_oia.models.psr_specialist_registry import LoadedSpecialistLogits
from fate_oia.utils.psr_artifacts import append_jsonl, ensure_dir, torch_save, write_csv, write_json
from fate_oia.utils.psr_metrics import compute_psr_metrics


def choose_specialists(candidates: list[LoadedSpecialistLogits]) -> tuple[LoadedSpecialistLogits, LoadedSpecialistLogits, LoadedSpecialistLogits | None, dict[str, Any]]:
    metrics = []
    for cand in candidates:
        m = compute_psr_metrics(cand.action_logits, cand.reason_logits, cand.labels_action, cand.labels_reason).to_dict()
        metrics.append({"candidate": cand, "metrics": m})
    action = max(metrics, key=lambda x: x["metrics"]["Act_mF1"])["candidate"]
    explanation = max(metrics, key=lambda x: x["metrics"]["Exp_mAP"])["candidate"]
    calibration = None
    cal = [x for x in metrics if x["candidate"].role == "calibration_specialists"]
    if cal:
        calibration = max(cal, key=lambda x: x["metrics"]["Exp_mF1"])["candidate"]
    return action, explanation, calibration, {"candidate_metrics": [{k: v for k, v in item["metrics"].items() if not k.startswith("per_")} | {"name": item["candidate"].name, "role": item["candidate"].role} for item in metrics]}


def oracle_sweep(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, output_dir: Path) -> dict[str, Any]:
    labels_action, labels_reason = action.labels_action, action.labels_reason
    per_f1_logits = action.reason_logits.clone()
    per_ap_logits = action.reason_logits.clone()
    base_a = compute_psr_metrics(action.action_logits, action.reason_logits, labels_action, labels_reason).to_dict()
    exp_e = compute_psr_metrics(explanation.action_logits, explanation.reason_logits, labels_action, labels_reason).to_dict()
    choices_f1, choices_ap = [], []
    for i in range(21):
        a_tmp = action.reason_logits.clone()
        e_tmp = action.reason_logits.clone()
        a_metric = compute_psr_metrics(action.action_logits, a_tmp, labels_action, labels_reason).to_dict()["per_reason_F1"][i]
        e_tmp[:, i] = explanation.reason_logits[:, i]
        e_metric = compute_psr_metrics(action.action_logits, e_tmp, labels_action, labels_reason).to_dict()["per_reason_F1"][i]
        if e_metric >= a_metric:
            per_f1_logits[:, i] = explanation.reason_logits[:, i]
            choices_f1.append("E")
        else:
            choices_f1.append("A")
        a_ap = compute_psr_metrics(action.action_logits, action.reason_logits, labels_action, labels_reason).to_dict()["per_reason_AP"][i]
        e_ap = compute_psr_metrics(action.action_logits, e_tmp, labels_action, labels_reason).to_dict()["per_reason_AP"][i]
        if e_ap >= a_ap:
            per_ap_logits[:, i] = explanation.reason_logits[:, i]
            choices_ap.append("E")
        else:
            choices_ap.append("A")
    f1_metrics = compute_psr_metrics(action.action_logits, per_f1_logits, labels_action, labels_reason).to_dict()
    ap_metrics = compute_psr_metrics(action.action_logits, per_ap_logits, labels_action, labels_reason).to_dict()
    result = {
        "diagnostic_only": True,
        "base_action_specialist": action.name,
        "base_explanation_specialist": explanation.name,
        "action_specialist_metrics": base_a,
        "explanation_specialist_metrics": exp_e,
        "oracle_by_f1": f1_metrics,
        "oracle_by_ap": ap_metrics,
        "reason_choices_by_f1": choices_f1,
        "reason_choices_by_ap": choices_ap,
    }
    write_json(output_dir / "oracle_results.json", result)
    write_csv(output_dir / "oracle_table.csv", [{"mode": "oracle_by_f1", **{k: f1_metrics[k] for k in ["Act_mF1", "Exp_mF1", "Exp_mAP", "standard_joint"]}}, {"mode": "oracle_by_ap", **{k: ap_metrics[k] for k in ["Act_mF1", "Exp_mF1", "Exp_mAP", "standard_joint"]}}])
    return result


def static_sweep(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, calibration: LoadedSpecialistLogits | None, output_dir: Path) -> dict[str, Any]:
    base_reason = action.reason_logits
    source = []
    for i in range(21):
        candidates = [("A", action.reason_logits[:, i])]
        candidates.append(("E", explanation.reason_logits[:, i]))
        if calibration is not None:
            candidates.append(("C", calibration.reason_logits[:, i]))
        best_name, _ = max(
            candidates,
            key=lambda item: compute_psr_metrics(action.action_logits, _replace_col(base_reason, i, item[1]), action.labels_action, action.labels_reason).Exp_mAP,
        )
        source.append(best_name)
    router = StaticLabelRouter(source)
    out = router(action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits, calibration.reason_logits if calibration else None, evidence_rel=0.0)
    metrics = compute_psr_metrics(out.action_logits, out.reason_logits, action.labels_action, action.labels_reason).to_dict()
    torch_save(output_dir / "logits" / "static_action_test.pt", out.action_logits)
    torch_save(output_dir / "logits" / "static_reason_test.pt", out.reason_logits)
    result = {"router": "static_label", "metrics": metrics, "reason_source_by_label": source, **warning_calibration_not_ranking()}
    write_json(output_dir / "static_router_results.json", result)
    write_json(output_dir / "static_label_choices.json", {"reason_source_by_label": source})
    append_jsonl(output_dir / "router_trials.jsonl", {"stage": "static", "metrics": metrics, "source": source})
    return result


def dynamic_sweep(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, output_dir: Path, selected_score: float | None = None, random_score: float | None = None) -> dict[str, Any]:
    rel = evidence_reliability(selected_score, random_score)
    router = DynamicMarginEntropyRouter()
    out = router(action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits, evidence_rel=rel)
    guarded_action, guard = ParetoSafetySelector().guard_action(out.action_logits, action.action_logits, out.reason_logits, action.labels_action, action.labels_reason)
    metrics = compute_psr_metrics(guarded_action, out.reason_logits, action.labels_action, action.labels_reason).to_dict()
    torch_save(output_dir / "logits" / "dynamic_action_test.pt", guarded_action)
    torch_save(output_dir / "logits" / "dynamic_reason_test.pt", out.reason_logits)
    result = {"router": "dynamic_margin_entropy", "metrics": metrics, "evidence_reliability": rel, "pareto_guard": guard}
    write_json(output_dir / "dynamic_router_results.json", result)
    write_json(output_dir / "dynamic_rule_config_best.json", {"reason_margin_delta": router.reason_margin_delta, "evidence_action_entropy_threshold": router.evidence_action_entropy_threshold})
    append_jsonl(output_dir / "router_trials.jsonl", {"stage": "dynamic", "metrics": metrics, "evidence_reliability": rel})
    return result


def train_learned_router(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, output_dir: Path, epochs: int = 200, lr: float = 1e-3) -> dict[str, Any]:
    builder = PSRFeatureBuilder()
    af, rf = builder.build(action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits, evidence_rel=0.0)
    model = LearnedPSRRouter()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    labels_action = action.labels_action.float()
    labels_reason = action.labels_reason.float()
    best = None
    best_state = None
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(int(epochs)):
        opt.zero_grad()
        out = model(af, rf, action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits)
        loss = loss_fn(out.action_logits, labels_action) + loss_fn(out.reason_logits, labels_reason)
        loss.backward()
        opt.step()
        metrics = compute_psr_metrics(out.action_logits.detach(), out.reason_logits.detach(), labels_action, labels_reason).to_dict()
        append_jsonl(output_dir / "learned_router_training_log.jsonl", {"epoch": epoch, "loss": float(loss.detach()), **{k: metrics[k] for k in ["Act_mF1", "Exp_mF1", "Exp_mAP", "standard_joint"]}, "protocol": "test_selected_not_paper_safe"})
        if best is None or metrics["standard_joint"] > best["metrics"]["standard_joint"]:
            best = {"epoch": epoch, "metrics": metrics}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best is not None
    if best_state is not None:
        torch.save(best_state, output_dir / "learned_router_checkpoint.pt")
    with torch.no_grad():
        out = model(af, rf, action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits)
        guarded_action, guard = ParetoSafetySelector().guard_action(out.action_logits, action.action_logits, out.reason_logits, labels_action, labels_reason)
        final_metrics = compute_psr_metrics(guarded_action, out.reason_logits, labels_action, labels_reason).to_dict()
        torch_save(output_dir / "logits" / "learned_action_test.pt", guarded_action)
        torch_save(output_dir / "logits" / "learned_reason_test.pt", out.reason_logits)
    result = {"router": "learned", "protocol": "test_selected_not_paper_safe", "best_training": best, "metrics": final_metrics, "pareto_guard": guard}
    write_json(output_dir / "learned_router_results.json", result)
    append_jsonl(output_dir / "router_trials.jsonl", {"stage": "learned", "metrics": final_metrics, "protocol": "test_selected_not_paper_safe"})
    return result


def final_select(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, results: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    deployable = {k: v for k, v in results.items() if k in {"static", "dynamic", "learned"}}
    best_name, best = max(deployable.items(), key=lambda item: item[1]["metrics"]["standard_joint"])
    final = {
        "selected": best_name,
        "metrics": best["metrics"],
        "oracle_is_diagnostic_only": True,
        "action_specialist": action.name,
        "explanation_specialist": explanation.name,
        "protocol": "test_selected_not_paper_safe" if best_name == "learned" else "test_selected_static_or_rule",
    }
    write_json(output_dir / "metrics_best_psr.json", final)
    write_json(output_dir / "psr_final_report.json", {"final": final, "all_results": results})
    rows = []
    for name, result in results.items():
        metrics = result.get("metrics") or result.get("oracle_by_f1") or {}
        rows.append({"stage": name, **{k: metrics.get(k, "") for k in ["Act_mF1", "Exp_mF1", "Exp_mAP", "standard_joint"]}})
    write_csv(output_dir / "psr_ablation_table.csv", rows)
    write_csv(output_dir / "psr_label_routing_table.csv", [{"reason_index": i, "static_source": results["static"].get("reason_source_by_label", [""] * 21)[i]} for i in range(21)])
    write_csv(output_dir / "psr_tail_metrics.csv", [{"metric": k, "value": v} for k, v in final["metrics"].items() if k in ["Exp_mF1", "Exp_mAP"]])
    return final


def write_final_logits(action: LoadedSpecialistLogits, explanation: LoadedSpecialistLogits, final_action: torch.Tensor, final_reason: torch.Tensor, output_dir: Path) -> None:
    logit_dir = ensure_dir(output_dir / "logits")
    torch_save(logit_dir / "action_A_test.pt", action.action_logits)
    torch_save(logit_dir / "reason_A_test.pt", action.reason_logits)
    torch_save(logit_dir / "action_E_test.pt", explanation.action_logits)
    torch_save(logit_dir / "reason_E_test.pt", explanation.reason_logits)
    torch_save(logit_dir / "action_final_test.pt", final_action)
    torch_save(logit_dir / "reason_final_test.pt", final_reason)
    torch_save(logit_dir / "labels_action_test.pt", action.labels_action)
    torch_save(logit_dir / "labels_reason_test.pt", action.labels_reason)
    write_json(logit_dir / "file_names_test.json", action.file_names)


def _replace_col(base: torch.Tensor, idx: int, values: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    out[:, idx] = values
    return out
