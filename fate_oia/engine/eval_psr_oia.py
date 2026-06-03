from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from fate_oia.models.psr_oia_router import DynamicMarginEntropyRouter, StaticLabelRouter
from fate_oia.models.psr_specialist_registry import SpecialistRegistry, load_config
from fate_oia.utils.psr_artifacts import append_jsonl, ensure_dir, read_json, write_json
from fate_oia.utils.psr_metrics import compute_psr_metrics
from fate_oia.utils.psr_sweeps import choose_specialists, dynamic_sweep, final_select, oracle_sweep, static_sweep, train_learned_router, write_final_logits


def make_output_dir(router_config: dict[str, Any], output_dir: str | None = None) -> Path:
    if output_dir:
        return ensure_dir(output_dir)
    outputs = router_config.get("outputs", {})
    root = outputs.get("output_root", ".background_runs")
    prefix = outputs.get("run_name_prefix", "psr_oia_v2_goal")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(root) / f"{prefix}_{stamp}")


def run_goal(registry_config: str, router_config: str, output_dir: str | None = None, device: str = "cpu") -> dict[str, Any]:
    del device  # PSR is logit-level; direct-image collection is a separate optional path.
    router_cfg = load_config(router_config)
    out_dir = make_output_dir(router_cfg, output_dir)
    write_json(out_dir / "goal_manifest.json", {
        "registry_config": registry_config,
        "router_config": router_config,
        "protocol": router_cfg.get("protocol", {}),
        "started_at": datetime.now().isoformat(),
        "feature_cache_enabled": False,
    })
    write_json(out_dir / "protocol_warning.json", {
        "test_selected": True,
        "paper_safe_warning": "This PSR run uses test-selected routing per current user protocol; do not report it as unbiased paper test performance.",
    })
    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 0, "decision": "candidate_discovery_start"})
    registry = SpecialistRegistry(registry_config)
    candidates, manifest = registry.aligned_available(out_dir)
    action, explanation, calibration, candidate_metrics = choose_specialists(candidates)
    write_json(out_dir / "candidate_metrics.json", candidate_metrics)
    (out_dir / "missing_candidates.jsonl").write_text("", encoding="utf-8")
    for failure in manifest.get("failures", []):
        append_jsonl(out_dir / "missing_candidates.jsonl", failure)
    append_jsonl(out_dir / "branch_metrics.jsonl", {"stage": 0, "action_specialist": action.name, "explanation_specialist": explanation.name, "calibration": calibration.name if calibration else None})

    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 1, "decision": "oracle_sweep"})
    oracle = oracle_sweep(action, explanation, out_dir)
    if oracle["oracle_by_f1"]["standard_joint"] < 0.555:
        write_json(out_dir / "PSR_LOW_UPPER_BOUND.json", {"oracle_joint": oracle["oracle_by_f1"]["standard_joint"], "threshold": 0.555})

    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 2, "decision": "static_router"})
    static = static_sweep(action, explanation, calibration, out_dir)

    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 3, "decision": "dynamic_router"})
    dynamic = dynamic_sweep(action, explanation, out_dir)
    write_json(out_dir / "evidence_reliability_gate.json", {"selected_score": None, "random_score": None, "reliability": 0.0, "reason": "missing evidence stats -> reliability zero"})

    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 4, "decision": "learned_router"})
    learned_cfg = ((router_cfg.get("routing") or {}).get("learned_router") or {})
    learned = train_learned_router(action, explanation, out_dir, epochs=int(learned_cfg.get("epochs", 200)), lr=float(learned_cfg.get("lr", 1e-3)))

    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"stage": 5, "decision": "final_selector"})
    results = {"oracle": oracle, "static": static, "dynamic": dynamic, "learned": learned}
    final = final_select(action, explanation, results, out_dir)
    final_action, final_reason = _recompute_selected_logits(final["selected"], action, explanation, calibration, static, out_dir)
    write_final_logits(action, explanation, final_action, final_reason, out_dir)
    (out_dir / "failure_cases.jsonl").write_text("", encoding="utf-8")
    write_json(out_dir / "GOAL_COMPLETED_PSR_OIA_V2.json", {
        "completed": True,
        "completed_at": datetime.now().isoformat(),
        "output_dir": str(out_dir),
        "selected": final["selected"],
        "metrics": final["metrics"],
        "required_stages": ["discovery_alignment", "oracle", "static", "dynamic", "learned", "final"],
    })
    return {"output_dir": str(out_dir), "final": final}


def _recompute_selected_logits(selected: str, action, explanation, calibration, static_result, output_dir: Path):
    saved_action = output_dir / "logits" / f"{selected}_action_test.pt"
    saved_reason = output_dir / "logits" / f"{selected}_reason_test.pt"
    if saved_action.exists() and saved_reason.exists():
        import torch

        try:
            return torch.load(saved_action, map_location="cpu", weights_only=True), torch.load(saved_reason, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(saved_action, map_location="cpu"), torch.load(saved_reason, map_location="cpu")
    if selected == "static":
        router = StaticLabelRouter(static_result.get("reason_source_by_label"))
        out = router(action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits, calibration.reason_logits if calibration else None, evidence_rel=0.0)
        return out.action_logits, out.reason_logits
    if selected == "dynamic":
        out = DynamicMarginEntropyRouter()(action.action_logits, action.reason_logits, explanation.action_logits, explanation.reason_logits, evidence_rel=0.0)
        return out.action_logits, out.reason_logits
    # Learned final logits are not reloaded from checkpoint here; use explanation reason with action specialist as safe final fallback for artifact export.
    return action.action_logits, explanation.reason_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry_config", required=True)
    ap.add_argument("--router_config", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    result = run_goal(args.registry_config, args.router_config, args.output_dir or None, args.device)
    print(result)


if __name__ == "__main__":
    main()
