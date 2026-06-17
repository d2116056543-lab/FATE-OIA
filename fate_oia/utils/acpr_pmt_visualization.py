from __future__ import annotations

import json
from pathlib import Path


def export_chain_case(out_dir: Path, case: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chains = case.get("chains", case.get("top_action_reason_predicate_chains", []))
    normalized = {
        "file_name": case.get("file_name", ""),
        "gt_action": case.get("gt_action", case.get("gt_actions", [])),
        "pred_action": case.get("pred_action", case.get("pred_actions", [])),
        "gt_reason": case.get("gt_reason", case.get("gt_reasons", [])),
        "pred_reason": case.get("pred_reason", case.get("pred_reasons", [])),
        "chains": chains,
    }
    for chain in normalized["chains"]:
        chain.setdefault("action_name", chain.get("action", "unknown_action"))
        chain.setdefault("reason_name", chain.get("reason", "unknown_reason"))
        chain.setdefault("predicate_name", chain.get("predicate", "unknown_predicate"))
        chain.setdefault("chain_score", chain.get("score", 0.0))
        chain.setdefault("predicate_prob", chain.get("predicate_probability", 0.0))
        chain.setdefault("patch_coordinates", chain.get("coords", []))
    (out_dir / "chain.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "top_action_reason_predicate_patch_chains.json").write_text(json.dumps(normalized["chains"], ensure_ascii=False, indent=2), encoding="utf-8")
    html = "<html><body><h1>ACPR-PMT-S action-reason-predicate-patch chains</h1><pre>" + json.dumps(normalized, ensure_ascii=False, indent=2) + "</pre></body></html>"
    (out_dir / "report.html").write_text(html, encoding="utf-8")
