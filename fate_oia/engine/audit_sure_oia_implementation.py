from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.losses.gradnorm import GradNormBalancer
from fate_oia.losses.sure_losses import compute_sure_losses, make_sure_criterion
from fate_oia.models.sure_oia_model import SUREOIAFeatureModel
from fate_oia.utils.sure_artifacts import write_json
from fate_oia.utils.sure_review_gates import assert_no_forbidden_supervisor_patterns, assert_test_only_manifest


def _fake_meta() -> list[dict]:
    return [{"patch_grid_h": 8, "patch_grid_w": 12}]


def run_synthetic_smoke(out_dir: Path) -> dict:
    torch.manual_seed(7)
    model = SUREOIAFeatureModel(dim=64, action_dim=4, reason_dim=21, relation_queries=8, max_edges_total=32)
    balancer = GradNormBalancer()
    tokens = torch.randn(2, 96, 64)
    structured = [
        {"objects": [{"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 300}}], "lanes": [], "drivable": {"has_map": True}},
        {"objects": [], "lanes": [{"category": "lane", "vertices": [(1, 1), (2, 2)]}], "drivable": {}},
    ]
    out = model(tokens, structured=structured, image_meta=_fake_meta())
    criterion = make_sure_criterion("bce")
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    losses = compute_sure_losses(out, action, reason, criterion)
    total, grad_stats = balancer(losses)
    total.backward()
    smoke_dir = out_dir / "smoke" / "epoch_000"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "smoke" / "run_manifest.json", {"eval_splits": ["test"], "uses_val": False, "uses_feature_cache": False})
    write_json(smoke_dir / "metrics_summary.json", {"synthetic": True})
    write_json(smoke_dir / "relation_stats.json", out["relation_stats"])
    write_json(smoke_dir / "gradnorm_stats.json", grad_stats)
    write_json(smoke_dir / "action_safe_stats.json", out["action_safe_stats"])
    return {
        "synthetic_loss": float(total.detach()),
        "selected_edges": out["relation_stats"]["selected_edges"],
        "candidate_edges": out["relation_stats"]["candidate_edges"],
        "memory_gate_mean": float(out["memory_gate"].detach().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit SURE-OIA v2 implementation before full training.")
    ap.add_argument("--config", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA/data")
    ap.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--eval_splits", default="test")
    args = ap.parse_args()
    if args.eval_splits != "test":
        raise ValueError("SURE audit requires eval_splits=test")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = BDDOIAMultiTaskDataset(data_root=args.data_root, raw_root=args.raw_root, split="test", load_image=False)
    names = [s.file_name for s in ds.samples[:128]]
    audit = BDD100KStructuredIndex(args.bdd100k_root).audit_samples(names, "test")
    write_json(out_dir / "structured_bdd100k_audit.json", audit)
    smoke = run_synthetic_smoke(out_dir)
    manifest = {"eval_splits": ["test"], "uses_val": False, "uses_feature_cache": False}
    assert_test_only_manifest(manifest)
    assert_no_forbidden_supervisor_patterns(
        [
            "fate_oia/engine/supervise_sure_oia_foreground.py",
            "scripts/FATE_OIA_sure_oia_v2_foreground.ps1",
        ]
    )
    review = {
        "passed": True,
        "test_only": True,
        "direct_image_no_cache": True,
        "bdd100k_audit": audit,
        "synthetic_smoke": smoke,
        "blocking_items": [],
    }
    write_json(out_dir / "review_report.json", review)
    (out_dir / "REVIEW_PASS_SURE_OIA_V2.txt").write_text("SURE-OIA v2 audit pass\n" + json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "sure_audit_pass", "output_dir": str(out_dir), "match_rate": audit["match_rate"], "selected_edges": smoke["selected_edges"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
