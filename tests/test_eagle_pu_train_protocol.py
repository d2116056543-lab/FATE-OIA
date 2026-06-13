from __future__ import annotations

from pathlib import Path
import yaml
from fate_oia.engine.audit_eagle_pu_implementation import run_static_audit

def test_eagle_pu_config_forbids_cache_val_and_compression():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_eagle_pu_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["runtime"]["test_only"] is True
    assert cfg["runtime"]["no_feature_cache"] is True
    assert cfg["runtime"]["require_no_token_compression"] is True
    assert cfg["training"]["best_selection_split"] == "test"
    assert cfg["training"]["eval_splits"] == "test"
    assert cfg["model"]["token_compression"] == "none"
    assert cfg["model"]["feature_cache_enabled"] is False
    assert cfg["training"]["reference_effective_batch"] == 32
    assert cfg["training"]["fallback_ladder"] == [[6,5], [5,6], [4,8], [3,11], [2,16]]

def test_static_audit_reports_required_functional_checks():
    result = run_static_audit(Path("."), Path("configs/fate_oia_train_360x640_eagle_pu_v1.yaml"))
    expected = ["Dataset and targets", "DINO field", "Ego encoding", "Ontology", "State bank", "Label decision trunk", "Positive-unlabeled reason loss", "Prototype transport", "State-grounded graph", "Action-set auxiliary", "Calibration", "Full model forward", "Training protocol", "Foreground supervisor"]
    for name in expected:
        assert name in result["functional_checks"]
