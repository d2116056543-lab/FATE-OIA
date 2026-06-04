from pathlib import Path

import yaml


def test_config_is_no_cache_test_only_and_best_on_test():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_p3le_pair_oia_v1.yaml").read_text(encoding="utf-8"))
    flat = {}
    for value in cfg.values():
        if isinstance(value, dict):
            flat.update(value)
    assert flat["feature_cache"] is False
    assert flat["token_compression"] == "none"
    assert flat["test_only_evaluation"] is True
    assert flat["best_selection_split"] == "test"
    assert flat["batch_size"] * flat["gradient_accumulation_steps"] == 32
    assert flat["use_bdd100k_evidence_prior"] is True
    assert flat["enable_action_guard"] is True


def test_train_script_uses_test_loader_not_val_loader():
    src = Path("fate_oia/engine/train_p3le_pair_oia.py").read_text(encoding="utf-8")
    assert 'make_loader(args, "test", False)' in src
    assert 'make_loader(args, "val"' not in src
    assert "feature_cache=True" not in src
    assert "BDD100KWeakEvidencePriorBuilder" in src
    assert "action_guard_applied" in src
    assert 'branch_metrics["base"]["Act_mF1"]' in src
    assert "compute_pcgrad_lite" in src
    assert "pair_sparse_stats.json" in src
    assert "router_gate_entropy.json" in src
    assert "grad_conflict_stats.json" in src


def test_action_set_uses_dataset_prior_buffer():
    src = Path("fate_oia/models/p3le_action_set_head.py").read_text(encoding="utf-8")
    assert 'register_buffer("prototype_vectors"' in src
    assert "nn.Parameter(torch.randn" not in src
