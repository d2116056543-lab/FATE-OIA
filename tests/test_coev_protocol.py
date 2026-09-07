from pathlib import Path

import numpy as np
import torch
import yaml

from types import SimpleNamespace

from fate_oia.engine.evaluate_coev_oia import _intervention_flips
from fate_oia.engine.train_coev_oia import fixed_multilabel_subset_indices
from fate_oia.utils.coev_contracts import formal_total_updates


def test_fixed_training_protocol_and_no_cache_compression():
    cfg=yaml.safe_load(Path("configs/coev_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["epochs"]==18 and cfg["training"]["effective_batch"]==32
    assert cfg["data"]["history_frames"]==8 and cfg["data"]["epoch_budget"]["epoch_size"]==6400
    assert cfg["data"]["epoch_budget"]["missing_history_quota"]==320
    assert cfg["training"]["total_updates"]==formal_total_updates(6400,epochs=18)[1]==3600
    assert cfg["training"]["warmup_updates"]==180 and cfg["training"]["eval_split"]=="test"
    assert cfg["runtime"]["feature_cache_enabled"] is False and cfg["runtime"]["token_compression"]=="none"
    assert cfg["runtime"]["eval_batch_size"]==8
    assert cfg["training"]["threshold"]==.5 and cfg["training"]["no_metric_early_stop"] is True
    assert cfg["memory_probe"]["selected"]==[16,2]
    assert cfg["memory_probe"]["measured_samples_per_second"]["batch16"] > cfg["memory_probe"]["measured_samples_per_second"]["batch8"]
    assert "COEV_EPOCH_BUDGET_AMENDMENT.md" in cfg["experiment"]["spec_sha256"]


def test_preflight_binds_epoch_budget_amendment_and_dino_runtime_source():
    source=Path("fate_oia/utils/coev_preflight.py").read_text(encoding="utf-8")
    assert '"vision_transformer.py"' in source
    assert '"docs/superpowers/plans/2026-09-07-coev-epoch-budget-amendment.md"' in source
    assert "amendment_hash_matches_spec" in source
    owner=Path("tests/run_coev_owner_audit.py").read_text(encoding="utf-8")
    assert "g[:,13]" not in owner and "g[:,6]" not in owner
    assert "g.shape[1]//2" in owner and "g.shape[1]-1" in owner


def test_foreground_supervisor_is_attached_and_has_heartbeat():
    source=Path("fate_oia/utils/coev_supervisor.py").read_text(encoding="utf-8")
    assert "subprocess.Popen" in source and "coev_supervisor_heartbeat" in source and "child_pid" in source
    for forbidden in ("DETACHED_PROCESS","CREATE_NEW_PROCESS_GROUP","start_new_session","Start-Process","TaskScheduler","nohup"):
        assert forbidden not in source
    launcher=Path("scripts/run_coev_foreground.ps1").read_text(encoding="utf-8")
    assert '$PythonExe = "E:\\Anaconda\\envs\\sbw39\\python.exe"' in launcher
    assert '& $PythonExe @argsList' in launcher


def test_only_test_is_evaluated_and_completion_is_strict():
    source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    assert "evaluate(model, test_loader, device)" in source
    assert 'formal = epochs == cfg["training"]["epochs"] and args.max_train is None and args.max_test is None' in source
    assert "ReduceLROnPlateau" not in source and "test_loader" in source
    assert 'epoch_sampler_stats["pool_covered_after_epoch_rate"]' in source
    assert 'epoch_sampler_stats["exposure_min_after_epoch"]' in source


def test_intervention_flip_metrics_accept_numpy_scores_and_count_all_directions():
    truth=np.array([[1,1,0,0]],dtype=bool)
    full=np.array([[.8,.8,.2,.2]],dtype=np.float32)
    changed=np.array([[.7,.3,.7,.3]],dtype=np.float32)
    result=_intervention_flips(truth,full,changed)
    assert result["full_TP_to_intervention_FN"]==1
    assert result["full_TN_to_intervention_FP"]==1
    assert result["full_FP_to_intervention_TN"]==0
    assert result["full_FN_to_intervention_TP"]==0
    assert np.isclose(result["mean_abs_probability_delta"],.30)


def test_eval_uses_independent_batch_and_bf16_without_changing_training_batch():
    train_source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    eval_source=Path("fate_oia/engine/evaluate_coev_oia.py").read_text(encoding="utf-8")
    assert 'batch_size=int(cfg["runtime"].get("eval_batch_size",1))' in train_source
    assert 'enabled=device.type == "cuda"' in eval_source


def test_epoch_test_budget_is_proportional_fixed_and_final_eval_is_full():
    cfg=yaml.safe_load(Path("configs/coev_oia_v1.yaml").read_text(encoding="utf-8"))
    budget=cfg["data"]["test_epoch_budget"]
    assert budget["enabled"] is True
    assert budget["sample_count"]==1912
    assert abs(budget["sample_count"] / cfg["data"]["test_count"] -
               cfg["data"]["epoch_budget"]["epoch_size"] / cfg["data"]["train_count"]) < .001
    assert abs(cfg["runtime"]["traffic_audit_samples_per_epoch"] / 512 -
               cfg["data"]["epoch_budget"]["epoch_size"] / cfg["data"]["train_count"]) < .001
    assert cfg["runtime"]["final_full_test_metrics"] is True
    source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    assert 'evaluation_scope="fixed_stratified_epoch_subset"' in source
    assert 'output/"final_metrics_full_test.json"' in source


def test_fixed_multilabel_test_subset_is_deterministic_and_preserves_labels():
    records=[]
    for index in range(100):
        action=[int(index % 2 == 0),int(index % 3 == 0),int(index % 5 == 0),int(index % 7 == 0)]
        reason=[int(index % divisor == 0) for divisor in range(2,23)]
        records.append(SimpleNamespace(action=action,reason=reason,history_available=index % 4 != 0))
    first=fixed_multilabel_subset_indices(records,42,77,32)
    second=fixed_multilabel_subset_indices(records,42,77,32)
    assert first==second and len(first)==len(set(first))==42
    selected=torch.tensor([records[i].action+records[i].reason for i in first])
    pool=torch.tensor([r.action+r.reason for r in records])
    assert torch.all(selected.sum(0)>0)
    assert float((selected.float().mean(0)-pool.float().mean(0)).abs().mean()) < .08
