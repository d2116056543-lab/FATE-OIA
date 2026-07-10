from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.engine.mosaic_schedule import mosaic_phase_controls
from fate_oia.engine.train_acpr_mosaic_ad import (
    _apply_phase,
    build_model_components,
    build_optimizers,
    load_config,
    train_representation_epoch,
)
from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient


ROOT = Path(__file__).resolve().parents[1]


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


class _NoGroundingIndex:
    def lookup(self, _file_name):
        return SimpleNamespace(label_json=None, drivable_map=None)


def test_one_batch_phase_a_training_executes_the_real_integrated_path(tmp_path) -> None:
    torch.manual_seed(101)
    config_path = ROOT / "configs" / "fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml"
    config = load_config(config_path)
    config["model"]["use_mock_dino"] = True
    config["model"]["highres_topk"] = 8
    config["model"]["midres_topk"] = 4
    config["model"]["decoder_layers"] = 1
    config["training"]["print_every"] = 1000
    device = torch.device("cpu")
    model, selective, threshold, action_queue, reason_queue = build_model_components(
        config, config_path, device
    )
    optimizer, _ = build_optimizers(model, selective, threshold, config)
    controls = mosaic_phase_controls(0)
    _apply_phase(model, selective, optimizer, controls)
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    batch = {
        "image": torch.randn(1, 3, 360, 640),
        "action": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        "reason": torch.zeros(1, 21).scatter_(1, torch.tensor([[2]]), 1.0),
        "file_name": ["synthetic.jpg"],
        "image_path": ["synthetic.jpg"],
        "split": ["train"],
    }
    before = model.action_decoder.visual_decoder.classifier_weight.detach().clone()
    rows, updates = train_representation_epoch(
        model=model,
        selective=selective,
        action_queue=action_queue,
        reason_queue=reason_queue,
        loader=[batch],
        optimizer=optimizer,
        action_anchor=MOSAICActionAnchoredGradient(),
        grounding_builder=MOSAICGroundingObservationBuilder(model.schema_bundle["factors"]),
        grounding_index=_NoGroundingIndex(),
        multiview=MOSAICWeakMultiView(factor_names, flip_probability=0.0, seed=1),
        controls=controls,
        config=config,
        device=device,
        epoch=0,
        grad_accum=1,
        global_update=0,
        total_updates=1,
    )
    after = model.action_decoder.visual_decoder.classifier_weight.detach()
    assert updates == 1
    assert not torch.equal(before, after)
    assert action_queue.count == 1
    assert reason_queue.count == 0  # no stale pre-posterior targets may enter the Phase C queue
    assert rows["loss_components.jsonl"]
    assert rows["factor_grounding_stats.jsonl"]
    assert rows["posterior_recovery_stats.jsonl"][-1]["summary"] is True
    assert not _contains_tensor(rows)


def test_one_batch_phase_d_executes_posterior_and_action_anchor() -> None:
    torch.manual_seed(103)
    config_path = ROOT / "configs" / "fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml"
    config = load_config(config_path)
    config["model"].update({"use_mock_dino": True, "highres_topk": 8, "midres_topk": 4, "decoder_layers": 1})
    config["selective_observation"]["synthetic_missing_positive_fraction"] = 1.0
    config["training"]["print_every"] = 1000
    device = torch.device("cpu")
    model, selective, threshold, action_queue, reason_queue = build_model_components(config, config_path, device)
    optimizer, _ = build_optimizers(model, selective, threshold, config)
    controls = mosaic_phase_controls(9)
    _apply_phase(model, selective, optimizer, controls)
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    batch = {
        "image": torch.randn(1, 3, 360, 640),
        "action": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "reason": torch.zeros(1, 21).scatter_(1, torch.tensor([[9]]), 1.0),
        "file_name": ["phase_d.jpg"], "image_path": ["phase_d.jpg"], "split": ["train"],
    }
    rows, updates = train_representation_epoch(
        model=model, selective=selective, action_queue=action_queue, reason_queue=reason_queue,
        loader=[batch], optimizer=optimizer, action_anchor=MOSAICActionAnchoredGradient(),
        grounding_builder=MOSAICGroundingObservationBuilder(model.schema_bundle["factors"]),
        grounding_index=_NoGroundingIndex(), multiview=MOSAICWeakMultiView(factor_names, flip_probability=0.0, seed=2),
        controls=controls, config=config, device=device, epoch=9, grad_accum=1,
        global_update=0, total_updates=1,
    )
    assert updates == 1
    assert rows["action_anchor_stats.jsonl"][0]["available"] if "available" in rows["action_anchor_stats.jsonl"][0] else True
    assert rows["action_anchor_stats.jsonl"][0]["constraint_pass"]
    assert rows["posterior_recovery_stats.jsonl"][-1]["hidden_positive_count"] == 1
    assert rows["selective_observation_stats.jsonl"][0]["posterior_available"] is True
    assert not _contains_tensor(rows)
