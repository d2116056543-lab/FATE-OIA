from pathlib import Path

import torch
from torch import nn

from fate_oia.engine.train_acpr_meter_oia import (
    _diagnostic_due,
    initialize_model_from_checkpoint,
    load_meter_config,
)
from fate_oia.losses.meter_action_losses import action_delta_pairwise_ranking_loss, meter_action_loss
from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk
from fate_oia.models.meter_calalign_foundation import METERCalAlignFoundation
from fate_oia.models.meter_semantic_action import FactorSpecificActionTransport


def test_transport_keeps_an_absolute_delta_cap_when_visual_logits_explode() -> None:
    transport = FactorSpecificActionTransport(
        dim=8,
        action_dim=2,
        factor_dim=3,
        rank=2,
        max_visual_rms=5.0,
        max_action_delta=1.0,
    )
    output = transport(
        torch.tensor([[1.0e6, -1.0e6], [8.0e5, -8.0e5]]),
        torch.randn(2, 2, 8),
        torch.randn(2, 3, 8),
        torch.ones(2, 3),
        torch.ones(2, 3),
        progress=1.0,
        update_running_stats=True,
    )

    assert float(output["action_correction_kappa"].max()) <= 1.0 + 1e-6
    assert float(output["action_evidence_delta"].abs().max()) <= 1.0 + 1e-6
    assert float(output["action_visual_rms_raw"].max()) >= 1.0e5


def test_delta_ranking_is_smooth_bounded_and_has_finite_gradients() -> None:
    delta = torch.tensor(
        [[0.0, 1.0e6], [1.0e6, -1.0e6], [0.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    loss = action_delta_pairwise_ranking_loss(delta, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert float(loss) <= 2.2
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()


def test_config_preserves_the_named_null_loss_weight() -> None:
    config = load_meter_config(
        Path(__file__).parents[1]
        / "configs"
        / "fate_oia_train_360x640_acpr_meter_oia_v2_tesa.yaml"
    )

    assert config["loss_weights"]["null"] == 0.03
    assert "None" not in config["loss_weights"]


def test_action_logit_guard_bounds_pathological_action_head_outputs() -> None:
    trunk = ACPRLabelTrunk(
        dim=8, action_dim=4, reason_dim=2, action_logit_norm_cap=0.25
    )
    with torch.no_grad():
        trunk.action_visual_head[-1].weight.fill_(1.0e5)
        trunk.reason_to_action.weight.fill_(1.0e5)
    output = trunk(torch.randn(3, 2, 9, 8))

    assert float(output["action_logits_direct"].norm(dim=-1).max()) <= 0.25001
    assert float(output["action_visual_preclip_norm"].max()) > 0.25


def test_foundation_receives_configured_action_logit_guard() -> None:
    foundation = METERCalAlignFoundation(
        dim=384,
        action_dim=4,
        reason_dim=21,
        action_logit_norm_cap=20.0,
        use_mock_dino=True,
    )

    assert foundation.trunk.action_logit_norm_cap == 20.0


def test_weight_only_checkpoint_initialization_excludes_optimizer_state(tmp_path: Path) -> None:
    source = nn.Linear(3, 2)
    checkpoint_path = tmp_path / "epoch1.pth"
    torch.save(
        {
            "epoch": 1,
            "optimizer_step": 880,
            "model": source.state_dict(),
            "optimizer": {"poisoned_momentum": True},
        },
        checkpoint_path,
    )
    target = nn.Linear(3, 2)

    initialization = initialize_model_from_checkpoint(target, checkpoint_path)

    for actual, expected in zip(target.parameters(), source.parameters()):
        assert torch.equal(actual, expected)
    assert initialization == {
        "mode": "weights_only",
        "source_epoch": 1,
        "source_optimizer_step": 880,
        "path": str(checkpoint_path),
    }


def test_guarded_continuation_defaults_preserve_epoch_one_logit_range() -> None:
    config = load_meter_config(
        Path(__file__).parents[1]
        / "configs"
        / "fate_oia_train_360x640_acpr_meter_oia_v2_tesa.yaml"
    )

    assert config["model"]["action_logit_norm_cap"] == 20.0
    assert config["training"]["foundation_grad_clip"] == 0.25
    assert config["training"]["lr_foundation"] == 0.00005


def test_diagnostic_schedule_keeps_primary_test_fast_and_audits_final_epoch() -> None:
    assert not _diagnostic_due(epoch=0, total_epochs=5, interval=5)
    assert not _diagnostic_due(epoch=3, total_epochs=5, interval=5)
    assert _diagnostic_due(epoch=4, total_epochs=5, interval=5)
    assert _diagnostic_due(epoch=0, total_epochs=5, interval=1)

def test_transport_admission_starts_as_an_exact_legacy_equivalent() -> None:
    transport = FactorSpecificActionTransport(dim=8, action_dim=2, factor_dim=3, rank=2)
    output = transport(
        torch.randn(2, 2), torch.randn(2, 2, 8), torch.randn(2, 3, 8),
        torch.ones(2, 3), torch.ones(2, 3), progress=1.0,
    )

    assert torch.equal(output["action_evidence_admission_gate"], torch.ones(2, 2))
    assert torch.allclose(
        output["action_logits_final"],
        output["action_logits_visual"] + output["action_evidence_delta_pre_admission"],
    )


def test_evidence_free_action_correction_does_not_force_a_delta() -> None:
    logits = torch.tensor([[0.4, -0.4]])
    losses = meter_action_loss(
        {
            "action_logits_visual": logits,
            "action_logits_final": logits.clone(),
            "action_transport_support": torch.zeros_like(logits),
            "action_evidence_admission_gate": torch.zeros_like(logits),
        },
        torch.tensor([[1.0, 0.0]]),
    )


def test_legacy_best_checkpoint_allows_only_new_admission_parameters(tmp_path: Path) -> None:
    source = FactorSpecificActionTransport(dim=8, action_dim=2, factor_dim=3, rank=2)
    state = source.state_dict()
    state.pop("action_evidence_admission.weight")
    state.pop("action_evidence_admission.bias")
    checkpoint_path = tmp_path / "legacy-best.pth"
    torch.save({"epoch": 7, "optimizer_step": 100, "model": state}, checkpoint_path)

    target = FactorSpecificActionTransport(dim=8, action_dim=2, factor_dim=3, rank=2)
    initialization = initialize_model_from_checkpoint(target, checkpoint_path)

    assert initialization["mode"] == "weights_only"
    assert torch.equal(target.action_evidence_admission.weight, torch.zeros_like(target.action_evidence_admission.weight))
    assert torch.equal(target.action_evidence_admission.bias, torch.zeros_like(target.action_evidence_admission.bias))
