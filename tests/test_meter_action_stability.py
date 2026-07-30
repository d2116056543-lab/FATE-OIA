from pathlib import Path

import torch

from fate_oia.engine.train_acpr_meter_oia import load_meter_config
from fate_oia.losses.meter_action_losses import action_delta_pairwise_ranking_loss
from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk
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
    assert float(output["action_direct_preclip_norm"].max()) > 0.25
