from pathlib import Path

import torch
import yaml

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder
from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_heca_factor_allocation_ignores_uniform_token_rescaling() -> None:
    module = StateConditionedActionCredit(dim=4, action_dim=1, factor_dim=3, rank=2)
    with torch.no_grad():
        module.action_query.weight.copy_(torch.eye(4))
        module.factor_key.weight.copy_(torch.eye(4))
    kwargs = dict(
        action_logits_visual=torch.zeros(1, 1),
        action_nodes=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        factor_state_prob_credit=torch.full((1, 3, 3), 1.0 / 3.0),
        factor_reliability=torch.ones(1, 3),
        factor_action_ownership=torch.ones(3),
        progress=1.0,
    )
    bridge = torch.tensor(
        [[[0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )
    base = module(factor_action_bridge_token=bridge, **kwargs)
    scaled = module(factor_action_bridge_token=bridge * 100.0, **kwargs)
    torch.testing.assert_close(
        base["action_factor_weights"], scaled["action_factor_weights"], atol=1e-5, rtol=1e-5
    )


def test_heca_reason_evidence_residual_removes_constant_label_offset() -> None:
    decoder = METERPrivateReasonDecoder(dim=4, reason_dim=2, action_dim=4)
    with torch.no_grad():
        decoder.correction_vector.fill_(1.0)
    output = decoder(
        reason_logits_calalign=torch.zeros(3, 2),
        reason_nodes=torch.randn(3, 2, 4),
        factor_measurement_token=torch.ones(3, 2, 4),
        factor_reliability=torch.ones(3, 2),
        factor_groundable_mask=torch.ones(2),
        progress=1.0,
        update_running_stats=True,
    )
    torch.testing.assert_close(
        output["reason_evidence_delta"], torch.zeros_like(output["reason_evidence_delta"])
    )


def test_heca_root_cause_guard_config_matches_pilot_fix() -> None:
    config = yaml.safe_load(
        Path("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["action_measurement_grad_scale"] == 0.20
    assert config["loss_weights"]["action_nonreg"] == 0.15
    assert config["loss_weights"]["action_nonreg_boundary_margin"] == 0.05
