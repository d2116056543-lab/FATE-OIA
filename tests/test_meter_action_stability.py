from __future__ import annotations

import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def _credit(
    visual: torch.Tensor,
    *,
    ownership: torch.Tensor | None = None,
    progress: float = 1.0,
) -> tuple[StateConditionedActionCredit, dict[str, torch.Tensor]]:
    torch.manual_seed(31)
    module = StateConditionedActionCredit(
        dim=8,
        action_dim=2,
        factor_dim=3,
        rank=2,
        correction_fraction=0.20,
        max_action_delta=1.0,
    )
    output = module(
        visual,
        torch.randn(visual.shape[0], 2, 8),
        torch.randn(visual.shape[0], 3, 8),
        torch.softmax(torch.randn(visual.shape[0], 3, 3), dim=-1),
        torch.ones(visual.shape[0], 3),
        torch.ones(3) if ownership is None else ownership,
        progress=progress,
        update_running_stats=True,
    )
    return module, output


def test_heca_credit_keeps_absolute_delta_cap_when_visual_logits_explode() -> None:
    _, output = _credit(
        torch.tensor([[1.0e6, -1.0e6], [8.0e5, -8.0e5]])
    )

    assert float(output["action_correction_kappa"].max()) <= 1.0 + 1e-6
    assert float(output["action_evidence_delta"].abs().max()) <= 1.0 + 1e-6
    assert torch.isfinite(output["action_logits_final"]).all()


def test_heca_credit_kappa_respects_the_action_visual_trust_region() -> None:
    visual = torch.tensor([[0.02, -0.04], [0.03, -0.05]])
    _, output = _credit(visual)
    visual_rms = visual.square().mean(0).sqrt()

    assert torch.all(
        output["action_correction_kappa"].squeeze(0) <= 0.20 * visual_rms + 1e-6
    )


def test_heca_final_action_is_exact_visual_anchor_plus_bounded_credit() -> None:
    _, output = _credit(torch.randn(2, 2))

    torch.testing.assert_close(
        output["action_logits_final"],
        output["action_logits_visual"] + output["action_evidence_delta"],
    )
    torch.testing.assert_close(
        output["action_evidence_delta"],
        output["action_credit_ramp"] * output["action_evidence_delta_unramped"],
    )


def test_heca_zero_progress_preserves_visual_anchor_exactly() -> None:
    _, output = _credit(torch.randn(2, 2), progress=0.0)

    assert torch.count_nonzero(output["action_evidence_delta"]) == 0
    torch.testing.assert_close(
        output["action_logits_final"], output["action_logits_visual"]
    )


def test_heca_non_owned_factor_cannot_receive_action_credit() -> None:
    _, output = _credit(
        torch.randn(2, 2), ownership=torch.tensor([1.0, 0.0, 1.0])
    )

    assert output["action_factor_weights"][..., 1].eq(0).all()
    assert output["action_factor_contribution"][..., 1].eq(0).all()
