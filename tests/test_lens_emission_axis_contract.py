import torch


def test_identity_emission_interprets_positive_counter_unknown_state_order():
    from fate_oia.models.lens_annotation_emission import LENSAnnotationEmission

    emission = LENSAnnotationEmission(reason_dim=1, group_ids=torch.tensor([0]))
    # LENSLatentState publishes states as [positive, counter, unknown].
    states = torch.eye(3).view(3, 1, 3)
    result = emission(states, torch.zeros(3, 1), progress=0.0)

    assert torch.allclose(
        result["reason_prob_latent"].flatten(),
        torch.tensor([1.0 - 1e-6, 1e-6, 0.5]),
        atol=2e-6,
    )


def test_responsibility_exposes_both_named_axis_orders():
    from fate_oia.losses.lens_latent_losses import conflict_discounted_responsibility

    payload = conflict_discounted_responsibility(
        state_prob=torch.tensor([[[0.70, 0.20, 0.10]]]),
        emission_prob=torch.tensor([[0.05, 0.50, 0.95]]),
        observed_reason=torch.ones(1, 1),
        action_state_logits=torch.zeros(1, 1, 3, 4),
        action_targets=torch.zeros(1, 4),
        lambda_action=1.0,
    )

    assert torch.allclose(
        payload["gamma_state_order"],
        payload["gamma_emission_order"][..., [2, 0, 1]],
    )


def test_conflict_safe_logits_convert_state_order_before_emission_contract():
    from fate_oia.losses.lens_latent_losses import conflict_safe_reason_logits

    result = conflict_safe_reason_logits(
        state_prob=torch.tensor([[[1.0, 0.0, 0.0]]]),
        source_reason=torch.zeros(1, 1),
        emission_prob=torch.tensor([[0.0, 0.5, 1.0]]),
        observed_reason=torch.ones(1, 1),
        gamma=torch.tensor([[[1.0, 0.0, 0.0]]]),
        conflict=torch.zeros(1, 1),
        alpha_reason=1.0,
    )

    assert result["reason_logits_latent_train"].item() > 10.0
