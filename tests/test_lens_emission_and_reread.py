import torch


def test_ordered_emission_is_identity_at_progress_zero_and_strictly_ordered_afterwards():
    from fate_oia.models.lens_annotation_emission import LENSAnnotationEmission

    module = LENSAnnotationEmission(reason_dim=21, group_ids=torch.tensor([0] * 7 + [1] * 7 + [2] * 7))
    state = torch.softmax(torch.randn(2, 21, 3), dim=-1)
    source = torch.randn(2, 21)
    zero = module(state, source, progress=0.0)
    later = module(state, source, progress=1.0)
    assert torch.equal(zero["reason_logits_formal"], source)
    emission = later["emission_prob"]
    assert torch.all(emission[:, 2] > emission[:, 1])
    assert torch.all(emission[:, 1] > emission[:, 0])


def test_emission_frequency_initialization_targets_ordered_probabilities():
    from fate_oia.models.lens_annotation_emission import LENSAnnotationEmission

    module=LENSAnnotationEmission(reason_dim=21,group_ids=torch.tensor([0]*7+[1]*7+[2]*7))
    module.initialize_from_frequency(torch.linspace(0.02,0.40,21))
    probability=module.emission_probabilities()
    assert torch.all(probability[:,2]>probability[:,1])
    assert torch.all(probability[:,1]>probability[:,0])
    assert float(torch.minimum(probability[:,1]-probability[:,0],probability[:,2]-probability[:,1]).min())>0.02


def test_action_reread_contribution_is_additive_and_unknown_has_no_named_credit():
    from fate_oia.models.lens_action_reread import LENSActionReread

    module = LENSActionReread(dim=16, action_dim=4, reason_dim=21)
    out = module(
        action_nodes=torch.randn(2, 4, 16),
        detail_tokens=torch.randn(2, 3600, 16),
        source_action_attention=torch.softmax(torch.randn(2, 4, 3600), dim=-1),
        evidence_map=torch.softmax(torch.randn(2, 21, 3600), dim=-1),
        evidence_token=torch.randn(2, 21, 16),
        state_prob=torch.softmax(torch.randn(2, 21, 3), dim=-1),
        state_token=torch.randn(2, 21, 16),
        state_embeddings=torch.randn(21, 3, 16),
        action_logits_base=torch.randn(2, 4),
        progress=1.0,
    )
    assert out["action_logits_final"].shape == (2, 4)
    assert torch.allclose(out["factor_contribution_state"][..., 2], torch.zeros_like(out["factor_contribution_state"][..., 2]))
    assert not torch.allclose(out["factor_contribution_state"][..., 0], -out["factor_contribution_state"][..., 1])
    assert float(out["contribution_reconstruction_error"]) < 1e-5
