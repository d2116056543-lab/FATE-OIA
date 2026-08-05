import torch


def test_adaptive_evidence_has_explicit_null_mass_and_gradient():
    from fate_oia.models.lens_adaptive_evidence import LENSAdaptiveEvidence

    module = LENSAdaptiveEvidence(dim=16, reason_dim=21, layer_count=3)
    nodes = torch.randn(2, 21, 16, requires_grad=True)
    patches = torch.randn(2, 3, 3600, 16, requires_grad=True)
    out = module(nodes, patches)
    total = out["evidence_map"].sum(-1) + out["evidence_null_mass"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert float(out["evidence_temperature"].min()) >= 0.35
    assert float(out["evidence_temperature"].max()) <= 2.0
    out["evidence_token"].sum().backward()
    assert module.query_proj.weight.grad is not None
    assert module.query_proj.weight.grad.abs().sum() > 0


def test_identifiable_state_progress_zero_is_clean_visual_log_odds():
    from fate_oia.models.lens_latent_state import LENSLatentState

    module = LENSLatentState(dim=16, reason_dim=21)
    source = torch.randn(2, 21)
    out = module(torch.randn(2, 21, 16), torch.randn(2, 21, 16), source, torch.rand(2, 21), torch.rand(2, 21), torch.rand(2, 21), progress=0.0)
    assert torch.allclose(out["state_unknown_prob"], torch.zeros_like(source))
    assert torch.equal(out["state_support_logit"], source)
    assert torch.allclose(out["state_prob"].sum(-1), torch.ones_like(source))
