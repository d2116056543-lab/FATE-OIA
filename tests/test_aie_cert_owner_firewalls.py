import torch
from fate_oia.models.aie_cert_oia_model import AIECertOIAModel
from fate_oia.losses.aie_cert_loss_registry import exact_owner_parameter_groups


def _grad_sum(parameters): return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def test_owner_exact_cover_and_gradient_firewalls():
    model=AIECertOIAModel(use_mock_dino=True,mock_dim=384); groups=exact_owner_parameter_groups(model)
    output=model(torch.randn(1,3,360,640),action_scale=.2,reason_budget_max=.2,predicate_prior_scale=.2,transport_gamma_cap=.08)
    output['reason_logits_final_train'].sum().backward(retain_graph=True)
    assert _grad_sum(groups['action_evidence'])==0 and _grad_sum(groups['action_contribution'])==0
    model.zero_grad(set_to_none=True); output['name_quality'].sum().backward()
    assert _grad_sum(groups['action_evidence'])==0 and _grad_sum(groups['predicate_visual'])==0
    assert _grad_sum(groups['naming_readout'])>0
