import torch

from fate_oia.utils.precise_gradient_ownership import parameter_ownership, project_target_credit_gradient
from fate_oia.models.precise_oia_model import PRECISEOIAModel


def test_target_credit_projection_preserves_grounding_alignment_and_cap():
    grounding = torch.tensor([3.0, 0.0])
    target = torch.tensor([-5.0, 7.0])
    projected = project_target_credit_gradient(grounding, target)
    assert torch.dot(projected, grounding).item() >= -1e-8
    assert projected.norm().item() <= 0.2 * grounding.norm().item() + 1e-7


def test_latent_evidence_parameters_are_grounding_owned_not_reason_owned():
    model = PRECISEOIAModel(use_mock_dino=True)
    owners = parameter_ownership(model)
    latent_ids = {id(parameter) for parameter in model.evidence_fields.latent_parameters()}
    assert latent_ids <= {id(parameter) for parameter in owners["evidence_core"]}
    assert latent_ids.isdisjoint({id(parameter) for parameter in owners["reason_semantic"]})
    assert latent_ids.isdisjoint({id(parameter) for parameter in owners["reason_latent"]})


def test_main_task_logits_cannot_bypass_projection_to_train_evidence_core():
    model = PRECISEOIAModel(use_mock_dino=True)
    output = model(torch.randn(1, 3, 360, 640))
    evidence_parameters = parameter_ownership(model)["evidence_core"]
    action_grads = torch.autograd.grad(output["action_logits_final_raw"].sum(), evidence_parameters, retain_graph=True, allow_unused=True)
    reason_grads = torch.autograd.grad(output["reason_logits_semantic"].sum(), evidence_parameters, retain_graph=True, allow_unused=True)
    assert all(value is None or value.abs().max().item() == 0.0 for value in action_grads)
    assert all(value is None or value.abs().max().item() == 0.0 for value in reason_grads)


def test_cached_intervention_path_explicitly_allows_projectable_evidence_gradient():
    model = PRECISEOIAModel(use_mock_dino=True)
    output = model(torch.randn(1, 3, 360, 640))
    decoded = model.decode_cached_exchange(
        output["action_tokens_reread"],
        output["reason_tokens_reread"],
        output["explicit_evidence_tokens"],
        output["evidence_reliability"],
        output["reason_latent_delta"],
    )
    grads = torch.autograd.grad(decoded["action_logits"].sum() + decoded["reason_logits"].sum(), parameter_ownership(model)["evidence_core"], allow_unused=True)
    assert any(value is not None and value.abs().sum().item() > 0.0 for value in grads)
