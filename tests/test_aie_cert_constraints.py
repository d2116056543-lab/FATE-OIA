import torch
from fate_oia.losses.aie_cert_constraints import AIECertDualState
from fate_oia.losses.aie_cert_losses import evidence_constraints


def test_duals_are_buffers_updated_after_violation_and_checkpointed():
    d=AIECertDualState(); assert not list(d.parameters())
    d.train(); before=d.lambda_effect.clone(); d.update({'effect':torch.tensor(1.)})
    assert d.lambda_effect > before
    clone=AIECertDualState(); clone.load_state_dict(d.state_dict()); assert torch.equal(clone.lambda_effect,d.lambda_effect)


def test_effect_constraint_uses_only_selected_action_atom_for_each_sample():
    contribution = torch.zeros(3, 4, 4)
    contribution[0, 1, 2] = 0.3
    contribution[1, 3, 0] = 0.4
    contribution[2, 0, 1] = 0.5
    output = {"bounded_contribution": contribution, "action_delta": torch.zeros(3, 4),
              "reason_delta": torch.zeros(3, 21)}
    certificate = {"valid_mask": torch.tensor([True, True, True]),
                   "certificate": torch.tensor([0.3, 0.4, 0.5]),
                   "reliability": torch.ones(3), "action_id": torch.tensor([1, 3, 0]),
                   "atom_id": torch.tensor([2, 0, 1])}
    constraints, availability = evidence_constraints(output, certificate, None)
    assert availability["effect"]
    assert constraints["effect"].ndim == 0
    assert torch.isfinite(constraints["effect"])
