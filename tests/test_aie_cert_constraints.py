import torch
from fate_oia.losses.aie_cert_constraints import AIECertDualState


def test_duals_are_buffers_updated_after_violation_and_checkpointed():
    d=AIECertDualState(); assert not list(d.parameters())
    d.train(); before=d.lambda_effect.clone(); d.update({'effect':torch.tensor(1.)})
    assert d.lambda_effect > before
    clone=AIECertDualState(); clone.load_state_dict(d.state_dict()); assert torch.equal(clone.lambda_effect,d.lambda_effect)
