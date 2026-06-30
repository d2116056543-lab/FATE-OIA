import torch
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_observation_builder import NativeTextObservationBuilder
from fate_oia.models.acpr_ntmcal_pu_state import NativeTextPUReasonState

def test_pu_schedule():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    obs = NativeTextObservationBuilder(bank, "configs/acpr_ntmcal_reason_formulas.yaml")
    pu = NativeTextPUReasonState(obs.support, obs.contra)
    y = torch.zeros(2,21); q = torch.rand(2,len(bank.specs)); rho = torch.rand_like(q)
    assert pu(y,q,rho,0)["hard_negative_mask"].sum() == 0
    assert pu(y,q,rho,7)["support_score"].shape == (2,21)
