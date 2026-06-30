import torch
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_observation_builder import NativeTextObservationBuilder

def test_observation_train_and_test():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    b = NativeTextObservationBuilder(bank, "configs/acpr_ntmcal_reason_formulas.yaml")
    y = torch.zeros(2,21); y[:,0]=1
    obs = b(y, split="train")
    assert obs["obs_mask"].sum() > 0
    test = b(y, split="test")
    assert test["obs_mask"].sum() == 0
