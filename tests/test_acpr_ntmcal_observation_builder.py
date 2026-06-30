import torch
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_observation_builder import NativeTextObservationBuilder


def test_observation_builder_train_and_test_batch_size():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    b = NativeTextObservationBuilder(bank, "configs/acpr_ntmcal_reason_formulas.yaml")
    y = torch.zeros(2, 21)
    y[:, 0] = 1
    train = b(y, split="train")
    assert train["obs_mask"].shape == (2, len(bank.specs))
    assert train["source_stats"]["text_obs_positive_count"] > 0
    test = b(None, split="test", batch_size=2)
    assert test["obs_mask"].shape == (2, len(bank.specs))
    assert test["obs_mask"].sum() == 0
    assert test["source_stats"]["test_ignored"] is True
