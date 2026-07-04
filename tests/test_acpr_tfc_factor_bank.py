from fate_oia.models.tfc_factor_bank import TFCFactorBank


def test_tfc_factor_bank_contract():
    bank = TFCFactorBank.from_yaml("configs/acpr_tfc_factors.yaml")
    mats = bank.compatibility_matrices()
    assert bank.num_factors >= 10
    assert mats["factor_to_action_support"].shape == (bank.num_factors, 4)
    assert mats["factor_to_reason_support"].shape == (bank.num_factors, 21)
    assert mats["factor_conflict"].shape == (bank.num_factors, bank.num_factors)
    assert mats["native_similarity"].isfinite().all()
