from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank

def test_predicate_bank_required_names():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    assert len(bank.specs) >= 40
    assert "traffic_light_red" in bank.name_to_id
    assert "drivable_right" in bank.name_to_id
