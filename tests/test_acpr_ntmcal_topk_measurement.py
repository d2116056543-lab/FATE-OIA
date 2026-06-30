import torch
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_text_atoms import NativeTextAtomEncoder
from fate_oia.models.acpr_ntmcal_topk_predicate_measurement import NativeTextTopKPredicateMeasurement

def test_topk_measurement_shapes():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    enc = NativeTextAtomEncoder(bank.atom_vocab)
    m = NativeTextTopKPredicateMeasurement(bank, enc, topk=8)
    out = m(torch.randn(2,3,3600,384))
    assert out["predicate_q"].shape == (2, len(bank.specs))
    assert out["predicate_topk_attention"].shape[-1] == 8
    assert out["predicate_stats"]["dense_bpnd_materialized"] is False
