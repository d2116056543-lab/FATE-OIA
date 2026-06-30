import torch
from pathlib import Path
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_text_atoms import NativeTextAtomEncoder
from fate_oia.models.acpr_ntmcal_topk_predicate_measurement import NativeTextTopKPredicateMeasurement


def test_topk_measurement_shapes_and_no_dense_expand():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    enc = NativeTextAtomEncoder(bank.atom_vocab, dim=16)
    m = NativeTextTopKPredicateMeasurement(bank, enc, dim=16, topk=5)
    out = m(torch.randn(2, 3, 3600, 16))
    assert out["predicate_q"].shape == (2, len(bank.specs))
    assert out["predicate_topk_indices"].shape[-1] == 5
    assert out["predicate_stats"]["dense_bpnd_materialized"] is False
    src = Path("fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py").read_text(encoding="utf-8")
    assert "expand(-1, p, -1, -1)" not in src
