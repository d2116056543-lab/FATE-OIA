import torch
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_text_atoms import NativeTextAtomEncoder, native_text_structure_loss

def test_text_atoms_encode_and_grad():
    bank = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
    enc = NativeTextAtomEncoder(bank.atom_vocab)
    out = enc.encode_predicates(bank.specs)
    assert out.shape == (len(bank.specs), 384)
    loss = native_text_structure_loss(enc, bank.specs)["native_text_structure_loss"]
    loss.backward()
    assert any(p.grad is not None for p in enc.parameters())
