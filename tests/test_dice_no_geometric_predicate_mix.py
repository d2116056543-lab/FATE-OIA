from pathlib import Path


def test_atom_reconstructor_uses_arithmetic_not_log_map_mixture():
    source=Path("fate_oia/models/dice_atom_reconstructor.py").read_text(encoding="utf-8")
    assert 'torch.einsum("bakp,bpn->bakn", predicate_mixture, pattn)' in source
    assert "pattn.log()" not in source
