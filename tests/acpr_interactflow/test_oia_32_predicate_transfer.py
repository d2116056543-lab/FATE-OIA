from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.predicate_transfer import TextPredicateTransfer, load_oia_predicate_queries


def test_oia_predicate_transfer_loads_exact_checkpoint_tensor(tmp_path):
    source = torch.randn(32, 384)
    ckpt = tmp_path / "source.pth"
    torch.save({"model": {"predicate_head.predicate_queries": source}}, ckpt)

    loaded, report = load_oia_predicate_queries(ckpt)

    assert torch.allclose(loaded, source.float())
    assert report["source_tensor_key"] == "predicate_head.predicate_queries"
    assert report["source_shape"] == [32, 384]
    assert len(report["source_checkpoint_sha256"]) == 64


def test_text_predicate_transfer_uses_oia_source_and_residual(tmp_path):
    source = torch.randn(32, 384)
    ckpt = tmp_path / "source.pth"
    torch.save({"model": {"predicate_head.predicate_queries": source}}, ckpt)
    names = [f"oia_predicate_{i}" for i in range(32)] + [f"psi_predicate_{i}" for i in range(16)]

    module = TextPredicateTransfer(
        names,
        dim=384,
        source_checkpoint=str(ckpt),
        oia_predicate_names=names[:32],
        require_source_checkpoint=True,
        require_transformer_text=False,
    )
    out = module(torch.randn(2, 48, 384))
    report = module.report()

    assert out["transferred_predicate_tokens"].shape == (2, 48, 384)
    assert out["transfer_gate"].shape == (48,)
    assert report["source_loaded"] is True
    assert report["oia_name_order_verified"] is True
    assert report["oia_transfer_formula"] == "W_o q_oia + W_n E_text(name) + residual"
