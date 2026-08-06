import torch

from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface


def test_evidence_interface_contract_shapes():
    module = AIEEvidenceInterface(dim=32, num_predicates=32, grid_hw=(4, 5), local_points_per_layer=3)
    out = module(
        torch.randn(2, 4, 32),
        torch.randn(2, 3, 20, 32),
        torch.softmax(torch.randn(2, 32, 20), -1),
        torch.sigmoid(torch.randn(2, 32)),
    )
    assert out["probe_queries"].shape == (2, 4, 4, 32)
    assert out["global_attention"].shape == (2, 4, 4, 20)
    assert out["evidence_token"].shape == (2, 4, 4, 32)
    assert out["evidence_map"].shape == (2, 4, 4, 20)
    assert out["sampling_offsets"].shape == (2, 4, 4, 3, 3, 2)
    assert out["sampling_weights"].shape == (2, 4, 4, 3, 3)
    torch.testing.assert_close(out["evidence_map"].sum(-1), torch.ones(2, 4, 4), atol=1e-5, rtol=0)


def test_probe_chunking_is_numerically_equivalent():
    module = AIEEvidenceInterface(dim=32, num_predicates=32, grid_hw=(4, 5), probe_chunk_size=16).eval()
    arguments = (
        torch.randn(2, 4, 32),
        torch.randn(2, 3, 20, 32),
        torch.softmax(torch.randn(2, 32, 20), -1),
        torch.rand(2, 32),
    )
    with torch.no_grad():
        full = module(*arguments)
        module.probe_chunk_size = 8
        chunked = module(*arguments)
    torch.testing.assert_close(full["evidence_map"], chunked["evidence_map"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full["evidence_token"], chunked["evidence_token"], atol=1e-6, rtol=1e-5)

