import torch

from fate_oia.models.dice_atom_reconstructor import DICEAtomReconstructor


def test_coherent_token_is_read_from_the_returned_final_map():
    torch.manual_seed(4)
    module = DICEAtomReconstructor(dim=8, num_layers=2, num_predicates=5, grid_hw=(2, 3))
    evidence = torch.randn(2, 4, 4, 8)
    field = torch.randn(2, 2, 6, 8)
    predicate_attention = torch.softmax(torch.randn(2, 5, 6), -1)
    predicate_probs = torch.sigmoid(torch.randn(2, 5))
    masks = {name: torch.rand(6) for name in module.region_names}
    out = module(evidence, field, predicate_attention, predicate_probs, masks)
    expected = torch.einsum("bakl,bakn,blnd->bakd", out["layer_weights"], out["coherent_map"], out["projected_values"])
    assert torch.allclose(out["coherent_token"], expected, atol=1e-6)
    assert out["predicate_top2_count"].max() <= 2
    assert out["coherent_map"].shape == (2, 4, 4, 6)


def test_sparse_bhattacharyya_path_has_finite_gradients():
    torch.manual_seed(5)
    module=DICEAtomReconstructor(dim=8,num_layers=2,num_predicates=5,grid_hw=(2,3))
    masks={name:torch.ones(6) for name in module.region_names}
    out=module(torch.randn(1,4,4,8),torch.randn(1,2,6,8),
               torch.softmax(torch.randn(1,5,6),-1),torch.sigmoid(torch.randn(1,5)),masks)
    out["predicate_agreement"].sum().backward()
    gradients=[parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
