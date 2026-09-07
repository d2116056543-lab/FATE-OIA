import torch

from fate_oia.models.coev_evidence_readout import CoEVEvidenceReadout
from fate_oia.models.coev_path_lift import CoEVCoupledPathLift
from fate_oia.utils.coev_contracts import flip_labels, formal_total_updates


def test_flip_is_involution_and_schedule_is_fixed():
    a, r = torch.arange(4), torch.arange(21)
    aa, rr = flip_labels(*flip_labels(a, r))
    assert torch.equal(aa, a) and torch.equal(rr, r)
    assert formal_total_updates(6400, epochs=18) == (200, 3600)


def test_path_lift_shapes_persistence_and_direction():
    lift = CoEVCoupledPathLift()
    t = torch.linspace(-5, 0, 15).unsqueeze(0)
    values = torch.zeros(1, 15, 14)
    valid = torch.ones_like(values, dtype=torch.bool)
    values[..., 0] = 0.7
    out = lift(values, valid, t)
    assert out["unary"].shape == (1, 28, 7)
    assert out["pair"].shape == (1, 210, 14)
    assert out["unary"][0, 0, 2] > 0.69
    assert out["pair"].isfinite().all()


def test_invalid_gap_does_not_create_delta():
    t = torch.tensor([[-5.0, -4.0, -1.0, 0.0]])
    x = torch.zeros(1, 4, 14); x[0, :, 0] = torch.tensor([0.0, 1.0, 10.0, 11.0])
    valid = torch.ones_like(x, dtype=torch.bool); valid[:, 1:3, 0] = False
    out = CoEVCoupledPathLift()(x, valid, t)
    assert out["unary"][0, 14, 3].abs() < 1e-6


def test_directed_signature_area_changes_sign_under_reversal():
    t = torch.tensor([[-4., -3., -2., -1., 0.]])
    square = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    def area(path):
        values = torch.zeros(1, 5, 14); values[0, :, :2] = path
        valid = torch.ones_like(values, dtype=torch.bool)
        return CoEVCoupledPathLift()(values, valid, t)["pair"][0, 105, 11]
    assert torch.allclose(area(square), torch.tensor(20.0), atol=1e-5)
    assert torch.allclose(area(square.flip(0)), torch.tensor(-20.0), atol=1e-5)


def test_readout_reconstructs_and_invalid_factors_are_zero():
    b = 2
    lift = {"unary": torch.randn(b, 28, 7), "pair": torch.randn(b, 210, 14),
            "unary_valid": torch.ones(b, 28, dtype=torch.bool),
            "pair_valid": torch.ones(b, 210, dtype=torch.bool)}
    model = CoEVEvidenceReadout(dim=16)
    out = model(torch.randn(b, 25, 16), lift)
    reconstructed = out["visual_logits"] + out["factor_contribution"].sum(-1)
    assert torch.allclose(out["logits"], reconstructed, atol=1e-6)
    lift["unary_valid"].zero_(); lift["pair_valid"].zero_()
    assert model(torch.randn(b, 25, 16), lift)["factor_contribution"].abs().max() == 0
