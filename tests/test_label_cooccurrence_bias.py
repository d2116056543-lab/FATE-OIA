import torch

from fate_oia.models.label_cooccurrence import build_label_statistics, pmi_bias_matrix
from fate_oia.models.label_correlation import LabelCorrelationBlock


def test_pmi_bias_matrix_is_finite_and_zero_diagonal():
    labels = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    stats = build_label_statistics(labels, smoothing=1.0)
    bias = pmi_bias_matrix(stats, clip=3.0, zero_diagonal=True)
    assert bias.shape == (4, 4)
    assert torch.isfinite(bias).all()
    assert torch.allclose(torch.diag(bias), torch.zeros(4))


def test_label_correlation_accepts_pmi_bias_and_residual_init_zero_is_identity():
    block = LabelCorrelationBlock(
        dim=8,
        num_labels=4,
        num_heads=2,
        bias_mode="pmi",
        residual_init=0.0,
        bias_matrix=torch.zeros(4, 4),
    )
    tokens = torch.randn(2, 4, 8)
    out = block(tokens)
    assert torch.allclose(out, tokens, atol=1e-6)

