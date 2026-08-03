import torch

from fate_oia.losses.save_faithfulness_losses import (
    compute_named_unnamed_contributions,
)


def test_named_and_unnamed_contributions_conserve_fp32_and_bf16() -> None:
    for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 5e-4)):
        raw = torch.tensor(
            [[[0.25, -0.50, 0.75, 0.10], [-0.20, 0.30, -0.40, 0.60]]],
            dtype=dtype,
        )
        candidate_weight = torch.tensor(
            [[[0.50, 0.25, 0.15, 0.10], [0.20, 0.40, 0.10, 0.30]]],
            dtype=dtype,
        )
        predicate_map = torch.tensor(
            [[[0.9, 0.1, 0.2, 0.0], [0.2, 0.8, 0.1, 0.3], [0.1, 0.1, 0.7, 0.2]]],
            dtype=dtype,
        )
        eligibility = torch.tensor([1.0, 0.5, 0.0], dtype=dtype)

        output = compute_named_unnamed_contributions(
            raw,
            candidate_weight,
            predicate_map,
            named_eligibility=eligibility,
        )

        expected = raw.float().sum(-1)
        reconstructed = (
            output["action_named_contribution"].sum(-1)
            + output["action_unnamed_contribution"]
        )
        torch.testing.assert_close(reconstructed, expected, atol=tolerance, rtol=0)
        torch.testing.assert_close(
            output["action_responsibility_sum"],
            torch.ones_like(output["action_responsibility_sum"]),
            atol=tolerance,
            rtol=0,
        )
        assert torch.count_nonzero(output["action_named_contribution"][..., 2]) == 0
        assert bool((output["action_unnamed_responsibility"] > 0).all())
        assert float(output["action_conservation_error"].abs().max()) <= tolerance


def test_missing_named_eligibility_fails_closed() -> None:
    try:
        compute_named_unnamed_contributions(
            torch.ones(1, 1, 4),
            torch.full((1, 1, 3), 1.0 / 3.0),
            torch.full((1, 2, 4), 0.25),
        )
    except ValueError as error:
        assert "named eligibility" in str(error).lower()
    else:
        raise AssertionError("missing named eligibility must fail closed")
