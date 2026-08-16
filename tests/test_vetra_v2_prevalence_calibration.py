import torch

from fate_oia.utils.aie_calibration import fit_posthoc_thresholds


def test_prevalence_shrinkage_moves_rare_label_toward_score_quantile():
    logits = torch.tensor([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    baseline = fit_posthoc_thresholds(logits, targets, [[0]], shrinkage_support=0.0)
    calibrated = fit_posthoc_thresholds(
        logits,
        targets,
        [[0]],
        shrinkage_support=0.0,
        target_prevalence=torch.tensor([0.5]),
        prevalence_multiplier=torch.tensor([1.0]),
        prevalence_support_prior=100.0,
    )
    expected_quantile = torch.quantile(logits.sigmoid().flatten(), 0.5)
    assert torch.abs(calibrated["threshold_prob"][0] - expected_quantile) < torch.abs(
        baseline["threshold_prob"][0] - expected_quantile
    )
    assert calibrated["calibration_mode"] == "prevalence_shrinkage"


def test_prevalence_multiplier_is_label_specific():
    logits = torch.linspace(-3, 3, 20).unsqueeze(1).repeat(1, 2)
    targets = torch.zeros(20, 2)
    targets[-2:, :] = 1
    result = fit_posthoc_thresholds(
        logits,
        targets,
        [[0, 1]],
        target_prevalence=torch.tensor([0.1, 0.1]),
        prevalence_multiplier=torch.tensor([1.0, 3.0]),
        prevalence_support_prior=100.0,
    )
    assert result["threshold_prob"][1] < result["threshold_prob"][0]
