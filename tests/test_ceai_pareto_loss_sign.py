import torch

from fate_oia.losses.ceai_losses import pareto_safety_penalty


def test_pareto_penalty_zero_when_final_better_than_base():
    final = torch.tensor(0.10)
    base = torch.tensor(0.50)
    penalty = pareto_safety_penalty(final, base, margin=0.01)
    assert float(penalty) == 0.0


def test_pareto_penalty_positive_when_final_worse_than_base():
    final = torch.tensor(0.60)
    base = torch.tensor(0.50)
    penalty = pareto_safety_penalty(final, base, margin=0.01)
    assert float(penalty) > 0.05
