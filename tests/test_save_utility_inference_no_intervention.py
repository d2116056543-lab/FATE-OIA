import torch

from fate_oia.models.save_utility_bridge import SAVEUtilityBridge


def test_eval_forward_only_predicts_utility_and_does_not_run_counterfactual_decoder() -> None:
    bridge = SAVEUtilityBridge(dim=8, action_dim=2, factor_dim=3)
    bridge.eval()
    calls = []

    def forbidden_decoder(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("counterfactual decoder must not run during inference")

    output = bridge(
        action_global_token=torch.randn(1, 2, 8),
        predicate_token=torch.randn(1, 3, 8),
        predicate_state_summary=torch.randn(1, 3, 8),
        predicate_reliability=torch.ones(1, 3),
        base_predicate_overlap=torch.rand(1, 2, 3),
        global_detail_query_similarity=torch.rand(1, 2, 3),
        detail_field=torch.randn(1, 32, 8),
        predicate_map=torch.rand(1, 3, 32),
        action_contribution=torch.rand(1, 2, 32),
        base_action_logits=torch.zeros(1, 2),
        action_targets=torch.ones(1, 2),
        optimizer_update=4,
        teacher_decoder=forbidden_decoder,
    )

    assert not calls
    assert output["teacher_plan"] is None
    assert output["utility_teacher_due"] is False
    assert output["utility_prob"].shape == (1, 2, 3)


def _training_inputs():
    return {
        "action_global_token": torch.randn(1, 2, 8),
        "predicate_token": torch.randn(1, 3, 8),
        "predicate_state_summary": torch.randn(1, 3, 8),
        "predicate_reliability": torch.ones(1, 3),
        "base_predicate_overlap": torch.rand(1, 2, 3),
        "global_detail_query_similarity": torch.rand(1, 2, 3),
        "detail_field": torch.randn(1, 32, 8),
        "predicate_map": torch.rand(1, 3, 32),
        "action_contribution": torch.rand(1, 2, 32),
        "base_action_logits": torch.zeros(1, 2),
        "action_targets": torch.ones(1, 2),
        "optimizer_update": 4,
    }


def test_due_training_teacher_requires_decoder_and_cannot_be_disabled() -> None:
    bridge = SAVEUtilityBridge(dim=8, action_dim=2, factor_dim=3)
    bridge.train()
    for extra in ({}, {"run_teacher": False}):
        try:
            bridge(**_training_inputs(), **extra)
        except RuntimeError as error:
            assert "teacher" in str(error).lower()
        else:
            raise AssertionError("due training teacher must fail without a decoder")
