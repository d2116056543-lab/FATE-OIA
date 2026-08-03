import inspect

import torch

from fate_oia.losses import save_faithfulness_losses
from fate_oia.models.save_utility_bridge import SAVEUtilityBridge


def test_utility_and_conservation_paths_do_not_materialize_factor_patch_tokens() -> None:
    utility_source = inspect.getsource(SAVEUtilityBridge)
    loss_source = inspect.getsource(save_faithfulness_losses)
    forbidden_shape_markers = ("bfnd", "bafnd", "factor_patch_tokens", "factor_token_by_patch")
    assert not any(marker in utility_source for marker in forbidden_shape_markers)
    assert not any(marker in loss_source for marker in forbidden_shape_markers)

    bridge = SAVEUtilityBridge(dim=8, action_dim=2, factor_dim=3)
    output = bridge(
        action_global_token=torch.randn(2, 2, 8),
        predicate_token=torch.randn(2, 3, 8),
        predicate_state_summary=torch.randn(2, 3, 8),
        predicate_reliability=torch.rand(2, 3),
        base_predicate_overlap=torch.rand(2, 2, 3),
        global_detail_query_similarity=torch.rand(2, 2, 3),
    )
    assert output["predicate_candidate_weight"].shape == (2, 2, 4)
    assert output["utility_logit"].shape == (2, 2, 3)
    assert output["utility_prob"].shape == (2, 2, 3)

