import torch
from fate_oia.models.egcaf_oia_model import EGCafOIAModel


def test_full_model_forward_fake_image_required_keys():
    m = EGCafOIAModel(hidden_dim=32, lightweight_backbone=True)
    y = m(torch.randn(2,3,64,96), return_artifacts=True)
    required = ["action_core_logits","action_final_logits","action_logits","guarded_action_logits","reason_logits","factor_embeddings","factor_region_masks","factor_boxes","factor_type_logits","factor_scores","factor_weights","selected_indices","selected_weights","selected_factor_sources","selected_factor_types","selected_factor_boxes","z_selected_only","z_without_selected","z_without_random","scene_state_logits","lambda_exp","selector_entropy","factor_judge_stats"]
    for k in required:
        assert k in y
    assert y["action_core_logits"].shape == (2,4)
    assert y["reason_logits"].shape == (2,21)
