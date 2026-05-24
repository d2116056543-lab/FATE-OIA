import torch

from fate_oia.engine.eval_counterfactual import counterfactual_delta_summary
from fate_oia.explain.snna_plus import label_attention_to_token_scores, topk_deletion_mask
from fate_oia.grounding.losses import attention_grounding_bce, mask_iou, pointing_game_hit
from fate_oia.grounding.mask_builder import objects_to_mask
from fate_oia.models.reason_to_action_bottleneck import ReasonToActionBottleneck


def test_grounding_mask_and_losses():
    objs = [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}]
    mask = objects_to_mask(objs, image_size=(100, 100), output_size=(10, 10), categories={"car"})
    assert mask.sum() > 0
    attn = mask.clone()
    assert torch.isfinite(attention_grounding_bce(attn, mask))
    assert pointing_game_hit(attn, mask) == 1.0
    assert mask_iou(attn, mask) > 0.9


def test_reason_bottleneck_and_snna_plus_scores():
    bottleneck = ReasonToActionBottleneck(reason_dim=21, action_dim=4)
    assert bottleneck(torch.randn(2, 21)).shape == (2, 4)
    attn = torch.rand(2, 3, 25, 8)
    scores = label_attention_to_token_scores(attn, label_index=4)
    assert scores.shape == (2, 8)
    assert topk_deletion_mask(scores).shape == (2, 8)


def test_counterfactual_delta_summary():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    original = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    masked = torch.zeros_like(original)
    summary = counterfactual_delta_summary(original, masked, labels)
    assert summary["mean_delta_bce"] > 0