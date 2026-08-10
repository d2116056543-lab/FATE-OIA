import torch

from fate_oia.losses.dice_rank_sketch import DistributionalRankSketch


def test_rank_sketch_tracks_each_action_independently_and_is_serializable():
    sketch = DistributionalRankSketch(num_labels=4, quantiles=16)
    logits = torch.tensor([[2., -2., 1., -1.], [-1., 1., -2., 2.]])
    target = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    sketch.update(logits, target, update=7)
    stats = sketch.stats(7)
    assert stats["labels_with_positive"] == 4
    assert stats["labels_with_negative"] == 4
    restored = DistributionalRankSketch(num_labels=4, quantiles=16)
    restored.load_state_dict(sketch.state_dict())
    assert torch.equal(restored.positive_quantiles, sketch.positive_quantiles)
