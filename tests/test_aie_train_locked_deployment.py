import torch
from torch import nn

from fate_oia.models.aie_train_locked_deployment import AIETrainLockedDeployment


class _FakeAIE(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_action_scale = None
        self.last_reason_action_scale = None
        self.last_reason_scale = None

    def encode_images(self, images):
        return {"images": images}

    def decode_from_field(self, field, *, action_scale, reason_scale, reason_action_scale):
        self.last_action_scale = action_scale.detach().clone()
        self.last_reason_action_scale = reason_action_scale
        self.last_reason_scale = reason_scale
        images = field["images"]
        batch = images.shape[0]
        return {
            "action_logits_final": torch.zeros(batch, 4, device=images.device),
            "reason_logits_final": torch.ones(batch, 21, device=images.device) * reason_scale,
        }


def test_train_locked_deployment_applies_checkpointed_scales_and_thresholds():
    base = _FakeAIE()
    thresholds = [0.5] * 4 + [torch.sigmoid(torch.tensor(0.6)).item()] * 21
    model = AIETrainLockedDeployment(
        base, [0.0, 0.25, 0.75, 0.25], thresholds, reason_scale=0.6
    )

    out = model(torch.randn(2, 3, 8, 8))

    torch.testing.assert_close(base.last_action_scale, torch.tensor([0.0, 0.25, 0.75, 0.25]))
    assert base.last_reason_action_scale == 0.0
    assert base.last_reason_scale == 0.6
    torch.testing.assert_close(out["action_logits_deploy"], torch.zeros(2, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["reason_logits_deploy"], torch.zeros(2, 21), atol=1e-6, rtol=0)
    assert "action_scales" in model.state_dict()
    assert "threshold_prob" in model.state_dict()


def test_train_locked_deployment_rejects_invalid_parameter_shapes():
    try:
        AIETrainLockedDeployment(_FakeAIE(), [1.0, 1.0], [0.5] * 25)
    except ValueError as error:
        assert "action_scales" in str(error)
    else:
        raise AssertionError("invalid action scales were accepted")
