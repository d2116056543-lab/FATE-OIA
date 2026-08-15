import torch

from fate_oia.losses.aie_loss_registry import owner_trainability
from fate_oia.models.aie_oia_model import AIEOIAModel


def test_reason_only_owner_freezes_every_action_path():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    owner_trainability(model, ("reason_private",))
    assert any(parameter.requires_grad for parameter in model.reason_private.parameters())
    assert not any(parameter.requires_grad for parameter in model.foundation.parameters())
    assert not any(parameter.requires_grad for parameter in model.action_evidence.parameters())
    assert not any(parameter.requires_grad for parameter in model.action_contribution.parameters())


def test_reason_only_update_preserves_action_logits_exactly():
    torch.manual_seed(8)
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    trainable = owner_trainability(model, ("reason_private",))["reason_private"]
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    image = torch.randn(1, 3, 360, 640)
    before = model(image)["action_logits_final"].detach().clone()
    loss = model(image)["reason_logits_final_train"].square().mean()
    loss.backward(); optimizer.step()
    after = model(image)["action_logits_final"].detach()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
