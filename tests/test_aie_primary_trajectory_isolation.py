import copy
import torch

from fate_oia.models.aie_oia_model import AIEOIAModel


def test_final_losses_do_not_change_primary_trajectory():
    torch.manual_seed(4)
    base = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    full = copy.deepcopy(base)
    image = torch.randn(1, 3, 360, 640)
    for model, include_final in ((base, False), (full, True)):
        optimizer = torch.optim.SGD([p for p in model.foundation.parameters() if p.requires_grad], lr=1e-4)
        out = model(image)
        loss = out["action_logits_primary"].square().mean() + out["reason_logits_primary"].square().mean()
        if include_final:
            loss = loss + out["action_logits_final_train"].square().mean() + out["reason_logits_final_train"].square().mean()
        loss.backward(); optimizer.step()
    for left, right in zip(base.foundation.parameters(), full.foundation.parameters()):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=0)

