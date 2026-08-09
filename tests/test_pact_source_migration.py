import torch

from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.models.pact_oia_model import PACTOIAModel


def test_pact_migration_is_source_equivalent_in_compatibility_mode():
    torch.manual_seed(7)
    source = AIEOIAModel(use_mock_dino=True).eval()
    pact = PACTOIAModel(use_mock_dino=True).eval()
    assert pact.migrate_from_aie_state_dict(source.state_dict()) == {"missing_keys": [], "unexpected_keys": []}
    image = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        expected = source(image, action_scale=0.35, reason_scale=0.20)
        actual = pact(
            image, semantic_share_license=1.0, action_scale=0.35,
            reason_budget=0.20, compatibility_mode=True,
        )
    for key in ("action_logits_primary", "action_logits_final", "reason_logits_primary", "reason_logits_final"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=0)


def test_pact_calls_dino_once():
    model = PACTOIAModel(use_mock_dino=True).eval()
    calls = 0
    original = model.dino.forward

    def counted(images):
        nonlocal calls
        calls += 1
        return original(images)

    model.dino.forward = counted
    with torch.no_grad():
        model(torch.randn(1, 3, 360, 640))
    assert calls == 1
