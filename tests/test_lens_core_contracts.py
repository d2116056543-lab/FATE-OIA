import torch


def test_lens_progress_zero_recovers_source_action_and_reason():
    from fate_oia.models.lens_oia_model import LENSOIAModel

    model = LENSOIAModel(use_mock_dino=True)
    out = model(torch.randn(2, 3, 360, 640), progress=0.0)
    assert torch.equal(out["action_logits_base"], out["action_logits_source"])
    assert torch.equal(out["action_logits_final"], out["action_logits_source"])
    assert torch.equal(out["reason_logits_formal"], out["reason_logits_source"])
    assert out["evidence_map"].shape == (2, 21, 3600)
    assert out["action_logits_state_substitution"].shape == (2, 21, 3, 4)


def test_lens_action_does_not_depend_on_emission_parameters():
    from fate_oia.models.lens_oia_model import LENSOIAModel

    model = LENSOIAModel(use_mock_dino=True)
    image = torch.randn(1, 3, 360, 640)
    before = model(image, progress=1.0)["action_logits_final"]
    with torch.no_grad():
        for parameter in model.annotation_emission.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
    after = model(image, progress=1.0)["action_logits_final"]
    assert torch.allclose(before, after, atol=1e-6, rtol=0.0)
