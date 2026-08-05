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


def test_lens_action_base_uses_clean_latent_log_odds_after_progress_zero():
    from fate_oia.models.lens_oia_model import LENSOIAModel

    model = LENSOIAModel(use_mock_dino=True)
    out = model(torch.randn(1, 3, 360, 640), progress=1.0)
    expected_reason = model.foundation.trunk.reason_to_action(out["clean_observable_log_odds"])
    expected = out["action_fusion_gate_source"] * out["action_visual_source"] + (1 - out["action_fusion_gate_source"]) * expected_reason
    assert torch.allclose(out["action_logits_base"], expected)


def test_all_diagnostic_branches_share_one_encoded_field():
    from fate_oia.models.lens_oia_model import LENSOIAModel

    model=LENSOIAModel(use_mock_dino=True)
    calls=0; original=model.foundation.dino.forward
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        return original(*args,**kwargs)
    model.foundation.dino.forward=counted
    field=model.encode_images(torch.randn(1,3,360,640))
    branches=model.decode_branches_from_field(field,progress=1.0)
    assert calls==1
    assert set(branches)=={"source_calalign","lens_base","lens_final","factor_only","reread_off","latent_state_off","unknown_off","emission_identity","evidence_map_shuffle","wrong_factor"}
