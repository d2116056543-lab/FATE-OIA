import torch

from fate_oia.models.aie_oia_model import AIEOIAModel


def test_ablation_decodes_reuse_one_encoded_field():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True); calls = 0; original = model.foundation.dino.forward
    def counted(x):
        nonlocal calls; calls += 1; return original(x)
    model.foundation.dino.forward = counted
    field = model.encode_images(torch.randn(1, 3, 360, 640))
    model.decode_from_field(field, action_scale=1, reason_scale=1)
    model.decode_from_field(field, action_scale=1, reason_scale=1, predicate_bias_enabled=False)
    assert calls == 1


def test_target_specific_ablation_changes_evidence_without_reencoding():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True).eval()
    calls = 0
    original = model.foundation.dino.forward
    def counted(x):
        nonlocal calls
        calls += 1
        return original(x)
    model.foundation.dino.forward = counted
    field = model.encode_images(torch.randn(2, 3, 360, 640))
    base = model.decode_from_field(field, action_scale=1, reason_scale=1)
    shuffled = model.decode_from_field(field, action_scale=1, reason_scale=1, action_evidence_shuffle=True)
    wrong = model.decode_from_field(field, action_scale=1, reason_scale=1, wrong_action_evidence=True)
    global_only = model.decode_from_field(
        field, action_scale=1, reason_scale=1, local_reread_enabled=False, group_attention_enabled=False
    )
    assert calls == 1
    assert not torch.allclose(base["evidence_token"], shuffled["evidence_token"])
    assert not torch.allclose(base["evidence_token"], wrong["evidence_token"])
    assert not torch.allclose(base["evidence_token"], global_only["evidence_token"])
