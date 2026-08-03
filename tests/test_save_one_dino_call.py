from __future__ import annotations

import torch

from fate_oia.models.save_oia_model import SAVEOIAModel


def test_save_forward_and_all_decodes_reuse_one_frozen_dino_encoding() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    images = torch.randn(2, 3, 360, 640)

    field = model.encode_images(images)
    first = model.decode_from_field(field, progress=0.0)
    second = model.decode_from_field(field, progress=0.5, diagnostic_modes=("evidence_off",))

    assert model.encode_call_count == 1
    assert model.foundation.ordinary_dino_calls == 1
    assert first["patch_tokens_by_layer"].data_ptr() == field["patch_tokens_by_layer"].data_ptr()
    assert second["patch_tokens_by_layer"].data_ptr() == field["patch_tokens_by_layer"].data_ptr()
    assert all(not parameter.requires_grad for parameter in model.foundation.dino.parameters())


def test_save_stages_global_utility_detail_without_evidence_forward(monkeypatch) -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    events = []
    original_global = model.action_evidence.read_global
    original_detail = model.action_evidence.read_detail
    original_utility = model.utility_bridge.forward

    def read_global(*args, **kwargs):
        events.append("read_global")
        return original_global(*args, **kwargs)

    def utility(*args, **kwargs):
        events.append("utility_bridge")
        return original_utility(*args, **kwargs)

    def read_detail(*args, **kwargs):
        events.append("read_detail")
        return original_detail(*args, **kwargs)

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("SAVE decode must not call action_evidence.forward")

    monkeypatch.setattr(model.action_evidence, "read_global", read_global)
    monkeypatch.setattr(model.utility_bridge, "forward", utility)
    monkeypatch.setattr(model.action_evidence, "read_detail", read_detail)
    monkeypatch.setattr(model.action_evidence, "forward", forbidden_forward)

    model.decode_from_field(model.encode_images(torch.randn(1, 3, 360, 640)), progress=0.25)

    assert events == ["read_global", "utility_bridge", "read_detail"]
