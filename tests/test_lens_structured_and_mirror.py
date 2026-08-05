import torch


def test_mirror_roundtrip_restores_all_semantics():
    from fate_oia.datasets.lens_mirror import mirror_lens_batch

    image = torch.randn(2, 3, 12, 20)
    action = torch.randn(2, 4)
    reason = torch.randn(2, 21)
    state = torch.randn(2, 21, 3)
    maps = torch.randn(2, 21, 24)
    once = mirror_lens_batch(image, action, reason, state, maps)
    twice = mirror_lens_batch(once["image"], once["action"], once["reason"], once["state_target"], once["map_target"])
    assert torch.equal(twice["image"], image)
    assert torch.equal(twice["action"], action)
    assert torch.equal(twice["reason"], reason)
    assert torch.equal(twice["state_target"], state)
    assert torch.equal(twice["map_target"], maps)


def test_structured_builder_never_turns_missing_source_into_counter(tmp_path):
    from fate_oia.datasets.lens_structured_evidence import LENSStructuredEvidenceBuilder, UNKNOWN

    reasons = {i: {"support_sources": [f"support_{i}"], "counter_sources": [f"counter_{i}"], "default_region": "objects", "complete_source_required": True} for i in range(21)}
    schema = tmp_path / "schema.yaml"
    import yaml
    schema.write_text(yaml.safe_dump({"reasons": reasons}), encoding="utf-8")
    result = LENSStructuredEvidenceBuilder(schema, grid_hw=(2, 3)).build([{"explicit_attributes": {}, "complete_sources": {}}])
    assert result.state_mask.sum() == 0
    assert torch.all(result.state_target[..., UNKNOWN] == 1)
    assert result.map_mask.sum() == 0

