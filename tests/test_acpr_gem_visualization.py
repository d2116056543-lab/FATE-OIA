from fate_oia.utils.acpr_gem_artifacts import build_evidence_chain


def test_evidence_chain_contains_action_evidence_predicate_reason_patch():
    chain = build_evidence_chain(
        action="forward",
        evidence_slot="front_object",
        predicate="front_vehicle_close",
        reason="obstacle: vehicle",
        patch_xy=(40, 22),
    )

    assert chain["action"] == "forward"
    assert chain["evidence_slot"] == "front_object"
    assert chain["predicate"] == "front_vehicle_close"
    assert chain["reason"] == "obstacle: vehicle"
    assert chain["patch_xy"] == [40, 22]
