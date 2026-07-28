from fate_oia.engine.eval_acpr_meter_oia import ACTION_BRANCHES, REASON_BRANCHES


def test_evaluator_exposes_all_required_branches() -> None:
    assert {"visual", "semantic", "peer", "final", "factor_off", "factor_shuffle", "meta_off"} <= set(ACTION_BRANCHES)
    assert {"calalign", "global", "local", "mix", "final", "annotation_off", "factor_context_off", "map_shuffle", "meta_off"} <= set(REASON_BRANCHES)
