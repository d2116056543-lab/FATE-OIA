from fate_oia.engine.eval_precise_oia import EVAL_BRANCHES


def test_eval_contract_includes_raw_deploy_and_semantic_reason_branches():
    assert {"action_direct", "action_final_raw", "action_deploy", "reason_direct", "reason_semantic", "reason_observed", "reason_deploy"} <= set(EVAL_BRANCHES)
