from fate_oia.engine.eval_precise_oia import EVAL_BRANCHES
from pathlib import Path


def test_eval_contract_includes_raw_deploy_and_semantic_reason_branches():
    assert {"action_direct", "action_final_raw", "action_deploy", "reason_direct", "reason_semantic", "reason_observed", "reason_deploy"} <= set(EVAL_BRANCHES)
    for mode in ("explicit_only", "latent_only", "exchange_off", "evidence_shuffled", "reason_token_shuffled", "annotation_off"):
        assert f"action_{mode}" in EVAL_BRANCHES
        assert f"reason_{mode}" in EVAL_BRANCHES


def test_counterfactual_artifact_contains_sign_and_per_target_breakdown():
    source = (Path(__file__).resolve().parents[1] / "fate_oia" / "engine" / "eval_precise_oia.py").read_text(encoding="utf-8")
    for field in ("sign_agreement", "per_action", "per_reason"):
        assert field in source
