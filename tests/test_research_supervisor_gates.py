from fate_oia.engine.supervise_head_zoo_oia import head_zoo_decision
from fate_oia.engine.supervise_oia_research_pipeline import choose_winner_head


def _row(epoch, joint, exp_map, exp_mf1=None, act=0.7):
    return {
        "epoch": epoch,
        "test_joint": joint,
        "test_Exp_mAP": exp_map,
        "test_Exp_mF1": exp_mf1 if exp_mf1 is not None else joint * 0.7,
        "test_Act_mF1": act,
    }


def test_head_zoo_gate_does_not_stop_before_min_gate_epoch():
    rows = [_row(i, 0.40, 0.20) for i in range(1, 8)]
    decision = head_zoo_decision(rows, min_gate_epoch=14, run_c_joint=0.547844, run_c_map=0.367822)
    assert decision.continue_run
    assert decision.reason == "before_min_gate_epoch"


def test_head_zoo_gate_allows_extension_when_recent_metrics_rise_near_runc():
    rows = [_row(i, 0.535 + i * 0.001, 0.355 + i * 0.001) for i in range(1, 16)]
    decision = head_zoo_decision(rows, min_gate_epoch=14, max_epoch=15, extension_epoch=20, run_c_joint=0.547844, run_c_map=0.367822)
    assert decision.continue_run
    assert decision.reason == "near_runc_and_recently_improving"


def test_head_zoo_gate_stops_after_14_when_clearly_below_and_not_improving():
    rows = [_row(i, 0.49, 0.31) for i in range(1, 15)]
    decision = head_zoo_decision(rows, min_gate_epoch=14, run_c_joint=0.547844, run_c_map=0.367822)
    assert not decision.continue_run
    assert decision.reason == "below_gate_after_min_epoch"


def test_choose_winner_head_requires_meaningful_gain_or_map_gain():
    rows = {
        "h0": {"best": _row(14, 0.544, 0.370, exp_mf1=0.382)},
        "h1": {"best": _row(14, 0.551, 0.368, exp_mf1=0.384)},
    }
    winner = choose_winner_head(rows, run_c_joint=0.547844, run_c_map=0.367822, run_c_exp_mf1=0.381301)
    assert winner["head_name"] == "h1"
    assert winner["winner_reason"] == "joint_gain"
