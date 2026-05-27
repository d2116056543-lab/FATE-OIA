import json
from pathlib import Path

from fate_oia.engine.offline_score_branch_summary import summarize


def test_offline_score_summary_decisions(tmp_path: Path):
    alpha = {"rows":[{"alpha":0.0,"joint":0.50},{"alpha":0.1,"joint":0.501}]}
    threshold = {"threshold_gain_exp_mF1":0.02}
    failure = {"tail_mean_AP":0.2,"tail_best_possible_F1":0.4,"top_failed_reason_indices":[9,12]}
    a=tmp_path/"a.json"; t=tmp_path/"t.json"; f=tmp_path/"f.json"; o=tmp_path/"o.json"
    a.write_text(json.dumps(alpha), encoding="utf-8")
    t.write_text(json.dumps(threshold), encoding="utf-8")
    f.write_text(json.dumps(failure), encoding="utf-8")
    out = summarize(str(a), str(t), str(f), str(o))
    assert out["calibration_is_major_bottleneck"] is True
    assert out["fusion_fix_priority"] is False
    assert out["tail_calibration_problem"] is True
