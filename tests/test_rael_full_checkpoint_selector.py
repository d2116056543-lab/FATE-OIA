from types import SimpleNamespace

from fate_oia.engine.supervise_acpr_rael_oia_foreground import _save_full_checkpoints


class _Trainer:
    def state_dict(self):
        return {"sentinel": 1}


def test_full_checkpoint_selector_accepts_evaluator_global_only_branch(tmp_path):
    evaluation = {
        "deploy_metrics": {
            "metrics": {
                "joint": 0.30,
                "action": {"mF1": 0.40},
                "reason": {"mF1": 0.20, "mAP": 0.10},
            }
        },
        "branch_metrics": {
            "branches": [
                {"name": "global_only", "metrics": {"action": {"mF1": 0.35}}}
            ]
        },
        "selection": {"joint": 0.30},
    }
    flags = _save_full_checkpoints(
        output_dir=tmp_path,
        runtime=SimpleNamespace(trainer=_Trainer()),
        epoch=0,
        evaluation=evaluation,
        best={
            "deploy_joint": -1.0,
            "action_mf1": -1.0,
            "exp_mf1": -1.0,
            "exp_map": -1.0,
            "global_action": -1.0,
        },
    )
    assert flags["global_action"] is True
    assert (tmp_path / "checkpoint_latest.pth").exists()
    assert (tmp_path / "checkpoint_best_test_global_action.pth").exists()
