from fate_oia.engine.train_aie_oia import checkpoint_selection_criteria


def test_checkpoint_selection_tracks_genuine_fixed_metrics_separately_from_deploy():
    selected = {
        "final": {"joint": 0.54, "Act_mF1": 0.72, "Exp_mF1": 0.36, "Act_mAP": 0.79, "Exp_mAP": 0.38},
        "deploy": {"joint": 0.57, "Act_mF1": 0.73, "Exp_mF1": 0.41},
    }

    criteria = checkpoint_selection_criteria(selected)

    assert criteria["fixed_joint"] == 0.54
    assert criteria["fixed_action_mF1"] == 0.72
    assert criteria["fixed_reason_mF1"] == 0.36
    assert criteria["deploy_joint"] == 0.57
