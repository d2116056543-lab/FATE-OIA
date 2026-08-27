import torch

from fate_oia.engine.export_tida_traffic_evidence_report import (
    compute_action_metrics,
    extract_metric_views,
)


def test_extract_metric_views_uses_locked_deploy_and_corrected_action():
    metrics = {
        "online": {
            "deploy": {
                "image": {"Act_mF1": 0.781, "Exp_mF1": 0.463},
                "video_stable": {
                    "Act_mF1": 0.786,
                    "Act_oF1": 0.81,
                    "Act_mAP": 0.86,
                    "Exp_mF1": 0.464,
                },
            }
        }
    }
    corrector = {"test": {"full_Act_mF1": 0.791}}

    image, video, final = extract_metric_views(metrics, corrector)

    assert image == {"Act_mF1": 0.781, "Exp_mF1": 0.463}
    assert video["Act_mF1"] == 0.786
    assert video["Act_oF1"] == 0.81
    assert video["Act_mAP"] == 0.86
    assert video["Exp_mF1"] == 0.464
    assert final["Act_mF1"] == 0.791
    assert final["Exp_mF1"] == 0.464
    assert "Act_oF1" not in final
    assert "Act_mAP" not in final
    assert final["Act_ranking_metric_status"] == "not_defined_for_discrete_router"
    assert video is not final


def test_extract_metric_views_rejects_missing_locked_deploy_view():
    try:
        extract_metric_views({"online": {"deploy": {}}}, {"test": {}})
    except ValueError as error:
        assert "locked deploy" in str(error)
    else:
        raise AssertionError("missing deploy metrics were accepted")


def test_compute_action_metrics_uses_final_predictions_and_probabilities():
    target = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bool)
    prediction = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
    probability = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])

    metrics = compute_action_metrics(prediction, probability, target)

    assert abs(metrics["Act_mF1"] - (1.0 + 2.0 / 3.0) / 2.0) < 1e-7
    assert abs(metrics["Act_oF1"] - 6.0 / 7.0) < 1e-7
    assert metrics["Act_mAP"] == 1.0
    assert abs(metrics["Act_exact_match"] - 2.0 / 3.0) < 1e-7


def test_compute_action_metrics_omits_map_without_a_deployed_continuous_score():
    target = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    prediction = target.clone()

    metrics = compute_action_metrics(prediction, None, target)

    assert metrics["Act_mF1"] == 1.0
    assert "Act_mAP" not in metrics
    assert metrics["Act_ranking_metric_status"] == "not_defined_for_discrete_router"
