import torch

from fate_oia.engine.train_tida_oia import calibrate_logit_flow_deployment
from fate_oia.models.tida_logit_flow import TIDALogitFlowReader


def _reader(num_labels=3):
    module = TIDALogitFlowReader(num_labels=num_labels, hidden_dim=8, cap=0.05)
    with torch.no_grad():
        module.candidate_output.weight.normal_(0.0, 0.1)
    return module


def _inputs(num_labels=3):
    return (
        torch.randn(2, 3, num_labels),
        torch.randn(2, num_labels),
        torch.tensor([[-1.0, -0.6, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 4, dtype=torch.bool),
    )


def test_logit_flow_train_calib_policy_can_use_centered_candidate_directly():
    module = _reader()
    center = torch.tensor([0.01, -0.02, 0.005])
    module.set_deployment_policy(
        gate=torch.ones(3),
        scale=torch.ones(3),
        cutoff=torch.zeros(3),
        center=center,
        use_utility=torch.zeros(3),
        source="train_calib_oof",
    )

    output = module(*_inputs())

    expected = output["candidate_delta"] - center[None]
    assert torch.allclose(output["centered_candidate_delta"], expected)
    assert torch.allclose(output["deploy_delta"], expected)
    assert output["deployment_source"] == "train_calib_oof"


def test_logit_flow_zero_policy_is_exact_image_fallback():
    module = _reader()
    module.set_deployment_policy(
        gate=torch.zeros(3),
        scale=torch.ones(3),
        cutoff=torch.zeros(3),
        center=torch.zeros(3),
        use_utility=torch.zeros(3),
        source="train_calib_oof",
    )
    inputs = _inputs()

    output = module(*inputs)

    assert torch.count_nonzero(output["deploy_delta"]) == 0
    assert torch.equal(output["deploy_logits"], inputs[1])


def test_reason_logit_flow_policy_is_fit_only_from_train_calib_rows():
    module = _reader(num_labels=3)

    class Model:
        logit_flow_enabled = True
        reason_logit_flow = module
        action_logit_flow = _reader(num_labels=2)

    count = 20
    target = torch.zeros(count, 3)
    target[:10, 0] = 1.0
    image = torch.full((count, 3), -1.0)
    image[:, 0] = -0.01
    candidate = torch.zeros_like(image)
    candidate[:10, 0] = 0.03
    utility = torch.zeros_like(image)
    rows = {
        "image_action": torch.zeros(count, 2),
        "video_action": torch.zeros(count, 2),
        "action_target": torch.zeros(count, 2),
        "image_reason": image,
        "video_reason": image,
        "reason_target": target,
        "logit_flow_action_candidate_delta": torch.zeros(count, 2),
        "logit_flow_action_utility_probability": torch.zeros(count, 2),
        "logit_flow_reason_candidate_delta": candidate,
        "logit_flow_reason_utility_probability": utility,
    }
    fit = calibrate_logit_flow_deployment(
        Model(),
        rows,
        {
            "locked_image_thresholds": [0.5] * 5,
            "locked_image_threshold_source": "train_calib_fixture",
            "logit_flow_policy_scales": [0.0, 1.0],
            "logit_flow_policy_cutoffs": [0.0],
            "logit_flow_policy_oof_folds": 5,
        },
    )

    assert fit["source"] == "train_calib_oof"
    assert fit["test_labels_used"] is False
    assert fit["reason"]["candidate_center_source"] == "train_calib_median"
    assert module.deployment_label_gate[0] == 1
    assert module.deployment_use_utility[0] == 0
    assert torch.count_nonzero(module.deployment_label_gate[1:]) == 0
