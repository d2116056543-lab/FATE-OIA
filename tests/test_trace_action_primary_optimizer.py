from types import SimpleNamespace
from fate_oia.models.trace_oia_model import TraceOIAModel
from fate_oia.utils.trace_optimizer_groups import build_action_primary_trace_optimizer


def test_action_bias_and_reason_alpha_are_grouped_once_and_backbone_absent():
    model = TraceOIAModel(dim=32, action_dim=4, reason_dim=21, action_final_mode="action_safe_selector")
    args = SimpleNamespace(lr_action_head=3e-4, lr_reason_head=2e-4, lr_transport=1e-4, lr_label_corr=5e-5, lr_reason_alpha=5e-5, lr_action_bias=1e-3, weight_decay=1e-4)
    opt, report = build_action_primary_trace_optimizer(model, args)
    assert "action_bias" in report["param_to_group"]
    assert "reason_alpha" in report["param_to_group"]
    assert report["param_to_group"]["action_bias"] == "action_bias"
    assert report["param_to_group"]["reason_alpha"] == "reason_alpha"
    assert report["missing_trainable"] == []
    assert report["duplicate_trainable"] == []
    grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert grouped == report["trainable_param_tensors"]
