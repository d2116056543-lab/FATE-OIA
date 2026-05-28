import torch

from fate_oia.engine.train_head_zoo_oia import HeadZooModel, build_head, compute_head_zoo_loss


def test_build_head_accepts_required_plan_names():
    for name in [
        "h0_runc_compatible",
        "h1_q2l_decoder",
        "h2_ml_decoder_g8",
        "h3_ctran_masked",
        "h4_runc_mrc_aux",
        "h5_runc_calibrated",
    ]:
        head = build_head(name, dim=384, action_dim=4, reason_dim=21)
        assert head is not None


def test_head_zoo_model_splits_action_reason_logits():
    layers = [torch.randn(2, 8, 384) for _ in range(4)]
    model = HeadZooModel("h1_q2l_decoder", dim=384, action_dim=4, reason_dim=21, n_last_blocks=4)
    out = model(layers)
    assert out["logits"].shape == (2, 25)
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["layer_weights"].shape == (4,)


def test_compute_loss_contains_rank_and_sigmoid_f1_terms():
    out = {
        "action_logits": torch.randn(3, 4, requires_grad=True),
        "reason_logits": torch.randn(3, 21, requires_grad=True),
        "aux_losses": {"aux": torch.tensor(0.25, requires_grad=True)},
    }
    labels = torch.zeros(3, 25)
    labels[:, 0] = 1
    labels[:, 4] = 1
    labels[:, 5] = 1
    total, parts = compute_head_zoo_loss(
        out,
        labels,
        action_dim=4,
        loss_action_weight=1.0,
        loss_reason_weight=1.5,
        reason_ranking_weight=0.05,
        sigmoid_f1_weight=0.05,
    )
    assert total.requires_grad
    assert parts["ranking_loss"] >= 0
    assert parts["sigmoid_f1_loss"] >= 0
    assert parts["aux"] == 0.25
