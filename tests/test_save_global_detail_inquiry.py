import torch

from fate_oia.models.save_action_evidence import SAVEActionEvidence, evidence_ramp


def _inputs(batch: int = 1, actions: int = 4, patches: int = 3600, dim: int = 16):
    torch.manual_seed(7)
    nodes = torch.randn(batch, actions, dim)
    global_field = torch.randn(batch, patches, dim)
    detail_field = torch.randn(batch, patches, dim)
    base_logits = torch.randn(batch, actions)
    base_attention = torch.full((batch, actions, patches), 1.0 / patches)
    return nodes, global_field, detail_field, base_logits, base_attention


def test_save_reads_global_and_detail_fields_with_distinct_full_patch_inquiries():
    nodes, global_field, detail_field, base_logits, base_attention = _inputs()
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)

    output = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )

    assert output["action_global_token"].shape == (1, 4, 16)
    assert output["action_detail_token"].shape == (1, 4, 16)
    assert output["action_global_attention"].shape == (1, 4, 3600)
    assert output["action_detail_attention"].shape == (1, 4, 3600)
    torch.testing.assert_close(
        output["action_global_attention"].sum(dim=-1),
        torch.ones(1, 4),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        output["action_detail_attention"].sum(dim=-1),
        torch.ones(1, 4),
        atol=1e-6,
        rtol=0,
    )

    global_loss = output["action_global_token"].square().mean()
    detail_loss = output["action_detail_token"].square().mean()
    (global_loss + detail_loss).backward()

    assert model.global_inquiry.in_proj_weight.grad is not None
    assert model.global_inquiry.in_proj_weight.grad.abs().sum() > 0
    assert model.detail_query.weight.grad is not None
    assert model.detail_query.weight.grad.abs().sum() > 0


def test_detail_inquiry_is_not_a_reuse_of_global_field():
    nodes, global_field, detail_field, base_logits, base_attention = _inputs()
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    first = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )
    changed_detail = model(
        nodes,
        global_field,
        detail_field + 0.25,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )

    assert not torch.allclose(
        first["action_detail_token"], changed_detail["action_detail_token"]
    )


def test_formal_final_residual_trains_both_inquiries_and_detail_value_context():
    nodes, global_field, detail_field, base_logits, base_attention = _inputs(batch=2)
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    output = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )

    output["action_logits_final"].square().mean().backward()

    parameters = {
        "global": model.global_inquiry.in_proj_weight,
        "detail_query": model.detail_query.weight,
        "detail_key": model.detail_key.weight,
        "detail_value": model.detail_value.weight,
        "detail_output": model.detail_output.weight,
        "action_value": model.patch_action_value.weight,
        "patch_value": model.patch_value.weight,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_detail_value_branch_functionally_changes_formal_signed_evidence():
    nodes, global_field, detail_field, base_logits, base_attention = _inputs()
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    first = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )
    with torch.no_grad():
        model.detail_value.weight.add_(0.10)
    changed = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
        calalign_action_attention=base_attention,
    )

    assert not torch.allclose(first["action_detail_token"], changed["action_detail_token"])
    assert not torch.allclose(first["action_evidence_raw"], changed["action_evidence_raw"])
    assert not torch.allclose(first["action_logits_final"], changed["action_logits_final"])


def test_global_null_bypass_is_functional_when_global_update_is_zero():
    nodes, global_field, detail_field, base_logits, _ = _inputs()
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    with torch.no_grad():
        model.global_inquiry.out_proj.weight.zero_()
        model.global_inquiry.out_proj.bias.zero_()

    first = model(nodes, global_field, detail_field, base_logits)
    changed = model(nodes, global_field + 100.0, detail_field, base_logits)

    torch.testing.assert_close(first["action_global_token"], nodes, atol=0, rtol=0)
    torch.testing.assert_close(first["action_global_bypass"], nodes, atol=0, rtol=0)
    torch.testing.assert_close(
        changed["action_global_token"], first["action_global_token"], atol=0, rtol=0
    )


def test_ramp_gain_kappa_saturation_and_direction_preserving_cap_contract():
    assert evidence_ramp(0.0) == 0.0
    assert evidence_ramp(0.025) == 0.25
    assert evidence_ramp(0.05) == 0.5
    assert evidence_ramp(0.10) == 1.0
    assert evidence_ramp(0.75) == 1.0

    model = SAVEActionEvidence(
        dim=6,
        action_dim=3,
        num_heads=2,
        action_logit_cap=0.5,
    )
    torch.testing.assert_close(
        torch.sigmoid(model.evidence_gain_raw),
        torch.full((3,), 0.05),
        atol=1e-7,
        rtol=0,
    )
    with torch.no_grad():
        model.running_action_rms.copy_(torch.tensor([0.01, 1.0, 10.0]))
        model.evidence_gain_raw.fill_(20.0)
        model.global_inquiry.out_proj.weight.zero_()
        model.global_inquiry.out_proj.bias.zero_()
        model.detail_query.weight.zero_()
        model.detail_key.weight.zero_()
        model.detail_value.weight.copy_(torch.eye(6))
        model.detail_output.weight.copy_(torch.eye(6))
        model.patch_action_value.weight.copy_(50.0 * torch.eye(6))
        model.patch_value.weight.copy_(50.0 * torch.eye(6))

    nodes = torch.ones(1, 3, 6)
    global_field = torch.ones(1, 3600, 6)
    detail_field = torch.ones(1, 3600, 6)
    base_logits = torch.tensor([[30.0, 0.0, 40.0]])
    output = model(nodes, global_field, detail_field, base_logits, progress=0.05)

    torch.testing.assert_close(
        output["action_credit_ramp"], torch.tensor(0.5), atol=0, rtol=0
    )
    torch.testing.assert_close(
        output["action_correction_kappa"],
        torch.tensor([[0.10, 0.20, 1.00]]),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        output["action_evidence_bounded"].abs(),
        output["action_correction_kappa"].expand_as(output["action_evidence_bounded"]),
        atol=1e-5,
        rtol=0,
    )
    assert output["action_logit_uncapped_final"].norm(dim=-1).item() > 0.5
    torch.testing.assert_close(
        output["action_logits_final"].norm(dim=-1),
        torch.tensor([0.5]),
        atol=1e-6,
        rtol=0,
    )
    ratio = output["action_logits_final"] / output["action_logit_uncapped_final"]
    torch.testing.assert_close(ratio[:, :1], ratio[:, 2:], atol=1e-6, rtol=0)

    zero = model(nodes, global_field, detail_field, base_logits, progress=0.0)
    torch.testing.assert_close(zero["action_logits_final"], base_logits, atol=0, rtol=0)
