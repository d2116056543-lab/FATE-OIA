import torch
from fate_oia.losses.dice_losses import directional_effect_loss, directional_license_loss, route_directional_license_targets


def test_effect_loss_prefers_target_signed_direction():
    target=torch.tensor([1.])
    good=directional_effect_loss(torch.tensor([.2]),target,torch.tensor([.2]))
    bad=directional_effect_loss(torch.tensor([-.2]),target,torch.tensor([.2]))
    assert good<bad


def test_directional_effect_is_eventwise_not_cross_event():
    atom=torch.tensor([.2,-.3],requires_grad=True)
    target=torch.tensor([1.,0.])
    effect=torch.tensor([.2,.3])
    loss=directional_effect_loss(atom,target,effect)
    assert loss.item()==0.0
    loss.backward()
    assert torch.equal(atom.grad,torch.zeros_like(atom))


def test_negative_effect_is_retained_as_signed_counter_supervision():
    zero=directional_effect_loss(torch.tensor([0.]),torch.tensor([1.]),torch.tensor([-.4]))
    matching_counter=directional_effect_loss(torch.tensor([-.4]),torch.tensor([1.]),torch.tensor([-.4]))
    assert matching_counter<zero


def test_target_relative_certificate_is_routed_to_raw_action_direction():
    support=torch.tensor([.9,.9]); counter=torch.tensor([.1,.1]); target=torch.tensor([1.,0.])
    raw_support,raw_counter=route_directional_license_targets(support,counter,target)
    assert torch.allclose(raw_support,torch.tensor([.9,.1]))
    assert torch.allclose(raw_counter,torch.tensor([.1,.9]))


def test_license_loss_is_bf16_autocast_safe():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        loss=directional_license_loss(torch.zeros(2,device=device),torch.zeros(2,device=device),
            torch.tensor([.9,.9],device=device),torch.tensor([.1,.1],device=device),torch.tensor([1.,0.],device=device))
    assert torch.isfinite(loss)
