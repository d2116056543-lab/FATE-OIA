import torch

from fate_oia.engine.evaluate_coev_oia import is_better
from fate_oia.engine.train_coev_oia import build_model, build_optimizer
from fate_oia.losses.coev_losses import CoEVLoss, binary_gce
from fate_oia.models.coev_observers import PredicateObserver, fit_background_affine, inverse_grid_sample_affine
from fate_oia.models.coev_oia_model import CoEVOIAModel
from fate_oia.utils.coev_contracts import CoEVTargets


def test_predicates_are_independent_sigmoids_and_grounding_owner_is_live():
    model=PredicateObserver(dim=8); low=torch.randn(2,1,4,8,requires_grad=True)
    maps=model(low,(2,2)); assert maps.shape==(2,1,8,2,2) and (maps.sum((-1,-2))-1).abs().mean()>.01


def test_affine_pathology_falls_back_to_identity():
    src=torch.zeros(1,11,2);dst=torch.ones_like(src);w=torch.ones(1,11)
    affine,valid=fit_background_affine(src,dst,w)
    assert not valid.item(); assert torch.allclose(affine,torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]))


def test_grid_sample_affine_correspondence_uses_inverse_direction():
    theta=torch.tensor([[[1.,0.,.2],[0.,1.,-.1]]]);source=torch.tensor([[[.4,.3]]])
    output=inverse_grid_sample_affine(theta,source)
    reconstructed=torch.einsum("bij,bnj->bni",theta[:,:,:2],output)+theta[:,:,2].unsqueeze(1)
    assert torch.allclose(reconstructed,source,atol=1e-6)


def test_binary_gce_is_finite_and_best_tie_uses_map_then_epoch():
    assert torch.isfinite(binary_gce(torch.tensor([[30.,-30.]]),torch.tensor([[1.,0.]])))
    a={"joint":.5,"Act_mAP":.6,"Exp_mAP":.4}; b={"joint":.5,"Act_mAP":.5,"Exp_mAP":.4}
    assert is_better(a,b,2,1); assert not is_better(a,a,2,1)


def test_mock_factory_has_one_backbone_and_exact_optimizer_cover():
    cfg={"backbone":{"pretrained_weights":"unused","activation_checkpointing":False},"model":{"history_chunk_size":1,"reason_soft_bias_verified":False},
         "training":{"lr_upper_dino":1e-5,"lr_new_modules":2e-4,"weight_decay":.05}}
    model=build_model(cfg,use_mock_dino=True); opt=build_optimizer(model,cfg)
    assert model.visual_field.activation_checkpointing is False
    ids=[id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
    assert {g["owner"] for g in opt.param_groups} == {"upper_decay","upper_no_decay","new_decay","new_no_decay"}
    assert all(g["weight_decay"] == 0 for g in opt.param_groups if g["owner"].endswith("no_decay"))


def test_task_norm_receives_frozen_layer_gradient_but_measurement_norm_does_not():
    model=CoEVOIAModel("unused",use_mock_dino=True)
    field=model.visual_field
    out=field.encode_frame(torch.randn(1,3,16,16))
    (out["task4"].sum()+out["task8"].sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in field.task_norm.parameters())
    assert all(p.grad is None for p in field.measurement_norm.parameters())
