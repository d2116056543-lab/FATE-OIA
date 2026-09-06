import torch
from pathlib import Path

from fate_oia.engine.evaluate_coev_oia import is_better
from fate_oia.engine.train_coev_oia import build_optimizer
from fate_oia.explain.coev_interventions import input_intervention
from fate_oia.models.coev_evidence_readout import CoEVEvidenceReadout
from fate_oia.models.coev_observers import PredicateObserver
from fate_oia.models.coev_path_lift import CoEVCoupledPathLift
from fate_oia.models.coev_oia_model import CoEVOIAModel
from fate_oia.utils.coev_contracts import CoEVInputs


def _lift(values, valid, times):
    return CoEVCoupledPathLift()(values, valid, times)


def test_mutation_frozen_upper_blocks_is_killed():
    model=CoEVOIAModel("unused",use_mock_dino=True)
    assert all(not p.requires_grad for p in model.visual_field.backbone.blocks[:8].parameters())
    assert all(p.requires_grad for p in model.visual_field.backbone.blocks[8:].parameters())


def test_mutation_constant_time_is_killed_by_duration_channel():
    x=torch.zeros(1,3,14);v=torch.ones_like(x,dtype=torch.bool)
    short=_lift(x,v,torch.tensor([[-1.,-.5,0.]]))["unary"][0,14,5]
    long=_lift(x,v,torch.tensor([[-5.,-2.5,0.]]))["unary"][0,14,5]
    assert short != long


def test_mutation_absolute_area_is_killed():
    t=torch.tensor([[-4.,-3.,-2.,-1.,0.]])
    path=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    def area(p):
        x=torch.zeros(1,5,14);x[0,:,:2]=p
        return _lift(x,torch.ones_like(x,dtype=torch.bool),t)["pair"][0,105,11]
    assert area(path)>0 and area(path.flip(0))<0


def test_mutation_spatial_softmax_is_killed():
    maps=PredicateObserver(dim=8)(torch.randn(2,1,4,8),(2,2))
    assert not torch.allclose(maps.sum((-1,-2)),torch.ones(2,1,8),atol=1e-3)


def test_mutation_gap_crossing_is_killed():
    x=torch.zeros(1,4,14);x[0,:,0]=torch.tensor([0.,1.,10.,11.])
    v=torch.ones_like(x,dtype=torch.bool);v[:,1:3,0]=False
    assert _lift(x,v,torch.tensor([[-5.,-4.,-1.,0.]]))["unary"][0,14,3].abs()<1e-6


def test_mutation_missing_signature_half_term_is_killed_by_subdivision():
    coarse=torch.zeros(1,3,14);coarse[0,:,:2]=torch.tensor([[0.,0.],[1.,1.],[2.,0.]])
    fine=torch.zeros(1,5,14);fine[0,:,:2]=torch.tensor([[0.,0.],[.5,.5],[1.,1.],[1.5,.5],[2.,0.]])
    a=_lift(coarse,torch.ones_like(coarse,dtype=torch.bool),torch.tensor([[-2.,-1.,0.]]))["pair"][0,105]
    b=_lift(fine,torch.ones_like(fine,dtype=torch.bool),torch.linspace(-2,0,5).unsqueeze(0))["pair"][0,105]
    assert torch.allclose(a,b,atol=1e-5)


def test_mutation_reason_label_forward_input_is_killed_by_contract():
    assert set(CoEVInputs.__dataclass_fields__) == {"target_rgb","history_rgb","actual_t","valid","geometry_meta"}


def test_mutation_intervention_reuses_no_old_state():
    class Counter:
        def __init__(self): self.calls=0
        def __call__(self, inputs): self.calls+=1; return {"logits":inputs.history_rgb.mean((1,2,3,4)).unsqueeze(-1)}
    model=Counter();inputs=CoEVInputs(torch.ones(1,3,360,640),torch.zeros(1,14,3,256,448),
        torch.linspace(-5,0,15).unsqueeze(0),torch.ones(1,15,dtype=torch.bool),{})
    out=input_intervention(model,inputs,"repeated_last")
    assert model.calls==1 and out["logits"].item()==1


def test_mutation_repeated_history_as_real_is_detected_by_input_dependence():
    class MeanModel:
        def __call__(self, inputs): return {"logits":inputs.history_rgb.mean().reshape(1,1)}
    inputs=CoEVInputs(torch.ones(1,3,360,640),torch.zeros(1,14,3,256,448),
        torch.linspace(-5,0,15).unsqueeze(0),torch.ones(1,15,dtype=torch.bool),{})
    original=MeanModel()(inputs)["logits"]
    repeated=input_intervention(MeanModel(),inputs,"repeated_last")["logits"]
    assert not torch.allclose(original,repeated)


def test_mutation_main_output_swap_is_killed_by_reconstruction():
    model=CoEVEvidenceReadout(dim=8);lift={"unary":torch.randn(1,28,7),"pair":torch.randn(1,210,14),
        "unary_valid":torch.ones(1,28,dtype=torch.bool),"pair_valid":torch.ones(1,210,dtype=torch.bool)}
    out=model(torch.randn(1,25,8),lift)
    assert torch.allclose(out["logits"],out["visual_logits"]+out["factor_contribution"].sum(-1))
    assert not torch.allclose(out["logits"],out["visual_logits"])


def test_mutation_wrong_best_view_is_killed():
    incumbent={"joint":.6,"Act_mAP":.4,"Exp_mAP":.4}; candidate={"joint":.59,"Act_mAP":.99,"Exp_mAP":.99}
    assert not is_better(candidate,incumbent,2,1)


def test_mutation_missing_optimizer_owner_is_killed():
    cfg={"backbone":{"pretrained_weights":"unused"},"model":{"history_chunk_size":1,"reason_soft_bias_verified":False},
         "training":{"lr_upper_dino":1e-5,"lr_new_modules":2e-4,"weight_decay":.05}}
    model=CoEVOIAModel("unused",use_mock_dino=True);opt=build_optimizer(model,cfg)
    owned={id(p) for group in opt.param_groups for p in group["params"]}
    assert owned=={id(p) for p in model.parameters() if p.requires_grad}


def test_mutation_early_train_completed_is_killed_by_formal_guard():
    source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    assert 'formal = epochs == cfg["training"]["epochs"] and args.max_train is None and args.max_test is None' in source
    assert "if formal:" in source and 'output/"TRAIN_COMPLETED.json"' in source


def test_mutation_test_feedback_into_scheduler_is_killed_by_source_boundary():
    source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    assert "ReduceLROnPlateau" not in source and "metrics[\"joint\"]" not in source
