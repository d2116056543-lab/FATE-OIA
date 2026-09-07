import random

import numpy as np
import pytest
import torch

from fate_oia.engine.train_coev_oia import verify_resume_or_eval_budget_migration
from fate_oia.utils.coev_checkpoint import atomic_torch_save, capture_rng, restore_rng, verify_resume_identity
from fate_oia.utils.tida_stateful_sampler import TIDAStatefulRandomSampler


def test_atomic_checkpoint_and_rng_roundtrip(tmp_path):
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    state=capture_rng();expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
    restore_rng(state);actual=(random.random(),float(np.random.rand()),float(torch.rand(())))
    assert actual==expected
    target=tmp_path/"checkpoint.pth";atomic_torch_save({"optimizer_boundary":True,"rng":state},target)
    assert target.is_file() and not target.with_suffix(".pth.tmp").exists()
    assert torch.load(target,weights_only=False)["optimizer_boundary"] is True


def test_sampler_cursor_resume_replays_exact_next_sample():
    source=list(range(19));first=TIDAStatefulRandomSampler(source,seed=77)
    prefix=list(iter(first))[:7];first.mark_consumed(7);state=first.state_dict()
    resumed=TIDAStatefulRandomSampler(source,seed=77);resumed.load_state_dict(state)
    assert list(iter(resumed))[0]==list(iter(first))[0]
    assert all(index_seed[0] not in {x[0] for x in prefix} for index_seed in iter(resumed))


def test_resume_identity_mismatch_is_fatal():
    verify_resume_identity({"git_head":"a","config_sha256":"b"},{"git_head":"a","config_sha256":"b"})
    with pytest.raises(RuntimeError,match="identity mismatch"):
        verify_resume_identity({"git_head":"a","config_sha256":"b"},{"git_head":"x","config_sha256":"b"})


def test_only_exact_eval_budget_identity_migration_is_allowed():
    old={"git_head":"old","git_tree":"tree","config_sha256":"old-cfg","data_audit_sha256":"data"}
    new={"git_head":"new","git_tree":"new-tree","config_sha256":"new-cfg","data_audit_sha256":"data"}
    migration={"enabled":True,"from_identity":old}
    assert verify_resume_or_eval_budget_migration(old,new,migration) is True
    with pytest.raises(RuntimeError):
        verify_resume_or_eval_budget_migration({**old,"config_sha256":"other"},new,migration)
    with pytest.raises(RuntimeError):
        verify_resume_or_eval_budget_migration(old,{**new,"data_audit_sha256":"other"},migration)


def test_optimizer_boundary_resume_matches_uninterrupted_next_updates():
    def make():
        torch.manual_seed(11);model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:max(.1,1-step/10));return model,opt,sched
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10;y=torch.arange(16,dtype=torch.float32).reshape(8,2)/10
    def steps(model,opt,sched,start,end):
        for index in range(start,end):
            opt.zero_grad();torch.nn.functional.mse_loss(model(x[index:index+1]),y[index:index+1]).backward();opt.step();sched.step()
    full,full_opt,full_sched=make();steps(full,full_opt,full_sched,0,8)
    partial,opt,sched=make();steps(partial,opt,sched,0,4)
    state={"model":partial.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"rng":capture_rng(),"optimizer_boundary":True}
    resumed,resumed_opt,resumed_sched=make();resumed.load_state_dict(state["model"]);resumed_opt.load_state_dict(state["optimizer"]);resumed_sched.load_state_dict(state["scheduler"]);restore_rng(state["rng"])
    steps(resumed,resumed_opt,resumed_sched,4,8)
    assert all(torch.equal(a,b) for a,b in zip(full.parameters(),resumed.parameters()))
    assert full_sched.state_dict()==resumed_sched.state_dict()
