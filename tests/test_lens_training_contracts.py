import torch


def test_accumulation_flushes_last_partial_window():
    from fate_oia.engine.train_lens_oia import should_optimizer_step

    assert not should_optimizer_step(0, 6, 5)
    assert should_optimizer_step(4, 6, 5)
    assert should_optimizer_step(5, 6, 5)
    assert should_optimizer_step(0, 1, 5)


def test_update_schedule_uses_optimizer_updates_not_epochs():
    from fate_oia.engine.train_lens_oia import mechanism_progress

    assert mechanism_progress(0, 100, 0.10) == 0.0
    assert mechanism_progress(5, 100, 0.10) == 0.5
    assert mechanism_progress(10, 100, 0.10) == 1.0


def test_optimizer_groups_are_not_consumed_by_duplicate_check():
    from fate_oia.engine.train_lens_oia import make_optimizer
    from fate_oia.models.lens_oia_model import LENSOIAModel

    model=LENSOIAModel(use_mock_dino=True)
    cfg={"training":{"lr_foundation":1e-3,"lr_adaptive_evidence":1e-3,"lr_latent_state":1e-3,"lr_action_reread":1e-3,"lr_annotation_emission":1e-3,"weight_decay":0.0}}
    optimizer=make_optimizer(model,cfg)
    assert all(len(group["params"])>0 for group in optimizer.param_groups)
    parameter=optimizer.param_groups[0]["params"][0]
    before=parameter.detach().clone(); parameter.grad=torch.ones_like(parameter); optimizer.step()
    assert not torch.equal(before,parameter)


def test_iterative_split_is_disjoint_and_preserves_rare_labels():
    from fate_oia.datasets.lens_splits import make_lens_splits

    labels = torch.zeros(100, 25)
    labels[:, 0] = 1
    labels[::10, 20] = 1
    labels[::20, 24] = 1
    split = make_lens_splits([f"x{i}" for i in range(100)], labels, seed=3)
    groups = [set(split[name]) for name in ("train_main", "train_audit", "train_calib")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(map(len, groups)) == 100
    for name in ("train_audit", "train_calib"):
        indices = split[name]
        assert labels[indices, 20].sum() >= 1
        assert labels[indices, 24].sum() >= 1
