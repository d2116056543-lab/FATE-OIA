import torch
from fate_oia.losses.dice_rank_sketch import DistributionalRankSketch
from fate_oia.engine.train_dice_oia_probe import restore_training_state, summarize_cf_rows


def test_rank_state_round_trip_preserves_update_age():
    sketch=DistributionalRankSketch(); sketch.update(torch.randn(4,4),torch.eye(4),9)
    restored=DistributionalRankSketch(); restored.load_state_dict(sketch.state_dict())
    assert restored.stats(10)==sketch.stats(10)


def test_resume_restores_model_optimizer_sketch_and_counters(tmp_path):
    model=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    sketch=DistributionalRankSketch(); sketch.update(torch.randn(4,4),torch.eye(4),9)
    checkpoint=tmp_path/"resume.pth"
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "rank_sketch":sketch.state_dict(),"epoch":3,"update":17},checkpoint)
    expected=[value.detach().clone() for value in model.parameters()]
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    restored=DistributionalRankSketch()
    epoch,update=restore_training_state(model,optimizer,restored,checkpoint)
    assert (epoch,update)==(4,17)
    assert all(torch.equal(actual,reference) for actual,reference in zip(model.parameters(),expected))
    assert restored.stats(18)==sketch.stats(18)


def test_mechanism_summary_aggregates_all_epochs():
    rows=[]
    for action_id in range(4):
        for index in range(300):
            rows.append({"action_id":action_id,"effect":.2 if index<240 else -.1,
                         "support_hat":.9 if index<240 else .1,"support_target":1.0 if index<240 else 0.0,
                         "counter_hat":.1 if index<240 else .9,"counter_target":0.0 if index<240 else 1.0,
                         "target_signed_contribution":float(index)})
    summary=summarize_cf_rows(rows)
    assert summary["valid_events"]==1200
    assert all(summary["per_action"][str(action)]["count"]==300 for action in range(4))
    assert summary["certificate_positive_rate_lcb95"]>.55
    assert summary["license_prediction_auc"]>.65
