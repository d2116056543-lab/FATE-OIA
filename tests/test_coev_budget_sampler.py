import json
from types import SimpleNamespace

from fate_oia.datasets.coev_video_dataset import requested_times
from fate_oia.utils.coev_budget_sampler import CoEVBudgetedStratifiedSampler


class Records:
    def __init__(self):
        self.records=[]
        for index in range(10):
            action=[float(index%4==label) for label in range(4)]
            reason=[float((index+label)%7==0) for label in range(21)]
            self.records.append(SimpleNamespace(file_name=f"f{index}",action=action,reason=reason))
    def __len__(self): return len(self.records)


def test_nine_frame_quadratic_schedule_keeps_five_second_span():
    times=requested_times(9)
    assert len(times)==9 and times[0].item()==-5 and times[-1].item()==0
    assert times.tolist()==sorted(times.tolist())
    assert abs(times[-2].item()+.078125)<1e-6


def test_budget_sampler_is_unique_stratified_rotating_and_resumable(tmp_path):
    metadata=tmp_path/"novelty.jsonl"
    metadata.write_text("".join(json.dumps({"file_name":f"f{i}","temporal_novelty_score":i/9,
        "history_available":i!=0})+"\n" for i in range(10)),encoding="utf-8")
    data=Records();kwargs=dict(seed=3,metadata_path=metadata,epoch_size=4,
        quotas={"dynamic":2,"balanced":1,"static":1},missing_history_quota=0,candidate_draws=3)
    sampler=CoEVBudgetedStratifiedSampler(data,**kwargs)
    first=[index for index,_ in sampler];stats=sampler.current_stats()
    assert len(first)==len(set(first))==4
    assert stats["stratum_counts"]=={"dynamic":2,"balanced":1,"static":1}
    sampler.mark_consumed(4);sampler.advance_epoch()
    assert abs(sampler.current_stats()["pool_covered_once_rate"]-.4)<1e-6
    state=sampler.state_dict();restored=CoEVBudgetedStratifiedSampler(data,**kwargs);restored.load_state_dict(state)
    assert list(restored)==list(sampler)


def test_budget_sampler_covers_every_pool_member_before_avoidable_repeats(tmp_path):
    metadata=tmp_path/"novelty.jsonl"
    metadata.write_text("".join(json.dumps({"file_name":f"f{i}","temporal_novelty_score":i/9,
        "history_available":i!=0})+"\n" for i in range(10)),encoding="utf-8")
    sampler=CoEVBudgetedStratifiedSampler(Records(),seed=3,metadata_path=metadata,epoch_size=5,
        quotas={"dynamic":2,"balanced":1,"static":2},missing_history_quota=1,candidate_draws=3)
    for _ in range(3):
        sampler.mark_consumed(5)
        sampler.advance_epoch()
    assert sampler.exposure.min().item() >= 1
    stats=sampler.current_stats()
    assert stats["missing_history_count"] == 1
    assert stats["stratum_counts"] == {"dynamic":2,"balanced":1,"static":2}
    assert 0.0 <= stats["history_available_rate"] <= 1.0
    assert set(stats["history_available_by_stratum"]) == {"dynamic","balanced","static"}
