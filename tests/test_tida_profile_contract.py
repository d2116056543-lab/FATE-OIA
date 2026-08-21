import torch
import yaml

from fate_oia.engine.profile_tida_oia import candidate_specs, normalized_growth_per_100_samples


def test_profile_contract_matches_plan():
    config = yaml.safe_load(open("configs/fate_oia_train_tida_oia_v1_15f.yaml", encoding="utf-8"))
    assert config["memory_probe"]["warmup_updates"] == 10
    assert config["memory_probe"]["measured_updates"] == 50
    assert [(row["batch_size"], row["gradient_accumulation_steps"], row["context_chunk_size"])
            for row in candidate_specs(config)] == [(4, 8, 2), (3, 10, 3), (2, 15, 5), (1, 30, 7)]


def test_memory_growth_is_normalized_to_100_measured_microsteps():
    flat = normalized_growth_per_100_samples(torch.tensor([2.0, 2.0, 2.0]))
    rising = normalized_growth_per_100_samples(torch.arange(50, dtype=torch.float32) * 0.001)
    assert abs(flat) < 1e-8
    assert 0.09 < rising < 0.11
