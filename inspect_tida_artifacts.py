from pathlib import Path

import torch


root = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")
for name in (
    "traffic_action_attention_test.pt",
    "relational_action_attention_test.pt",
    "relational_reason_attention_test.pt",
    "cotracker_tracks_test.pt",
    "cotracker_motion_features_test.pt",
):
    value = torch.load(root / name, map_location="cpu")
    print(name, tuple(value.shape), value.dtype, float(value.float().min()), float(value.float().max()))
