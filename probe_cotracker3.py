from __future__ import annotations

import json
import time

import torch


model = torch.hub.load(r"E:\sbw\deps\co-tracker", "cotracker3_offline", source="local")
model = model.cuda().eval()
video = torch.rand(1, 15, 3, 192, 344, device="cuda") * 255.0
torch.cuda.synchronize()
started = time.perf_counter()
with torch.no_grad():
    tracks, visibility = model(video, grid_size=8)
torch.cuda.synchronize()
print(json.dumps({
    "class": type(model).__name__,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "elapsed_seconds": time.perf_counter() - started,
    "tracks_shape": list(tracks.shape),
    "visibility_shape": list(visibility.shape),
    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
}))
