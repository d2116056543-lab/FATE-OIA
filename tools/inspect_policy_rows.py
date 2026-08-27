import sys

import torch


payload = torch.load(sys.argv[1], map_location="cpu")
print(type(payload).__name__)
if isinstance(payload, dict):
    for key, value in payload.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape), value.dtype)
        else:
            print(key, type(value).__name__, value)
