import os
import torch

paths = [
    "ckp/classifier.pth.tar",
    "ckp/reference/dino_deitsmall8_pretrain.pth",
    "ckp/reference/dino_deitsmall8_linearweights.pth",
    "ckp/backbone_200.pth",
]

for p in paths:
    print("---", p, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else None)
    if not os.path.exists(p):
        continue
    obj = torch.load(p, map_location="cpu")
    print(type(obj))
    if isinstance(obj, dict):
        print("keys", list(obj.keys())[:30])
        for k in ["epoch", "best_acc", "state_dict", "teacher", "student", "optimizer", "linear"]:
            if k in obj:
                v = obj[k]
                try:
                    n = len(v)
                except Exception:
                    n = v
                print(k, type(v), n)
