from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fate_oia.engine.train_sure_oia import evaluate, make_sure_loader
from fate_oia.engine.train_fate_oia import build_backbone
from fate_oia.losses.gradnorm import GradNormBalancer
from fate_oia.models.sure_oia_model import SUREOIAFeatureModel
from fate_oia.utils.sure_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SURE-OIA v2 checkpoint on test split only.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="E:/sbw/BDD-OIA/data")
    ap.add_argument("--raw_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = SUREOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim).to(device)
    balancer = GradNormBalancer().to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if "balancer" in ckpt:
        balancer.load_state_dict(ckpt["balancer"], strict=False)
    loader = make_sure_loader(args, "test", False)
    out_dir = Path(args.output_dir)
    stats = evaluate(args, backbone, model, loader, device, out_dir / "epoch_eval")
    write_json(out_dir / "final_eval_test.json", stats["metrics"])


if __name__ == "__main__":
    main()
