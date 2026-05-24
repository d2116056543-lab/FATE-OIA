from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
import vision_transformer as vits
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.models.snna25_head import SNNA25Head


def _build_transform(image_height: int, image_width: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _build_backbone(args, device: torch.device) -> tuple[nn.Module, int]:
    if args.arch not in vits.__dict__:
        raise ValueError(f"Unsupported SNNA ViT arch: {args.arch}")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, embed_dim


@torch.no_grad()
def _extract_features(model: nn.Module, images: torch.Tensor, n_last_blocks: int, avgpool_patchtokens: bool, arch: str) -> torch.Tensor:
    if "vit" in arch:
        intermediate_output = model.get_intermediate_layers(images, n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
        return output
    return model(images)


def _labels_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([batch["action"].float(), batch["reason"].float()], dim=1)


def _limited(dataset, max_samples: int) -> Iterable:
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def _make_loader(args, split: str, shuffle: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=_build_transform(args.image_height, args.image_width),
    )
    max_samples = args.max_train_samples if split == "train" else args.max_val_samples
    ds = _limited(ds, max_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def _criterion(args) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    return nn.BCEWithLogitsLoss()


def _run_epoch(args, model, head, loader, criterion, optimizer, device, train: bool) -> dict:
    head.train(train)
    total_loss = 0.0
    count = 0
    logits_all = []
    labels_all = []
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = _labels_from_batch(batch).to(device, non_blocking=True)
        with torch.no_grad():
            features = _extract_features(model, images, args.n_last_blocks, args.avgpool_patchtokens, args.arch)
        out = head(features)
        loss = criterion(out["logits"], labels)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        count += bs
        logits_all.append(out["logits"].detach().cpu())
        labels_all.append(labels.detach().cpu())
        if step % args.log_every == 0:
            print(json.dumps({"event": "snna25_batch", "train": train, "step": step, "loss": float(loss.item()), "batch_size": bs}), flush=True)
    logits = torch.cat(logits_all, 0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    labels = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    metrics = evaluate_snna25(logits, labels, args.action_dim, threshold_mode="fixed", fixed_threshold=args.eval_threshold)["metrics"]
    return {"loss": total_loss / max(count, 1), "count": count, "metrics": metrics, "logits": logits, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SNNA-25 action+reason head on frozen SNNA/DINO features.")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=4)
    ap.add_argument("--avgpool_patchtokens", action="store_true")
    ap.add_argument("--action_dim", type=int, default=4, choices=[4, 5])
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=224)
    ap.add_argument("--image_width", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, in_dim = _build_backbone(args, device)
    head = SNNA25Head(in_dim, args.action_dim, args.reason_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = _criterion(args)
    train_loader = _make_loader(args, "train", True)
    val_loader = _make_loader(args, "val", False)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        train_stats = _run_epoch(args, backbone, head, train_loader, criterion, optimizer, device, True)
        val_stats = _run_epoch(args, backbone, head, val_loader, criterion, optimizer, device, False)
        score = float(val_stats["metrics"].get("Act_mF1", 0.0)) + float(val_stats["metrics"].get("Exp_mF1", 0.0))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"], "val_metrics": val_stats["metrics"], "selection_score": score}
        history.append(row)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        latest = {"epoch": epoch, "head": head.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "in_dim": in_dim, "best_score": max(best, score)}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        torch.save(val_stats["logits"], out_dir / "val_logits_latest.pt")
        torch.save(val_stats["labels"], out_dir / "val_labels_latest.pt")
        if score >= best:
            best = score
            torch.save(latest, out_dir / "checkpoint_best.pth")
            torch.save(val_stats["logits"], out_dir / "val_logits_best.pt")
            torch.save(val_stats["labels"], out_dir / "val_labels_best.pt")
        print(json.dumps({"event": "snna25_epoch", **row}), flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()